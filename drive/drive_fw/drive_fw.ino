#include <SimpleFOC.h>
#include <Preferences.h>
#include <Update.h>
// Recon drone drive firmware. Pi <-> ESP32 over serial (USB/CH340 and/or header UART2), 115200 8N1, newline-terminated ASCII.
// Pi -> board:
//   V <left> <right>   wheel speed targets, rad/s at the motor shaft (+ = forward). Resend at >= 5 Hz.
//   S                  stop now (no ramp, motors released)
//   R                  reprint the last self-test results (stored in flash)
//   U <bytes> <md5>    firmware update (see serialUpdate; host side is pi/fw_update.py)
//   A <rad/s^2>        ramp rate until reboot (default ACCEL), for kick tests
//   K <rate> <tilt>    hub damping gains until reboot (default DAMP_RATE DAMP_TILT; 0 0 = off), for floor tuning
// Board -> Pi, every STATUS_MS:
//   T <ms> <angL> <angR> <velL> <velR> <state>   angles rad (cumulative, + = forward), velocities rad/s, both
//     relative to the ground (encoder + hub rotation) when HUB_COMP is on and the IMU works
//   state: IDLE, RUN, WDOG (stopped, no V for WATCHDOG_MS), FAULT (initFOC failed, will not drive),
//          NOVIN (no motor supply at boot; waiting, motors not initialised)
//   I <ms> <ax> <ay> <az> <gx> <gy> <gz> <loopHz> <imuErr> <rejL> <rejR>
//     accel g, gyro deg/s (sensor axes; all 0 if no MPU6050), main loop rate, failed IMU reads, rejected encoder reads
//   H <ms> <tilt> <rate> <scale> <damp>   hub tilt from rest (deg), hub rate (deg/s), drive scale 0..1 from the tilt
//     limiter, damping correction (rad/s)
// Board -> Pi, every POWER_MS:
//   P <ms> <vin> <temp>   motor supply V (filtered, as used for PWM scaling), IMU die temperature C (nan if no IMU)
// Lines starting with # are info/errors; the Pi should ignore or log them.
#define POLE_PAIRS   7       // supply voltage for PWM scaling is measured at boot (VIN_MEA), so 0.9 V stays 0.9 V on 12 V or 4S
#define VOLT_LIMIT   1.2    // 0.9 V oscillated with P0.2 I2 Tf0.01 (2026-10-18); retrying with softer PID (2026-10-21). 1.2 V for the yoga mat lip (2026-11-05)
                             // 1.0 V: PLA track on the right struggled to drive at 0.9 (2026-10-30)
#define RECAL        false   // true = measure zero and direction at boot, wheels off the ground. Redo if a motor or magnet moves
#define ZERO_L       3.03    // measured wheels free 2026-11-05 (2.97, 3.09). On the floor the kickstand stops the
#define ZERO_R       0.5384  // alignment spin and the left zero read up to 3.92, so these are fixed instead
#define MAX_VEL      15.0   // rad/s clamp (~0.75 m/s on 100 mm wheels)
#define ACCEL        5.0     // rad/s^2 ramp on V targets; S and watchdog stops are immediate. Was 20: the kick
                             //   swung the hub (no tail, centre of mass only 8 mm below the axle) (2026-10-30)
#define HUB_COMP     true    // control wheel speed over the ground (encoder + hub rotation from the IMU), not vs the hub
#define HUB_DAMP     true    // while driving, ease the push as the hub swings back and add it as it swings forward
#define DAMP_RATE    0.3     //   wheel rad/s per hub rad/s, and...
#define DAMP_TILT    1.0     //   ...per hub rad of tilt from rest. Weak start, untested on the floor (2026-11-01)
#define DAMP_MAX     3.0     //   rad/s cap on the correction
#define TILT_SOFT    30.0    // deg of hub tilt from rest where the drive starts easing off...
#define TILT_MAX     60.0    //   ...down to nothing here, so the hub can't roll over
#define WATCHDOG_MS  300
#define STATUS_MS    50
#define POWER_MS     250     // VIN read + P line
#define INVERT_L     true    // motors are mirrored on the chassis: flip so + drives forward on each side
#define INVERT_R     true    // set from the assembled drone in the vice (2026-10-30); bench guesses were L false, R true
#define GATE_OFF_IDLE true   // drop shared gate-driver supply when both wheels are released (silent, saves power)
#define M_EN         12      // shared gate-driver supply enable for BOTH motors (GPIO22 is just CS0)
#define IMU_ON       true    // MPU6050 on its own header (IO4 SDA, IO15 SCL), software I2C. Skipped if not found
#define IMU_MS       20      // IMU read period (50 Hz); each 14-byte read takes ~1 ms
#define MPU          0x68
#define IMU_SDA      4       // on Wire (shared with M0 encoder) the IDF I2C driver crashed or hung on noisy IMU
#define IMU_SCL      15      //   reads under motor drive (2026-10-21); software I2C just fails the read
#define VIN_PIN      13      // VIN_MEA divider. initFOC on USB power alone browned out USB every ~2 s (2026-10-22)
#define VIN_DIV      8.46    // VIN / pin voltage, calibrated on the 12 V bench PSU (2026-10-22)
#define VIN_MIN      9.0     // V; USB alone backfeeds ~4.3 V onto VIN
#define SELFTEST     false   // true = floor test (2026-10-27): every power-up, 10 s after READY, drives SEQ unless any
                             // serial byte arrives first (PC/Pi takes over). Results saved to flash, printed at next boot.
                             // Off since 2026-10-30: the Pi drives it.
#define LINK_RX      16      // 4-pin header "RXD" (Pi TXD, pin 8). The header is NOT the USB serial (probed 2026-10-30)
#define LINK_TX      17      // 4-pin header "TXD" (Pi RXD, pin 10)

// Rejects single bad I2C reads (junk angle jumps) so they can't kick the velocity loop
class FilteredAS5600 : public MagneticSensorI2C {
 public:
  FilteredAS5600() : MagneticSensorI2C(AS5600_I2C) {}
  uint32_t rejected = 0;
  float getSensorAngle() override {
    float a = MagneticSensorI2C::getSensorAngle();
    if (last >= 0 && bad < 3) {
      float d = fabsf(a - last); if (d > _PI) d = _2PI - d;
      if (d > 0.5f) { bad++; rejected++; return last; }
    }
    bad = 0; last = a; return a;
  }
 private:
  float last = -1; uint8_t bad = 0;
};

BLDCMotor motorL(POLE_PAIRS), motorR(POLE_PAIRS);           // M0 = left, M1 = right: driving from the phone, + left
BLDCDriver3PWM driverL(32, 33, 25), driverR(26, 27, 14);     //   - right turned the drone CCW, so M1/M0 were swapped (2026-10-30)
FilteredAS5600 sensorL, sensorR;
const float sgnL = INVERT_L ? -1 : 1, sgnR = INVERT_R ? -1 : 1;
float targetL = 0, targetR = 0, cmdL = 0, cmdR = 0, accel = ACCEL, kRate = DAMP_RATE, kTilt = DAMP_TILT;
uint32_t lastV = 0;
bool fault = false, wdog = false, okL = false, okR = false;   // a failed motor is skipped; V still drives the other (bench tests)
float supplyV = 12, imuTemp = 0;
bool hostSeen = false;   // any serial input cancels the self-test

// Everything printed goes to both USB (Serial) and the Pi header (Serial2); commands are read from both
class Tee : public Print {
 public:
  size_t write(uint8_t c) override { Serial.write(c); return Serial2.write(c); }
  size_t write(const uint8_t *b, size_t n) override { Serial.write(b, n); return Serial2.write(b, n); }
} Out;

bool setupMotor(BLDCMotor &m, BLDCDriver3PWM &d, MagneticSensorI2C &s, TwoWire &bus, const char *name, float zero, Direction dir) {
  s.init(&bus);
  d.voltage_power_supply = supplyV; d.voltage_limit = VOLT_LIMIT; d.init();
  m.linkDriver(&d); m.linkSensor(&s);
  m.voltage_limit = VOLT_LIMIT; m.voltage_sensor_align = VOLT_LIMIT;
  m.PID_velocity.P = 0.1; m.PID_velocity.I = 1; m.PID_velocity.D = 0; m.LPF_velocity.Tf = 0.02;
  m.controller = MotionControlType::velocity;
  m.init();
  if (!RECAL) { m.zero_electric_angle = zero; m.sensor_direction = dir; }   // skips the alignment spin
  bool ok = m.initFOC();
  bus.beginTransmission(0x36);                // skipping alignment skips its encoder check, so ask the AS5600 directly
  if (bus.endTransmission() != 0) { Out.printf("# %s encoder not answering\n", name); ok = false; }
  Out.printf("# %s initFOC %s  zero_electric_angle %.4f  dir %d\n", name, ok ? "OK" : "FAILED",
                m.zero_electric_angle, (int)m.sensor_direction);
  if (!ok) { s.update(); Out.printf("# %s encoder angle now %.3f rad\n", name, s.getAngle()); }
  m.disable();
  return ok;
}

// De-energise at target 0 (velocity hold at 0 buzzes against cogging). move() still updates angle/velocity.
void moveOrRelease(BLDCMotor &m, float t) {
  if (t == 0) { if (m.enabled) m.disable(); }
  else if (!m.enabled) { m.PID_velocity.reset(); m.enable(); }
  m.move(t);
}

void gate(bool on) {
  static bool g = true;
  if (!GATE_OFF_IDLE || on == g) return;
  g = on; digitalWrite(M_EN, on ? HIGH : LOW);
  if (on) delay(2);  // let the gate-driver supply come up before PWM
}

void stopNow() { targetL = targetR = cmdL = cmdR = 0; }

void printSelftest();
float vin();

// U <bytes> <md5>: new app image (drive_fw.ino.bin) over this port. Motors stop, then the host sends 1024-byte
// chunks and waits for "#A" after each. Written to the spare OTA slot; reboots into it only if the MD5 matches.
void serialUpdate(Stream &p, char *args) {
  char *e; uint32_t size = strtoul(args, &e, 10);
  while (*e == ' ') e++;
  stopNow();
  if (okL) moveOrRelease(motorL, 0);
  if (okR) moveOrRelease(motorR, 0);
  gate(false);
  if (!size || strlen(e) != 32 || !Update.begin(size)) { p.printf("# UPDATE FAIL begin %s\n", Update.errorString()); return; }
  Update.setMD5(e);
  p.println("# UPDATE ready");
  uint8_t buf[1024]; uint32_t got = 0;
  while (got < size) {
    size_t want = min<uint32_t>(sizeof(buf), size - got), n = 0;
    uint32_t t = millis();
    while (n < want && millis() - t < 3000) { if (p.available()) n += p.readBytes(buf + n, want - n); else delay(1); }
    if (n < want) { Update.abort(); p.println("# UPDATE FAIL timeout"); return; }
    if (Update.write(buf, n) != n) { p.printf("# UPDATE FAIL write %s\n", Update.errorString()); Update.abort(); return; }
    got += n; p.println("#A");
  }
  if (!Update.end()) { p.printf("# UPDATE FAIL end %s\n", Update.errorString()); return; }
  p.println("# UPDATE OK, rebooting"); p.flush(); delay(100);
  ESP.restart();
}

void handle(char *l, Stream &p) {
  if (l[0] == 'S') { stopNow(); wdog = false; return; }
  if (l[0] == 'R') { printSelftest(); return; }
  if (l[0] == 'U') { serialUpdate(p, l + 1); return; }
  if (l[0] == 'A') { accel = constrain(strtof(l + 1, NULL), 1.0f, 1000.0f); Out.printf("# ACCEL %.0f\n", accel); return; }
  if (l[0] == 'K') {
    char *e; kRate = constrain(strtof(l + 1, &e), 0.0f, 5.0f); kTilt = constrain(strtof(e, NULL), 0.0f, 10.0f);
    Out.printf("# DAMP rate %.2f tilt %.2f\n", kRate, kTilt); return;
  }
  if (l[0] == 'V') {
    char *p = l + 1, *e1, *e2;
    float a = strtof(p, &e1), b = strtof(e1, &e2);
    if (e1 == p || e2 == e1) { Out.println("# ERR bad V"); return; }
    if (!okL && !okR) { Out.println("# ERR fault, not driving"); return; }
    targetL = constrain(a, -MAX_VEL, MAX_VEL); targetR = constrain(b, -MAX_VEL, MAX_VEL);
    lastV = millis(); wdog = false;
    return;
  }
  if (l[0]) Out.println("# ERR unknown command");
}

void readPort(Stream &p, char *buf, uint8_t &n) {
  while (p.available()) {
    char c = p.read(); hostSeen = true;
    if (c == '\r') continue;
    if (c != '\n') { if (n < 47) buf[n++] = c; continue; }
    buf[n] = 0; n = 0;
    handle(buf, p);
  }
}

void readSerial() {
  static char b0[48], b2[48]; static uint8_t n0 = 0, n2 = 0;
  readPort(Serial, b0, n0); readPort(Serial2, b2, n2);
}

float ramp(float cur, float tgt, float step) { return cur + constrain(tgt - cur, -step, step); }

bool imuOk = false;
float imu[6];   // ax ay az (g), gx gy gz (deg/s)
float accF[3];  // low-passed accel, for body tilt in the self-test

// Software I2C (open drain, ~150 kHz). Pull-ups are on the GY-521. Any failure returns false, never blocks long.
void sw(uint8_t p, bool v) { digitalWrite(p, v); delayMicroseconds(2); }
bool sclHigh() {
  digitalWrite(IMU_SCL, HIGH);
  uint32_t t = micros();
  while (!digitalRead(IMU_SCL)) if (micros() - t > 500) return false;   // clock stretch or stuck
  delayMicroseconds(2); return true;
}
void i2cStart() { sw(IMU_SDA, 1); sclHigh(); sw(IMU_SDA, 0); sw(IMU_SCL, 0); }
void i2cStop() { sw(IMU_SDA, 0); sclHigh(); sw(IMU_SDA, 1); }
void i2cRecover() { sw(IMU_SDA, 1); for (int i = 0; i < 9; i++) { sclHigh(); sw(IMU_SCL, 0); } i2cStop(); }  // free a stuck slave
bool i2cWrite(uint8_t b) {   // true = ACK
  for (int i = 7; i >= 0; i--) { sw(IMU_SDA, b >> i & 1); if (!sclHigh()) return false; sw(IMU_SCL, 0); }
  sw(IMU_SDA, 1); if (!sclHigh()) return false;
  bool ack = !digitalRead(IMU_SDA); sw(IMU_SCL, 0); return ack;
}
uint8_t i2cRead(bool ack) {
  uint8_t b = 0; sw(IMU_SDA, 1);
  for (int i = 0; i < 8; i++) { sclHigh(); b = b << 1 | digitalRead(IMU_SDA); sw(IMU_SCL, 0); }
  sw(IMU_SDA, !ack); sclHigh(); sw(IMU_SCL, 0); sw(IMU_SDA, 1); return b;
}
bool mpuRead(uint8_t reg, uint8_t *buf, uint8_t n) {
  i2cStart();
  if (!i2cWrite(MPU << 1) || !i2cWrite(reg)) { i2cStop(); return false; }
  i2cStart();   // repeated start
  if (!i2cWrite(MPU << 1 | 1)) { i2cStop(); return false; }
  for (uint8_t i = 0; i < n; i++) buf[i] = i2cRead(i < n - 1);
  i2cStop(); return true;
}
bool mpuWrite(uint8_t reg, uint8_t v) {
  i2cStart(); bool ok = i2cWrite(MPU << 1) && i2cWrite(reg) && i2cWrite(v); i2cStop(); return ok;
}

bool imuInit() {
  digitalWrite(IMU_SDA, HIGH); digitalWrite(IMU_SCL, HIGH);
  pinMode(IMU_SDA, OUTPUT_OPEN_DRAIN); pinMode(IMU_SCL, OUTPUT_OPEN_DRAIN);   // input stays enabled for reads
  i2cRecover();
  uint8_t who;
  if (!mpuRead(0x75, &who, 1)) return false;   // WHO_AM_I
  bool ok = mpuWrite(0x6B, 0x01)    // wake, gyro PLL clock
         && mpuWrite(0x1A, 0x03)    // DLPF ~44 Hz
         && mpuWrite(0x1B, 0x08)    // gyro +-500 deg/s
         && mpuWrite(0x1C, 0x08);   // accel +-4 g
  Out.printf("# IMU found, WHO_AM_I 0x%02X%s\n", who, ok ? "" : ", config write FAILED");  // 0x68 genuine; clones differ
  return ok;
}

bool imuRead() {
  uint8_t b[14];
  if (!mpuRead(0x3B, b, 14)) { i2cRecover(); return false; }   // ACCEL_XOUT_H .. GYRO_ZOUT_L
  int16_t r[7];
  for (int i = 0; i < 7; i++) r[i] = (int16_t)(b[2 * i] << 8 | b[2 * i + 1]);
  for (int i = 0; i < 3; i++) { imu[i] = r[i] / 8192.0f; imu[i + 3] = r[i + 4] / 65.5f; }
  imuTemp = r[3] / 340.0f + 36.53f;   // MPU6050 datasheet scaling
  for (int i = 0; i < 3; i++) accF[i] += 0.2f * (imu[i] - accF[i]);
  return true;
}

// Hub pitch about the axle = IMU y axis (hub tilted by hand, 2026-10-30). + = the way a wheel turns driving forward.
// The encoders measure wheel vs hub, so wheel speed over the ground = encoder speed + hubRate.
float gyBias = 0, hubAng = 0, hubRate = 0, hubRest = 0;   // deg/s, rad, rad/s, rad
int hubInit = 0;   // steady reads so far; bias = their mean. Taking the first read gave 23 deg/s, the first 0.5 s gave 9
                   // (hub still swinging after motor calibration, 2026-10-30), so wait for 20 steady reads in a row
float gyPrev = 0;
void hubUpdate(bool idle) {
  float a = atan2f(imu[2], -imu[0]);
  if (hubInit < 20) {
    if (fabsf(imu[4] - gyPrev) < 0.5f) { gyBias += imu[4] / 20; hubInit++; } else gyBias = hubInit = 0;
    gyPrev = imu[4];
    hubAng = hubRest = a;
    return;
  }
  float g = imu[4] - gyBias;
  bool still = idle && fabsf(g) < 3;
  if (still) gyBias += 0.02f * g;   // ~1.8 deg/s at rest, drifts with warm-up
  hubRate = -g * DEG_TO_RAD;
  hubAng = 0.98f * (hubAng + hubRate * IMU_MS * 1e-3f) + 0.02f * a;
  if (still) hubRest += 0.02f * (hubAng - hubRest);   // rest angle moves with cable pull
}
float hubTilt() { return (hubAng - hubRest) * RAD_TO_DEG; }

// ---- Self-test: drive a fixed sequence on the floor with no host, keep a summary in flash ----
// Aborts on: any serial input, a wheel stalled (>= 1 rad/s asked, < 0.3 measured) for 1.5 s,
// or the body swinging > 60 deg from where it sat at the start (the hub rolling instead of the wheels).
struct Seg { float l, r; uint16_t ms; };
const Seg SEQ[] = {{2, 2, 3000}, {0, 0, 2000}, {-2, -2, 3000}, {0, 0, 2000},   // 100 mm wheels: 2 rad/s = 0.1 m/s
                   {4, 4, 2000}, {0, 0, 2000}, {-4, -4, 2000}, {0, 0, 2000},
                   {2, -2, 2000}, {0, 0, 2000}, {-2, 2, 2000}, {0, 0, 1000}};
const int NSEG = sizeof(SEQ) / sizeof(SEQ[0]);
struct Res { float velL, velR, tilt; uint16_t n; };   // mean measured speeds after the ramp, max body swing (deg)
Res res[NSEG];
const char *END_TXT[] = {"none", "complete", "STALL left", "STALL right", "TILT > 60 deg", "cancelled by serial input",
                         "not run (motor fault)"};
uint8_t stEnd = 0; float stVin = 0;
Preferences prefs;

float tiltFrom(const float *ref) {
  float d = 0, a = 0, b = 0;
  for (int i = 0; i < 3; i++) { d += accF[i] * ref[i]; a += accF[i] * accF[i]; b += ref[i] * ref[i]; }
  return acosf(constrain(d / sqrtf(a * b + 1e-9f), -1.0f, 1.0f)) * 57.3f;
}

void printSelftest() {
  prefs.begin("selftest", true);
  if (prefs.getBytesLength("res") != sizeof(res)) { prefs.end(); Out.println("# SELFTEST no stored result"); return; }
  prefs.getBytes("res", res, sizeof(res)); stEnd = prefs.getUChar("end", 0); stVin = prefs.getFloat("vin", 0);
  uint32_t runs = prefs.getUInt("runs", 0); prefs.end();
  Out.printf("# SELFTEST last run (#%lu): %s, VIN %.1f V, limit %.1f V\n", runs, END_TXT[stEnd], stVin, VOLT_LIMIT);
  for (int i = 0; i < NSEG; i++) {
    if (SEQ[i].l == 0 && SEQ[i].r == 0) continue;
    Out.printf("# SELFTEST seg %d  target L %.1f R %.1f  measured L %.2f R %.2f rad/s  max swing %.1f deg  n %u\n",
                  i, SEQ[i].l, SEQ[i].r, res[i].velL, res[i].velR, res[i].tilt, res[i].n);
  }
}

void selftestEnd(uint8_t why) {
  stopNow(); stEnd = why;
  prefs.begin("selftest", false);
  prefs.putBytes("res", res, sizeof(res)); prefs.putUChar("end", why); prefs.putFloat("vin", stVin);
  prefs.putUInt("runs", prefs.getUInt("runs", 0) + 1); prefs.end();
  Out.printf("# SELFTEST end: %s\n", END_TXT[why]);
  printSelftest();
}

void selftest(uint32_t now) {
  static int8_t seg = -2;              // -2 countdown, -1 done, >= 0 running
  static uint32_t t0 = 0, tSeg = 0, stallL = 0, stallR = 0, tPrint = 0;
  static float ref[3];
  if (seg == -1) return;
  if (seg == -2) {
    if (!t0) t0 = now;
    if (fault) { seg = -1; Out.println("# SELFTEST not run: motor fault"); return; }
    if (hostSeen) { seg = -1; Out.println("# SELFTEST skipped: serial host present"); return; }
    if (now - tPrint >= 1000) { tPrint = now; Out.printf("# SELFTEST starts in %lu s\n", (10000 - (now - t0)) / 1000); }
    if (now - t0 < 10000) return;
    memset(res, 0, sizeof(res)); memcpy(ref, accF, sizeof(ref)); stVin = vin();
    seg = 0; tSeg = now; stallL = stallR = 0;
    Out.println("# SELFTEST start");
  }
  if (hostSeen) { seg = -1; selftestEnd(5); return; }
  const Seg &s = SEQ[seg];
  Res &r = res[seg];
  float vL = sgnL * motorL.shaft_velocity, vR = sgnR * motorR.shaft_velocity;
  if (imuOk) { float t = tiltFrom(ref); if (t > r.tilt) r.tilt = t; if (t > 60) { seg = -1; selftestEnd(4); return; } }
  if (now - tSeg > 500 && (s.l || s.r)) {   // past the ramp (4 rad/s at 20 rad/s^2 = 0.2 s)
    r.velL += (vL - r.velL) / (r.n + 1); r.velR += (vR - r.velR) / (r.n + 1); r.n++;
    stallL = fabsf(s.l) >= 1 && fabsf(vL) < 0.3f ? (stallL ? stallL : now) : 0;
    stallR = fabsf(s.r) >= 1 && fabsf(vR) < 0.3f ? (stallR ? stallR : now) : 0;
    if (stallL && now - stallL > 1500) { seg = -1; selftestEnd(2); return; }
    if (stallR && now - stallR > 1500) { seg = -1; selftestEnd(3); return; }
  }
  targetL = s.l; targetR = s.r; lastV = now;   // feeds the normal ramp + watchdog path
  if (now - tSeg >= s.ms) {
    tSeg = now; stallL = stallR = 0;
    if (++seg >= NSEG) { seg = -1; selftestEnd(1); }
  }
}

float vin() { return analogReadMilliVolts(VIN_PIN) * VIN_DIV / 1000; }

// AS5600 raw angle (0..4095), or -(Wire error code) if it doesn't answer
int encRaw(TwoWire &bus) {
  bus.beginTransmission(0x36); bus.write(0x0C);
  int e = bus.endTransmission(false);
  if (e) return -e;
  if (bus.requestFrom(0x36, 2) != 2) return -9;
  int hi = bus.read(); return (hi & 0x0F) << 8 | bus.read();
}

// Don't touch the motors until the PSU is on, so USB alone stays up for flashing and monitoring
void waitVin() {
  if (vin() >= VIN_MIN) { Out.printf("# VIN %.1f V\n", vin()); return; }
  uint32_t t = 0;
  while (vin() < VIN_MIN) {
    while (Serial.available()) Serial.read();
    while (Serial2.available()) Serial2.read();
    if (millis() - t >= 1000) {
      t = millis(); Out.printf("# no VIN (%.1f V), waiting for PSU\n", vin());
      Out.printf("# ENC L(M0) %d  R(M1) %d   (raw 0..4095, negative = not answering)\n", encRaw(Wire), encRaw(Wire1));
    }
    Out.printf("T %lu 0 0 0 0 NOVIN\n", millis());
    delay(STATUS_MS);
  }
  Out.printf("# VIN %.1f V, starting\n", vin());
  delay(500);   // let the supply settle
}

void setup() {
  Serial.begin(115200);
  Serial2.setTxBufferSize(1024);   // so T/I lines queue instead of stalling the FOC loop
  Serial2.setRxBufferSize(2048);   // room for a 1024-byte update chunk
  Serial2.begin(115200, SERIAL_8N1, LINK_RX, LINK_TX); delay(500);
  Out.println("# BOOT drive_fw, built " __DATE__ " " __TIME__);
  pinMode(M_EN, OUTPUT); digitalWrite(M_EN, LOW);
  if (SELFTEST) printSelftest();   // before waitVin, so results read over USB alone
  Wire.begin(19, 18, 400000);   // M0 bus
  Wire1.begin(23, 5, 100000);   // M1 bus: slower for noise tolerance (had NACKs at 400k)
  Wire.setTimeOut(5); Wire1.setTimeOut(5);   // 50 ms hung the board after a NACK (2026-10-21)
  waitVin();
  supplyV = constrain(vin(), VIN_MIN, 17.0f);
  Out.printf("# supply for PWM scaling %.1f V\n", supplyV);
  digitalWrite(M_EN, HIGH);
  if (IMU_ON && !(imuOk = imuInit())) Out.println("# IMU not found on IO4/IO15, continuing without it");
  SimpleFOCDebug::enable(&Out);   // initFOC prints why it fails (lines start with MOT:, not #)
  okL = setupMotor(motorL, driverL, sensorL, Wire, "L(M0)", ZERO_L, Direction::CW);
  okR = setupMotor(motorR, driverR, sensorR, Wire1, "R(M1)", ZERO_R, Direction::CCW);
  fault = !(okL && okR);
  gate(false);
  Out.println(!fault ? "# READY" : okL || okR ? "# FAULT one motor failed initFOC, only the other will drive"
                                                 : "# FAULT initFOC failed, not driving. Check PSU and reset.");
}

void loop() {
  if (okL) motorL.loopFOC();   // a dead encoder bus would flood NACKs and drop the loop to ~36 Hz
  if (okR) motorR.loopFOC();
  readSerial();
  uint32_t now = millis();
  if (SELFTEST) selftest(now);
  if ((targetL || targetR || cmdL || cmdR) && now - lastV > WATCHDOG_MS) { stopNow(); wdog = true; }

  static uint32_t tPrev = micros();
  uint32_t t = micros();
  float step = accel * (t - tPrev) * 1e-6f; tPrev = t;
  cmdL = ramp(cmdL, targetL, step); cmdR = ramp(cmdR, targetR, step);

  // A hub tilting away from rest means the wheels push harder than the low centre of mass can hold: ease off.
  // Scale the PI output limit, not the target: SimpleFOC clamps the integral to it, so the push itself fades
  // (scaling the target left the built-up integral pushing and the hub went over, 2026-10-30)
  bool hub = HUB_COMP && imuOk;
  float hr = hub ? hubRate : 0;
  // Only tilt the push would add to counts: forward push swings the hub negative, reverse positive, a spin neither way
  float s = cmdL + cmdR, tilt = s > 0.01f ? -hubTilt() : s < -0.01f ? hubTilt() : fabsf(hubTilt());
  float k = hub ? constrain((TILT_MAX - tilt) / (TILT_MAX - TILT_SOFT), 0.0f, 1.0f) : 1;
  motorL.PID_velocity.limit = motorR.PID_velocity.limit = k * VOLT_LIMIT;
  // Damping: the push swings the hub the other way (forward push -> negative tilt). Taking hub rate and tilt
  // off the target eases the push as the hub swings back and adds to it as it swings forward. Same on both wheels.
  float damp = hub && HUB_DAMP ? constrain(kRate * hubRate + kTilt * hubTilt() * DEG_TO_RAD, -DAMP_MAX, DAMP_MAX) : 0;
  gate(cmdL != 0 || cmdR != 0);
  if (okL) moveOrRelease(motorL, cmdL ? sgnL * (cmdL + damp - hr) : 0);
  if (okR) moveOrRelease(motorR, cmdR ? sgnR * (cmdR + damp - hr) : 0);

  static uint32_t ti = 0, tl = 0, loops = 0, hz = 0, imuErr = 0;
  loops++;
  if (now - tl >= 1000) { hz = loops; loops = 0; tl = now; }
  if (imuOk && now - ti >= IMU_MS) { ti = now; if (imuRead()) hubUpdate(!cmdL && !cmdR); else imuErr++; }

  static uint32_t ts = 0;
  if (now - ts >= STATUS_MS) {
    ts = now;
    const char *st = fault ? "FAULT" : wdog ? "WDOG" : (cmdL || cmdR) ? "RUN" : "IDLE";
    float ha = hub ? hubAng : 0;
    Out.printf("T %lu %.4f %.4f %.3f %.3f %s\n", now, sgnL * motorL.shaft_angle + ha, sgnR * motorR.shaft_angle + ha,
                  sgnL * motorL.shaft_velocity + hr, sgnR * motorR.shaft_velocity + hr, st);
    Out.printf("I %lu %.3f %.3f %.3f %.1f %.1f %.1f %lu %lu %lu %lu\n", now, imu[0], imu[1], imu[2],
                             imu[3], imu[4], imu[5], hz, imuErr, sensorL.rejected, sensorR.rejected);
    if (imuOk) Out.printf("H %lu %.1f %.1f %.2f %.2f\n", now, hubTilt(), hubRate * RAD_TO_DEG, k, damp);
  }

  // Follow VIN for PWM scaling, so VOLT_LIMIT stays true as the battery sags (or the PSU is turned while running)
  static uint32_t tp = 0;
  if (now - tp >= POWER_MS) {
    tp = now;
    supplyV += 0.3f * (constrain(vin(), VIN_MIN, 17.0f) - supplyV);
    driverL.voltage_power_supply = driverR.voltage_power_supply = supplyV;
    Out.printf("P %lu %.2f %.1f\n", now, supplyV, imuOk ? imuTemp : NAN);
  }
}

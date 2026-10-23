#include <SimpleFOC.h>
// Recon drone drive firmware. Pi <-> ESP32 over serial (USB/CH340 or UART), 115200 8N1, newline-terminated ASCII.
// Pi -> board:
//   V <left> <right>   wheel speed targets, rad/s at the motor shaft (+ = forward). Resend at >= 5 Hz.
//   S                  stop now (no ramp, motors released)
// Board -> Pi, every STATUS_MS:
//   T <ms> <angL> <angR> <velL> <velR> <state>   angles rad (cumulative, + = forward), velocities rad/s
//   state: IDLE, RUN, WDOG (stopped, no V for WATCHDOG_MS), FAULT (initFOC failed, will not drive),
//          NOVIN (no motor supply at boot; waiting, motors not initialised)
//   I <ms> <ax> <ay> <az> <gx> <gy> <gz> <loopHz> <imuErr> <rejL> <rejR>
//     accel g, gyro deg/s (sensor axes; all 0 if no MPU6050), main loop rate, failed IMU reads, rejected encoder reads
// Lines starting with # are info/errors; the Pi should ignore or log them.
#define SUPPLY_V     12.0
#define POLE_PAIRS   7
#define VOLT_LIMIT   0.9     // 0.9 V oscillated with P0.2 I2 Tf0.01 (2026-10-18); retrying with softer PID (2026-10-21)
#define MAX_VEL      15.0    // rad/s clamp (~0.75 m/s on 100 mm wheels)
#define ACCEL        20.0    // rad/s^2 ramp on V targets; S and watchdog stops are immediate
#define WATCHDOG_MS  300
#define STATUS_MS    50
#define INVERT_L     false   // motors are mirrored on the chassis: flip so + drives forward on each side
#define INVERT_R     true
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

BLDCMotor motorL(POLE_PAIRS), motorR(POLE_PAIRS);           // M0 = left, M1 = right
BLDCDriver3PWM driverL(32, 33, 25), driverR(26, 27, 14);
FilteredAS5600 sensorL, sensorR;
const float sgnL = INVERT_L ? -1 : 1, sgnR = INVERT_R ? -1 : 1;
float targetL = 0, targetR = 0, cmdL = 0, cmdR = 0;
uint32_t lastV = 0;
bool fault = false, wdog = false, okL = false, okR = false;   // a failed motor is skipped; V still drives the other (bench tests)

bool setupMotor(BLDCMotor &m, BLDCDriver3PWM &d, MagneticSensorI2C &s, TwoWire &bus, const char *name) {
  s.init(&bus);
  d.voltage_power_supply = SUPPLY_V; d.voltage_limit = VOLT_LIMIT; d.init();
  m.linkDriver(&d); m.linkSensor(&s);
  m.voltage_limit = VOLT_LIMIT; m.voltage_sensor_align = VOLT_LIMIT;
  m.PID_velocity.P = 0.1; m.PID_velocity.I = 1; m.PID_velocity.D = 0; m.LPF_velocity.Tf = 0.02;
  m.controller = MotionControlType::velocity;
  m.init();
  bool ok = m.initFOC();
  // zero_electric_angle + direction can later be hard-coded to skip alignment (wheels loaded on the ground)
  Serial.printf("# %s initFOC %s  zero_electric_angle %.4f  dir %d\n", name, ok ? "OK" : "FAILED",
                m.zero_electric_angle, (int)m.sensor_direction);
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

void handle(char *l) {
  if (l[0] == 'S') { stopNow(); wdog = false; return; }
  if (l[0] == 'V') {
    char *p = l + 1, *e1, *e2;
    float a = strtof(p, &e1), b = strtof(e1, &e2);
    if (e1 == p || e2 == e1) { Serial.println("# ERR bad V"); return; }
    if (!okL && !okR) { Serial.println("# ERR fault, not driving"); return; }
    targetL = constrain(a, -MAX_VEL, MAX_VEL); targetR = constrain(b, -MAX_VEL, MAX_VEL);
    lastV = millis(); wdog = false;
    return;
  }
  if (l[0]) Serial.println("# ERR unknown command");
}

void readSerial() {
  static char buf[48]; static uint8_t n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c != '\n') { if (n < sizeof(buf) - 1) buf[n++] = c; continue; }
    buf[n] = 0; n = 0;
    handle(buf);
  }
}

float ramp(float cur, float tgt, float step) { return cur + constrain(tgt - cur, -step, step); }

bool imuOk = false;
float imu[6];   // ax ay az (g), gx gy gz (deg/s)

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
  Serial.printf("# IMU found, WHO_AM_I 0x%02X%s\n", who, ok ? "" : ", config write FAILED");  // 0x68 genuine; clones differ
  return ok;
}

bool imuRead() {
  uint8_t b[14];
  if (!mpuRead(0x3B, b, 14)) { i2cRecover(); return false; }   // ACCEL_XOUT_H .. GYRO_ZOUT_L
  int16_t r[7];
  for (int i = 0; i < 7; i++) r[i] = (int16_t)(b[2 * i] << 8 | b[2 * i + 1]);
  for (int i = 0; i < 3; i++) { imu[i] = r[i] / 8192.0f; imu[i + 3] = r[i + 4] / 65.5f; }  // r[3] is temperature
  return true;
}

float vin() { return analogReadMilliVolts(VIN_PIN) * VIN_DIV / 1000; }

// Don't touch the motors until the PSU is on, so USB alone stays up for flashing and monitoring
void waitVin() {
  if (vin() >= VIN_MIN) { Serial.printf("# VIN %.1f V\n", vin()); return; }
  uint32_t t = 0;
  while (vin() < VIN_MIN) {
    while (Serial.available()) Serial.read();
    if (millis() - t >= 1000) { t = millis(); Serial.printf("# no VIN (%.1f V), waiting for PSU\n", vin()); }
    Serial.printf("T %lu 0 0 0 0 NOVIN\n", millis());
    delay(STATUS_MS);
  }
  Serial.printf("# VIN %.1f V, starting\n", vin());
  delay(500);   // let the supply settle
}

void setup() {
  Serial.begin(115200); delay(500);
  Serial.println("# BOOT drive_fw");
  pinMode(M_EN, OUTPUT); digitalWrite(M_EN, LOW);
  waitVin();
  digitalWrite(M_EN, HIGH);
  Wire.begin(19, 18, 400000);   // M0 bus
  Wire1.begin(23, 5, 100000);   // M1 bus: slower for noise tolerance (had NACKs at 400k)
  Wire.setTimeOut(5); Wire1.setTimeOut(5);   // 50 ms hung the board after a NACK (2026-10-21)
  if (IMU_ON && !(imuOk = imuInit())) Serial.println("# IMU not found on IO4/IO15, continuing without it");
  okL = setupMotor(motorL, driverL, sensorL, Wire, "L(M0)");
  okR = setupMotor(motorR, driverR, sensorR, Wire1, "R(M1)");
  fault = !(okL && okR);
  gate(false);
  Serial.println(!fault ? "# READY" : okL || okR ? "# FAULT one motor failed initFOC, only the other will drive"
                                                 : "# FAULT initFOC failed, not driving. Check PSU and reset.");
}

void loop() {
  if (okL) motorL.loopFOC();   // a dead encoder bus would flood NACKs and drop the loop to ~36 Hz
  if (okR) motorR.loopFOC();
  readSerial();
  uint32_t now = millis();
  if ((targetL || targetR || cmdL || cmdR) && now - lastV > WATCHDOG_MS) { stopNow(); wdog = true; }

  static uint32_t tPrev = micros();
  uint32_t t = micros();
  float step = ACCEL * (t - tPrev) * 1e-6f; tPrev = t;
  cmdL = ramp(cmdL, targetL, step); cmdR = ramp(cmdR, targetR, step);

  gate(cmdL != 0 || cmdR != 0);
  if (okL) moveOrRelease(motorL, sgnL * cmdL);
  if (okR) moveOrRelease(motorR, sgnR * cmdR);

  static uint32_t ti = 0, tl = 0, loops = 0, hz = 0, imuErr = 0;
  loops++;
  if (now - tl >= 1000) { hz = loops; loops = 0; tl = now; }
  if (imuOk && now - ti >= IMU_MS) { ti = now; if (!imuRead()) imuErr++; }

  static uint32_t ts = 0;
  if (now - ts >= STATUS_MS) {
    ts = now;
    const char *st = fault ? "FAULT" : wdog ? "WDOG" : (cmdL || cmdR) ? "RUN" : "IDLE";
    Serial.printf("T %lu %.4f %.4f %.3f %.3f %s\n", now, sgnL * motorL.shaft_angle, sgnR * motorR.shaft_angle,
                  sgnL * motorL.shaft_velocity, sgnR * motorR.shaft_velocity, st);
    Serial.printf("I %lu %.3f %.3f %.3f %.1f %.1f %.1f %lu %lu %lu %lu\n", now, imu[0], imu[1], imu[2],
                             imu[3], imu[4], imu[5], hz, imuErr, sensorL.rejected, sensorR.rejected);
  }
}

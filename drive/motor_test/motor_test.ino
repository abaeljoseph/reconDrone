#include <SimpleFOC.h>
#define OPEN_LOOP   false
#define SUPPLY_V    12.0   // set to real supply voltage
#define POLE_PAIRS  7
#define VOLT_LIMIT  0.8    // 0.9 V made velocity PID oscillate (2026-10-18); tune PID before raising

BLDCMotor motor0(POLE_PAIRS), motor1(POLE_PAIRS);
#define M_EN        12     // shared gate-driver supply enable for BOTH motors (GPIO22 is just CS0)
BLDCDriver3PWM driver0(32, 33, 25), driver1(26, 27, 14);
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
FilteredAS5600 sensor0, sensor1;
float target0 = 0, target1 = 0;
Commander command(Serial);
void doA(char *c) { command.scalar(&target0, c); }
void doB(char *c) { command.scalar(&target1, c); }

// De-energise at target 0: open loop holds full voltage_limit at standstill, and closed-loop
// velocity hold at 0 limit-cycles (loud buzz) against cogging. Reset PID so restarts don't jump.
void moveOrRelease(BLDCMotor &m, float t) {
  if (t == 0) { if (m.enabled) m.disable(); }
  else if (!m.enabled) { m.PID_velocity.reset(); m.enable(); }
  m.move(t);  // still updates shaft angle/velocity while disabled (needed for odometry)
}

void setupMotor(BLDCMotor &m, BLDCDriver3PWM &d, MagneticSensorI2C &s, TwoWire &bus, const char *name) {
  s.init(&bus);
  d.voltage_power_supply = SUPPLY_V; d.voltage_limit = VOLT_LIMIT; d.init();
  m.linkDriver(&d);
  m.voltage_limit = VOLT_LIMIT; m.voltage_sensor_align = VOLT_LIMIT;
  m.PID_velocity.P = 0.2; m.PID_velocity.I = 2; m.PID_velocity.D = 0; m.LPF_velocity.Tf = 0.01;
  if (OPEN_LOOP) m.controller = MotionControlType::velocity_openloop;
  else { m.linkSensor(&s); m.controller = MotionControlType::velocity; }
  Serial.printf("--- %s ---\n", name);
  m.init();
  if (!OPEN_LOOP) m.initFOC();
}

void setup() {
  Serial.begin(115200); delay(500);
  SimpleFOCDebug::enable(&Serial);
  pinMode(M_EN, OUTPUT); digitalWrite(M_EN, HIGH);
  Wire.begin(19, 18, 400000);
  Wire1.begin(23, 5, 100000);   // M1 bus: slower for noise tolerance (had NACKs at 400k)
  Wire.setTimeOut(5); Wire1.setTimeOut(5);
  setupMotor(motor0, driver0, sensor0, Wire, "M0");
  setupMotor(motor1, driver1, sensor1, Wire1, "M1");
  command.add('A', doA, "M0 target rad/s");
  command.add('B', doB, "M1 target rad/s");
  Serial.println("Ready. A3 / B3 to spin, A0 / B0 to stop.");
}

void loop() {
  motor0.loopFOC(); motor1.loopFOC();
  moveOrRelease(motor0, target0); moveOrRelease(motor1, target1);
  command.run();
  static uint32_t t = 0;
  if (millis() - t > 200) {
    t = millis();
    if (OPEN_LOOP) { sensor0.update(); sensor1.update();
      Serial.printf("E0 %7.2f rad   E1 %7.2f rad\n", sensor0.getAngle(), sensor1.getAngle()); }
    else Serial.printf("M0 vel %6.2f   M1 vel %6.2f rad/s   rej %lu %lu\n", motor0.shaft_velocity,
                       motor1.shaft_velocity, sensor0.rejected, sensor1.rejected);
  }
}

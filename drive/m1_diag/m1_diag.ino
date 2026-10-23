#include <SimpleFOC.h>
// Diagnostic: drive both motors open loop at a steady speed (forward then reverse) and log
// commanded vs encoder angle. Error fixed to rotor angle in both directions = sensor/magnet offset.
// Error that flips sign with direction = drag/friction. Send G to start, any other char stops.
#define SUPPLY_V    12.0
#define POLE_PAIRS  7
#define VOLT_LIMIT  0.6
#define SPEED       3.0     // rad/s
#define RUN_MS      16000   // half forward, half reverse
#define M_EN        12

BLDCMotor motor0(POLE_PAIRS), motor1(POLE_PAIRS);
BLDCDriver3PWM driver0(32, 33, 25), driver1(26, 27, 14);
MagneticSensorI2C sensor0(AS5600_I2C), sensor1(AS5600_I2C);
bool running = false;
uint32_t t0 = 0;

void setupMotor(BLDCMotor &m, BLDCDriver3PWM &d, MagneticSensorI2C &s, TwoWire &bus) {
  s.init(&bus);
  d.voltage_power_supply = SUPPLY_V; d.voltage_limit = VOLT_LIMIT; d.init();
  m.linkDriver(&d);
  m.voltage_limit = VOLT_LIMIT;
  m.controller = MotionControlType::velocity_openloop;
  m.init(); m.disable();
}

void stopAll() {
  motor0.disable(); motor1.disable();
  if (running) Serial.println("DONE");
  running = false;
}

void setup() {
  Serial.begin(115200); delay(500);
  pinMode(M_EN, OUTPUT); digitalWrite(M_EN, HIGH);
  Wire.begin(19, 18, 400000);
  Wire1.begin(23, 5, 100000);   // M1 bus: slower for noise tolerance (had NACKs at 400k)
  Wire.setTimeOut(5); Wire1.setTimeOut(5);
  setupMotor(motor0, driver0, sensor0, Wire);
  setupMotor(motor1, driver1, sensor1, Wire1);
  Serial.println("Ready. G to run diagnostic.");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'G' && !running) {
      motor0.enable(); motor1.enable();
      running = true; t0 = millis();
      Serial.println("t_ms,cmd0,enc0,cmd1,enc1");
    } else if (c != 'G' && c != '\n' && c != '\r') stopAll();
  }
  if (!running) return;
  uint32_t el = millis() - t0;
  float v = el < RUN_MS / 2 ? SPEED : -SPEED;
  motor0.move(v); motor1.move(v);
  static uint32_t tl = 0;
  if (millis() - tl >= 10) {
    tl = millis();
    sensor0.update(); sensor1.update();
    Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f\n", el, motor0.shaft_angle, sensor0.getAngle(),
                  motor1.shaft_angle, sensor1.getAngle());
  }
  if (el > RUN_MS) stopAll();
}

#include <Wire.h>
#define AS5600 0x36

void scan(TwoWire &bus, const char *name) {
  Serial.printf("Scan %s:", name);
  for (uint8_t a = 1; a < 127; a++) {
    bus.beginTransmission(a);
    if (bus.endTransmission() == 0) Serial.printf(" 0x%02X", a);
  }
  Serial.println();
}

int readReg(TwoWire &bus, uint8_t reg, uint8_t n) {
  bus.beginTransmission(AS5600);
  bus.write(reg);
  if (bus.endTransmission(false) != 0) return -1;
  if (bus.requestFrom((uint8_t)AS5600, n) != n) return -1;
  int v = 0;
  for (int i = 0; i < n; i++) v = (v << 8) | bus.read();
  return v;
}

void report(TwoWire &bus, const char *name) {
  int st = readReg(bus, 0x0B, 1), raw = readReg(bus, 0x0C, 2), agc = readReg(bus, 0x1A, 1);
  if (st < 0 || raw < 0 || agc < 0) { Serial.printf("%s: NO RESPONSE        ", name); return; }
  const char *mag = !(st & 0x20) ? "NO MAGNET" : (st & 0x08) ? "TOO CLOSE" : (st & 0x10) ? "TOO FAR" : "OK";
  Serial.printf("%s: %6.1f deg  agc %3d  %-9s   ", name, (raw & 0x0FFF) * 360.0 / 4096, agc, mag);
}

void setup() {
  Serial.begin(115200); delay(500);
  Wire.begin(19, 18, 400000);
  Wire1.begin(23, 5, 400000);
  scan(Wire, "I2C_0"); scan(Wire1, "I2C_1");
}

void loop() { report(Wire, "E0"); report(Wire1, "E1"); Serial.println(); delay(100); }

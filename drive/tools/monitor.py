import serial, time, sys
# Usage: monitor.py PORT SECONDS [reset]
# Opens with DTR/RTS released so the ESP32 runs normally; optional RTS reset pulse.
port, secs = sys.argv[1], float(sys.argv[2])
s = serial.Serial()
s.port, s.baudrate, s.timeout = port, 115200, 0.1
s.dtr, s.rts = False, False
s.open()
if len(sys.argv) > 3 and sys.argv[3] == 'reset':
    s.rts = True; s.dtr = False; time.sleep(0.1); s.rts = False; s.dtr = False
t = time.time()
while time.time() - t < secs:
    line = s.readline()
    if line: print(line.decode('ascii', errors='ignore').rstrip(), flush=True)
s.close()

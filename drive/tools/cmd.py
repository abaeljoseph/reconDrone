import serial, time, sys
# Usage: cmd.py PORT SECONDS "A3" ["B3" ...]
# Sends commands, logs output for SECONDS, then always sends A0 and B0.
port, secs, cmds = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
s = serial.Serial()
s.port, s.baudrate, s.timeout = port, 115200, 0.1
s.dtr, s.rts = False, False
s.open()
def log(dur):
    t = time.time()
    while time.time() - t < dur:
        line = s.readline()
        if line: print(line.decode('ascii', errors='ignore').rstrip(), flush=True)
try:
    for c in cmds:
        print('>>', c, flush=True); s.write((c + '\n').encode())
    log(secs)
finally:
    print('>> A0 / B0', flush=True)
    s.write(b'A0\n'); s.write(b'B0\n')
    log(1.0)
    s.close()

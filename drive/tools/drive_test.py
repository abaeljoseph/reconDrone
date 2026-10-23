import serial, sys, time
# Usage: drive_test.py PORT SECONDS LEFT RIGHT [nostop]
# Also the reference for the Pi side: open with DTR/RTS released (otherwise the board resets or
# enters download mode), resend V at 20 Hz (board watchdog is 300 ms), always send S on exit.
# nostop: go silent instead of sending S, to test the watchdog.
port, secs, l, r = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
nostop = 'nostop' in sys.argv[5:]
s = serial.Serial()
s.port, s.baudrate, s.timeout = port, 115200, 0
s.dtr, s.rts = False, False
s.open()
buf = b''
def pump():
    global buf
    buf += s.read(4096)
    *lines, buf = buf.split(b'\n')
    for x in lines: print(x.decode('ascii', 'ignore').rstrip(), flush=True)
try:
    t0 = time.time()
    while time.time() - t0 < secs:
        s.write(f'V {l} {r}\n'.encode()); pump(); time.sleep(0.05)
finally:
    if not nostop: s.write(b'S\n')
    t = time.time()
    while time.time() - t < 1.5: pump(); time.sleep(0.05)
    s.write(b'S\n'); s.close()

import hashlib, sys, time, serial
# Flash drive_fw over the Pi <-> ESP header UART (drive_fw's U command). Nothing else may have the port open.
#   fw_update.py /dev/ttyAMA0 drive_fw.ino.bin      (the app .bin from the build dir, NOT the .merged.bin)
port, path = sys.argv[1], sys.argv[2]
data = open(path, 'rb').read()
if data[0] != 0xE9: sys.exit(f'{path} is not an ESP32 app image')
md5 = hashlib.md5(data).hexdigest()

s = serial.Serial()
s.port, s.baudrate, s.timeout = port, 115200, 0.1
s.open(); s.rts = False; s.dtr = False   # same order as drive.py, so a USB port won't reset the board

def wait(prefix, secs):
    end = time.time() + secs
    while time.time() < end:
        l = s.readline().decode('ascii', 'ignore').strip()
        if l.startswith('# UPDATE FAIL'): sys.exit(l)
        if l.startswith(prefix): return l
    sys.exit(f'timed out waiting for "{prefix}"')

s.write(b'S\n'); time.sleep(0.2); s.reset_input_buffer()
s.write(f'U {len(data)} {md5}\n'.encode())
wait('# UPDATE ready', 10)
t0 = time.time()
for i in range(0, len(data), 1024):
    s.write(data[i:i + 1024]); wait('#A', 10)
    if i % (64 * 1024) == 0: print(f'{i // 1024:4d}/{len(data) // 1024} KB', flush=True)
print(wait('# UPDATE OK', 15), f'({time.time() - t0:.0f} s)')
s.close()

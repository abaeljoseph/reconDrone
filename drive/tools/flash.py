import serial, subprocess, sys, time, os
# Usage: flash.py PORT SKETCH_DIR
# Board auto-reset timing fails on this PC, so: hold DTR (GPIO0 low), user taps RST,
# wait for "waiting for download", then write the merged image with --before no_reset.
port, sketch = sys.argv[1], os.path.abspath(sys.argv[2])
name = os.path.basename(sketch)
img = os.path.join(sketch, 'build', name + '.ino.merged.bin')
esptool = os.path.expandvars(r'%LOCALAPPDATA%\Arduino15\packages\esp32\tools\esptool_py\4.9.dev3\esptool.exe')

s = serial.Serial()
s.port, s.baudrate, s.timeout = port, 115200, 0.1
s.dtr, s.rts = True, False
s.open()
s.dtr, s.rts = True, False
print('Holding GPIO0 low. Tap RST now.', flush=True)
t, buf = time.time(), b''
while time.time() - t < 150:
    buf += s.read(512)
    if b'waiting for download' in buf: break
    if b'SPI_FAST_FLASH_BOOT' in buf:
        print('Booted normally, tap RST again.', flush=True); buf = b''
else:
    sys.exit('Timed out waiting for download mode')
s.close()
print('In download mode, flashing', img, flush=True)
r = subprocess.run([esptool, '--chip', 'esp32', '--port', port, '--baud', '921600',
                    '--before', 'no_reset', '--after', 'hard_reset', 'write_flash', '-z', '0x0', img])
sys.exit(r.returncode)

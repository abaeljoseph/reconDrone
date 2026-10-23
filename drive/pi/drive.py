import math, os, serial, threading, time
# Pi-side driver for drive_fw. Import and use:
#   d = Drive('/dev/ttyUSB0'); d.wheels(4, 4)  or  d.twist(0.2, 0)  ...  d.pose() -> (x m, y m, heading rad)
#   d.close()   (also runs on exit; the board watchdog stops the motors 300 ms after this process dies)
# CLI test: drive.py PORT [SECONDS LEFT RIGHT]   prints pose; with no speeds just tracks (turn wheels by hand)
WHEEL_D = 0.100    # m, wheel outer diameter (brief says 108.6, user said assume 100 for now)
TRACK   = 0.200    # m, centre of left wheel to centre of right wheel. PLACEHOLDER until measured
MAX_VEL = 15.0     # rad/s, same clamp as drive_fw
SEND_HZ = 20

class Drive:
    def __init__(self, port, log=None):
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 0.02
        # DTR/RTS drive the board's auto-reset. Never release RTS while DTR is released (that pulls EN low).
        if os.name == 'nt':
            self.s.dtr, self.s.rts = False, False; self.s.open()   # applied atomically on Windows (tested)
        else:
            self.s.open(); self.s.rts = False; self.s.dtr = False  # Linux asserts both on open; drop RTS first
        self.lock = threading.Lock()
        self.cmd = None                     # (left, right) rad/s, None = stopped
        self.x = self.y = self.th = 0.0
        self.prev = None                    # (ms, angL, angR) of last T line
        self.state, self.velL, self.velR, self.last_t = '?', 0.0, 0.0, 0.0
        self.imu, self.diag = None, None    # (ax ay az g, gx gy gz deg/s), (loopHz, imuErr, rejL, rejR)
        self.log = log or (lambda line: None)
        self.alive = True
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()
        import atexit; atexit.register(self.close)

    # --- commands ---
    def wheels(self, left, right):
        """Wheel speeds in rad/s, + = forward. Resent at 20 Hz until changed or stop()."""
        with self.lock:
            self.cmd = (max(-MAX_VEL, min(MAX_VEL, left)), max(-MAX_VEL, min(MAX_VEL, right)))

    def twist(self, v, w):
        """v m/s forward, w rad/s CCW (left turn)."""
        r = WHEEL_D / 2
        self.wheels((v - w * TRACK / 2) / r, (v + w * TRACK / 2) / r)

    def stop(self):
        with self.lock: self.cmd = None
        self._send('S')

    def close(self):
        if not self.alive: return
        self.alive = False
        try: self.s.write(b'S\n'); self.s.flush(); self.s.write(b'S\n'); self.s.close()
        except Exception: pass

    # --- state ---
    def pose(self):
        with self.lock: return self.x, self.y, self.th

    def reset_pose(self):
        with self.lock: self.x = self.y = self.th = 0.0

    def ok(self):
        """True if a status line arrived in the last 0.3 s and the board is not in FAULT or NOVIN (PSU off)."""
        return time.time() - self.last_t < 0.3 and self.state not in ('FAULT', 'NOVIN')

    # --- internals ---
    def _send(self, line):
        try: self.s.write((line + '\n').encode())
        except Exception: pass

    def _odom(self, ms, aL, aR):
        if self.prev is None or ms < self.prev[0]:   # first line, or board rebooted: re-anchor, no jump
            self.prev = (ms, aL, aR); return
        r = WHEEL_D / 2
        dL, dR = (aL - self.prev[1]) * r, (aR - self.prev[2]) * r
        self.prev = (ms, aL, aR)
        if abs(dL) > 0.5 or abs(dR) > 0.5: return      # > 0.5 m in 50 ms is junk
        ds, dth = (dL + dR) / 2, (dR - dL) / TRACK
        mid = self.th + dth / 2
        self.x += ds * math.cos(mid); self.y += ds * math.sin(mid)
        self.th = math.atan2(math.sin(self.th + dth), math.cos(self.th + dth))

    def _parse(self, line):
        p = line.split()
        if len(p) == 12 and p[0] == 'I':
            try: v = [float(x) for x in p[2:]]
            except ValueError: self.log(line); return
            with self.lock: self.imu, self.diag = tuple(v[:6]), tuple(int(x) for x in v[6:])
            return
        if len(p) != 7 or p[0] != 'T': self.log(line); return
        try: ms, aL, aR, vL, vR = int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
        except ValueError: self.log(line); return
        with self.lock:
            self._odom(ms, aL, aR)
            self.velL, self.velR, self.state, self.last_t = vL, vR, p[6], time.time()

    def _run(self):
        buf, nxt = b'', 0.0
        while self.alive:
            try: buf += self.s.read(4096)
            except Exception: time.sleep(0.1); continue
            *lines, buf = buf.split(b'\n')
            for l in lines:
                l = l.decode('ascii', 'ignore').strip()
                if l: self._parse(l)
            if time.time() >= nxt:
                nxt = time.time() + 1 / SEND_HZ
                with self.lock: c = self.cmd
                if c: self._send(f'V {c[0]:.3f} {c[1]:.3f}')

if __name__ == '__main__':
    import sys
    port = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    d = Drive(port, log=lambda l: print(l, flush=True))
    try:
        t0 = time.time()
        while time.time() - t0 < 1.0 and not d.ok(): time.sleep(0.05)
        if len(sys.argv) > 4: d.wheels(float(sys.argv[3]), float(sys.argv[4]))
        while time.time() - t0 < secs:
            x, y, th = d.pose()
            print(f'{time.time() - t0:5.1f}s  {d.state:5s} vel {d.velL:6.2f} {d.velR:6.2f}  '
                  f'x {x:7.3f} y {y:7.3f} hdg {math.degrees(th):7.1f}'
                  + (f'  acc {d.imu[0]:6.2f} {d.imu[1]:6.2f} {d.imu[2]:6.2f}  gyro {d.imu[3]:6.1f} {d.imu[4]:6.1f} '
                     f'{d.imu[5]:6.1f}  loop {d.diag[0]} Hz  imuErr {d.diag[1]} rej {d.diag[2]} {d.diag[3]}'
                     if d.imu else ''), flush=True)
            time.sleep(0.5)
    finally:
        d.close()

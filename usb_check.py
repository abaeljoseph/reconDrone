#!/usr/bin/env python3
"""
D435 USB link diagnostic — tells you whether a cable is giving you USB 3 or
silently falling back to USB 2.

Run once per cable:
    python3 usb_check.py

Reports the negotiated link speed, what librealsense thinks it's connected
over, which stream profiles the camera is willing to offer, and the actual
delivered throughput under a deliberately heavy config.

USB 2 is the bottleneck to watch for: the D435 drops its high-bandwidth modes
entirely (no 848x480@90, no 1280x720 depth) rather than just running slower.
"""

import argparse
import glob
import threading
import time

import pyrealsense2 as rs

RS_VID = "8086"


def link_speed():
    """Negotiated USB speed in Mbps, straight from the kernel."""
    for path in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            with open(path) as fh:
                if fh.read().strip() != RS_VID:
                    continue
            base = path.rsplit("/", 1)[0]
            with open(f"{base}/speed") as fh:
                speed = fh.read().strip()
            with open(f"{base}/product") as fh:
                product = fh.read().strip()
            return float(speed), product, base.rsplit("/", 1)[1]
        except OSError:
            continue
    return None, None, None


def usb3_buses():
    """(total SuperSpeed root hubs, how many have anything attached).

    A USB 3 device in a USB 3 port enumerates on a 5000M bus. If every
    SuperSpeed bus is empty while the camera sits on a 480M one, the
    SuperSpeed pairs aren't making it through — cable or port.
    """
    total = occupied = 0
    for path in glob.glob("/sys/bus/usb/devices/usb*"):
        try:
            with open(f"{path}/speed") as fh:
                if float(fh.read().strip()) < 5000:
                    continue
        except (OSError, ValueError):
            continue
        total += 1
        bus = path.rsplit("/", 1)[1].replace("usb", "")
        # "{bus}-1" is a real device; "{bus}-0:1.0" is the root hub's own
        # interface and is always present, so filter those out.
        kids = [p for p in glob.glob(f"/sys/bus/usb/devices/{bus}-*")
                if ":" not in p.rsplit("/", 1)[1]]
        if kids:
            occupied += 1
    return total, occupied


def rs_link():
    """(speed_mbps, busid, devnum) for the RealSense right now, or Nones.

    Re-scanned on every poll rather than cached: a re-enumeration moves the
    camera to a different sysfs path (4-1 → 3-1), which is exactly the event
    we're trying to catch.
    """
    for path in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            with open(path) as fh:
                if fh.read().strip() != RS_VID:
                    continue
            base = path.rsplit("/", 1)[0]
            with open(f"{base}/speed") as fh:
                speed = float(fh.read().strip())
            with open(f"{base}/devnum") as fh:
                devnum = fh.read().strip()
            return speed, base.rsplit("/", 1)[1], devnum
        except (OSError, ValueError):
            continue
    return None, None, None


def depth_modes(dev):
    modes = set()
    for sensor in dev.query_sensors():
        for prof in sensor.get_stream_profiles():
            if prof.stream_type() != rs.stream.depth:
                continue
            try:
                vp = prof.as_video_stream_profile()
            except Exception:
                continue
            modes.add((vp.width(), vp.height(), prof.fps()))
    return modes


class LoadGenerator(threading.Thread):
    """Streams flat out so the link is under real traffic while you flex it.

    A marginal cable is far more likely to drop SuperSpeed when it's actually
    carrying data than when it's idle.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.frames = 0
        self.stalls = 0
        self.restarts = 0

    def run(self):
        while self.running:
            pipe = rs.pipeline()
            try:
                devs = rs.context().query_devices()
                if len(devs) == 0:
                    time.sleep(0.5)
                    continue
                modes = depth_modes(devs[0])
                if not modes:
                    time.sleep(0.5)
                    continue
                w, h, f = max(modes, key=lambda m: m[0] * m[1] * m[2])
                cfg = rs.config()
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
                cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, f)
                pipe.start(cfg)
            except RuntimeError:
                time.sleep(0.5)
                self.restarts += 1
                continue

            try:
                while self.running:
                    try:
                        pipe.wait_for_frames(1500)
                        self.frames += 1
                    except RuntimeError:
                        self.stalls += 1
                        break        # device likely re-enumerated — rebuild
            finally:
                try:
                    pipe.stop()
                except Exception:
                    pass

    def stop(self):
        self.running = False


def flex_test(duration, with_load):
    """Watch the link while you bend the cable, logging every speed change."""
    print("\n  D435 Cable Flex Test")
    print("  ────────────────────")
    print(f"  Watching for {duration}s. Bend, twist and wiggle the cable —")
    print("  especially right at the connector hoods, where cables fail first.")
    print("  Any drop from 5000 to 480 Mbps means the cable is losing SuperSpeed.\n")

    load = None
    if with_load:
        load = LoadGenerator()
        load.start()
        time.sleep(2.0)       # let the first pipeline come up
        print("  Load     : streaming at max mode\n")
    else:
        print("  Load     : none (idle link)\n")

    speed, busid, devnum = rs_link()
    if speed is None:
        print("  [!] RealSense not found — check the cable is seated.\n")
        if load:
            load.stop()
        return

    print(f"  t=0.0s   baseline: {speed:.0f} Mbps on {busid} (device {devnum})")
    start = time.time()
    events = []
    time_at = {}
    last_change = start
    prev = (speed, busid, devnum)

    try:
        while time.time() - start < duration:
            time.sleep(0.05)
            cur = rs_link()
            if cur == prev:
                continue

            now = time.time()
            time_at[prev[0]] = time_at.get(prev[0], 0.0) + (now - last_change)
            last_change = now
            t = now - start

            p_speed, p_bus, p_dev = prev
            c_speed, c_bus, c_dev = cur
            if c_speed is None:
                msg = "DISCONNECTED"
            elif p_speed is None:
                msg = f"reconnected at {c_speed:.0f} Mbps on {c_bus}"
            elif c_speed < p_speed:
                msg = f"DROPPED {p_speed:.0f} → {c_speed:.0f} Mbps"
            elif c_speed > p_speed:
                msg = f"recovered {p_speed:.0f} → {c_speed:.0f} Mbps"
            else:
                msg = f"re-enumerated (device {p_dev} → {c_dev})"

            print(f"  t={t:5.1f}s  {msg}")
            events.append((t, msg))
            prev = cur
    except KeyboardInterrupt:
        print("\n  (stopped early)")

    now = time.time()
    time_at[prev[0]] = time_at.get(prev[0], 0.0) + (now - last_change)
    if load:
        load.stop()
        load.join(timeout=3)

    print("\n  ── Result ──")
    total = sum(time_at.values()) or 1.0
    for spd in sorted(time_at, key=lambda s: -(s or 0)):
        label = "disconnected" if spd is None else f"{spd:.0f} Mbps"
        pct = 100 * time_at[spd] / total
        print(f"  {label:>14} : {time_at[spd]:6.1f}s  ({pct:.0f}%)")
    if load:
        print(f"  frames streamed: {load.frames}  stalls: {load.stalls}"
              f"  pipeline restarts: {load.restarts}")

    drops = [e for e in events if "DROPPED" in e[1] or "DISCONNECT" in e[1]]
    print()
    if not events:
        print("  VERDICT: rock solid — no link changes at all under flexing.")
    elif drops:
        print(f"  VERDICT: FAILED — {len(drops)} link drop(s) while flexing.")
        print("           This cable will silently fall back to USB 2 on a")
        print("           moving chassis, costing you the 60/90 fps modes.")
    else:
        print(f"  VERDICT: {len(events)} link change(s), but no drop below "
              "SuperSpeed.")
    print()


def verdict(mbps, descriptor):
    if mbps is None:
        return "UNKNOWN — RealSense not found on the USB bus"
    if mbps >= 5000:
        return "USB 3 — full bandwidth (cable and port are both fine)"
    if mbps >= 480:
        return ("USB 2 — BOTTLENECKED. Either the cable is USB 2 (or a "
                "charge-only/ribbon type missing the SuperSpeed pairs), or "
                "it's plugged into a USB 2 port.")
    return f"{mbps} Mbps — unexpected"


def main():
    mbps, product, busid = link_speed()

    print("\n  D435 USB Link Diagnostic")
    print("  ────────────────────────")
    print(f"  Device        : {product or 'not found'}  ({busid or '-'})")
    print(f"  Kernel link   : {mbps if mbps else '?'} Mbps")

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("\n  [!] No RealSense device found — check the cable is seated.\n")
        return
    dev = devices[0]

    try:
        descriptor = dev.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:
        descriptor = "?"
    try:
        fw = dev.get_info(rs.camera_info.firmware_version)
    except Exception:
        fw = "?"

    print(f"  librealsense  : USB {descriptor}")
    print(f"  Firmware      : {fw}")

    total3, busy3 = usb3_buses()
    print(f"  SuperSpeed    : {busy3}/{total3} USB 3 buses have a device attached")
    print(f"\n  VERDICT: {verdict(mbps, descriptor)}")
    if mbps and mbps < 5000:
        if total3 and not busy3:
            print("           Every USB 3 bus is empty — so the SuperSpeed pairs")
            print("           aren't getting through. Try the other cable, and make")
            print("           sure it's in a BLUE port (the black ones are USB 2).")
        elif busy3:
            print("           Another device is using SuperSpeed fine, so the port")
            print("           works — suspect this cable.")
    print()

    # Which high-bandwidth modes does the camera actually offer on this link?
    print("  High-bandwidth depth modes offered:")
    wanted = [(848, 480, 90), (848, 480, 60), (1280, 720, 30), (640, 480, 60)]
    available = set()
    for sensor in dev.query_sensors():
        for prof in sensor.get_stream_profiles():
            try:
                vp = prof.as_video_stream_profile()
            except Exception:
                continue
            if prof.stream_type() == rs.stream.depth:
                available.add((vp.width(), vp.height(), prof.fps()))
    for w, h, f in wanted:
        mark = "yes" if (w, h, f) in available else "NO  (USB2 restriction)"
        print(f"    {w}x{h} @ {f:>3} fps : {mark}")

    # Measured throughput under the heaviest mode this link actually allows.
    print("\n  Throughput test (depth + IR, highest mode this link allows)...")
    usable = [m for m in available if m in wanted]
    if usable:
        w, h, f = max(usable, key=lambda m: m[0] * m[1] * m[2])
    elif available:
        # USB2: fall back to whatever the camera is willing to offer.
        w, h, f = max(available, key=lambda m: m[0] * m[1] * m[2])
    else:
        print("    no usable depth profile found")
        return

    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
    cfg.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, f)
    pipe = rs.pipeline()
    try:
        pipe.start(cfg)
    except RuntimeError as exc:
        print(f"    could not start {w}x{h}@{f}: {exc}")
        return

    frames = 0
    bytes_rx = 0
    drops = 0
    last_num = None
    t0 = time.time()
    try:
        while time.time() - t0 < 8.0:
            try:
                fs = pipe.wait_for_frames(2000)
            except RuntimeError:
                drops += 1
                continue
            frames += 1
            num = fs.get_frame_number()
            if last_num is not None and num > last_num + 1:
                drops += num - last_num - 1
            last_num = num
            d = fs.get_depth_frame()
            ir = fs.get_infrared_frame(1)
            if d:
                bytes_rx += d.get_data_size()
            if ir:
                bytes_rx += ir.get_data_size()
    finally:
        pipe.stop()

    elapsed = time.time() - t0
    fps = frames / elapsed
    mbytes = bytes_rx / elapsed / 1e6
    print(f"    config      : {w}x{h} @ {f} fps  (depth + IR)")
    print(f"    delivered   : {fps:.1f} fps  ({100 * fps / f:.0f}% of requested)")
    print(f"    throughput  : {mbytes:.1f} MB/s  ({mbytes * 8:.0f} Mbit/s)")
    print(f"    dropped     : {drops} frames")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="D435 USB link diagnostic")
    ap.add_argument("--flex", action="store_true",
                    help="watch the link while you bend the cable")
    ap.add_argument("--seconds", type=int, default=60,
                    help="how long to watch in --flex mode (default 60)")
    ap.add_argument("--no-load", action="store_true",
                    help="don't stream during --flex (watch an idle link)")
    args = ap.parse_args()

    if args.flex:
        flex_test(args.seconds, with_load=not args.no_load)
    else:
        main()

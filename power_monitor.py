#!/usr/bin/env python3
"""
Live Pi 5 power monitor — run in a second terminal while realsense_yolo.py runs.

    python3 power_monitor.py

Samples the Pi 5's PMIC as fast as it can (~30/s) so short voltage dips are
caught, not just the averages a multimeter shows. Ctrl+C prints a summary.

What the Pi CAN'T tell you: the total current coming in on its 5V input. The
PMIC measures the Pi's own internal rails (CPU, RAM, I/O), not the USB ports,
so the D435 and thermal cam aren't included in "Pi rails". For the whole
system's draw, read the input power on the PSU and use:

    current into the Pi ≈ PSU watts × ~0.9 (buck efficiency) / 5 V
"""

import re
import subprocess
import time

UV_WARN = 4.75   # Pi 5 flags under-voltage a little above 4.6V; warn early

# The rail readings are measured AFTER the Pi's own regulators, so to turn them
# into current drawn from the 5V input we divide by their conversion
# efficiency. 85% is a rough typical figure for these regulators, not measured.
PMIC_EFF = 0.85

THROTTLE_BITS = {
    0: "UNDER-VOLTAGE NOW", 1: "freq capped now", 2: "THROTTLED NOW",
    3: "temp limit now",
    16: "under-voltage since boot", 17: "freq capped since boot",
    18: "throttled since boot", 19: "temp limit since boot",
}


def adc():
    out = subprocess.run(["vcgencmd", "pmic_read_adc"],
                         capture_output=True, text=True).stdout
    return {(n, k): float(v)
            for n, k, v in re.findall(r"(\S+)_(A|V) \w+\(\d+\)=([\d.]+)", out)}


def throttled():
    out = subprocess.run(["vcgencmd", "get_throttled"],
                         capture_output=True, text=True).stdout
    val = int(out.strip().split("=")[1], 16)
    return val, [msg for bit, msg in THROTTLE_BITS.items() if val & (1 << bit)]


def temp():
    out = subprocess.run(["vcgencmd", "measure_temp"],
                         capture_output=True, text=True).stdout
    return float(re.search(r"([\d.]+)", out).group(1))


def main():
    print("\n  Pi 5 power monitor — Ctrl+C to stop and show a summary")
    print("  EXT5V = voltage reaching the Pi.  Pi rails = Pi's own draw "
          "(excludes USB cameras).")
    print(f"  Amps = estimated current from the 5V input for the Pi itself "
          f"(rails ÷ EXT5V ÷ {PMIC_EFF:.0%} efficiency).\n")

    ext, rails, core, amps = [], [], [], []
    dips = 0
    t0 = last_print = time.time()
    try:
        while True:
            v = adc()
            e = v.get(("EXT5V", "V"), 0.0)
            r = sum(v[(n, "V")] * v[(n, "A")] for (n, k) in v
                    if k == "A" and (n, "V") in v)
            c = v.get(("VDD_CORE", "V"), 0) * v.get(("VDD_CORE", "A"), 0)
            a = r / e / PMIC_EFF if e > 0 else 0.0
            ext.append(e); rails.append(r); core.append(c); amps.append(a)
            if e < UV_WARN:
                dips += 1

            if time.time() - last_print >= 0.5:
                last_print = time.time()
                _, flags = throttled()
                state = ", ".join(flags) if flags else "OK"
                print(f"\r  EXT5V {e:.3f} V (min {min(ext):.3f})  |  "
                      f"Pi rails {r:4.2f} W ≈ {a:4.2f} A  core {c:4.2f} W  |  "
                      f"{temp():4.1f}°C  |  {state}" + " " * 10,
                      end="", flush=True)
    except KeyboardInterrupt:
        pass

    dur = time.time() - t0
    mean = lambda a: sum(a) / len(a)
    val, flags = throttled()
    print("\n\n  ── Summary ──")
    print(f"  {len(ext)} samples over {dur:.0f}s")
    print(f"  EXT5V     : min {min(ext):.3f} V  mean {mean(ext):.3f} V  "
          f"max {max(ext):.3f} V")
    print(f"  below {UV_WARN} V : {dips} samples ({100 * dips / len(ext):.1f}%)")
    print(f"  Pi rails  : {mean(rails):.2f} W avg, {max(rails):.2f} W peak")
    print(f"  Pi current: ≈{mean(amps):.2f} A avg, ≈{max(amps):.2f} A peak "
          f"at 5V  (estimate, Pi only — USB cameras not included)")
    print(f"  CPU core  : {mean(core):.2f} W avg, {max(core):.2f} W peak")
    print(f"  throttled : 0x{val:x}  {', '.join(flags) if flags else '(clean)'}\n")


if __name__ == "__main__":
    main()

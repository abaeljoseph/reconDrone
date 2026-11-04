import collections, json, logging, math, os, subprocess, sys, threading, time
import cv2, numpy as np, pyrealsense2 as rs
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from drive import Drive, TRACK, WHEEL_D
from leds import Lights
import thermal as th
# Phone remote: RealSense IR (or colour, or thermal) feed with person detection as MJPEG plus game-style touch sticks.
# Detection is the playground vision system (realsense_yolo.py): its ncnn_pose.py and YOLOv8n-pose NCNN models,
# IR imager 2 with the emitter off, a 90 deg rotated pass for people lying down, distance from depth. Plus body-heat
# blobs from the thermal camera (thermal.py) mapped onto the D435 view, for people lying down or half covered.
# Each new person in view is snapshotted to ~/snaps (jpg + json, browse at /snaps/) and pinned on the odometry map.
#   python3 web.py [PORT] [HTTP_PORT]   then open http://<pi>:8080 on the phone (same WiFi)
# Left stick: distance from centre = speed, angle = how hard it curves (sideways pivots on the inner wheel),
# pulled back to one side = spin on the spot that way. Right stick left/right = spin on the spot.
# No reverse: with only the rear kickstand the hub tips forward instead of backing up (2026-10-30).
FWD_MAX   = 8.0    # rad/s outer wheel speed at full left stick
SPIN_MAX  = 3.0     # rad/s per side at full right stick
DEADMAN   = 0.3     # s without a /cmd from the page -> stop
ACCEL     = 5.0     # rad/s^2, drive_fw ramp; used to stop the 180 spin early so the ramp-down lands on it
JPEG_Q    = 70
VISION_DIR = os.path.expanduser('~/playground')
MODEL_MAIN, SHAPE_MAIN = 'yolov8n-pose-169_ncnn_model', (192, 320)   # landscape
MODEL_ROT,  SHAPE_ROT  = 'yolov8n-pose-rot_ncnn_model', (320, 192)   # portrait, for the rotated pass
NCNN_THREADS, INFER_MAX_FPS = 2, 30                                  # as realsense_yolo.py
CONF_IR, CONF_RGB = 0.4, 0.25   # the COCO-trained model misfires more on IR; RGB as realsense_yolo.py
# MegaDetector v6 (md_net.py), on trial for IR: trained on camera-trap photos, many of them night IR. Boxes only.
MD_DIR = os.path.expanduser('~/models')
MODEL_MD, SHAPE_MD = 'md6c_256x320_ncnn_model', (256, 320)
MODEL_MD_ROT, SHAPE_MD_ROT = 'md6c_320x256_ncnn_model', (320, 256)
CONF_MD = 0.25
# IR: when pose finds nobody, MegaDetector on a contrast-equalised copy. Off: in the report_tests lit run (2026-10-31)
# it only helped sitting, which heat covers too, and halved the rate (16 fps) with nobody in view.
FALLBACK_MD = False
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# Thermal: the only detector that found people lying across or half covered in the lit run (99%, no false alarms).
# Same direction as the D435, narrower view: thermal px -> D435 px is a scale fx / TH_FX and an offset.
TH_FX = 237.8          # thermal focal length, px of its 256 wide frame (calib_thermal.py, 2026-11-01, 619 pairs at 2 m)
TH_DX, TH_DY = 3.2, 11.4   # thermal px the D435 centre sits off the thermal centre (same run; not mirrored)
TH_MAX_AGE = 0.3       # s: older heat frames are not used
HEAT_CONF = (0.7, 0.6)    # heat-only person (upright, lying): blob peak this x 10 C above the scene, 7 / 6 C. People
                          # lying scored 0.69-0.83, the couch's warm patch after sitting 0.56 (2026-11-01)
VOTE_N, VOTE_K = 8, 5           # people show once someone was seen in VOTE_K of the last VOTE_N frames
                                # (was 3 of 5: brief false flashes on colour, 2026-11-01)
PERSON_SPAN = (0.4, 2.4)        # m: box size x depth must fit a person, head and shoulders up to lying full length
PIN_HITS = 5                    # detections merged into a pin before it shows on the map
IR_INDEX = 2        # imager beside the thermal cam, as realsense_yolo.py
# 848x480 is the D435's full 16:9 view but needs USB 3; on USB 2 it only runs at 10 fps, so fall back to 640x480
IR_MODES  = [(848, 480, 30), (640, 480, 30)]
RGB_MODES = [(640, 480, 30), (640, 480, 15)]   # colour + depth may not fit USB 2 at 30
STOP_DIST = 0.45    # m: anything this close ahead blocks forward drive; the ramp-down from FWD_MAX takes ~0.15 m of it
SNAP_DIR  = os.path.expanduser('~/snaps')
PIN_MERGE = 1.0     # m: a detection this close to a map pin is the same person
CONTACT_GAP = 2.0   # s with nobody in view before a detection counts as a new contact
CELLS = 4           # 2x 2S 18650 packs in series
# 18650 resting voltage per cell -> charge %, rough; under load it reads a little low. 3.0 V/cell (12 V) is empty.
CELL_CURVE = ([3.00, 3.30, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00, 4.10, 4.20], [0, 5, 10, 20, 35, 50, 62, 75, 88, 100])
BATT_WINDOW = 600   # s of charge history for the time-left estimate...
BATT_MIN_TIME, BATT_MIN_DROP = 180, 2   # ...shown once it spans 3 min and the charge fell 2 %
BATT_REST = 3       # s after the wheels stop before the voltage counts (it recovers from the motor sag)
BATT_SWAP = 5       # % rise that means a fresh battery
STATS_DIR, STATS_EVERY = os.path.expanduser('~/stats'), 10   # one CSV per run, a row every 10 s
# Bar RGB, as the HUD: blue all fine, orange a person in view, blinking red motor board or battery trouble. Kept dim:
# all six on full white is 360 mA off the Pi's 5 V rail.
RGB_BLUE, RGB_ORANGE, RGB_RED, RGB_WHITE = (0, 40, 120), (120, 50, 0), (120, 0, 0), (120, 120, 120)

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)   # /cmd at 20 Hz would flood the log
d = None
lights = None
ir, rgb = False, 'status'                     # IR LEDs on, RGB mode: 'status', 'white' or 'off'
lock = threading.Lock()
last_cmd, spinning, jpeg = 0.0, False, None
infer_cv = threading.Condition()
infer_in, infer_ver, people = None, 0, []    # people: (x1, y1, x2, y2, conf, dist m, lying)
people_n = []                                 # map pin number of each, None until its pin shows
batt_hist, batt, batt_mins = [], None, None   # [(time, %)], resting charge %, minutes left at the current drain
view, autohalt, halted, blocked = 'ir', False, False, False   # halted: stopped for a new person until sticks let go
overlay = True                                # body heat tinted over the IR / colour view
avoid = True                                  # obstacle stop; on again every restart
overlay_cache = (None, None)                  # (key, (colour, alpha) warped to the view)
obstacle, ai_fps, frame = None, 0.0, None     # nearest thing ahead m, inference rate, last annotated frame
path, pins, contacts, snap_count = [], [], 0, 0
infer_n = 0                                   # inferences since start, for the true average rate in stats
heat = (0.0, None, [])                        # last thermal frame: time, temps C, blobs in thermal px

def battery_pct(v): return float(np.interp(v / CELLS, *CELL_CURVE))

def battery():
    """Charge from the resting voltage only (the motors sag it, which swung the time left from 13 to 300+ min),
    smoothed and never rising unless it jumps by BATT_SWAP. Time left from its drop over BATT_WINDOW, once that is
    long enough and has dropped by BATT_MIN_DROP; None until then, and on the bench PSU."""
    global batt, batt_mins
    moved = 0.0
    while True:
        time.sleep(2)
        now = time.time()
        if d.cmd or spinning: moved = now
        if d.vin is None or now - moved < BATT_REST: continue
        p = battery_pct(d.vin)
        batt = p if batt is None or p > batt + BATT_SWAP else min(batt, 0.9 * batt + 0.1 * p)
        if batt_hist and batt > batt_hist[-1][1] + 1: batt_hist.clear()      # charged or swapped
        batt_hist.append((now, batt))
        while batt_hist[0][0] < now - BATT_WINDOW: batt_hist.pop(0)
        t, q = np.array(batt_hist).T
        if t[-1] - t[0] < BATT_MIN_TIME or q[0] - q[-1] < BATT_MIN_DROP: batt_mins = None; continue
        slope = np.polyfit(t - now, q, 1)[0]                 # %/s
        batt_mins = batt / -slope / 60 if slope < 0 else None

def pi_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f: return int(f.read()) / 1000

FAN_HW = '/sys/devices/platform/cooling_fan/hwmon'
fan_low_since = None
def fan():
    """(rpm, stalled): stalled once the Pi has driven the fan for 3 s and it reads under 300 rpm (a wire caught in
    the blades on 2026-11-01 stopped it and the Pi overheated). (None, False) without a fan."""
    global fan_low_since
    try:
        h = os.path.join(FAN_HW, os.listdir(FAN_HW)[0])
        rpm, pwm = (int(open(os.path.join(h, n)).read()) for n in ('fan1_input', 'pwm1'))
    except (OSError, IndexError, ValueError): return None, False
    if pwm > 0 and rpm < 300: fan_low_since = fan_low_since or time.time()
    else: fan_low_since = None
    return rpm, bool(fan_low_since and time.time() - fan_low_since > 3)

def cpu_times():
    with open('/proc/stat') as f: v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]                # total, idle + iowait

def stats():
    """~/stats/<start>.csv: average AI rate, CPU, temperatures and supply over each STATS_EVERY s, to see heat-up."""
    os.makedirs(STATS_DIR, exist_ok=True)
    name = os.path.join(STATS_DIR, time.strftime('%Y%m%d-%H%M%S') + '.csv')
    with open(name, 'w') as f:
        f.write('time,pi_up_min,view,ai_fps,cpu_pct,pi_temp,board_temp,vin,wheel_rad_s,people,throttled\n')
    n0, c0, t0 = infer_n, cpu_times(), time.time()
    while True:
        time.sleep(STATS_EVERY)
        n, c, t = infer_n, cpu_times(), time.time()
        busy = 100 * (1 - (c[1] - c0[1]) / max(1, c[0] - c0[0]))
        with open('/proc/uptime') as f: up = float(f.read().split()[0]) / 60
        try: thr = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True).stdout.strip().split('=')[-1]
        except Exception: thr = ''
        row = [time.strftime('%H:%M:%S'), f'{up:.1f}', view, f'{(n - n0) / (t - t0):.1f}', f'{busy:.0f}',
               f'{pi_temp():.1f}', '' if d.temp is None else f'{d.temp:.1f}', '' if d.vin is None else f'{d.vin:.2f}',
               f'{(abs(d.velL) + abs(d.velR)) / 2:.1f}', len(people), thr]
        with open(name, 'a') as f: f.write(','.join(map(str, row)) + '\n')
        n0, c0, t0 = n, c, t

def start_camera(want):
    """IR2 or colour plus depth at the first mode the USB link takes. Returns pipe, depth scale, k, align, (fx, cx)."""
    rgb = want == 'rgb'
    for w, h, fps in RGB_MODES if rgb else IR_MODES:
        pipe = rs.pipeline(); cfg = rs.config()
        if rgb: cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        else: cfg.enable_stream(rs.stream.infrared, IR_INDEX, w, h, rs.format.y8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        try: prof = pipe.start(cfg)
        except RuntimeError: continue
        sensor = prof.get_device().first_depth_sensor()
        sensor.set_option(rs.option.emitter_enabled, 1 if rgb else 0)   # the dots only show in IR, and help depth
        im = prof.get_stream(rs.stream.color) if rgb else prof.get_stream(rs.stream.infrared, IR_INDEX)
        intr = im.as_video_stream_profile().get_intrinsics()
        # Depth is seen from IR1, the IR image from IR2: a point sits k/z px further along in the depth map.
        # Colour gets depth aligned to it instead, so no shift.
        k = 0.0 if rgb else intr.fx * prof.get_stream(rs.stream.depth).get_extrinsics_to(im).translation[0]
        print(f'camera: {"colour" if rgb else f"IR{IR_INDEX}"} + depth {w}x{h} @{fps}', flush=True)
        return pipe, sensor.get_depth_scale(), k, rs.align(rs.stream.color) if rgb else None, (intr.fx, w - 1 - intr.ppx)
    raise RuntimeError(f'no {want} + depth mode would start')

def camera():
    global jpeg, frame, obstacle, infer_in, infer_ver
    while True:
        pipe, want = None, 'rgb' if view == 'rgb' else 'ir'   # thermal view keeps the D435 on IR for depth
        try:
            pipe, scale, k, align, lens = start_camera(want)
            while (view == 'rgb') == (want == 'rgb'):
                fs = pipe.wait_for_frames(2000)
                if align: fs = align.process(fs)
                im = fs.get_color_frame() if align else fs.get_infrared_frame(IR_INDEX)
                dep = fs.get_depth_frame()
                if not im or not dep: continue
                # D435 is mounted upside down; depth is turned with it so display coords index both
                img = cv2.rotate(np.asanyarray(im.get_data()), cv2.ROTATE_180)
                if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                z = cv2.rotate(np.asanyarray(dep.get_data()), cv2.ROTATE_180).astype(np.float32) * scale
                obstacle = nearest(z)
                with infer_cv:
                    infer_in, infer_ver = (img, z, k, lens, want), infer_ver + 1; infer_cv.notify()
                t0, t, _ = heat
                if view == 'thermal' and t is not None and time.time() - t0 < 1:
                    out = cv2.cvtColor(th.gray(t), cv2.COLOR_GRAY2BGR)
                    draw(out, [view_to_th(p, lens, img.shape[0], out.shape[1::-1]) for p in people], people_n)
                else:
                    out = img.copy()
                    if overlay and t is not None and time.time() - t0 < 1: heat_tint(out, t0, t, lens)
                    draw(out, people, people_n)
                frame = out
                if out.shape[1] * 9 < out.shape[0] * 16:     # 4:3: crop the middle to 16:9 so it fills a landscape phone
                    c = (out.shape[0] - out.shape[1] * 9 // 16) // 2; out = out[c:out.shape[0] - c]
                jpeg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1].tobytes()
            pipe.stop()                                     # view switched
        except Exception as e:
            print('camera:', e, flush=True); jpeg = frame = obstacle = None
            try: pipe and pipe.stop()
            except Exception: pass
            time.sleep(2)

def heat_cam():
    """Thermal frames and their body-heat blobs into heat; retries every 5 s if the camera is missing."""
    global heat
    while True:
        cam = None
        try:
            cam = th.Thermal(); cam.warm(); print('thermal: on', flush=True)
            while True:
                t = cam.read(); heat = (time.time(), t, th.heat_people(t, (th.W, th.H)))
        except Exception as e:
            print('thermal:', e, flush=True); heat = (0.0, None, [])
            if cam: cam.close()
            time.sleep(5)

def th_M(lens, h):
    """Thermal px -> D435 display px as an affine map. A live fit to the person box was tried and jittered (2026-11-01)."""
    s = lens[0] / TH_FX
    return np.float32([[s, 0, lens[1] + s * (TH_DX - th.W / 2)], [0, s, h / 2 + s * (TH_DY - th.H / 2)]])

def th_to_view(x, y, lens, h):
    """Thermal px -> D435 display px."""
    M = th_M(lens, h)
    return M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]

def heat_tint(img, t0, t, lens):
    """Blend body heat onto the D435 view in place: inferno colours, fading in from 2 C above the scene median, so
    cool background stays the plain camera image. Warped once per thermal frame."""
    global overlay_cache
    h, w = img.shape[:2]
    key = (t0, w, h, lens)
    if overlay_cache[0] != key:
        rel = t - float(np.median(t))
        g = np.clip((rel - th.T_SPAN[0]) * (255 / (th.T_SPAN[1] - th.T_SPAN[0])), 0, 255).astype(np.uint8)
        a = np.clip((rel - 2) / 8, 0, 1).astype(np.float32) * 0.7
        M = th_M(lens, h); s = M[0, 0]
        ys, xs = np.nonzero(a)                               # warp only the warm part's box
        x0 = y0 = x1 = y1 = 0
        if len(xs):
            x0, x1 = max(0, int(s * (xs.min() - 1) + M[0, 2])), min(w, int(s * (xs.max() + 2) + M[0, 2]) + 1)
            y0, y1 = max(0, int(s * (ys.min() - 1) + M[1, 2])), min(h, int(s * (ys.max() + 2) + M[1, 2]) + 1)
        if x1 > x0 and y1 > y0:
            M[:, 2] -= (x0, y0); size = (x1 - x0, y1 - y0)
            col, a = cv2.warpAffine(cv2.applyColorMap(g, cv2.COLORMAP_INFERNO), M, size), cv2.warpAffine(a, M, size)
            overlay_cache = (key, ((slice(y0, y1), slice(x0, x1)), col, a, 1 - a))
        else: overlay_cache = (key, None)
    if overlay_cache[1]:                                     # one C call on the warm part: AI fell 30 -> 20 fps blending all in numpy
        roi, col, a, b = overlay_cache[1]
        img[roi] = cv2.blendLinear(col, img[roi], a, b)

def view_to_th(p, lens, h, size):
    """A detection in D435 display px -> thermal display px at size, for drawing on the thermal view."""
    M, sx, sy = th_M(lens, h), size[0] / th.W, size[1] / th.H
    x1, x2 = (int((x - M[0, 2]) / M[0, 0] * sx) for x in p[0:3:2])
    y1, y2 = (int((y - M[1, 2]) / M[1, 1] * sy) for y in p[1:4:2])
    return (x1, y1, x2, y2) + tuple(p[4:])

def heat_boxes(lens, shape):
    """Fresh heat blobs as D435 view boxes (x1, y1, x2, y2, conf, lying), or None if the thermal frame is stale."""
    t0, _, blobs = heat
    if time.time() - t0 > TH_MAX_AGE: return None
    h, w = shape[:2]; out = []
    for x1, y1, x2, y2, conf, _, lying in blobs:
        (a, b), (c, e) = th_to_view(x1, y1, lens, h), th_to_view(x2, y2, lens, h)
        out.append((int(max(0, a)), int(max(0, b)), int(min(w - 1, c)), int(min(h - 1, e)), conf, lying))
    return out

def warm(p, hb):
    """A lying (rotated pass) box counts only if body heat overlaps it: on colour at conf 0.25 the rotated pass called
    the couch a person lying (2026-11-01). With no fresh heat frame it needs CONF_IR instead."""
    if hb is None: return p[4] >= CONF_IR
    return any(min(p[2], q[2]) > max(p[0], q[0]) and min(p[3], q[3]) > max(p[1], q[1]) for q in hb)

def heat_found(z, k, lens, hb, found):
    """Fresh heat blobs (heat_boxes) with depth, except those centred in a box already found. Only the size
    maximum applies: a head showing from under debris is smaller than PERSON_SPAN allows. Heat alone has to be skin
    hot (HEAT_CONF): the warm patch left on the couch after sitting was called a person lying (2026-11-01)."""
    out = []
    for a, b, c, e, conf, lying in hb or []:
        if conf < HEAT_CONF[lying]: continue
        cx, cy = (a + c) // 2, (b + e) // 2
        if a >= c or b >= e or any(p[0] <= cx <= p[2] and p[1] <= cy <= p[3] for p in found): continue
        dist = dist_at(z, k, cx, cy)
        if not dist or max(c - a, e - b) * dist / lens[0] <= PERSON_SPAN[1]: out.append((a, b, c, e, conf, dist, lying))
    return out

def nearest(z):
    """Closest thing ahead (m): 5th percentile of the valid depth in the middle of the view, above the floor."""
    h, w = z.shape
    p = z[int(h * 0.25):int(h * 0.6), int(w * 0.3):int(w * 0.7)]
    p = p[p > 0]
    return float(np.percentile(p, 5)) if p.size > 500 else None

def dist_at(z, k, x, y):
    """Median depth (m) round a display point, shifted for the IR2/IR1 baseline; 0 if no reading."""
    h, w = z.shape; d = 0.0
    for _ in range(3):   # the shift depends on the depth we're looking up, so refine from the first guess
        sx = int(round(x + (k / d if d else 0)))    # + not -: the image is turned 180
        p = z[max(0, y - 4):y + 5, max(0, min(sx, w - 1) - 4):min(sx, w - 1) + 5]
        p = p[p > 0]
        if not p.size: return d
        d = float(np.median(p))
    return d

def box_dist(z, k, x1, y1, x2, y2):
    """Distance (m) of what fills a box: 20th percentile of the depth in its middle half, so the wall showing round a
    person side on doesn't count. dist_at at the centre read the wall and the size check threw out a standing person
    the pose net scored 0.83 (2026-11-01). Shifted for the IR2/IR1 baseline like dist_at."""
    h, w = z.shape; d = 0.0
    bw, bh = x2 - x1, y2 - y1
    for _ in range(2):
        s = int(round(k / d)) if d else 0
        p = z[max(0, y1 + bh // 4):min(h, y2 - bh // 4), max(0, x1 + bw // 4 + s):min(w, x2 - bw // 4 + s)]
        p = p[p > 0]
        if p.size < 20: return d
        d = float(np.percentile(p, 20))
    return d

def load_nets():
    sys.path.insert(0, VISION_DIR)
    from ncnn_pose import PoseNet
    nets = (PoseNet(os.path.join(VISION_DIR, MODEL_MAIN), SHAPE_MAIN, threads=NCNN_THREADS),
            PoseNet(os.path.join(VISION_DIR, MODEL_ROT), SHAPE_ROT, threads=NCNN_THREADS))
    print('vision: models loaded', flush=True)
    return nets

def load_md():
    from md_net import MegaDetector
    return (MegaDetector(os.path.join(MD_DIR, MODEL_MD), SHAPE_MD, threads=NCNN_THREADS),
            MegaDetector(os.path.join(MD_DIR, MODEL_MD_ROT), SHAPE_MD_ROT, threads=NCNN_THREADS))

def detect(net, net_rot, img, z, k, conf, rot=True):
    """People in one frame as (x1, y1, x2, y2, conf, dist m, lying). The rotated pass runs if the main one finds nobody."""
    h = img.shape[0]; found = []
    for b in net(img, conf=conf):
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        found.append((x1, y1, x2, y2, b.conf[0], box_dist(z, k, x1, y1, x2, y2), False))
    if rot and not found:   # someone lying flat looks standing when the frame is turned 90 deg
        for b in net_rot(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), conf=conf):
            rx1, ry1, rx2, ry2 = map(float, b.xyxy[0])   # 90 CW maps (x, y) -> (h-1-y, x)
            x1, y1, x2, y2 = int(ry1), int(h - 1 - rx2), int(ry2), int(h - 1 - rx1)
            if x1 < x2 and y1 < y2:
                found.append((x1, y1, x2, y2, b.conf[0], box_dist(z, k, x1, y1, x2, y2), True))
    return found

def find_people(nets, md, img, z, k, fx, src, rot=True):
    """Every box and the person-sized ones in a BGR frame: pose, then on IR MegaDetector on a CLAHE copy if pose
    finds nobody person sized."""
    h, w = img.shape[:2]
    raw = detect(*nets, img, z, k, CONF_RGB if src == 'rgb' else CONF_IR, rot)
    found = [p for p in raw if plausible(p, fx, w, h)]
    if md and src == 'ir' and not found:
        more = detect(*md, cv2.cvtColor(CLAHE.apply(img[..., 0]), cv2.COLOR_GRAY2BGR), z, k, CONF_MD, rot)
        raw += more; found = [p for p in more if plausible(p, fx, w, h)]
    return raw, found

def infer():
    global people, people_n, ai_fps, infer_n
    nets, md = load_nets(), load_md() if FALLBACK_MD else None
    seen, last, last_seen, votes = 0, 0.0, 0.0, collections.deque(maxlen=VOTE_N)
    while True:
        time.sleep(max(0.0, last + 1 / INFER_MAX_FPS - time.time()))
        with infer_cv:
            while infer_ver == seen: infer_cv.wait(1)
            (img, z, k, lens, src), seen = infer_in, infer_ver
        now = time.time(); ai_fps += 0.1 * (1 / max(now - last, 1e-3) - ai_fps); infer_n += 1
        last = now
        hb = heat_boxes(lens, img.shape)
        found = [p for p in find_people(nets, md, img, z, k, lens[0], src)[1] if not p[6] or warm(p, hb)]
        found += heat_found(z, k, lens, hb, found)
        votes.append(bool(found))
        if sum(votes) < VOTE_K: found = []                  # one-frame misfires never show
        pose = d.pose()
        qs = [pin(p, pose, lens) if p[5] else None for p in found]
        people, people_n = found, [q.get('n') if q else None for q in qs]
        if found:
            if now - last_seen > CONTACT_GAP: contact(img, found)
            last_seen = now

def plausible(p, fx, w, h):
    """Box size x depth is person sized. Whole-frame boxes fail. Boxes cut off by the frame edge skip the minimum,
    boxes with no depth pass."""
    x1, y1, x2, y2, _, dist, _ = p
    if x2 - x1 > 0.9 * w and y2 - y1 > 0.9 * h: return False
    if not dist: return True
    span = max(x2 - x1, y2 - y1) * dist / fx
    if x1 <= 2 or y1 <= 2 or x2 >= w - 3 or y2 >= h - 3: return span <= PERSON_SPAN[1]
    return PERSON_SPAN[0] <= span <= PERSON_SPAN[1]

def contact(img, ps):
    """Someone new came into view: halt if asked, and snapshot it as evidence."""
    global contacts, halted
    contacts += 1
    if autohalt:
        with lock: halted = True; d.stop()
    shot = img.copy(); draw(shot, ps); snapshot(shot, ps, 'contact')

def pin(p, pose, lens):
    """Put a detection on the map: bearing from the box centre, range from depth; merges into a pin within PIN_MERGE.
    Returns the pin. It gets its number n (1, 2, ...) when it shows, after PIN_HITS detections."""
    x, y, th = pose; fx, cx = lens
    b = math.atan(((p[0] + p[2]) / 2 - cx) / fx)            # + = right of the view axis
    r = p[5] / math.cos(b)                                  # depth is measured along the view axis
    px, py = x + r * math.cos(th - b), y + r * math.sin(th - b)
    for q in pins:
        if math.hypot(q['x'] - px, q['y'] - py) < PIN_MERGE:
            q['x'] += 0.2 * (px - q['x']); q['y'] += 0.2 * (py - q['y']); q['conf'] = max(q['conf'], float(p[4]))
            q['hits'] += 1
            if q['hits'] == PIN_HITS: q['n'] = 1 + max((r.get('n', 0) for r in pins), default=0)
            return q
    pins.append(dict(x=px, y=py, conf=float(p[4]), lying=bool(p[6]), hits=1))
    return pins[-1]

def snapshot(img, ps, why):
    """Save the frame and what was known about it to SNAP_DIR; returns the file name or None if no frame."""
    global snap_count
    if img is None: return None
    os.makedirs(SNAP_DIR, exist_ok=True)
    t = time.time(); name = time.strftime('%Y%m%d-%H%M%S', time.localtime(t)) + f'-{int(t * 1000) % 1000:03d}-{why}'
    cv2.imwrite(os.path.join(SNAP_DIR, name + '.jpg'), img)
    x, y, th = d.pose()
    with open(os.path.join(SNAP_DIR, name + '.json'), 'w') as f:
        json.dump(dict(time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t)), why=why, view=view,
                       x=x, y=y, heading_deg=math.degrees(th), vin=d.vin,
                       people=[dict(box=[int(v) for v in p[:4]], conf=float(p[4]), dist=float(p[5]), lying=bool(p[6]))
                               for p in ps]), f, indent=1)
    snap_count += 1
    return name

def track():
    """Odometry trail for the mini-map, a point every 5 cm."""
    while True:
        time.sleep(0.2)
        x, y, _ = d.pose()
        if not path or math.hypot(x - path[-1][0], y - path[-1][1]) > 0.05:
            path.append((round(x, 3), round(y, 3)))
            if len(path) > 5000: del path[:1000]

def draw(img, ps, ns=()):
    """Thin box round each person, clipped so every edge stays on screen, and above it the same numbered map pin as
    on the phone's compass (! until the pin shows) with the distance under it. Orange like the HUD's person colour."""
    h, w = img.shape[:2]
    col, font, aa = (26, 140, 255), cv2.FONT_HERSHEY_DUPLEX, cv2.LINE_AA
    for i, (x1, y1, x2, y2, conf, dist, lying) in enumerate(ps):
        x1, y1, x2, y2 = max(1, x1), max(1, y1), min(w - 2, x2), min(h - 2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        n = ns[i] if i < len(ns) and ns[i] else None
        label = ('LYING ' if lying else '') + (f'{dist:.1f} M' if dist else '')
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cx = min(max((x1 + x2) // 2, tw // 2 + 4, 14), w - max(tw // 2 + 4, 14))
        my = y1 - th - 30 if y1 > th + 44 else y1 + 16   # pin head above the box, or inside its top edge at the frame top
        # teardrop: head r11 with lines tangent to it meeting at a tip 24 px below its centre
        a = math.degrees(math.acos(11 / 24))
        head = cv2.ellipse2Poly((cx, my), (11, 11), 0, int(90 + a), int(450 - a), 10)
        cv2.fillPoly(img, [np.vstack([[(cx, my + 24)], head])], (31, 210, 255), aa)   # yellow, as on the compass
        t = str(n) if n else '!'
        (nw, nh), _ = cv2.getTextSize(t, font, 0.5, 1)
        cv2.putText(img, t, (cx - nw // 2, my + nh // 2), font, 0.5, (0, 0, 0), 1, aa)
        for tk, c in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(img, label, (cx - tw // 2, my + th + 30), font, 0.5, c, tk, aa)

def bar():
    """Bar RGB at 10 Hz. Sent every time, not on change, so LEDs reset by a dip on the 5 V rail come straight back."""
    while True:
        time.sleep(0.1)
        if rgb == 'white': c = RGB_WHITE
        elif rgb == 'off': c = (0, 0, 0)
        elif not d.ok() or (d.vin and batt is not None and batt < 10): c = RGB_RED if time.time() % 1 < 0.5 else (0, 0, 0)
        else: c = RGB_ORANGE if people else RGB_BLUE
        lights.rgb(*c)

def deadman():
    while True:
        time.sleep(0.05)
        with lock:
            if not spinning and d.cmd and time.time() - last_cmd > DEADMAN: d.stop()

def spin180():
    global spinning
    target = math.pi * TRACK / 2 / (WHEEL_D / 2)          # rad of wheel turn per side for half a turn
    early = SPIN_MAX ** 2 / (2 * ACCEL)                     # rad covered while ramping down
    a0 = d.prev[1:] if d.prev else None
    d.wheels(-SPIN_MAX, SPIN_MAX); t0 = time.time()
    while a0 and time.time() - t0 < 6:
        if ((d.prev[2] - a0[1]) - (d.prev[1] - a0[0])) / 2 >= target - early: break
        time.sleep(0.01)
    d.wheels(0, 0); time.sleep(SPIN_MAX / ACCEL + 0.1); d.stop()
    with lock: spinning = False

@app.route('/')
def index():
    r = send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web.html'))
    r.headers['Cache-Control'] = 'no-store'   # the phone kept showing an old page after updates
    return r

@app.route('/fonts/<name>')
def fonts(name):
    """Served from the Pi so the HUD font also works on the hotspot with no internet."""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts'), name, max_age=86400)

@app.route('/manifest.json')
def manifest():
    """Add to Home Screen then opens the page fullscreen and sideways."""
    return jsonify(name='Recon Drone', short_name='Recon', start_url='/', display='fullscreen',
                   orientation='landscape', background_color='#0a0e14', theme_color='#0a0e14')

@app.route('/video')
def video():
    def gen():
        sent = None
        while True:
            j = jpeg
            if j is None or j is sent: time.sleep(0.005); continue
            sent = j
            yield b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(j)).encode() + b'\r\n\r\n' + j + b'\r\n'
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=f')

@app.route('/frames')
def frames():
    """Same JPEGs as /video, each behind a 4-byte big-endian length; the page reads it with fetch so it can
    tell when the stream stalls and reopen it (an MJPEG <img> just freezes or goes black)."""
    def gen():
        sent = None
        while True:
            j = jpeg
            if j is None or j is sent: time.sleep(0.005); continue
            sent = j
            yield len(j).to_bytes(4, 'big') + j
    return Response(gen(), mimetype='application/octet-stream', headers={'Cache-Control': 'no-cache'})

@app.route('/cmd', methods=['POST'])
def cmd():
    """{lx, ly, rx} sticks -1..1, y up = +, x right = +; {spin: true} once for the 180."""
    global last_cmd, spinning, halted, blocked
    j = request.get_json(force=True, silent=True) or {}
    lx, ly, rx = (max(-1.0, min(1.0, float(j.get(k, 0)))) for k in ('lx', 'ly', 'rx'))
    with lock:
        last_cmd = time.time()
        if spinning: return jsonify(ok=True, spinning=True)
        if j.get('spin'):
            spinning = True; threading.Thread(target=spin180, daemon=True).start()
            return jsonify(ok=True, spinning=True)
        if halted:                                          # stopped for a new person: hold until the sticks are let go
            if not (lx or ly or rx): halted = False
            d.stop(); return jsonify(ok=True, halted=True)
    L = R = 0.0
    if ly >= 0:                                             # upper half drives
        m = min(1.0, math.hypot(lx, ly))
        v = FWD_MAX * m ** 2                                # squared for fine control at low speed
        r = math.atan2(lx, ly) / (math.pi / 2) if m else 0.0   # stick angle: -1 full left .. +1 full right
        L, R = v * (1 - max(0.0, -r)), v * (1 - max(0.0, r))   # slow the inner wheel, never reverse it
    blocked = avoid and bool(L or R) and obstacle is not None and obstacle < STOP_DIST
    if blocked: L = R = 0.0                                 # too close ahead: turning on the spot only
    # Right stick, or left stick pulled back to one side (straight back does nothing): on the spot, + = turn right
    s = max(-SPIN_MAX, min(SPIN_MAX, SPIN_MAX * (rx * abs(rx) + (lx * abs(lx) if ly < 0 else 0))))
    L, R = L + s, R - s
    if L or R: d.wheels(L, R)
    else: d.stop()
    return jsonify(ok=True, spinning=False)

@app.route('/stop', methods=['POST'])
def stop():
    d.stop(); return jsonify(ok=True)

def state():
    x, y, th = d.pose()
    return dict(state=d.state, ok=d.ok(), velL=d.velL, velR=d.velR, hub=d.hub, x=x, y=y, th=th,
                spinning=spinning, camera=jpeg is not None,
                people=[{'conf': p[4], 'dist': p[5], 'lying': p[6]} for p in people],
                vin=d.vin, batt=batt, batt_mins=batt_mins,
                board_temp=d.temp, pi_temp=pi_temp(), fan=fan(), view=view, overlay=overlay, autohalt=autohalt, halted=halted, ir=ir, rgb=rgb, avoid=avoid,
                obstacle=obstacle, blocked=blocked, stop_dist=STOP_DIST, ai_fps=ai_fps,                contacts=contacts, snaps=snap_count, found=sum(q['hits'] >= PIN_HITS for q in pins))

@app.route('/status')
def status(): return jsonify(state())

@app.route('/events')
def events():
    """state() as server-sent events at 10 Hz, so the heading and map keep up without polling."""
    def gen():
        while True:
            yield 'data: ' + json.dumps(state()) + '\n\n'
            time.sleep(0.1)
    return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache'})

@app.route('/ping')
def ping(): return 'ok'

@app.route('/opts', methods=['POST'])
def opts():
    """{view: 'ir' | 'rgb' | 'thermal', overlay: bool, autohalt: bool, ir: bool, rgb: 'status' | 'white' | 'off', avoid: bool}, any of them."""
    global view, overlay, autohalt, halted, ir, rgb, avoid
    j = request.get_json(force=True, silent=True) or {}
    if j.get('view') in ('ir', 'rgb', 'thermal'): view = j['view']
    if 'overlay' in j: overlay = bool(j['overlay'])
    if 'autohalt' in j: autohalt = bool(j['autohalt']); halted = halted and autohalt
    if 'ir' in j: ir = bool(j['ir']); lights.ir(ir)
    if 'avoid' in j: avoid = bool(j['avoid'])
    if j.get('rgb') in ('status', 'white', 'off'): rgb = j['rgb']
    return jsonify(ok=True)

@app.route('/map')
def map_():
    p = path[::max(1, len(path) // 500)] + path[-1:]       # thinned to ~500 points
    return jsonify(path=p, pins=[q for q in pins if q['hits'] >= PIN_HITS])

@app.route('/map/clear', methods=['POST'])
def map_clear():
    path.clear(); pins.clear(); d.reset_pose(); return jsonify(ok=True)

@app.route('/snap', methods=['POST'])
def snap(): return jsonify(ok=True, name=snapshot(frame, people, 'manual'))

@app.route('/snaps/')
@app.route('/snaps/<name>')
def gallery(name=None):
    if name: return send_from_directory(SNAP_DIR, name)
    names = sorted((f for f in os.listdir(SNAP_DIR) if f.endswith('.jpg')), reverse=True) if os.path.isdir(SNAP_DIR) else []
    return ('<!doctype html><meta name="viewport" content="width=device-width"><body style="margin:0;background:#000">'
            + ''.join(f'<a href="{n}"><img src="{n}" style="width:50%;vertical-align:top"></a>' for n in names))

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global rgb
    d.stop(); d.close(); rgb = 'off'; lights.close()
    subprocess.Popen(['sudo', '-n', 'shutdown', '-h', 'now'])
    return jsonify(ok=True)

if __name__ == '__main__':
    import sys
    d = Drive(sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyAMA0', log=lambda l: print(l, flush=True))
    lights = Lights()
    threading.Thread(target=bar, daemon=True).start()
    threading.Thread(target=camera, daemon=True).start()
    threading.Thread(target=infer, daemon=True).start()
    threading.Thread(target=heat_cam, daemon=True).start()
    threading.Thread(target=battery, daemon=True).start()
    threading.Thread(target=track, daemon=True).start()
    threading.Thread(target=deadman, daemon=True).start()
    threading.Thread(target=stats, daemon=True).start()
    app.run('0.0.0.0', int(sys.argv[2]) if len(sys.argv) > 2 else 8080, threaded=True)

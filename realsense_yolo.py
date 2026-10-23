#!/usr/bin/env python3
"""
Recon Drone — Human Detection System
Intel RealSense D435 + YOLOv8n-pose (NCNN) + Victor 328B Thermal

Controls (terminal):
  i        — toggle IR / RGB display
  d        — toggle detection source (IR / RGB)
  e        — toggle IR emitter on / off
  r        — start / stop recording
  s        — save screenshot (both panels)
  ox <N>   — shift thermal overlay X by N pixels (e.g. ox 30, ox -20)
  oy <N>   — shift thermal overlay Y by N pixels
  q        — quit

Geometry           : 848x480 native 16:9 throughout — preserves the D435's full
                     89.9° HFOV (a 4:3 mode would crop 10° off the width) and
                     feeds the 320x192 model with no letterbox waste.
IR detection mode  : IR + depth @ 60 FPS, no colour stream
RGB detection mode : colour + IR + depth @ 60 FPS
Display            : main feed + thermal overlay (left) | thermal panel (right)
                     RED crosshair  = global thermal hotspot
                     YELLOW crosshair = hotspot within detected person region
"""

import csv
import math
import os
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response

# Not ultralytics: it imports torch unconditionally, which takes ~55s cold off
# the SD card even though NCNN does all the inference. See ncnn_pose.py.
from ncnn_pose import PoseNet

# ── Config ────────────────────────────────────────────────────────────────────
# Geometry is 16:9 end to end. The D435 CROPS to make 4:3 — 848x480 gives the
# full 89.9° HFOV, while 640x480 would throw away 10° of scanning width and
# invalidate the thermal FOV constants below.
IR_W, IR_H = 848, 480       # capture + display panel, native (never resized)
TH_W, TH_H = 640, 480       # thermal panel stays 4:3 — the sensor is 256x192
COMB_W     = IR_W + TH_W    # combined canvas handed to the stream

# Inference shapes are (h, w) and must match the exported NCNN models: the
# reshape dims are frozen at export time, so these cannot be changed here alone.
MODEL_MAIN      = "yolov8n-pose-169_ncnn_model"   # 320x192 landscape
MODEL_ROT       = "yolov8n-pose-rot_ncnn_model"   # 192x320 portrait
INFER_SHAPE     = (192, 320)
INFER_SHAPE_ROT = (320, 192)

# The D435 is mounted upside down, so its streams are rotated 180° for display
# and inference. The thermal cam is assumed still upright — if it shares the
# bracket and got flipped too, its overlay will sit 180° off from the IR beneath
# it, so set FLIP_THERMAL_180 as well. Check with a warm object off to one side:
# it must appear on the same side in both panels.
FLIP_180         = True
FLIP_THERMAL_180 = False

# Which D435 IR imager to show and detect on. IR2 is the one at the far end from
# the RGB camera (64.8 mm vs 14.9 mm), right beside the thermal cam, so its view
# lines up best with the thermal overlay. Depth is computed from IR1's
# viewpoint, so depth_at() corrects for the 50 mm stereo baseline.
IR_INDEX = 2

# Cap on detection rate. The camera runs at 60 fps, but detections at 30 are
# plenty for a ground robot, and halving inference halves the CPU load — which
# is most of the Pi's current spikes. 0 = run on every camera frame.
INFER_MAX_FPS = 30

# NCNN worker threads. Measured on the Pi 5 at 320x192: 2 threads = 12.1 ms,
# 3 = 12.2 ms, 4 = 11.6 ms — but 4 burns ~1.3 cores vs ~0.7 for 2. The model is
# too small to parallelise well, so extra threads mostly add power draw.
NCNN_THREADS = 2

CONF_THRESHOLD = 0.25
PERSON_CLASS   = 0
CSV_PATH       = Path("detections.csv")
VIDEO_PATH     = Path("recording.avi")
CAPTURE_DIR    = Path("captures")
VIDEO_FPS      = 30
WIN_TITLE      = "Recon Drone — Human Detection"
FFC_STD_THRESH = 1.0        # true FFC blanks are near-zero std
STREAM_PORT    = 8000       # MJPEG feed: http://<pi-ip>:8000/
STREAM_FPS     = 20         # cap on encode rate, independent of capture FPS
STREAM_QUALITY = 70         # JPEG quality — lower = smaller payload = less WiFi latency

def find_thermal_device():
    """Find the Victor 328B via sysfs device names.

    The RealSense's /dev/video* node numbers shift between boots and whenever
    librealsense claims the device, and its entry in `v4l2-ctl --list-devices`
    can be empty while claimed — so identify the thermal cam by its stable
    sysfs name instead, then confirm it offers YUYV 256x192.
    """
    import glob
    import subprocess

    SKIP = ("realsense", "pispbe", "hevc", "isp", "codec")
    candidates = []
    for path in glob.glob("/sys/class/video4linux/video*"):
        num = int(path.rsplit("video", 1)[1])
        try:
            with open(f"{path}/name") as fh:
                name = fh.read().strip()
        except OSError:
            continue
        if any(s in name.lower() for s in SKIP):
            continue
        candidates.append((num, name))

    for num, name in sorted(candidates):
        try:
            out = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{num}",
                                  "--list-formats-ext"],
                                 capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "256x192" in out and "YUYV" in out:
            print(f"  Thermal : detected at /dev/video{num}  ({name})")
            return num

    print("  Thermal : no 256x192 YUYV device found — thermal disabled")
    return None
THERMAL_ALPHA  = 0.40

# FOV-based alignment — D435 IR: 87°×58°  Victor 328B: 56°×42°
D435_IR_FOV_H = 87.0
D435_IR_FOV_V = 58.0
VICTOR_FOV_H  = 56.0
VICTOR_FOV_V  = 42.0

# ── Shared state ──────────────────────────────────────────────────────────────
running         = True
ir_mode         = True
detect_rgb      = False
emitter_on      = False
latest_infer         = None
latest_boxes         = []
latest_rotated_boxes = []   # boxes from 90°-rotated pass, unrotated back to frame coords
infer_lock      = threading.Lock()
infer_cv        = threading.Condition(infer_lock)   # wakes inference on a new frame
infer_ver       = 0
boxes_lock      = threading.Lock()
rotated_lock    = threading.Lock()

# Thermal overlay tuning (adjusted via ox / oy / sz commands).
# therm_scale = the share of the IR frame's WIDTH the thermal image covers; its
# height follows from the sensor's native 4:3, so the picture is never stretched.
# The default comes from the FOV ratio in the tangent domain, not a plain angle
# ratio — image scale goes with tan(θ/2), and the naive version over-sizes the
# box by ~15%. The old hand-tuned 50/15 values also absorbed the letterbox bug
# in the 4:3 path (bars counted as FOV), so re-tune from here rather than reusing.
therm_offset_x  = 0      # pixel shift right (+) / left (-)
therm_offset_y  = 0      # pixel shift down (+) / up (-)
therm_scale     = int(round(100 * math.tan(math.radians(VICTOR_FOV_H / 2))
                                / math.tan(math.radians(D435_IR_FOV_H / 2))))

last_thermal        = np.zeros((TH_H, TH_W, 3), dtype=np.uint8)  # 4:3 panel
last_thermal_warped = np.zeros((IR_H, IR_W, 3), dtype=np.uint8)  # IR-sized canvas
therm_box           = (0, 0, 0, 0)   # where the 4:3 overlay actually sits in it
latest_therm_raw    = np.zeros((192, 256),      dtype=np.uint8)
thermal_lock        = threading.Lock()
thermal_ready       = False
# Percentiles cached by thermal thread at 25 Hz — read freely in main loop
therm_p5      = 18.0
therm_p82     = 28.0
therm_p99     = 37.0
therm_version = 0      # incremented each time thermal thread writes a new frame

latest_frame    = None
frame_lock      = threading.Lock()
take_screenshot = False

pipeline     = None
depth_sensor = None
depth_w      = IR_W
depth_h      = IR_H
stereo_k     = 0.0   # fx × baseline (px·m) from depth's viewpoint to IR_INDEX's

# ── Pipeline management ───────────────────────────────────────────────────────
def start_pipeline(rgb_detection):
    global pipeline, depth_sensor, depth_w, depth_h, emitter_on, stereo_k

    if pipeline:
        try:
            pipeline.stop()
        except Exception:
            pass

    pipeline = rs.pipeline()
    cfg      = rs.config()

    # Both modes run native 848x480 @ 60 now — USB 3 affords it, and keeping
    # one geometry means display coords == depth coords (lb_scale stays 1.0).
    if rgb_detection:
        cfg.enable_stream(rs.stream.color,    IR_W, IR_H, rs.format.bgr8, 60)
    cfg.enable_stream(rs.stream.infrared, IR_INDEX, IR_W, IR_H, rs.format.y8, 60)
    cfg.enable_stream(rs.stream.depth,              IR_W, IR_H, rs.format.z16, 60)
    depth_w, depth_h = IR_W, IR_H

    profile = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Read the real baseline from the camera's calibration rather than assuming
    # 50 mm. For IR1 this comes out ~0 and the correction switches itself off.
    d_prof  = profile.get_stream(rs.stream.depth)
    ir_prof = profile.get_stream(rs.stream.infrared, IR_INDEX)
    fx      = ir_prof.as_video_stream_profile().get_intrinsics().fx
    stereo_k = fx * d_prof.get_extrinsics_to(ir_prof).translation[0]
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if emitter_on else 0)

# ── Thermal thread ────────────────────────────────────────────────────────────
def thermal_loop(cap):
    global last_thermal, last_thermal_warped, latest_therm_raw, thermal_ready
    global therm_p5, therm_p82, therm_p99, therm_version, therm_box
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if grey.std() < FFC_STD_THRESH:
            continue
        if grey.shape[0] == 384:
            grey = grey[:192, :]
        if FLIP_THERMAL_180:
            # Flip the raw frame, so the panel, the overlay and every
            # thermal→IR coordinate mapping all stay consistent downstream.
            grey = cv2.rotate(grey, cv2.ROTATE_180)
        # Compute percentiles here at 25 Hz so main loop never has to
        flat = grey.flatten().astype(np.float32)
        new_p5  = float(np.percentile(flat,  5))
        new_p82 = float(np.percentile(flat, 82))
        new_p99 = float(np.percentile(flat, 99))
        inferno  = cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
        coloured = cv2.resize(inferno, (TH_W, TH_H))          # right-hand panel

        # Overlay keeps the sensor's own 4:3 aspect — stretching it to the IR
        # frame's 16:9 would smear every feature horizontally by 1.33x and stop
        # the thermal lining up with what the IR sees. therm_scale sets how much
        # of the IR width it covers; the height follows from the native aspect.
        th_h_src, th_w_src = grey.shape[:2]
        ow = max(1, int(IR_W * therm_scale / 100.0))
        oh = max(1, int(ow * th_h_src / th_w_src))
        small = cv2.resize(inferno, (ow, oh))

        warped = np.zeros((IR_H, IR_W, 3), dtype=np.uint8)
        x0 = (IR_W - ow) // 2 + therm_offset_x
        y0 = (IR_H - oh) // 2 + therm_offset_y
        # Clip against the IR frame so offsets can push it part-way off-screen
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        cw = min(ow - sx0, IR_W - dx0)
        ch = min(oh - sy0, IR_H - dy0)
        if cw > 0 and ch > 0:
            warped[dy0:dy0 + ch, dx0:dx0 + cw] = small[sy0:sy0 + ch, sx0:sx0 + cw]
            box = (dx0, dy0, dx0 + cw, dy0 + ch)
        else:
            box = (0, 0, 0, 0)
        with thermal_lock:
            last_thermal[:]                                   = coloured
            last_thermal_warped[:]                            = warped
            latest_therm_raw[:grey.shape[0], :grey.shape[1]] = grey
            therm_p5, therm_p82, therm_p99 = new_p5, new_p82, new_p99
            therm_box = box
            therm_version += 1
            thermal_ready = True

# ── MJPEG streaming server ────────────────────────────────────────────────────
flask_app = Flask(__name__)

def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

# Push-based handoff: main loop → encoder → viewers. No polling, no sleeps, so a
# frame reaches the browser as soon as it exists rather than on a timer tick.
frame_cv   = threading.Condition(frame_lock)
frame_ver  = 0

jpeg_cv    = threading.Condition()
jpeg_bytes = None
jpeg_ver   = 0

def jpeg_encoder_loop():
    """Encode each new frame exactly once, however many viewers are connected."""
    global jpeg_bytes, jpeg_ver
    params   = [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY]
    last_ver = 0
    while running:
        with frame_cv:
            while running and frame_ver == last_ver:
                frame_cv.wait(0.5)
            if not running:
                return
            frame    = latest_frame   # already a private copy made by the main loop
            last_ver = frame_ver
        if frame is None:
            continue
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            continue
        with jpeg_cv:
            jpeg_bytes = buf.tobytes()
            jpeg_ver  += 1
            jpeg_cv.notify_all()

def mjpeg_generator():
    last_ver = 0
    while running:
        with jpeg_cv:
            while running and jpeg_ver == last_ver:
                jpeg_cv.wait(0.5)
            if not running:
                return
            buf      = jpeg_bytes
            last_ver = jpeg_ver
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")

@flask_app.route("/")
def stream_index():
    return ("<html><head><title>Recon Drone</title></head>"
            "<body style='margin:0;background:#000'>"
            "<img src='/stream' style='width:100%'></body></html>")

@flask_app.route("/stream")
def stream_route():
    resp = Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["X-Accel-Buffering"] = "no"    # never buffer the stream
    return resp

def start_stream_server():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask_app.run(host="0.0.0.0", port=STREAM_PORT, threaded=True,
                  use_reloader=False, debug=False)

# ── Inference thread ──────────────────────────────────────────────────────────
def inference_loop(model, model_rot):
    global latest_boxes, latest_rotated_boxes
    # Block until the camera delivers a frame we haven't inferred yet. The old
    # loop spun flat out, re-running the network on the same frame on 4 cores —
    # ~2.5 cores of CPU for nothing, and the CPU current spikes were pulling
    # the 5V rail down to the brownout threshold.
    last_ver  = 0
    min_gap   = 1.0 / INFER_MAX_FPS if INFER_MAX_FPS else 0.0
    last_run  = 0.0
    while running:
        wait = last_run + min_gap - time.time()
        if wait > 0:
            time.sleep(wait)
        with infer_cv:
            while running and (latest_infer is None or infer_ver == last_ver):
                infer_cv.wait(0.5)
            if not running:
                return
            frame    = latest_infer     # main loop hands over a fresh array each time
            last_ver = infer_ver
        last_run = time.time()

        boxes = model(frame, conf=CONF_THRESHOLD)   # single-class (person) model

        # Rotated pass — only when primary finds nobody.
        # A person lying flat looks "standing" when the frame is rotated 90°.
        # Rotating 848x480 yields a 480x848 portrait frame, so this pass uses
        # the portrait-exported model; the landscape one would squash it into
        # a ~108px-wide strip and lose the detection entirely.
        rot_boxes = []
        if not boxes:
            frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            res_rot = model_rot(frame_rot, conf=CONF_THRESHOLD)
            if res_rot:
                # Unrotate: 90° CW maps (ox,oy) → (IR_H-1-oy, ox)
                # Inverse: (rx,ry) → ox=ry, oy=IR_H-1-rx
                for rb in res_rot:
                    rx1, ry1, rx2, ry2 = map(float, rb.xyxy[0])
                    conf = float(rb.conf[0])
                    ox1 = int(np.clip(ry1, 0, IR_W - 1))
                    oy1 = int(np.clip(IR_H - 1 - rx2, 0, IR_H - 1))
                    ox2 = int(np.clip(ry2, 0, IR_W - 1))
                    oy2 = int(np.clip(IR_H - 1 - rx1, 0, IR_H - 1))
                    if ox1 < ox2 and oy1 < oy2:
                        rot_boxes.append((ox1, oy1, ox2, oy2, conf))

        with boxes_lock:
            latest_boxes = boxes
        with rotated_lock:
            latest_rotated_boxes = rot_boxes

# ── Terminal input ────────────────────────────────────────────────────────────
def terminal_listener():
    global running, ir_mode, detect_rgb, emitter_on, recording, video_writer
    global latest_infer, therm_offset_x, therm_offset_y, therm_scale, take_screenshot

    while running:
        try:
            cmd = input().strip().lower()

            if cmd == 'i':
                if not detect_rgb:
                    print("  [!] IR detection mode — display locked to IR")
                else:
                    ir_mode = not ir_mode
                    print(f"  Display → {'IR' if ir_mode else 'RGB'}")

            elif cmd == 'd':
                detect_rgb = not detect_rgb
                with infer_lock:
                    latest_infer = None
                with boxes_lock:
                    latest_boxes.clear()
                print(f"  Detection → {'RGB' if detect_rgb else 'IR'}  — restarting pipeline...")
                start_pipeline(detect_rgb)
                if not detect_rgb:
                    ir_mode = True
                print(f"  Pipeline ready  ({depth_w}x{depth_h} @ {'30' if detect_rgb else '60'} FPS target)")

            elif cmd == 'e':
                emitter_on = not emitter_on
                depth_sensor.set_option(rs.option.emitter_enabled, 1 if emitter_on else 0)
                print(f"  IR emitter → {'ON' if emitter_on else 'OFF'}")

            elif cmd == 'r':
                if not recording:
                    video_writer = cv2.VideoWriter(
                        str(VIDEO_PATH),
                        cv2.VideoWriter_fourcc(*"XVID"),
                        VIDEO_FPS,
                        (IR_W, IR_H),
                    )
                    recording = True
                    print(f"  Recording started → {VIDEO_PATH.resolve()}")
                else:
                    recording = False
                    video_writer.release()
                    video_writer = None
                    print(f"  Recording saved → {VIDEO_PATH.resolve()}")

            elif cmd == 's':
                take_screenshot = True
                print("  Screenshot queued...")

            elif cmd.startswith('ox '):
                try:
                    therm_offset_x = int(cmd.split()[1])
                    print(f"  Thermal X offset → {therm_offset_x}px  (oy to adjust vertical)")
                except (IndexError, ValueError):
                    print("  Usage: ox <pixels>  e.g.  ox 30  or  ox -20")

            elif cmd.startswith('oy '):
                try:
                    therm_offset_y = int(cmd.split()[1])
                    print(f"  Thermal Y offset → {therm_offset_y}px")
                except (IndexError, ValueError):
                    print("  Usage: oy <pixels>  e.g.  oy 20  or  oy -10")

            elif cmd.startswith('sz '):
                try:
                    therm_scale = int(cmd.split()[1])
                    print(f"  Thermal scale → {therm_scale}%  (100=fill frame, 110=10% zoom in)")
                except (IndexError, ValueError):
                    print("  Usage: sz <percent>  e.g.  sz 110  or  sz 90")

            elif cmd == 'q':
                running = False
                break

        except EOFError:
            break

# ── Frame / coord helpers ─────────────────────────────────────────────────────
def letterbox(img, out_w=IR_W, out_h=IR_H):
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x = (out_w - nw) // 2
    y = (out_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas, x, y, scale

def ir_to_therm_raw(ix, iy):
    """
    Map IR display (IR_W×IR_H) → thermal sensor (256×192) using FOV angles.
    Points outside the thermal FOV are clamped to the thermal frame edge.
    """
    norm_h = (ix - IR_W / 2) / (IR_W / 2)
    norm_v = (iy - IR_H / 2) / (IR_H / 2)
    # Scale by FOV ratio — thermal is more zoomed in, so same angle → larger norm
    tx = 128 + norm_h * (D435_IR_FOV_H / VICTOR_FOV_H) * 128
    ty = 96  + norm_v * (D435_IR_FOV_V / VICTOR_FOV_V) * 96
    return int(np.clip(tx, 0, 255)), int(np.clip(ty, 0, 191))

def therm_raw_to_panel(hx, hy):
    """Map raw thermal (256×192) → IR display geometry, applying scale+offset.
    Used only for the warped overlay on the IR display — NOT for drawing on the thermal panel."""
    base_px = hx * IR_W / 256
    base_py = hy * IR_H / 192
    scale = therm_scale / 100.0
    px = int(IR_W / 2 + scale * (base_px - IR_W / 2) + therm_offset_x)
    py = int(IR_H / 2 + scale * (base_py - IR_H / 2) + therm_offset_y)
    return px, py

def therm_raw_to_panel_full(hx, hy):
    """Map raw thermal (256×192) → thermal panel (TH_W×TH_H) without scale/offset.
    Use this for drawing boxes/crosshairs on the right thermal panel."""
    return int(hx * TH_W / 256), int(hy * TH_H / 192)

def ir_to_therm_panel(ix, iy):
    """Map IR display → thermal panel display via FOV mapping + current scale/offset."""
    tx, ty = ir_to_therm_raw(ix, iy)
    return therm_raw_to_panel(tx, ty)

def therm_raw_to_ir(tx, ty):
    """Inverse of ir_to_therm_raw: thermal raw (256×192) → IR display (IR_W×IR_H)."""
    ix = IR_W / 2 + (IR_W / 2) * (tx - 128) * VICTOR_FOV_H / (128.0 * D435_IR_FOV_H)
    iy = IR_H / 2 + (IR_H / 2) * (ty - 96)  * VICTOR_FOV_V / (96.0  * D435_IR_FOV_V)
    return int(np.clip(ix, 0, IR_W - 1)), int(np.clip(iy, 0, IR_H - 1))

def depth_at(depth_frame, dx, dy):
    """Sample depth for a point given in *display* coordinates.

    librealsense hands us the depth frame in the sensor's own orientation, so
    when the feed is rotated 180° for the upside-down mount the lookup has to
    be mirrored back — otherwise every distance comes from the opposite corner
    of the scene.
    """
    if FLIP_180:
        dx = depth_w - 1 - dx
        dy = depth_h - 1 - dy

    def sample(x):
        x = max(0, min(int(round(x)), depth_w - 1))
        return depth_frame.get_distance(x, max(0, min(dy, depth_h - 1)))

    z = sample(dx)
    # Depth is seen from IR1, but the image is IR_INDEX's, so the same point
    # sits stereo_k / Z pixels further along in the depth map. Z is what we're
    # looking up, so refine twice from the first guess (converges fast, since
    # the shift only changes slowly with distance).
    if stereo_k and z > 0:
        for _ in range(2):
            z2 = sample(dx - stereo_k / z)
            if z2 <= 0:
                break
            z = z2
    return z

def thermal_person_candidates(t_raw):
    """Find warm blobs that could be a lying or obscured person.
    Returns list of (x1,y1,x2,y2) in thermal raw (256×192) coords."""
    if t_raw.max() == 0:
        return []
    binary = (t_raw >= therm_p82).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 280:   # too small to be a person
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        # Human proportions: standing ~0.3–0.8, lying/crouching ~0.8–5.0
        if 0.25 <= aspect <= 5.2:
            out.append((x, y, x + w, y + h))
    return out

def temp_from_raw(t_raw, hx, hy, patch=4):
    """Estimate °C using percentiles cached by thermal thread (no per-call computation)."""
    region = t_raw[max(0, hy - patch):hy + patch + 1,
                   max(0, hx - patch):hx + patch + 1]
    pix = int(region.max()) if region.size > 0 else int(t_raw[hy, hx])
    span = therm_p99 - therm_p5
    if span <= 0:
        return 30.0
    return 18.0 + (pix - therm_p5) / span * 19.0

lb_x, lb_y, lb_scale = 0, 0, 1.0

# ── HUD helpers ───────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX

def _deg_radius(scale):
    return max(2, int(round(scale * 4)))

def text_size(text, scale, thickness):
    """getTextSize that accounts for '°' being drawn as a circle, not a glyph."""
    parts = text.split("°")
    w, h = 0, 0
    for part in parts:
        if part:
            (pw, ph), _ = cv2.getTextSize(part, FONT, scale, thickness)
            w += pw
            h = max(h, ph)
    if h == 0:
        (_, h), _ = cv2.getTextSize("0", FONT, scale, thickness)
    w += (len(parts) - 1) * (2 * _deg_radius(scale) + 3)
    return w, h

def put_text(img, text, org, scale, colour, thickness):
    """cv2.putText with degree-sign support.

    Hershey fonts are ASCII-only, so a literal '°' renders as '??'. Draw each
    one as a small ring instead and lay out the surrounding text around it.
    """
    x, y = org
    parts = text.split("°")
    for i, part in enumerate(parts):
        if i:
            r = _deg_radius(scale)
            (_, ch), _ = cv2.getTextSize("C", FONT, scale, thickness)
            cv2.circle(img, (x + r + 1, y - ch + r), r, colour,
                       max(1, thickness - 1), cv2.LINE_AA)
            x += 2 * r + 3
        if part:
            cv2.putText(img, part, (x, y), FONT, scale, colour, thickness)
            x += cv2.getTextSize(part, FONT, scale, thickness)[0][0]

def hud_text(img, text, pos, scale=0.7, colour=(0, 220, 255), thickness=2):
    tw, th = text_size(text, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 4), (0, 0, 0), -1)
    put_text(img, text, (x, y), scale, colour, thickness)

def draw_crosshair(img, cx, cy, colour, size=16, thickness=2):
    cv2.drawMarker(img, (cx, cy), colour, cv2.MARKER_CROSS, size, thickness)

def draw_box(img, x1, y1, x2, y2, conf, dist_m, temp_c=None):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Person  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if dist_m > 0:
        dlabel = f"{dist_m:.2f} m"
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        (dw, dh), _ = cv2.getTextSize(dlabel, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(img, (mx - dw // 2 - 2, my - dh - 2),
                      (mx + dw // 2 + 2, my + 4), (0, 0, 0), -1)
        cv2.putText(img, dlabel, (mx - dw // 2, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    if temp_c is not None:
        tlabel = f"~{temp_c:.1f}°C"
        tw2, th2 = text_size(tlabel, 0.65, 2)
        cv2.rectangle(img, (x1 - 2, y2 + 2), (x1 + tw2 + 4, y2 + th2 + 8), (0, 0, 0), -1)
        put_text(img, tlabel, (x1 + 2, y2 + th2 + 4), 0.65, (0, 165, 255), 2)

# ── Init ──────────────────────────────────────────────────────────────────────
CAPTURE_DIR.mkdir(exist_ok=True)

print("\n  Recon Drone — Human Detection System")
print("  ─────────────────────────────────────")
print(f"  Model   : YOLOv8n-pose NCNN  {INFER_SHAPE[1]}x{INFER_SHAPE[0]} + rotated pass")
print(f"  Camera  : Intel RealSense D435  {IR_W}x{IR_H} @60  (89.9°×59° IR FOV)")
print("  Thermal : Victor 328B (56°×42° FOV — auto-detected)")
print("  Commands: i d e r s  |  ox <px>  oy <px>  sz <pct>  |  q=quit")
print("  Hotspot : RED=global  YELLOW=within detection\n")

model     = PoseNet(MODEL_MAIN, INFER_SHAPE,     threads=NCNN_THREADS)
model_rot = PoseNet(MODEL_ROT,  INFER_SHAPE_ROT, threads=NCNN_THREADS)
start_pipeline(detect_rgb)

thermal_idx = find_thermal_device()
thermal_cap = None
if thermal_idx is not None:
    # CAP_V4L2 explicitly — the default backend can fall through to FFMPEG,
    # which adds buffering latency and spams "Not a video capture device".
    thermal_cap = cv2.VideoCapture(thermal_idx, cv2.CAP_V4L2)
    thermal_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    thermal_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always read the freshest frame
    thermal_cap.set(cv2.CAP_PROP_FPS, 25)
    if not thermal_cap.isOpened():
        thermal_cap.release()
        thermal_cap = None
if thermal_cap is None:
    print("  [!] Thermal camera unavailable — running IR/depth only")

threading.Thread(target=terminal_listener, daemon=True).start()
threading.Thread(target=inference_loop, args=(model, model_rot), daemon=True).start()
if thermal_cap is not None:
    threading.Thread(target=thermal_loop, args=(thermal_cap,), daemon=True).start()
threading.Thread(target=jpeg_encoder_loop,   daemon=True).start()
threading.Thread(target=start_stream_server, daemon=True).start()
print(f"  Stream  : http://{lan_ip()}:{STREAM_PORT}/  (view on PC/phone browser)")

show_window = bool(os.environ.get("DISPLAY"))
if show_window:
    try:
        cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_TITLE, COMB_W, IR_H)
        cv2.moveWindow(WIN_TITLE, 0, 0)
    except cv2.error:
        show_window = False
if not show_window:
    print("  [!] No local display — skipping cv2 window (use the stream URL above, or 'q' + Enter to quit)")

csv_file   = CSV_PATH.open("w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "detect_src", "display", "confidence",
                     "distance_m", "temp_c", "x1", "y1", "x2", "y2"])
total_logged      = 0
last_log_time     = 0.0
LOG_INTERVAL      = 5.0   # seconds between repeat logs while person stays in frame
was_detecting      = False  # IR detection state
was_therm_detecting = False  # thermal fallback state
video_writer      = None
recording         = False
combined          = np.empty((IR_H, COMB_W, 3), dtype=np.uint8)  # pre-allocated, reused every frame
_last_therm_ver   = -1
t_coloured        = np.zeros((TH_H, TH_W, 3), dtype=np.uint8)
t_warped          = np.zeros((IR_H, IR_W, 3), dtype=np.uint8)

fps_counter = 0
fps_display = 0.0
fps_timer   = time.time()

STREAM_PERIOD   = 1.0 / STREAM_FPS
last_stream_pub = 0.0

print("  Streaming — IR detection active\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    while running:
        try:
            frames = pipeline.wait_for_frames(5000)
        except RuntimeError:
            try:
                time.sleep(0.05)
            except KeyboardInterrupt:
                break
            continue

        depth_frame  = frames.get_depth_frame()
        ir_frame     = frames.get_infrared_frame(IR_INDEX)
        colour_frame = frames.get_color_frame() if detect_rgb else None

        if not depth_frame or not ir_frame:
            continue
        if detect_rgb and not colour_frame:
            continue

        ir_raw = cv2.cvtColor(np.asanyarray(ir_frame.get_data()), cv2.COLOR_GRAY2BGR)
        if FLIP_180:
            ir_raw = cv2.rotate(ir_raw, cv2.ROTATE_180)
        if ir_raw.shape[:2] == (IR_H, IR_W):
            ir_bgr = ir_raw
            lb_x, lb_y, lb_scale = 0, 0, 1.0
        else:
            ir_bgr, lb_x, lb_y, lb_scale = letterbox(ir_raw)  # noqa: F841
        rgb = np.asanyarray(colour_frame.get_data()) if colour_frame else None
        if FLIP_180 and rgb is not None:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)

        with infer_cv:
            latest_infer = rgb.copy() if (detect_rgb and rgb is not None) else ir_bgr.copy()
            infer_ver   += 1
            infer_cv.notify()

        display = rgb.copy() if (not ir_mode and rgb is not None) else ir_bgr.copy()

        # Grab thermal state — skip big copies when thermal frame hasn't changed
        with thermal_lock:
            cur_ver = therm_version
            t_raw   = latest_therm_raw.copy()
            t_ready = thermal_ready
            t_box   = therm_box
            if cur_ver != _last_therm_ver:
                t_coloured = last_thermal.copy()
                t_warped   = last_thermal_warped.copy()
                _last_therm_ver = cur_ver

        # ── Thermal overlay — blend only where thermal data actually exists ────
        # Blending the whole frame would mix the IR with black everywhere the
        # thermal cam can't see, dimming the wider part of the view by 40%.
        if t_ready:
            bx1, by1, bx2, by2 = t_box
            if bx2 > bx1 and by2 > by1:
                display[by1:by2, bx1:bx2] = cv2.addWeighted(
                    display[by1:by2, bx1:bx2], 1.0 - THERMAL_ALPHA,
                    t_warped[by1:by2, bx1:bx2], THERMAL_ALPHA, 0)

        with boxes_lock:
            boxes = list(latest_boxes)
        with rotated_lock:
            rotated_boxes = list(latest_rotated_boxes)

        now        = datetime.now().isoformat(timespec="milliseconds")
        detect_src = "RGB" if detect_rgb else "IR"
        disp_mode  = "IR" if (ir_mode or not detect_rgb) else "RGB"

        # ── Thermal panel ─────────────────────────────────────────────────────
        thermal_panel = t_coloured.copy() if t_ready else last_thermal.copy()
        hud_text(thermal_panel, f"THERMAL  offset:{therm_offset_x},{therm_offset_y}  sz:{therm_scale}%",
                 (8, 28), colour=(255, 200, 100))

        # Global hotspot — red crosshair on thermal panel
        if t_ready and t_raw.max() > 0:
            g_hy, g_hx = np.unravel_index(t_raw.argmax(), t_raw.shape)
            g_px, g_py = therm_raw_to_panel_full(g_hx, g_hy)
            draw_crosshair(thermal_panel, g_px, g_py, (0, 0, 255), size=20)
            g_temp = temp_from_raw(t_raw, g_hx, g_hy)
            hud_text(thermal_panel, f"HOT ~{g_temp:.0f}°C",
                     (min(g_px + 12, 580), g_py + 6),
                     colour=(0, 0, 255), scale=0.55)

        # Per-detection box drawing + detection-zone hotspot
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx   = max(0, min((x1 + x2) // 2, IR_W - 1))
            cy   = max(0, min((y1 + y2) // 2, IR_H - 1))

            # Depth
            dcx    = max(0, min(int((cx - lb_x) / lb_scale), depth_w - 1))
            dcy    = max(0, min(int((cy - lb_y) / lb_scale), depth_h - 1))
            dist_m = depth_at(depth_frame, dcx, dcy)

            # Temperature at box centre
            hx_c, hy_c = ir_to_therm_raw(cx, cy)
            temp_c = temp_from_raw(t_raw, hx_c, hy_c) if t_ready else None

            draw_box(display, x1, y1, x2, y2, conf, dist_m, temp_c)

            # Project IR box onto thermal panel + detection-zone hotspot
            if t_ready:
                rx1, ry1 = ir_to_therm_raw(x1, y1)
                rx2, ry2 = ir_to_therm_raw(x2, y2)
                tp_x1, tp_y1 = therm_raw_to_panel_full(rx1, ry1)
                tp_x2, tp_y2 = therm_raw_to_panel_full(rx2, ry2)
                tp_x1 = max(0, min(tp_x1, TH_W - 1))
                tp_x2 = max(0, min(tp_x2, TH_W - 1))
                tp_y1 = max(0, min(tp_y1, TH_H - 1))
                tp_y2 = max(0, min(tp_y2, TH_H - 1))

                if tp_x1 < tp_x2 and tp_y1 < tp_y2:
                    cv2.rectangle(thermal_panel, (tp_x1, tp_y1),
                                  (tp_x2, tp_y2), (0, 255, 0), 2)

                    # Hotspot within detection region (yellow crosshair)
                    rox1 = max(0, min(rx1, 255))
                    roy1 = max(0, min(ry1, 191))
                    rox2 = max(0, min(rx2, 255))
                    roy2 = max(0, min(ry2, 191))
                    if rox1 < rox2 and roy1 < roy2:
                        roi_t = t_raw[roy1:roy2, rox1:rox2]
                        if roi_t.size > 0:
                            lhy, lhx = np.unravel_index(roi_t.argmax(), roi_t.shape)
                            lhx += rox1;  lhy += roy1
                            lp_x, lp_y = therm_raw_to_panel_full(lhx, lhy)
                            draw_crosshair(thermal_panel, lp_x, lp_y,
                                           (0, 255, 255), size=16)
                            l_temp = temp_from_raw(t_raw, lhx, lhy)
                            hud_text(thermal_panel,
                                     f"~{l_temp:.1f}°C",
                                     (min(lp_x + 10, 580), lp_y + 6),
                                     colour=(0, 255, 255), scale=0.55)

        # ── Rotated-pass detections (lying-down) — drawn in orange ──────────────
        for (x1r, y1r, x2r, y2r, confr) in rotated_boxes:
            cx_r = (x1r + x2r) // 2
            cy_r = (y1r + y2r) // 2
            dcx_r = max(0, min(int((cx_r - lb_x) / lb_scale), depth_w - 1))
            dcy_r = max(0, min(int((cy_r - lb_y) / lb_scale), depth_h - 1))
            dist_r = depth_at(depth_frame, dcx_r, dcy_r)
            hx_r, hy_r = ir_to_therm_raw(cx_r, cy_r)
            temp_r = temp_from_raw(t_raw, hx_r, hy_r) if t_ready else None
            # IR display — orange box
            cv2.rectangle(display, (x1r, y1r), (x2r, y2r), (0, 140, 255), 2)
            lbl_r = f"Person {confr:.0%} (lying)"
            (tw, th), _ = cv2.getTextSize(lbl_r, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display, (x1r, y1r - th - 8), (x1r + tw + 4, y1r), (0, 140, 255), -1)
            cv2.putText(display, lbl_r, (x1r + 2, y1r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            if dist_r > 0:
                dl = f"{dist_r:.2f} m"
                mx_r, my_r = (x1r + x2r) // 2, (y1r + y2r) // 2
                (dw2, dh2), _ = cv2.getTextSize(dl, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(display, (mx_r - dw2 // 2 - 2, my_r - dh2 - 2),
                              (mx_r + dw2 // 2 + 2, my_r + 4), (0, 0, 0), -1)
                cv2.putText(display, dl, (mx_r - dw2 // 2, my_r),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2)
            if temp_r is not None:
                tl_r = f"~{temp_r:.1f}°C"
                tw2, th2 = text_size(tl_r, 0.65, 2)
                cv2.rectangle(display, (x1r - 2, y2r + 2),
                              (x1r + tw2 + 4, y2r + th2 + 8), (0, 0, 0), -1)
                put_text(display, tl_r, (x1r + 2, y2r + th2 + 4), 0.65, (0, 140, 255), 2)
            # Thermal panel — project box
            if t_ready:
                trx1, try1 = ir_to_therm_raw(x1r, y1r)
                trx2, try2 = ir_to_therm_raw(x2r, y2r)
                ttp_x1, ttp_y1 = therm_raw_to_panel_full(trx1, try1)
                ttp_x2, ttp_y2 = therm_raw_to_panel_full(trx2, try2)
                ttp_x1 = max(0, min(ttp_x1, TH_W - 1)); ttp_x2 = max(0, min(ttp_x2, TH_W - 1))
                ttp_y1 = max(0, min(ttp_y1, TH_H - 1)); ttp_y2 = max(0, min(ttp_y2, TH_H - 1))
                if ttp_x1 < ttp_x2 and ttp_y1 < ttp_y2:
                    cv2.rectangle(thermal_panel, (ttp_x1, ttp_y1),
                                  (ttp_x2, ttp_y2), (0, 140, 255), 2)

        # ── Thermal fallback — when IR finds nobody, check warm blobs ───────────
        therm_candidates = []
        if not boxes and not rotated_boxes and t_ready:
            therm_candidates = thermal_person_candidates(t_raw)
            for tx1, ty1, tx2, ty2 in therm_candidates:
                # Draw on thermal panel
                tp_x1f, tp_y1f = therm_raw_to_panel_full(tx1, ty1)
                tp_x2f, tp_y2f = therm_raw_to_panel_full(tx2, ty2)
                cv2.rectangle(thermal_panel, (tp_x1f, tp_y1f), (tp_x2f, tp_y2f),
                              (0, 220, 255), 2)
                t_temp = temp_from_raw(t_raw, (tx1 + tx2) // 2, (ty1 + ty2) // 2)
                hud_text(thermal_panel, f"HEAT ~{t_temp:.1f}°C",
                         (tp_x1f, max(tp_y1f - 6, 16)), colour=(0, 220, 255), scale=0.5)
                # Project onto IR display
                ix1f, iy1f = therm_raw_to_ir(tx1, ty1)
                ix2f, iy2f = therm_raw_to_ir(tx2, ty2)
                cv2.rectangle(display, (ix1f, iy1f), (ix2f, iy2f), (0, 220, 255), 2)
                hud_text(display, f"POSSIBLE ~{t_temp:.1f}°C",
                         (ix1f, max(iy1f - 6, 16)), colour=(0, 220, 255), scale=0.5)

        # ── Logging — instant on new detection, throttled while person stays ──
        now_t        = time.time()
        all_detected  = boxes or rotated_boxes
        new_detection = all_detected and not was_detecting
        periodic_log  = all_detected and was_detecting and (now_t - last_log_time >= LOG_INTERVAL)

        if new_detection or periodic_log:
            for (x1r, y1r, x2r, y2r, confr) in rotated_boxes:
                cx_rl = (x1r + x2r) // 2; cy_rl = (y1r + y2r) // 2
                dcx_rl = max(0, min(int((cx_rl - lb_x) / lb_scale), depth_w - 1))
                dcy_rl = max(0, min(int((cy_rl - lb_y) / lb_scale), depth_h - 1))
                dist_rl = depth_at(depth_frame, dcx_rl, dcy_rl)
                hx_rl, hy_rl = ir_to_therm_raw(cx_rl, cy_rl)
                temp_rl = temp_from_raw(t_raw, hx_rl, hy_rl) if t_ready else None
                csv_writer.writerow([now, "IR-ROT", disp_mode,
                                     f"{confr:.4f}", f"{dist_rl:.3f}",
                                     f"{temp_rl:.1f}" if temp_rl else "",
                                     x1r, y1r, x2r, y2r])
                total_logged += 1
            for box in boxes:
                x1l, y1l, x2l, y2l = map(int, box.xyxy[0])
                conf_l = float(box.conf[0])
                cx_l   = max(0, min((x1l + x2l) // 2, IR_W - 1))
                cy_l   = max(0, min((y1l + y2l) // 2, IR_H - 1))
                dcx_l  = max(0, min(int((cx_l - lb_x) / lb_scale), depth_w - 1))
                dcy_l  = max(0, min(int((cy_l - lb_y) / lb_scale), depth_h - 1))
                dist_l = depth_at(depth_frame, dcx_l, dcy_l)
                hx_l, hy_l = ir_to_therm_raw(cx_l, cy_l)
                temp_l = temp_from_raw(t_raw, hx_l, hy_l) if t_ready else None
                csv_writer.writerow([now, detect_src, disp_mode,
                                     f"{conf_l:.4f}", f"{dist_l:.3f}",
                                     f"{temp_l:.1f}" if temp_l else "",
                                     x1l, y1l, x2l, y2l])
                total_logged += 1
            last_log_time = now_t

        was_detecting = bool(boxes) or bool(rotated_boxes)

        # Log thermal-only candidates (same instant/throttle logic)
        new_therm     = therm_candidates and not was_therm_detecting
        periodic_therm = therm_candidates and was_therm_detecting and (now_t - last_log_time >= LOG_INTERVAL)
        if new_therm or periodic_therm:
            for tx1, ty1, tx2, ty2 in therm_candidates:
                tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                ix_c, iy_c = therm_raw_to_ir(tcx, tcy)
                dcx_t = max(0, min(int((ix_c - lb_x) / lb_scale), depth_w - 1))
                dcy_t = max(0, min(int((iy_c - lb_y) / lb_scale), depth_h - 1))
                dist_t = depth_at(depth_frame, dcx_t, dcy_t)
                temp_t = temp_from_raw(t_raw, tcx, tcy)
                csv_writer.writerow([now, "THERM", disp_mode,
                                     "", f"{dist_t:.3f}", f"{temp_t:.1f}",
                                     tx1, ty1, tx2, ty2])
                total_logged += 1
            last_log_time = now_t
        was_therm_detecting = bool(therm_candidates)

        # ── HUD ──────────────────────────────────────────────────────────────
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer   = time.time()

        emitter_str = "EMT:ON" if emitter_on else "EMT:OFF"
        hud_text(display, f"FPS {fps_display:.0f}  [{disp_mode}]  detect:{detect_src}  {emitter_str}", (8, 28))
        if boxes:
            hud_text(display, f"Persons: {len(boxes)}", (8, 62))
        elif rotated_boxes:
            hud_text(display, f"Persons: {len(rotated_boxes)} (lying)", (8, 62), colour=(0, 140, 255))
        else:
            hud_text(display, "Persons: 0", (8, 62))
        if total_logged:
            hud_text(display, f"Logged: {total_logged}", (8, 96), colour=(180, 255, 180))
        if boxes:
            hud_text(display, "DETECTED", (IR_W - 120, 28), colour=(0, 60, 255), scale=0.8)
        elif rotated_boxes:
            hud_text(display, "DETECTED (lying)", (IR_W - 200, 28), colour=(0, 140, 255), scale=0.8)
        elif therm_candidates:
            hud_text(display, f"HEAT SIG x{len(therm_candidates)}", (IR_W - 160, 28),
                     colour=(0, 220, 255), scale=0.7)
        if recording:
            hud_text(display, "REC", (IR_W - 60, 62), colour=(0, 0, 255), scale=0.7)
            video_writer.write(display)

        combined[:, :IR_W]  = display
        combined[:, IR_W:]  = thermal_panel

        # Screenshot
        if take_screenshot:
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fpath = CAPTURE_DIR / f"capture_{ts}.png"
            cv2.imwrite(str(fpath), combined)
            print(f"  Screenshot saved → {fpath.resolve()}")
            take_screenshot = False

        # Publish to the stream — throttled to STREAM_FPS so JPEG encoding never
        # steals CPU from inference. `combined` is reused in place every frame,
        # so hand the encoder its own copy rather than a reference to it.
        if time.time() - last_stream_pub >= STREAM_PERIOD:
            with frame_cv:
                latest_frame = combined.copy()
                frame_ver   += 1
                frame_cv.notify_all()
            last_stream_pub = time.time()

        if show_window:
            cv2.imshow(WIN_TITLE, combined)

        key = cv2.waitKey(1) & 0xFF if show_window else 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            if detect_rgb:
                ir_mode = not ir_mode
        elif key == ord('d'):
            detect_rgb = not detect_rgb
            with infer_lock:
                latest_infer = None
            if not detect_rgb:
                ir_mode = True
            print(f"  Detection → {'RGB' if detect_rgb else 'IR'}  — restarting pipeline...")
            start_pipeline(detect_rgb)
            print(f"  Pipeline ready")
        elif key == ord('e'):
            emitter_on = not emitter_on
            depth_sensor.set_option(rs.option.emitter_enabled, 1 if emitter_on else 0)
            print(f"  IR emitter → {'ON' if emitter_on else 'OFF'}")
        elif key == ord('r'):
            if not recording:
                video_writer = cv2.VideoWriter(
                    str(VIDEO_PATH),
                    cv2.VideoWriter_fourcc(*"XVID"),
                    VIDEO_FPS, (IR_W, IR_H),
                )
                recording = True
            else:
                recording = False
                video_writer.release()
                video_writer = None
        elif key == ord('s'):
            take_screenshot = True

finally:
    running = False
    with frame_cv:            # wake the encoder and any connected viewers
        frame_cv.notify_all()
    with jpeg_cv:
        jpeg_cv.notify_all()
    time.sleep(0.3)           # let threads see running=False before releasing devices
    if recording and video_writer:
        video_writer.release()
    csv_file.close()
    if thermal_cap is not None:
        thermal_cap.release()
    try:
        pipeline.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)            # flush any pending window events
    print(f"\n  Stopped. {total_logged} detections saved to {CSV_PATH.resolve()}")

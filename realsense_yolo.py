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

IR detection mode  : IR + depth @ 60 FPS, no colour stream
RGB detection mode : colour + IR + depth @ 30 FPS
Display            : main feed + thermal overlay (left) | thermal panel (right)
                     RED crosshair  = global thermal hotspot
                     YELLOW crosshair = hotspot within detected person region
"""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
INFERENCE_SIZE = 320
CONF_THRESHOLD = 0.25
PERSON_CLASS   = 0
CSV_PATH       = Path("detections.csv")
VIDEO_PATH     = Path("recording.avi")
CAPTURE_DIR    = Path("captures")
VIDEO_FPS      = 30
WIN_TITLE      = "Recon Drone — Human Detection"
FFC_STD_THRESH = 1.0        # true FFC blanks are near-zero std

def find_thermal_device():
    """Find the Victor 328B by scanning for a UVC device that outputs 256x192."""
    import subprocess
    try:
        devs = subprocess.run(["v4l2-ctl", "--list-devices"],
                              capture_output=True, text=True).stdout
        import re
        for num in sorted(set(re.findall(r"/dev/video(\d+)", devs)), key=int):
            out = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{num}",
                                  "--list-formats-ext"],
                                 capture_output=True, text=True).stdout
            if "256x192" in out and "YUYV" in out:
                print(f"  Thermal : detected at /dev/video{num}")
                return int(num)
    except Exception:
        pass
    print("  Thermal : auto-detect failed — defaulting to /dev/video0")
    return 0
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
boxes_lock      = threading.Lock()
rotated_lock    = threading.Lock()

# Thermal overlay tuning (adjusted via ox / oy / sz commands)
therm_offset_x  = 15     # pixel shift right (+) / left (-)
therm_offset_y  = 0      # pixel shift down (+) / up (-)
therm_scale     = 50     # overlay scale %; 100 = fill frame, 50 = 50% zoom

last_thermal        = np.zeros((480, 640, 3), dtype=np.uint8)
last_thermal_warped = np.zeros((480, 640, 3), dtype=np.uint8)
latest_therm_raw    = np.zeros((192, 256),    dtype=np.uint8)
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
depth_w      = 480
depth_h      = 270

# ── Pipeline management ───────────────────────────────────────────────────────
def start_pipeline(rgb_detection):
    global pipeline, depth_sensor, depth_w, depth_h, emitter_on

    if pipeline:
        try:
            pipeline.stop()
        except Exception:
            pass

    pipeline = rs.pipeline()
    cfg      = rs.config()

    if rgb_detection:
        cfg.enable_stream(rs.stream.color,    640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8,  30)
        cfg.enable_stream(rs.stream.depth,    640, 480, rs.format.z16,  30)
        depth_w, depth_h = 640, 480
    else:
        cfg.enable_stream(rs.stream.infrared, 1, 480, 270, rs.format.y8,  60)
        cfg.enable_stream(rs.stream.depth,    480, 270, rs.format.z16,  60)
        depth_w, depth_h = 480, 270

    pipeline.start(cfg)
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if emitter_on else 0)

# ── Thermal thread ────────────────────────────────────────────────────────────
def thermal_loop(cap):
    global last_thermal, last_thermal_warped, latest_therm_raw, thermal_ready
    global therm_p5, therm_p82, therm_p99, therm_version
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
        # Compute percentiles here at 25 Hz so main loop never has to
        flat = grey.flatten().astype(np.float32)
        new_p5  = float(np.percentile(flat,  5))
        new_p82 = float(np.percentile(flat, 82))
        new_p99 = float(np.percentile(flat, 99))
        coloured = cv2.resize(cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO), (640, 480))
        scale = therm_scale / 100.0
        M = np.float32([
            [scale, 0, 320 * (1 - scale) + therm_offset_x],
            [0, scale, 240 * (1 - scale) + therm_offset_y],
        ])
        warped = cv2.warpAffine(coloured, M, (640, 480))
        with thermal_lock:
            last_thermal[:]                                   = coloured
            last_thermal_warped[:]                            = warped
            latest_therm_raw[:grey.shape[0], :grey.shape[1]] = grey
            therm_p5, therm_p82, therm_p99 = new_p5, new_p82, new_p99
            therm_version += 1
            thermal_ready = True

# ── Inference thread ──────────────────────────────────────────────────────────
def inference_loop(model):
    global latest_boxes, latest_rotated_boxes
    while running:
        with infer_lock:
            if latest_infer is None:
                continue
            frame = latest_infer.copy()

        results = model(frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD,
                        classes=[PERSON_CLASS], verbose=False)[0]
        boxes = list(results.boxes) if results.boxes else []

        # Rotated pass — only when primary finds nobody.
        # A person lying flat looks "standing" when the frame is rotated 90°.
        rot_boxes = []
        if not boxes:
            frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            res_rot = model(frame_rot, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD,
                            classes=[PERSON_CLASS], verbose=False)[0]
            if res_rot.boxes:
                # Unrotate boxes: 90° CW rotation maps (ox,oy) → (479-oy, ox)
                # Inverse: (rx,ry) → ox=ry, oy=479-rx
                for rb in res_rot.boxes:
                    rx1, ry1, rx2, ry2 = map(float, rb.xyxy[0])
                    conf = float(rb.conf[0])
                    ox1 = int(np.clip(ry1,       0, 639))
                    oy1 = int(np.clip(479 - rx2, 0, 479))
                    ox2 = int(np.clip(ry2,       0, 639))
                    oy2 = int(np.clip(479 - rx1, 0, 479))
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
                        (640, 480),
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
def letterbox(img, out_w=640, out_h=480):
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
    Map IR display (640×480) → thermal sensor (256×192) using FOV angles.
    Points outside the thermal FOV are clamped to the thermal frame edge.
    """
    norm_h = (ix - 320) / 320.0
    norm_v = (iy - 240) / 240.0
    # Scale by FOV ratio — thermal is more zoomed in, so same angle → larger norm
    tx = 128 + norm_h * (D435_IR_FOV_H / VICTOR_FOV_H) * 128
    ty = 96  + norm_v * (D435_IR_FOV_V / VICTOR_FOV_V) * 96
    return int(np.clip(tx, 0, 255)), int(np.clip(ty, 0, 191))

def therm_raw_to_panel(hx, hy):
    """Map raw thermal (256×192) → thermal panel display (640×480), applying scale+offset.
    Used only for the warped overlay on the IR display — NOT for drawing on the thermal panel."""
    base_px = hx * 640 / 256
    base_py = hy * 480 / 192
    scale = therm_scale / 100.0
    px = int(320 + scale * (base_px - 320) + therm_offset_x)
    py = int(240 + scale * (base_py - 240) + therm_offset_y)
    return px, py

def therm_raw_to_panel_full(hx, hy):
    """Map raw thermal (256×192) → thermal panel 640×480 without scale/offset.
    Use this for drawing boxes/crosshairs on the right thermal panel."""
    return int(hx * 640 / 256), int(hy * 480 / 192)

def ir_to_therm_panel(ix, iy):
    """Map IR display → thermal panel display via FOV mapping + current scale/offset."""
    tx, ty = ir_to_therm_raw(ix, iy)
    return therm_raw_to_panel(tx, ty)

def therm_raw_to_ir(tx, ty):
    """Inverse of ir_to_therm_raw: thermal raw (256×192) → IR display (640×480)."""
    ix = 320 + 320.0 * (tx - 128) * VICTOR_FOV_H / (128.0 * D435_IR_FOV_H)
    iy = 240 + 240.0 * (ty - 96)  * VICTOR_FOV_V / (96.0  * D435_IR_FOV_V)
    return int(np.clip(ix, 0, 639)), int(np.clip(iy, 0, 479))

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
def hud_text(img, text, pos, scale=0.7, colour=(0, 220, 255), thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)

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
        (tw2, th2), _ = cv2.getTextSize(tlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img, (x1 - 2, y2 + 2), (x1 + tw2 + 4, y2 + th2 + 8), (0, 0, 0), -1)
        cv2.putText(img, tlabel, (x1 + 2, y2 + th2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

# ── Init ──────────────────────────────────────────────────────────────────────
CAPTURE_DIR.mkdir(exist_ok=True)

print("\n  Recon Drone — Human Detection System")
print("  ─────────────────────────────────────")
print("  Model   : YOLOv8n-pose NCNN")
print("  Camera  : Intel RealSense D435  (87°×58° IR FOV)")
print("  Thermal : Victor 328B (56°×42° FOV — auto-detected)")
print("  Commands: i d e r s  |  ox <px>  oy <px>  sz <pct>  |  q=quit")
print("  Hotspot : RED=global  YELLOW=within detection\n")

model = YOLO("yolov8n-pose_ncnn_model", task="pose")
start_pipeline(detect_rgb)

thermal_cap = cv2.VideoCapture(find_thermal_device())
thermal_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not thermal_cap.isOpened():
    print("  [!] Thermal camera not found — thermal disabled")

threading.Thread(target=terminal_listener, daemon=True).start()
threading.Thread(target=inference_loop, args=(model,), daemon=True).start()
threading.Thread(target=thermal_loop,   args=(thermal_cap,), daemon=True).start()

cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_TITLE, 1280, 480)
cv2.moveWindow(WIN_TITLE, 0, 0)

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
combined          = np.empty((480, 1280, 3), dtype=np.uint8)  # pre-allocated, reused every frame
_last_therm_ver   = -1
t_coloured        = np.zeros((480, 640, 3), dtype=np.uint8)
t_warped          = np.zeros((480, 640, 3), dtype=np.uint8)

fps_counter = 0
fps_display = 0.0
fps_timer   = time.time()

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
        ir_frame     = frames.get_infrared_frame(1)
        colour_frame = frames.get_color_frame() if detect_rgb else None

        if not depth_frame or not ir_frame:
            continue
        if detect_rgb and not colour_frame:
            continue

        ir_raw = cv2.cvtColor(np.asanyarray(ir_frame.get_data()), cv2.COLOR_GRAY2BGR)
        if ir_raw.shape[:2] == (480, 640):
            ir_bgr = ir_raw
            lb_x, lb_y, lb_scale = 0, 0, 1.0
        else:
            ir_bgr, lb_x, lb_y, lb_scale = letterbox(ir_raw)  # noqa: F841
        rgb = np.asanyarray(colour_frame.get_data()) if colour_frame else None

        with infer_lock:
            latest_infer = rgb.copy() if (detect_rgb and rgb is not None) else ir_bgr.copy()

        display = rgb.copy() if (not ir_mode and rgb is not None) else ir_bgr.copy()

        # Grab thermal state — skip big copies when thermal frame hasn't changed
        with thermal_lock:
            cur_ver = therm_version
            t_raw   = latest_therm_raw.copy()
            t_ready = thermal_ready
            if cur_ver != _last_therm_ver:
                t_coloured = last_thermal.copy()
                t_warped   = last_thermal_warped.copy()
                _last_therm_ver = cur_ver

        # ── Thermal overlay — use cached warped frame (no per-frame warpAffine) ─
        if t_ready:
            display = cv2.addWeighted(
                display, 1.0 - THERMAL_ALPHA, t_warped, THERMAL_ALPHA, 0)

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
            cx   = max(0, min((x1 + x2) // 2, 639))
            cy   = max(0, min((y1 + y2) // 2, 479))

            # Depth
            dcx    = max(0, min(int((cx - lb_x) / lb_scale), depth_w - 1))
            dcy    = max(0, min(int((cy - lb_y) / lb_scale), depth_h - 1))
            dist_m = depth_frame.get_distance(dcx, dcy)

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
                tp_x1 = max(0, min(tp_x1, 639))
                tp_x2 = max(0, min(tp_x2, 639))
                tp_y1 = max(0, min(tp_y1, 479))
                tp_y2 = max(0, min(tp_y2, 479))

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
            dist_r = depth_frame.get_distance(dcx_r, dcy_r)
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
                (tw2, th2), _ = cv2.getTextSize(tl_r, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(display, (x1r - 2, y2r + 2),
                              (x1r + tw2 + 4, y2r + th2 + 8), (0, 0, 0), -1)
                cv2.putText(display, tl_r, (x1r + 2, y2r + th2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)
            # Thermal panel — project box
            if t_ready:
                trx1, try1 = ir_to_therm_raw(x1r, y1r)
                trx2, try2 = ir_to_therm_raw(x2r, y2r)
                ttp_x1, ttp_y1 = therm_raw_to_panel_full(trx1, try1)
                ttp_x2, ttp_y2 = therm_raw_to_panel_full(trx2, try2)
                ttp_x1 = max(0, min(ttp_x1, 639)); ttp_x2 = max(0, min(ttp_x2, 639))
                ttp_y1 = max(0, min(ttp_y1, 479)); ttp_y2 = max(0, min(ttp_y2, 479))
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
                dist_rl = depth_frame.get_distance(dcx_rl, dcy_rl)
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
                cx_l   = max(0, min((x1l + x2l) // 2, 639))
                cy_l   = max(0, min((y1l + y2l) // 2, 479))
                dcx_l  = max(0, min(int((cx_l - lb_x) / lb_scale), depth_w - 1))
                dcy_l  = max(0, min(int((cy_l - lb_y) / lb_scale), depth_h - 1))
                dist_l = depth_frame.get_distance(dcx_l, dcy_l)
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
                dist_t = depth_frame.get_distance(dcx_t, dcy_t)
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
            hud_text(display, "DETECTED", (640 - 120, 28), colour=(0, 60, 255), scale=0.8)
        elif rotated_boxes:
            hud_text(display, "DETECTED (lying)", (640 - 200, 28), colour=(0, 140, 255), scale=0.8)
        elif therm_candidates:
            hud_text(display, f"HEAT SIG x{len(therm_candidates)}", (640 - 160, 28),
                     colour=(0, 220, 255), scale=0.7)
        if recording:
            hud_text(display, "REC", (640 - 60, 62), colour=(0, 0, 255), scale=0.7)
            video_writer.write(display)

        combined[:, :640]  = display
        combined[:, 640:]  = thermal_panel

        # Screenshot
        if take_screenshot:
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            fpath = CAPTURE_DIR / f"capture_{ts}.png"
            cv2.imwrite(str(fpath), combined)
            print(f"  Screenshot saved → {fpath.resolve()}")
            take_screenshot = False

        with frame_lock:
            latest_frame = combined

        cv2.imshow(WIN_TITLE, combined)

        key = cv2.waitKey(1) & 0xFF
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
                    VIDEO_FPS, (640, 480),
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
    time.sleep(0.3)           # let threads see running=False before releasing devices
    if recording and video_writer:
        video_writer.release()
    csv_file.close()
    thermal_cap.release()
    try:
        pipeline.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
    cv2.waitKey(1)            # flush any pending window events
    print(f"\n  Stopped. {total_logged} detections saved to {CSV_PATH.resolve()}")

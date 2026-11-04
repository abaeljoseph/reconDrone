import argparse, collections, csv, http.server, itertools, os, threading, time
import cv2, numpy as np
import web
from web import (start_camera, load_nets, load_md, detect, find_people, plausible, draw, heat_boxes, warm,
                 heat_found, CLAHE, CONF_IR, CONF_RGB, CONF_MD, IR_INDEX, VOTE_N, VOTE_K, JPEG_Q)
from thermal import Thermal, gray, heat_people, W as TH_W, H as TH_H
# Detection tests for the report: the phone feed's pipeline (web.py: rotated pass, size check and 3-of-5 vote) on a
# fixed setup for TEST_S s, one row per test appended to ~/report_tests.csv, plus a frame from the middle of each test
# and the last frame where a box was dropped, in ~/report_shots/<start time>/.
# Detectors: pose = the YOLOv8n-pose the phone uses, md = MegaDetector v6 (md_net.py), pose+md = pose with the
# MegaDetector-on-CLAHE fallback (web.py find_people), heat = body-warm blobs in the thermal frame (thermal.py),
# fused = the live phone pipeline (web.py infer): pose, lying boxes kept only with body heat on them, plus heat-only people.
# Modes: ir, rgb (D435), thermal (Victor 328B, no depth so only the whole-frame part of the size check applies).
#   python3 report_tests.py                walks through PLAN, asking for each test's name (Enter keeps the planned one)
#   python3 report_tests.py NAME [--det D] [--mode M] [--rot] [--clahe] [--wait S]   runs one test
#   --view PORT streams the annotated feed at http://<pi>:PORT/ (live with the next test's setup between tests)
#   --auto S [--start SCENE] runs PLAN unattended, telling the person what to do on the stream, S s per scene change
#   --tag T appends _T to every test name, e.g. lit
#   --only A,B,... runs just those scenes (with --auto, or walking through), each with KEY_SETUPS
# Stop web.py first: only one process can hold the cameras.
TEST_S = 10
CSV_PATH = os.path.expanduser('~/report_tests.csv')
SHOT_DIR = os.path.expanduser('~/report_shots')
FIELDS = ['time', 'test', 'detector', 'mode', 'rotated', 'clahe', 'conf_min', 'frames', 'frames_person',
          'detection_pct', 'raw_pct', 'mean_conf', 'mean_dist_m', 'mean_fps']   # raw_pct: any box before size check and vote
POSTURES = ['standing', 'sitting', 'lying_along', 'lying_across', 'lying_debris']   # 2 m from the camera
# Each scene: pose IR with rotated pass and CLAHE each off/on, MegaDetector IR with CLAHE off/on, pose with the
# MegaDetector fallback, pose RGB baseline, then thermal: heat blobs, pose and MegaDetector on the white-hot image.
# Nobody in view first: any detection there is a false positive. Then range at 1, 2, 3 m.
SETUPS = [('pose', 'ir', 0, 0), ('pose', 'ir', 1, 0), ('pose', 'ir', 0, 1), ('pose', 'ir', 1, 1),
          ('md', 'ir', 1, 0), ('md', 'ir', 1, 1), ('pose+md', 'ir', 1, 0), ('fused', 'ir', 1, 0), ('pose', 'rgb', 1, 0),
          ('fused', 'rgb', 1, 0), ('heat', 'thermal', 0, 0), ('pose', 'thermal', 1, 0), ('md', 'thermal', 1, 0)]
DIST_SETUPS = [('pose', 'ir', 1, 0), ('md', 'ir', 1, 0), ('fused', 'ir', 1, 0), ('heat', 'thermal', 0, 0)]
# People behind or under things, after the face-heat change (2026-11-01). empty_after: straight after getting off the
# couch, for the warm patch left behind.
COVER = ['floor_cushions_2m', 'covered_2m', 'covered_head_2m', 'empty_after', 'chair_crouch', 'chair_sit', 'chair_peek',
         'chair_lying']
KEY_SETUPS = [('pose', 'ir', 1, 0), ('pose', 'rgb', 1, 0), ('fused', 'ir', 1, 0), ('fused', 'rgb', 1, 0),
              ('heat', 'thermal', 0, 0)]
PLAN = [(scene, *s) for scene in ['empty'] + [f'{p}_2m' for p in POSTURES] for s in SETUPS] \
     + [(f'dist_{m}m', *s) for m in (1, 2, 3) for s in DIST_SETUPS] + [(c, *s) for c in COVER for s in KEY_SETUPS]
PAGE = b'<html><body style="margin:0;background:#000"><img src="/mjpg" style="width:100%;height:100vh;object-fit:contain">'
# What the person does for each scene, shown on the stream in --auto
INSTRUCT = {'empty': 'Get out of view, nobody in frame',
            'standing_2m': 'Stand about 2 m away, facing the camera',
            'sitting_2m': 'Sit on the couch',
            'lying_along_2m': 'Lie on the floor, feet toward the camera',
            'lying_across_2m': 'Lie on the couch, sideways',
            'lying_debris_2m': 'Lie on the couch, half covered (blanket, cushions)',
            'dist_1m': 'Stand 1 m from the camera', 'dist_2m': 'Stand 2 m from the camera',
            'dist_3m': 'Stand 3 m from the camera',
            'floor_cushions_2m': 'Lie on the floor, cushions over your body, face out',
            'covered_2m': 'Lie on the couch, blanket over everything but your face',
            'covered_head_2m': 'Lie on the couch, blanket over your head too',
            'empty_after': 'Get up off the couch and out of view',
            'chair_crouch': 'Crouch behind the chair, only head and shoulders showing',
            'chair_sit': 'Sit on the floor behind the chair, hidden as much as you can',
            'chair_peek': 'Behind the chair, only your head peeking out the side',
            'chair_lying': 'Lie on the floor behind the chair'}
NO_DEPTH = np.zeros((480, 640), np.float32)
cam_lock = threading.Lock()
shown, viewing, nxt, banner, tag = None, False, None, '', ''   # latest JPEG, --view on, live view's setup, stream
                                                               # instruction, test name suffix
run_dir = os.path.join(SHOT_DIR, time.strftime('%Y%m%d-%H%M%S'))

class Cam:
    """The D435 in one mode at a time, and the thermal camera, kept open once used (it takes ~10 s to start) and read
    in the background into web.heat as web.py's heat_cam(), so the fused detector sees it beside the D435."""
    def __init__(self): self.mode = self.rs_mode = self.pipe = self.th = None; self.seen = 0.0

    def open(self, mode, heat=False):
        if (mode == 'thermal' or heat) and not self.th:
            self.th = Thermal(); self.th.warm(); print('camera: thermal', flush=True)
            threading.Thread(target=self.heat_loop, args=(self.th,), daemon=True).start()
            while web.heat[1] is None: time.sleep(0.05)
        if mode != 'thermal' and mode != self.rs_mode:
            if self.pipe: self.pipe.stop()
            self.pipe, self.scale, self.k, self.align, self.lens = start_camera(mode); self.rs_mode = mode
            for _ in range(30): self.pipe.wait_for_frames(2000)    # let auto exposure settle
        self.mode = mode

    def heat_loop(self, cam):
        try:
            while self.th is cam:
                t = cam.read(); web.heat = (time.time(), t, heat_people(t, (TH_W, TH_H)))
        except Exception as e: print('thermal:', e, flush=True)

    def close(self):
        if self.pipe: self.pipe.stop()
        th, self.th = self.th, None
        if th: th.close()
        self.mode = self.rs_mode = self.pipe = None

    def read(self):
        """Display-oriented image (gray for IR and thermal) and depth in m as web.py's camera(), or for thermal the
        temperature frame in C."""
        if self.mode == 'thermal':
            while web.heat[0] == self.seen: time.sleep(0.005)
            self.seen, t, _ = web.heat
            return gray(t), t
        while True:
            fs = self.pipe.wait_for_frames(2000)
            if self.align: fs = self.align.process(fs)
            im = fs.get_color_frame() if self.align else fs.get_infrared_frame(IR_INDEX)
            dep = fs.get_depth_frame()
            if im and dep: break
        img = cv2.rotate(np.asanyarray(im.get_data()), cv2.ROTATE_180)
        return img, cv2.rotate(np.asanyarray(dep.get_data()), cv2.ROTATE_180).astype(np.float32) * self.scale

class Nets:
    """Loads each detector the first time a test needs it."""
    def __init__(self): self.got = {}

    def __getitem__(self, det):
        if det == 'pose+md': return self['pose'], self['md']
        if det == 'fused': return self['pose']
        if det == 'heat': return None
        if det not in self.got: self.got[det] = load_md() if det == 'md' else load_nets()
        return self.got[det]

def conf_for(det, mode):
    if det == 'heat': return 0
    if det == 'pose+md': return f'{CONF_IR}/{CONF_MD}'
    return CONF_MD if det == 'md' else CONF_RGB if mode == 'rgb' else CONF_IR

def step(cam, nets, det, mode, rot, clahe):
    """One frame through the phone's pipeline up to the vote: image, every box, boxes passing the size check, fx."""
    cam.open(mode, det == 'fused'); net = nets[det]
    img, z = cam.read()
    if img.ndim == 2:
        if clahe: img = CLAHE.apply(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if mode == 'thermal':
        raw = heat_people(z, (w, h)) if det == 'heat' else detect(*net, img, NO_DEPTH, 0.0, conf_for(det, mode), rot)
        return img, raw, [p for p in raw if plausible(p, 1.0, w, h)], None
    if det == 'pose+md': return (img, *find_people(*net, img, z, cam.k, cam.lens[0], mode, rot), cam.lens[0])
    if det == 'fused':
        hb = heat_boxes(cam.lens, img.shape)
        raw, found = find_people(net, None, img, z, cam.k, cam.lens[0], mode, rot)
        found = [p for p in found if not p[6] or warm(p, hb)]
        more = heat_found(z, cam.k, cam.lens, hb, found)
        return img, raw + more, found + more, cam.lens[0]
    raw = detect(*net, img, z, cam.k, conf_for(det, mode), rot)
    return img, raw, [p for p in raw if plausible(p, cam.lens[0], w, h)], cam.lens[0]

def render(img, raw, found, counted, label, fx):
    """Counted people as on the phone. Boxes the vote held back in yellow, boxes the size check dropped in grey with
    why (whole frame, or the size in m). Label top left, the --auto instruction along the bottom."""
    out = img.copy(); h, w = out.shape[:2]
    for p in raw:
        if p in counted: continue
        x1, y1, x2, y2, _, dist, _ = map(float, p); pt = (int(x1), int(y1))
        if p in found: col, why = (0, 220, 255), 'vote'
        elif x2 - x1 > 0.9 * w and y2 - y1 > 0.9 * h: col, why = (128, 128, 128), 'whole frame'
        else: col, why = (128, 128, 128), f'{max(x2 - x1, y2 - y1) * dist / fx:.1f} m' if dist and fx else ''
        cv2.rectangle(out, pt, (int(x2), int(y2)), col, 1)
        cv2.putText(out, f'{why} {p[4]:.2f}', (pt[0] + 3, pt[1] + (44 if pt[1] < 30 else 14)),   # clear of the label
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, col, 1, cv2.LINE_AA)
    draw(out, counted)
    for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, col, th, cv2.LINE_AA)
    if banner:
        s = min(0.65, 0.65 * (w - 16) / cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)[0][0])
        cv2.rectangle(out, (0, h - 34), (w, h), (0, 0, 0), -1)
        cv2.putText(out, banner, (8, h - 11), cv2.FONT_HERSHEY_DUPLEX, s, (0, 255, 255), 1, cv2.LINE_AA)
    return out

def show(out):
    global shown
    if viewing: shown = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1].tobytes()

class View(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path != '/mjpg':
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(PAGE); return
        self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f'); self.end_headers()
        sent = None
        try:
            while True:
                j = shown
                if j is None or j is sent: time.sleep(0.01); continue
                sent = j
                self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(j) + j + b'\r\n')
        except (BrokenPipeError, ConnectionResetError): pass

def live(cam, nets):
    """Between tests: the next test's setup on every frame, no vote."""
    while True:
        s = nxt
        if s is None: time.sleep(0.2); continue
        name, det, mode, rot, cl = s
        try:
            with cam_lock: img, raw, found, fx = step(cam, nets, det, mode, rot, cl and mode == 'ir')
            show(render(img, raw, found, found, f'LIVE  next: {name}  {det} {mode.upper()} rot {"on" if rot else "off"} '
                                                f'CLAHE {"on" if cl else "off"}', fx))
        except Exception as e: print('live:', e, flush=True); time.sleep(1)

def run_test(cam, nets, name, det, mode, rot, clahe):
    name += f'_{tag}' if tag else ''
    clahe = clahe and mode == 'ir'
    conf = conf_for(det, mode)
    votes, n, hits, raw, confs, dists = collections.deque(maxlen=VOTE_N), 0, 0, 0, [], []
    mid = miss = None
    with cam_lock:
        cam.open(mode, det == 'fused'); nets[det]
        print(f'running {name} {det} {mode} for {TEST_S} s ...', flush=True)
        t0 = time.time()
        while time.time() - t0 < TEST_S:
            img, boxes, found, fx = step(cam, nets, det, mode, rot, clahe)
            raw += bool(boxes); n += 1
            votes.append(bool(found))
            hit = bool(found) and sum(votes) >= VOTE_K
            if hit:
                hits += 1; best = max(found, key=lambda p: p[4]); confs.append(float(best[4]))
                if best[5]: dists.append(best[5])
            counted = found if hit else []
            out = render(img, boxes, found, counted, f'TEST {name}  {det} {mode.upper()}  '
                                                     f'{TEST_S - (time.time() - t0):.0f} s  person {hits}/{n}', fx)
            show(out)
            if mid is None and time.time() - t0 >= TEST_S / 2: mid = out
            if len(boxes) > len(counted): miss = out
        dt = time.time() - t0
    os.makedirs(run_dir, exist_ok=True)
    stem = os.path.join(run_dir, f'{time.strftime("%H%M%S")}_{name}_{det}_{mode}_rot{int(rot)}_clahe{int(clahe)}')
    for kind, im in (('mid', mid), ('miss', miss)):
        if im is not None: cv2.imwrite(f'{stem}_{kind}.jpg', im)
    row = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), test=name, detector=det, mode=mode.upper(),
               rotated='on' if rot else 'off', clahe='on' if clahe else 'off', conf_min=conf, frames=n,
               frames_person=hits, detection_pct=f'{100 * hits / max(n, 1):.1f}', raw_pct=f'{100 * raw / max(n, 1):.1f}',
               mean_conf=f'{np.mean(confs):.3f}' if confs else '', mean_dist_m=f'{np.mean(dists):.2f}' if dists else '',
               mean_fps=f'{n / dt:.1f}')
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as f:
        wr = csv.DictWriter(f, FIELDS)
        if new: wr.writeheader()
        wr.writerow(row)
    print('  '.join(f'{k} {row[k]}' for k in FIELDS[1:]), flush=True)

def walk(cam, nets):
    global nxt
    i = 0
    while i < len(PLAN):
        name, det, mode, rot, cl = nxt = PLAN[i]
        a = input(f'\n[{i + 1}/{len(PLAN)}] {name}  {det}  {mode.upper()}  rotated {"on" if rot else "off"}  '
                  f'CLAHE {"on" if cl else "off"}\n  Enter = run, or type a test name; s skip, b back, q quit: ').strip()
        if a == 'q': return
        if a == 's': i += 1; continue
        if a == 'b': i = max(0, i - 1); continue
        for s in (3, 2, 1): print(s, flush=True); time.sleep(1)
        run_test(cam, nets, a or name, det, mode, rot, cl); i += 1

def auto(cam, nets, ready, start):
    """Runs PLAN unattended: each scene's instruction and a countdown on the stream, then its tests back to back."""
    global nxt, banner
    scenes = [(s, list(g)) for s, g in itertools.groupby(PLAN, key=lambda t: t[0])]
    if start: scenes = scenes[[s for s, _ in scenes].index(start):]
    for name, tests in scenes:
        nxt = tests[0]
        for left in range(int(ready), 0, -1):
            banner = f'GET READY {left}: {INSTRUCT[name]}'; time.sleep(1)
        for i, t in enumerate(tests):
            banner = f'HOLD STILL {i + 1}/{len(tests)}: {INSTRUCT[name]}'
            run_test(cam, nets, *t)
    banner = 'ALL DONE'; time.sleep(10)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?')
    ap.add_argument('--det', choices=('pose', 'md', 'pose+md', 'fused', 'heat'), default='pose')
    ap.add_argument('--mode', choices=('ir', 'rgb', 'thermal'), default='ir')
    ap.add_argument('--rot', action='store_true')
    ap.add_argument('--clahe', action='store_true')
    ap.add_argument('--wait', type=float, default=0, help='s before the test starts, to get into position')
    ap.add_argument('--view', type=int, metavar='PORT', help='stream the annotated feed on this port')
    ap.add_argument('--auto', type=float, metavar='S', help='run PLAN unattended, S s to get into each scene')
    ap.add_argument('--start', choices=list(INSTRUCT), help='with --auto: first scene')
    ap.add_argument('--tag', default='', help='suffix for every test name')
    ap.add_argument('--only', help='comma separated scenes, in that order, with KEY_SETUPS only')
    a = ap.parse_args()
    tag = a.tag
    if a.only: PLAN = [(s, *k) for s in a.only.split(',') for k in KEY_SETUPS]
    nets, cam = Nets(), Cam()
    if a.view:
        viewing = True
        srv = http.server.ThreadingHTTPServer(('', a.view), View); srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        threading.Thread(target=live, args=(cam, nets), daemon=True).start()
    try:
        if a.name:
            nxt = (a.name, a.det, a.mode, a.rot, a.clahe); time.sleep(a.wait); nxt = None
            run_test(cam, nets, a.name, a.det, a.mode, a.rot, a.clahe)
        elif a.auto is not None: auto(cam, nets, a.auto, a.start)
        else: walk(cam, nets)
    finally:
        nxt = None
        with cam_lock: cam.close()

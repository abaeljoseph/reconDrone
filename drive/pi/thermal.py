import time
import cv2, numpy as np
# Victor 328B thermal camera (USB 0bda:5840) over V4L2: YUYV 256x384 at 25 fps. The top 192 rows are its own image,
# the bottom 192 rows raw temperature, one uint16 per pixel, T = raw/64 - 273.15 C. For ~10 s after opening it repeats
# one frame. Upright as mounted, same direction as the D435 but a narrower view (~1.6x zoom).
DEV = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'   # /dev/videoN changes between boots
W, H = 256, 192
# All temperatures are relative to the scene median: the absolute reading drifts for minutes after opening
# (walls read 36 C, then 28 C a minute later, in a ~23 C room on 2026-10-31).
T_SPAN = (-4, 14)   # C shown black to white, for display and the detectors
HOT = 4             # C above for a pixel to count as body heat
T_MAX = 16          # C above: blobs this hot, or touching something this hot, are lamps and electronics
MIN_AREA = 120      # px of the 256x192 frame, about a head at 3 m
FACE = (30, 7)      # px, C above: smaller blobs count if this hot, a face out from under a blanket (60 px, +8 C at 2 m);
                    # the couch's warm patch after sitting peaked at +5.7 (2026-11-01)
GAP = 4             # px: blobs this close are one person (clothes and blankets split the body heat into pieces)
# Blobs above one another, sharing half their width and no further apart than the taller one's height, are one
# standing person: shorts left a 56 px gap between shirt and legs on 2026-11-01.

class Thermal:
    def __init__(self, dev=DEV):
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2 * H)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self.cap.isOpened(): raise RuntimeError(f'thermal: cannot open {dev}')

    def read(self):
        """Temperature frame, C, H x W float32."""
        ok, f = self.cap.read()
        if not ok: raise RuntimeError('thermal: no frame')
        return f.reshape(2 * H, W, 2)[H:].view(np.uint16).reshape(H, W).astype(np.float32) / 64 - 273.15

    def warm(self, timeout=20):
        """Read until the frames start changing."""
        first, t0 = self.read(), time.time()
        while time.time() - t0 < timeout:
            if not np.array_equal(self.read(), first): return
        raise RuntimeError('thermal: frames never changed')

    def close(self): self.cap.release()

def gray(t, size=(640, 480)):
    """White-hot 8-bit image of a temperature frame."""
    lo, hi = np.median(t) + np.array(T_SPAN)
    return cv2.resize(np.clip((t - lo) * (255 / (hi - lo)), 0, 255).astype(np.uint8), size)

def heat_people(t, size=(640, 480)):
    """Body-warm blobs as (x1, y1, x2, y2, conf, dist 0, lying) in size coords. conf grows with how far the blob's
    peak is above the scene median, 1 at 10 C; lying if it is wider than tall. Blobs within GAP px are one."""
    amb = float(np.median(t))
    m = cv2.morphologyEx(((t > amb + HOT) & (t < amb + T_MAX)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(cv2.dilate(m, np.ones((2 * GAP + 1, 2 * GAP + 1), np.uint8)))
    lab[m == 0] = 0                                  # grouped by the dilated mask, sized by the real one
    hot, sx, sy, bs = t >= amb + T_MAX, size[0] / W, size[1] / H, []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if not len(xs): continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        if hot[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2].any(): continue
        bs.append([x1, y1, x2, y2, len(xs), float(t[ys, xs].max())])
    merged = True
    while merged:                                    # head, shirt and legs into one box
        merged = False
        for a in bs:
            for b in bs:
                if a is b: continue
                share, gy = min(a[2], b[2]) - max(a[0], b[0]), max(a[1], b[1]) - min(a[3], b[3])
                if (share >= 0.5 * min(a[2] - a[0], b[2] - b[0]) and gy <= max(a[3] - a[1], b[3] - b[1])
                        or share >= -GAP and gy <= GAP):     # or the boxes touch: a leg beside the body's box
                    a[:] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4], max(a[5], b[5])]
                    bs.remove(b); merged = True; break
            if merged: break
    out = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), min(1.0, (peak - amb) / 10), 0.0,
            bool(x2 - x1 > 1.3 * (y2 - y1))) for x1, y1, x2, y2, area, peak in bs
           if area >= MIN_AREA or area >= FACE[0] and peak - amb >= FACE[1]]
    return sorted(out, key=lambda p: -p[4])

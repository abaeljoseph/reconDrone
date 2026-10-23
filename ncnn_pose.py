"""
Minimal YOLOv8-pose runner on raw NCNN — no torch, no ultralytics.

Ultralytics imports torch unconditionally even when the model is NCNN, and
torch takes ~33s to load cold off an SD card (~55s with the rest of
ultralytics). This does the same inference with only ncnn + numpy + cv2.

Only boxes and confidences are decoded; keypoints are ignored because nothing
downstream uses them. Preprocessing mirrors ultralytics (aspect-preserving
letterbox padded with 114, RGB, /255) so detections match.
"""

from pathlib import Path

import cv2
import ncnn
import numpy as np


class Box:
    """Mimics the bits of an ultralytics Box the detection code reads."""
    __slots__ = ("xyxy", "conf")

    def __init__(self, x1, y1, x2, y2, conf):
        self.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]
        self.conf = [float(conf)]


class PoseNet:
    def __init__(self, model_dir, shape_hw, threads=4):
        model_dir = Path(model_dir)
        self.h, self.w = shape_hw
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(str(model_dir / "model.ncnn.param"))
        self.net.load_model(str(model_dir / "model.ncnn.bin"))

    def __call__(self, img, conf=0.25, iou=0.7):
        ih, iw = img.shape[:2]
        r = min(self.w / iw, self.h / ih)
        nw, nh = int(round(iw * r)), int(round(ih * r))
        px, py = (self.w - nw) // 2, (self.h - nh) // 2

        canvas = np.full((self.h, self.w, 3), 114, dtype=np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(
            img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1), dtype=np.float32) / 255.0

        with self.net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(blob))
            _, out = ex.extract("out0")
        pred = np.array(out)           # (56, N): cx, cy, w, h, score, 17*3 kpts

        scores = pred[4]
        keep = scores > conf
        if not keep.any():
            return []
        cx, cy, bw, bh = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
        scores = scores[keep]

        # Undo the letterbox: model-input pixels → original image pixels
        x1 = (cx - bw / 2 - px) / r
        y1 = (cy - bh / 2 - py) / r
        x2 = (cx + bw / 2 - px) / r
        y2 = (cy + bh / 2 - py) / r

        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idx = cv2.dnn.NMSBoxes(rects, scores.tolist(), conf, iou)
        boxes = []
        for i in np.array(idx).flatten():
            boxes.append(Box(np.clip(x1[i], 0, iw - 1), np.clip(y1[i], 0, ih - 1),
                             np.clip(x2[i], 0, iw - 1), np.clip(y2[i], 0, ih - 1),
                             scores[i]))
        boxes.sort(key=lambda b: -b.conf[0])
        return boxes

import os
import cv2, numpy as np, ncnn
# MegaDetector v6 (MDV6-yolov10-c, Microsoft's camera-trap detector: classes animal, person, vehicle, trained on
# camera-trap photos including night IR-flash shots) exported to NCNN with ultralytics 8.4 at a fixed input size.
# Same call shape as ncnn_pose.PoseNet so web.detect can use either: net(img, conf) -> boxes with .xyxy[0], .conf[0].
PERSON = 1

class Box:
    def __init__(self, xyxy, conf): self.xyxy, self.conf = [xyxy], [conf]

class MegaDetector:
    def __init__(self, path, shape, threads=2, iou=0.45):
        self.net = ncnn.Net(); self.net.opt.num_threads = threads
        self.net.load_param(os.path.join(path, 'model.ncnn.param'))
        self.net.load_model(os.path.join(path, 'model.ncnn.bin'))
        (self.h, self.w), self.iou = shape, iou

    def __call__(self, img, conf=0.25, cls=PERSON):
        """Class cls boxes in img pixels, highest confidence first. img is BGR or gray."""
        ih, iw = img.shape[:2]; r = min(self.w / iw, self.h / ih)
        nw, nh = round(iw * r), round(ih * r); px, py = (self.w - nw) // 2, (self.h - nh) // 2
        pad = np.full((self.h, self.w) + img.shape[2:], 114, np.uint8)    # letterbox as ultralytics
        pad[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        kind = ncnn.Mat.PixelType.PIXEL_GRAY2RGB if img.ndim == 2 else ncnn.Mat.PixelType.PIXEL_BGR2RGB
        x = ncnn.Mat.from_pixels(pad, kind, self.w, self.h)
        x.substract_mean_normalize([], [1 / 255.0] * 3)
        with self.net.create_extractor() as ex:
            ex.input('in0', x); _, out = ex.extract('out0')
        p = np.array(out)                                   # (4 + classes, anchors): cx cy w h, class scores
        s = p[4 + cls]; i = np.flatnonzero(s > conf)
        if not i.size: return []
        cx, cy, w, h = p[:4, i]
        xywh = np.stack([(cx - w / 2 - px) / r, (cy - h / 2 - py) / r, w / r, h / r], 1)
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), s[i].tolist(), conf, self.iou)
        boxes = []
        for j in np.array(keep).flatten():
            x1, y1, bw, bh = xywh[j]
            boxes.append(Box((max(0.0, x1), max(0.0, y1), min(iw - 1.0, x1 + bw), min(ih - 1.0, y1 + bh)), float(s[i][j])))
        return sorted(boxes, key=lambda b: -b.conf[0])

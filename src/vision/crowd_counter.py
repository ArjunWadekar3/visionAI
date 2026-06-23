"""
Crowd counter + lightweight tracker.

- Detection: whole-frame YOLO (high imgsz) OR SAHI tiled inference (NSA_SAHI=1).
  Tiling slices the frame into overlapping patches and detects on each, so small
  / distant heads across the WHOLE frame (incl. edges) are found -- this is what
  pushes the count far past the whole-frame limit on dense crowds.

- Tracking: a self-contained IoU tracker assigns a stable unique ID to each
  person across frames, so every detected person is counted once. Gives:
    * count        -> people currently in the frame
    * unique_total -> distinct people seen since start (cumulative footfall)

Honest limit: detection-based counting still saturates on extreme density
(heads fully overlapping). SAHI + low conf + high imgsz gets the maximum a
detector can; true 10k+ exact counts need a density model.
"""

import os

import numpy as np

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class SimpleTracker:
    """IoU-based tracker -> stable unique IDs. Dependency-free, crowd-friendly."""

    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}      # id -> {"bbox": (x1,y1,x2,y2), "age": int}
        self.next_id = 1

    def update(self, boxes):
        """boxes: list of (x1,y1,x2,y2). Returns parallel list of track ids."""
        assigned = {}
        used = set()
        # Match existing tracks to the best-overlapping new detection
        for tid in list(self.tracks.keys()):
            tb = self.tracks[tid]["bbox"]
            best_i, best_iou = -1, self.iou_thresh
            for i, b in enumerate(boxes):
                if i in used:
                    continue
                v = _iou(tb, b)
                if v >= best_iou:
                    best_i, best_iou = i, v
            if best_i >= 0:
                self.tracks[tid] = {"bbox": boxes[best_i], "age": 0}
                assigned[best_i] = tid
                used.add(best_i)
            else:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        ids = []
        for i, b in enumerate(boxes):
            if i in assigned:
                ids.append(assigned[i])
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": b, "age": 0}
                ids.append(tid)
        return ids

    @property
    def unique_total(self):
        return self.next_id - 1


class CrowdCounter:
    def __init__(self, model, model_path, conf=0.25, classes=None,
                 use_sahi=False, slice_size=512, overlap=0.2, imgsz=1280,
                 track=True):
        self.model = model
        self.model_path = model_path
        self.conf = conf
        self.classes = classes
        self.slice_size = slice_size
        self.overlap = overlap
        self.imgsz = imgsz
        self.use_sahi = use_sahi
        self.tracker = SimpleTracker() if track else None
        self._sahi_model = None
        if use_sahi:
            self._init_sahi()

    def _init_sahi(self):
        try:
            from sahi import AutoDetectionModel
            last_err = None
            for mtype in ("ultralytics", "yolov8"):
                try:
                    self._sahi_model = AutoDetectionModel.from_pretrained(
                        model_type=mtype,
                        model_path=self.model_path,
                        confidence_threshold=self.conf,
                        device=_DEVICE,
                    )
                    print(f"[INFO] SAHI tiled inference ON "
                          f"(type={mtype}, device={_DEVICE}, tile={self.slice_size})")
                    return
                except Exception as e:
                    last_err = e
            raise last_err
        except Exception as e:
            print(f"[WARN] SAHI unavailable ({e}). Whole-frame detection. "
                  f"Install: pip install sahi")
            self.use_sahi = False

    def _detect(self, frame):
        """Return list of (x1,y1,x2,y2) person boxes for one frame."""
        if self.use_sahi and self._sahi_model is not None:
            from sahi.predict import get_sliced_prediction
            result = get_sliced_prediction(
                frame, self._sahi_model,
                slice_height=self.slice_size, slice_width=self.slice_size,
                overlap_height_ratio=self.overlap,
                overlap_width_ratio=self.overlap, verbose=0)
            boxes = []
            for obj in result.object_prediction_list:
                if self.classes is not None and obj.category.id not in self.classes:
                    continue
                bb = obj.bbox
                boxes.append((int(bb.minx), int(bb.miny),
                              int(bb.maxx), int(bb.maxy)))
            return boxes
        # whole-frame
        res = self.model.predict(frame, conf=self.conf, classes=self.classes,
                                 imgsz=self.imgsz, verbose=False)
        boxes = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes.xyxy.cpu().numpy():
                boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        return boxes

    def process(self, frame):
        """Return (count, unique_total, boxes, ids, centers)."""
        boxes = self._detect(frame)
        if self.tracker is not None:
            ids = self.tracker.update(boxes)
            unique_total = self.tracker.unique_total
        else:
            ids = list(range(len(boxes)))
            unique_total = len(boxes)
        centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in boxes]
        return len(boxes), unique_total, boxes, ids, centers

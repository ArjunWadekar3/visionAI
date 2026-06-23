"""
Crowd counter -- estimates how many people/heads are in each frame.

Backends:
  - Whole-frame YOLO detection (default).
  - SAHI tiled inference (set NSA_SAHI=1): slices the frame into overlapping
    tiles and detects on each, then merges. This finds small / distant heads in
    dense or aerial (drone) footage far better than whole-frame detection.

Honest limits: detection-based counting saturates on extreme density (heads
overlapping). For 10,000+ people approaching 90-95% you need a dedicated
crowd-counting (density) model fine-tuned on your footage -- see analytics.py
DensityEstimator hook. SAHI gets you much closer than plain YOLO, not all the way.
"""

import os

import numpy as np

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"


class CrowdCounter:
    def __init__(self, model, model_path, conf=0.3, classes=None,
                 use_sahi=False, slice_size=512, overlap=0.2, imgsz=1280):
        self.model = model                # ultralytics YOLO (whole-frame path)
        self.model_path = model_path
        self.conf = conf
        self.classes = classes
        self.slice_size = slice_size
        self.overlap = overlap
        self.imgsz = imgsz                # larger -> small aerial people detected better
        self.use_sahi = use_sahi
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
                    print(f"[INFO] SAHI tiled inference enabled "
                          f"(type={mtype}, device={_DEVICE}, tile={self.slice_size})")
                    return
                except Exception as e:
                    last_err = e
            raise last_err
        except Exception as e:
            print(f"[WARN] SAHI unavailable ({e}). Falling back to whole-frame "
                  f"detection. Install with: pip install sahi")
            self.use_sahi = False

    def count(self, frame):
        """Return (count, boxes, centers).
        boxes:   list of (x1, y1, x2, y2)
        centers: list of (cx, cy)
        """
        if self.use_sahi and self._sahi_model is not None:
            return self._count_sahi(frame)
        return self._count_whole(frame)

    def _count_whole(self, frame):
        res = self.model.predict(frame, conf=self.conf, classes=self.classes,
                                 imgsz=self.imgsz, verbose=False)
        boxes, centers = [], []
        if res and res[0].boxes is not None:
            for b in res[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b[:4]
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return len(boxes), boxes, centers

    def _count_sahi(self, frame):
        from sahi.predict import get_sliced_prediction
        ov = self.overlap
        result = get_sliced_prediction(
            frame,
            self._sahi_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=ov,
            overlap_width_ratio=ov,
            verbose=0,
        )
        boxes, centers = [], []
        for obj in result.object_prediction_list:
            if self.classes is not None and obj.category.id not in self.classes:
                continue
            bb = obj.bbox
            x1, y1, x2, y2 = bb.minx, bb.miny, bb.maxx, bb.maxy
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        return len(boxes), boxes, centers

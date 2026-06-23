"""
Crowd analytics + abnormal-activity detection.

CrowdAnalytics:
  - live person count (from the tracker)
  - crowd-density heatmap + crowd level (LOW/MEDIUM/HIGH/CRITICAL)
  - overcrowding alert when the count exceeds a threshold
  - per-track behaviour: loitering (stays in one place) and running (moves fast)

Density estimation note
-----------------------
Counting thousands of people accurately needs a dedicated crowd-counting model
(CSRNet / P2PNet) with pretrained weights, which are NOT pip-installable -- you
download them separately. This module ships a working *detection-based* density
heatmap + level estimate now, and exposes a hook (DENSITY_MODEL_PATH) so a real
density model can be plugged in later without touching the rest of the app.
"""

import math
import os
from collections import defaultdict, deque

import cv2
import numpy as np

# Optional: path to a real crowd-density model (e.g. exported CSRNet).
# If the file exists, DensityEstimator will try to use it; otherwise it falls
# back to the detection-based heatmap.
DENSITY_MODEL_PATH = os.environ.get("DENSITY_MODEL_PATH", "")


class CrowdAnalytics:
    def __init__(self, overcrowd_threshold=50, history=30):
        self.overcrowd_threshold = overcrowd_threshold
        self.track_history = defaultdict(lambda: deque(maxlen=history))
        self.peak_count = 0

    # ---- per-frame update -------------------------------------------------
    def update(self, tracks):
        """tracks: list of (track_id, x1, y1, x2, y2).
        Returns dict with count, level, overcrowded, abnormal events."""
        count = len(tracks)
        self.peak_count = max(self.peak_count, count)

        centers = []
        abnormal = []
        for track_id, x1, y1, x2, y2 in tracks:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers.append((cx, cy))

            hist = self.track_history[track_id]
            if hist:
                px, py = hist[-1]
                speed = math.hypot(cx - px, cy - py)
                if speed > 45:
                    abnormal.append({"id": int(track_id), "type": "running"})
            hist.append((cx, cy))

            # loitering: low movement variance over a full window
            if len(hist) >= hist.maxlen:
                xs = [p[0] for p in hist]
                ys = [p[1] for p in hist]
                if np.var(xs) < 400 and np.var(ys) < 400:
                    abnormal.append({"id": int(track_id), "type": "loitering"})

        level = self._crowd_level(count)
        overcrowded = count >= self.overcrowd_threshold
        if overcrowded:
            abnormal.append({"id": -1, "type": "overcrowding", "count": count})

        return {
            "count": count,
            "level": level,
            "overcrowded": overcrowded,
            "centers": centers,
            "abnormal": abnormal,
        }

    def _crowd_level(self, count):
        t = self.overcrowd_threshold
        if count >= t:
            return "CRITICAL"
        if count >= t * 0.6:
            return "HIGH"
        if count >= t * 0.3:
            return "MEDIUM"
        return "LOW"

    # ---- density heatmap --------------------------------------------------
    def density_overlay(self, frame, centers, alpha=0.4):
        """Blend a Gaussian density heatmap of person positions onto the frame."""
        if not centers:
            return frame
        h, w = frame.shape[:2]
        density = np.zeros((h, w), dtype=np.float32)
        for cx, cy in centers:
            ix, iy = int(cx), int(cy)
            if 0 <= ix < w and 0 <= iy < h:
                density[iy, ix] += 1.0
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=25, sigmaY=25)
        if density.max() > 0:
            density = density / density.max()
        heat = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(heat, alpha, frame, 1 - alpha, 0)


class DensityEstimator:
    """Hook for a real crowd-counting model. Falls back to detection count."""

    def __init__(self, model_path=DENSITY_MODEL_PATH):
        self.model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            try:
                # Placeholder: load your CSRNet/P2PNet weights here.
                # import torch; self.model = torch.load(model_path)
                print(f"[INFO] Density model found at {model_path} "
                      f"(plug load code into DensityEstimator).")
            except Exception as e:
                print(f"[WARN] Could not load density model: {e}")
        else:
            print("[INFO] No density model set -> using detection-based count. "
                  "Set DENSITY_MODEL_PATH to a CSRNet/P2PNet model for dense crowds.")

    def estimate(self, frame, detection_count):
        if self.model is None:
            return detection_count  # detection-based fallback
        # return self.model(frame) ...
        return detection_count

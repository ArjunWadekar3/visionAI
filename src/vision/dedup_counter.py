"""
Motion-compensated de-duplicating counter for a PANNING drone.

Problem: a moving drone revisits the same area, so per-frame detection + simple
tracking counts the same people again -> inflated totals.

Idea: estimate the camera motion between consecutive frames (ORB feature
matching + homography), accumulate it so every frame's detections can be mapped
into ONE global coordinate canvas, then count each ground position only once
(spatial grid de-dup).

Experimental limits: the accumulated homography can drift over long pans, on
low-texture scenes, or with strong parallax; moving people add noise. For
survey-grade exact counts, build an orthomosaic (e.g. OpenDroneMap) and count it
once. This is a practical live/video approximation, far better than re-counting.
"""

import cv2
import numpy as np


class DedupCounter:
    def __init__(self, dist_thresh=30):
        self.orb = cv2.ORB_create(1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_kp = None
        self.prev_des = None
        self.H = np.eye(3, dtype=np.float64)   # current frame -> global canvas
        self.dist = dist_thresh
        self.cell = dist_thresh * 2
        self.grid = {}                         # (gx,gy) cell -> list[(x,y)]
        self.unique = 0

    def _match_or_add(self, gx, gy):
        """Return the global id for this ground position (existing if already
        seen nearby, else a new id)."""
        cx, cy = int(gx // self.cell), int(gy // self.cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for (px, py, pid) in self.grid.get((cx + dx, cy + dy), ()):
                    if (px - gx) ** 2 + (py - gy) ** 2 < self.dist * self.dist:
                        return pid  # same person seen before -> same id
        self.unique += 1
        new_id = self.unique
        self.grid.setdefault((cx, cy), []).append((gx, gy, new_id))
        return new_id

    def update(self, frame, centers):
        """Update the global motion estimate and count any newly-seen positions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_des is not None and des is not None and len(kp) > 10:
            matches = self.matcher.knnMatch(self.prev_des, des, k=2)
            good = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
            if len(good) >= 12:
                src = np.float32([self.prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H_cur_to_prev, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
                if H_cur_to_prev is not None:
                    self.H = self.H @ H_cur_to_prev

        ids = []
        for (cx, cy) in centers:
            v = self.H @ np.array([cx, cy, 1.0])
            if v[2] != 0:
                ids.append(self._match_or_add(v[0] / v[2], v[1] / v[2]))
            else:
                ids.append(-1)

        self.prev_kp, self.prev_des = kp, des
        return self.unique, ids

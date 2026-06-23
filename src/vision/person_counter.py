"""
Line-crossing person counter.

Counts people who cross a virtual line (a "tripwire"), using stable track IDs
from the tracker (YOLO + ByteTrack). A person is counted once per crossing, and
the direction (A->B or B->A) is recorded along with a timestamp.

This is the reliable way to answer "how many people passed".  For sparse to
moderate flow it is very accurate; in extremely dense crowds (many people on the
line at once) accuracy degrades -- see analytics.CrowdAnalytics for density.
"""

import cv2


def _side(line, point):
    """Sign of which side of the line the point is on (cross product)."""
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


class LineCrossingCounter:
    def __init__(self, line=None):
        # line = ((x1, y1), (x2, y2)); set later from frame if None
        self.line = line
        self.prev_side = {}      # track_id -> last side sign
        self.counted_ids = set()  # ids already counted (avoid double count)
        self.count_a2b = 0       # crossings in one direction
        self.count_b2a = 0       # crossings in the other direction
        self.events = []         # list of dicts: {id, direction, time, frame}

    @property
    def total(self):
        return self.count_a2b + self.count_b2a

    def set_default_line(self, width, height):
        """Default horizontal tripwire at 60% of the frame height."""
        if self.line is None:
            y = int(height * 0.6)
            self.line = ((0, y), (width, y))

    def update(self, tracks, frame_idx, timestamp):
        """tracks: list of (track_id, x1, y1, x2, y2). Returns new events this frame."""
        if self.line is None:
            return []
        new_events = []
        for track_id, x1, y1, x2, y2 in tracks:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            s = _side(self.line, (cx, cy))
            sign = 1 if s > 0 else (-1 if s < 0 else 0)

            prev = self.prev_side.get(track_id)
            self.prev_side[track_id] = sign

            if prev is None or sign == 0 or prev == sign:
                continue
            if track_id in self.counted_ids:
                continue

            # Crossed the line
            self.counted_ids.add(track_id)
            direction = "A->B" if sign > prev else "B->A"
            if direction == "A->B":
                self.count_a2b += 1
            else:
                self.count_b2a += 1
            ev = {
                "id": int(track_id),
                "direction": direction,
                "time": timestamp,
                "frame": frame_idx,
            }
            self.events.append(ev)
            new_events.append(ev)
        return new_events

    def reset(self):
        """Reset counts (used when a new video segment / report window starts)."""
        self.prev_side.clear()
        self.counted_ids.clear()
        self.count_a2b = 0
        self.count_b2a = 0
        self.events = []

    def draw(self, frame):
        """Draw just a clean counting line. Counts are shown in the side panel."""
        if self.line is None:
            return
        (x1, y1), (x2, y2) = self.line
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.line(frame, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, p1, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p2, 5, (0, 0, 255), -1, cv2.LINE_AA)

"""
NeuralStream Vision - Monitoring System
=======================================

Main entry point. Ties together:
  - camera_source : Webcam / HDMI capture / Video folder
  - person_counter: YOLO + ByteTrack -> red-line crossing count
  - analytics     : crowd count, density heatmap, loitering/running, overcrowding
  - watchlist     : wanted-person face alerts (screen + log + beep + snapshot)
  - reporter      : interval reports (live) or per-video reports

Run from the project root:
    python src/vision/monitor.py

Controls (in the window):
    - Left-click + drag : draw the red counting line
    - d                 : toggle density heatmap
    - ESC               : quit

IMPORTANT (do not reorder): ultralytics (torch) MUST be imported before any
TensorFlow-backed library, otherwise the process segfaults on Linux.
"""

from ultralytics import YOLO  # keep first

import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# Force X11 (xcb) backend so the window can go truly fullscreen on Wayland/GNOME
# (Wayland fullscreen via OpenCV HighGUI is unreliable). Must be set before cv2.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

# Minimum YOLO confidence for a detection to count as a person. Raising this
# removes spurious "ghost" detections (e.g. counting 4-5 people when alone).
PERSON_CONF = 0.5
PANEL_W = 360  # width of the live side dashboard

# Make sibling modules importable and resolve data paths from the project root,
# so the app works no matter which directory you launch it from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from camera_source import select_source, FrameSource          # noqa: E402
from person_counter import LineCrossingCounter                # noqa: E402
from analytics import CrowdAnalytics, DensityEstimator        # noqa: E402
from watchlist import Watchlist                               # noqa: E402
from reporter import Reporter, ask_report_config              # noqa: E402

DATA = PROJECT_ROOT / "data"
WATCH_DIR = str(DATA / "watchlist")
REPORT_DIR = str(DATA / "reports")
ALERT_DIR = str(DATA / "alerts")

PERSON_CLASS = 0  # COCO class id for "person"


def beep():
    """Non-blocking-ish alert sound, best effort across platforms."""
    try:
        if os.name == "nt":
            import winsound
            winsound.Beep(1000, 300)
        else:
            # terminal bell; works in most Linux terminals
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


class LineDrawer:
    """Lets the user draw the counting line by dragging the mouse.

    The footage is stretched to fill the screen, so map mouse coords back to
    original frame coordinates using the current x/y scale.
    """
    def __init__(self, counter):
        self.counter = counter
        self.start = None
        self.dragging = False
        self.scale_x = 1.0
        self.scale_y = 1.0

    def _to_frame(self, x, y):
        return (int(x / self.scale_x), int(y / self.scale_y))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = self._to_frame(x, y)
            self.dragging = True
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.counter.line = (self.start, self._to_frame(x, y))
            self.counter.prev_side.clear()
            self.counter.counted_ids.clear()
            self.dragging = False
            print(f"[INFO] Counting line set: {self.counter.line}")


def load_model():
    for name in ("yolov8s.pt", "yolov8n.pt"):
        try:
            m = YOLO(name)
            print(f"[INFO] Loaded {name}")
            return m
        except Exception:
            continue
    raise RuntimeError("Could not load any YOLO model")


def draw_overlay(frame, stats):
    """Draw a clean translucent stats box on top of the full-screen footage."""
    F = cv2.FONT_HERSHEY_SIMPLEX
    bw, bh = 300, 190
    x0, y0 = 12, 12
    roi = frame[y0:y0 + bh, x0:x0 + bw]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)  # translucent dark panel
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), 1)

    level_color = {"LOW": (0, 255, 0), "MEDIUM": (0, 255, 255),
                   "HIGH": (0, 165, 255), "CRITICAL": (0, 0, 255)}.get(
                       stats['level'], (200, 200, 200))

    cv2.putText(frame, "LIVE MONITOR", (x0 + 14, y0 + 32), F, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Persons : {stats['persons']}", (x0 + 14, y0 + 68), F, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Crowd   : {stats['level']}", (x0 + 14, y0 + 100), F, 0.65, level_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Crossed : {stats['crossed']}", (x0 + 14, y0 + 132), F, 0.65, (60, 120, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Alerts  : {stats['alerts']}", (x0 + 14, y0 + 162), F, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {stats['fps']:.0f}", (x0 + bw - 80, y0 + 30), F, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if stats['overcrowded']:
        cv2.putText(frame, "!! OVERCROWDING !!", (x0 + 14, y0 + bh + 30),
                    F, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    print("=" * 60)
    print(" NeuralStream Vision - Monitoring System")
    print("=" * 60)

    source = select_source()
    enabled, interval, out_dir = ask_report_config(REPORT_DIR)

    mode = Reporter.MODE_LIVE if source.is_live else Reporter.MODE_VIDEO
    reporter = Reporter(mode, out_dir, interval_minutes=interval, enabled=enabled)
    if enabled:
        print(f"[INFO] Reports will be saved to: {out_dir}")

    model = load_model()
    counter = LineCrossingCounter()
    crowd = CrowdAnalytics(overcrowd_threshold=50)
    density_est = DensityEstimator()
    watch = Watchlist(WATCH_DIR)
    os.makedirs(ALERT_DIR, exist_ok=True)

    window = "NeuralStream Monitoring"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_w, screen_h = get_screen_size()
    print(f"[INFO] Display size: {screen_w}x{screen_h}")
    drawer = LineDrawer(counter)
    cv2.setMouseCallback(window, drawer.on_mouse)

    show_heatmap = False
    frame_idx = 0
    current_label = None
    active_alerts = []           # recent (name, expiry_time) for the banner
    recent_activity = deque(maxlen=6)  # (text, expiry) for the side panel
    alerts_total = 0
    fps = 0.0
    fps_t0 = time.time()
    fps_n = 0

    print("\n[INFO] Started. Drag mouse to set the red line. 'd'=heatmap, ESC=quit.\n")

    for frame, label, new_segment in source.frames():
        now = time.time()
        timestamp = datetime.now().isoformat(timespec="seconds")
        h, w = frame.shape[:2]
        counter.set_default_line(w, h)

        # Per-video report boundary
        if new_segment and mode == Reporter.MODE_VIDEO:
            if current_label is not None:
                reporter.flush()          # finish previous video's report
            counter.reset()
            reporter.set_label(label)
        if current_label is None:
            reporter.set_label(label)
        current_label = label

        # --- detection + tracking (persistent IDs) ---
        results = model.track(frame, persist=True, classes=[PERSON_CLASS],
                              conf=PERSON_CONF, iou=0.5,
                              tracker="bytetrack.yaml", verbose=False)
        tracks = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box, tid in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = box[:4]
                tracks.append((int(tid), float(x1), float(y1), float(x2), float(y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"#{int(tid)}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- line crossing ---
        for ev in counter.update(tracks, frame_idx, timestamp):
            reporter.log_crossing(ev)

        # --- crowd analytics ---
        info = crowd.update(tracks)
        reporter.update_peak(info["count"])
        for ab in info["abnormal"]:
            reporter.log_abnormal(ab, timestamp)
            if ab["type"] == "overcrowding":
                recent_activity.append((f"Overcrowding ({ab['count']})", now + 4))
            else:
                recent_activity.append((f"Person #{ab['id']}: {ab['type']}", now + 4))
        if show_heatmap:
            frame = crowd.density_overlay(frame, info["centers"])

        # --- watchlist (sample every 5th frame for speed) ---
        if frame_idx % 5 == 0:
            for m in watch.check(frame, now):
                snap = os.path.join(
                    ALERT_DIR,
                    f"alert_{m['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(snap, frame)
                reporter.log_alert(m["name"], timestamp, snapshot=snap)
                active_alerts.append((m["name"], now + 3.0))
                recent_activity.append((f"WANTED: {m['name']}", now + 5))
                alerts_total += 1
                beep()
                print(f"[ALERT] Watchlist match: {m['name']} -> {snap}")
                bx = m["box"]
                cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 3)

        # --- draw clean line + boxes on the feed; stats go to the side panel ---
        counter.draw(frame)

        # active watchlist banner (kept on the feed, it's important)
        active_alerts = [(n, t) for (n, t) in active_alerts if t > now]
        if active_alerts:
            names = ", ".join(n for n, _ in active_alerts)
            cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 255), -1)
            cv2.putText(frame, f"WANTED MATCH: {names}", (20, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS
        fps_n += 1
        if fps_n >= 15:
            fps = fps_n / (time.time() - fps_t0)
            fps_t0 = time.time()
            fps_n = 0

        # --- clean overlay on the full-screen footage ---
        stats = {
            "fps": fps, "persons": info["count"], "level": info["level"],
            "overcrowded": info["overcrowded"], "crossed": counter.total,
            "alerts": alerts_total,
        }
        draw_overlay(frame, stats)

        # stretch footage to fill the entire screen; keep mouse->line mapping
        disp = cv2.resize(frame, (screen_w, screen_h))
        drawer.scale_x = screen_w / w
        drawer.scale_y = screen_h / h

        # live interval report
        reporter.maybe_flush_live()

        cv2.imshow(window, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:        # ESC
            break
        if key == ord("d"):
            show_heatmap = not show_heatmap
        frame_idx += 1

    reporter.flush()  # final report
    cv2.destroyAllWindows()
    print("[INFO] Monitoring stopped.")


if __name__ == "__main__":
    main()

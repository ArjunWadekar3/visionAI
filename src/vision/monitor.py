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
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# Force X11 (xcb) backend so the window can go truly fullscreen on Wayland/GNOME
# (Wayland fullscreen via OpenCV HighGUI is unreliable). Must be set before cv2.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

# --- Detection config (all overridable by environment variables) ---
#
# Default COCO YOLO is fine for normal front/side webcam/CCTV footage.
# For DRONE / OVERHEAD footage (heads seen from above) COCO fails -- those angles
# aren't in its training data. Use a head-detection or aerial-trained model:
#     export NSA_MODEL_PATH=/path/to/head_or_visdrone_model.pt
#     export NSA_CONF=0.30        # smaller heads from height need a lower threshold
#     export NSA_CLASSES=0        # class id(s) to count (head model: usually 0)
#
MODEL_PATH = os.environ.get("NSA_MODEL_PATH", "")
PERSON_CONF = float(os.environ.get("NSA_CONF", "0.15"))   # low = catch more heads, geometry filter removes noise
_cls_env = os.environ.get("NSA_CLASSES", "0")
DETECT_CLASSES = [int(c) for c in _cls_env.split(",") if c.strip().lstrip("-").isdigit()]

# SAHI tiled inference -- big boost for small/dense/aerial heads. Slower per
# frame (many tiles), so off by default. Enable: export NSA_SAHI=1
USE_SAHI = os.environ.get("NSA_SAHI", "0") == "1"
USE_TILED = os.environ.get("NSA_TILED", "0") == "1"   # batched tiling (CPU-friendly, same detection)
SLICE_SIZE = int(os.environ.get("NSA_SLICE", "640"))     # larger tiles = faster (fewer tiles)
IMGSZ = int(os.environ.get("NSA_IMGSZ", "1024"))         # whole-frame inference size (reduced for speed)
OVERCROWD = int(os.environ.get("NSA_OVERCROWD", "200"))  # crowd-level threshold
DETECT_EVERY = max(1, int(os.environ.get("NSA_DETECT_EVERY", "1")))  # run detection every Nth frame (speed)
SHOW_IDS = os.environ.get("NSA_SHOW_IDS", "1") == "1"    # draw unique track id on each box
USE_DEDUP = os.environ.get("NSA_DEDUP", "0") == "1"      # motion-compensated unique count (panning drone)
MAX_DET = int(os.environ.get("NSA_MAX_DET", "2000"))     # raise YOLO's per-image 300 cap for crowds

# Make sibling modules importable and resolve data paths from the project root,
# so the app works no matter which directory you launch it from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from camera_source import select_source, FrameSource          # noqa: E402
from crowd_counter import CrowdCounter                        # noqa: E402
from dedup_counter import DedupCounter                        # noqa: E402
from analytics import CrowdAnalytics                          # noqa: E402
from reporter import Reporter, ask_report_config              # noqa: E402

DATA = PROJECT_ROOT / "data"
WATCH_DIR = str(DATA / "watchlist")
REPORT_DIR = str(DATA / "reports")
ALERT_DIR = str(DATA / "alerts")



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


def load_model():
    # Custom model (head / aerial / VisDrone) takes priority for drone footage.
    if MODEL_PATH:
        if os.path.exists(MODEL_PATH):
            print(f"[INFO] Loaded custom model: {MODEL_PATH} "
                  f"(conf={PERSON_CONF}, classes={DETECT_CLASSES})")
            return YOLO(MODEL_PATH)
        print(f"[WARN] NSA_MODEL_PATH set but not found: {MODEL_PATH} "
              f"-- falling back to default COCO model.")
    for name in ("yolov8s.pt", "yolov8n.pt"):
        try:
            m = YOLO(name)
            print(f"[INFO] Loaded {name} (default COCO -- not ideal for overhead/drone)")
            return m
        except Exception:
            continue
    raise RuntimeError("Could not load any YOLO model")


class DetectorThread:
    """Runs full-quality detection in the background so the live display stays
    smooth. Detection is NOT reduced -- it keeps processing the newest frame as
    fast as the CPU allows; the main loop just overlays the latest result."""

    def __init__(self, counter):
        self.counter = counter
        self._frame = None
        self._result = (0, 0, [], [], [])  # count, unique_total, boxes, ids, centers
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._result

    def _run(self):
        while self._running:
            with self._lock:
                f = self._frame
                self._frame = None
            if f is None:
                time.sleep(0.005)
                continue
            count, unique_total, boxes, ids, centers = self.counter.process(f)
            with self._lock:
                self._result = (count, unique_total, boxes, ids, centers)

    def stop(self):
        self._running = False


def draw_overlay(frame, stats):
    """Draw a clean translucent stats box on top of the full-screen footage."""
    F = cv2.FONT_HERSHEY_SIMPLEX
    bw, bh = 300, 168
    x0, y0 = 12, 12
    roi = frame[y0:y0 + bh, x0:x0 + bw]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)  # translucent dark panel
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), 1)

    level_color = {"LOW": (0, 255, 0), "MEDIUM": (0, 255, 255),
                   "HIGH": (0, 165, 255), "CRITICAL": (0, 0, 255)}.get(
                       stats['level'], (200, 200, 200))

    cv2.putText(frame, "CROWD MONITOR", (x0 + 14, y0 + 30), F, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"PEOPLE : {stats['persons']}", (x0 + 14, y0 + 78), F, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"TOTAL  : {stats['unique']}", (x0 + 14, y0 + 116), F, 0.8, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Crowd : {stats['level']}", (x0 + 14, y0 + 148), F, 0.6, level_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {stats['fps']:.0f}", (x0 + bw - 80, y0 + 28), F, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

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
    counter = CrowdCounter(model, MODEL_PATH or "yolov8s.pt", conf=PERSON_CONF,
                           classes=DETECT_CLASSES, use_sahi=USE_SAHI,
                           use_tiled=USE_TILED, slice_size=SLICE_SIZE, imgsz=IMGSZ,
                           track=True, max_det=MAX_DET)
    crowd = CrowdAnalytics(overcrowd_threshold=OVERCROWD)
    os.makedirs(ALERT_DIR, exist_ok=True)

    window = "NeuralStream Monitoring"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_w, screen_h = get_screen_size()
    print(f"[INFO] Display size: {screen_w}x{screen_h}")

    show_heatmap = False
    frame_idx = 0
    current_label = None
    peak_count = 0
    # De-dup needs sequential frames + their own detections, so it runs inline
    # (no detector thread). Otherwise live uses a background detection thread.
    dedup = DedupCounter() if USE_DEDUP else None
    detector = DetectorThread(counter) if (source.is_live and not USE_DEDUP) else None
    if detector is not None:
        detector.start()
        print("[INFO] Live mode: detection running in background thread.")
    if dedup is not None:
        print("[INFO] De-dup ON: unique count via camera-motion compensation.")
    fps = 0.0
    fps_t0 = time.time()
    fps_n = 0

    print("\n[INFO] Started. 'd'=density heatmap, ESC=quit.\n")

    for frame, label, new_segment in source.frames():
        now = time.time()
        timestamp = datetime.now().isoformat(timespec="seconds")
        h, w = frame.shape[:2]

        # Per-video report boundary
        if new_segment and mode == Reporter.MODE_VIDEO:
            if current_label is not None:
                reporter.flush()          # finish previous video's report
            peak_count = 0
            reporter.set_label(label)
        if current_label is None:
            reporter.set_label(label)
        current_label = label

        # --- crowd counting + unique-ID tracking ---
        # Live: detection runs in a background thread (smooth display, full
        # detection quality). Video file: detect every frame for accuracy.
        if detector is not None:
            detector.submit(frame)
            count, unique_total, boxes, ids, centers = detector.get()
        else:
            count, unique_total, boxes, ids, centers = counter.process(frame)
        if dedup is not None:
            unique_total, ids = dedup.update(frame, centers)
        peak_count = max(peak_count, count)
        reporter.update_peak(count)
        reporter.update_unique(unique_total)
        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if SHOW_IDS:
                cv2.putText(frame, str(tid), (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        level = crowd._crowd_level(count)
        overcrowded = count >= OVERCROWD
        if overcrowded:
            reporter.log_abnormal({"type": "overcrowding", "count": count}, timestamp)
        if show_heatmap:
            frame = crowd.density_overlay(frame, centers)

        # FPS
        fps_n += 1
        if fps_n >= 15:
            fps = fps_n / (time.time() - fps_t0)
            fps_t0 = time.time()
            fps_n = 0

        # --- clean overlay on the full-screen footage ---
        stats = {
            "fps": fps, "persons": count, "unique": unique_total,
            "level": level, "overcrowded": overcrowded,
        }
        draw_overlay(frame, stats)

        disp = cv2.resize(frame, (screen_w, screen_h))

        # live interval report
        reporter.maybe_flush_live()

        cv2.imshow(window, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:        # ESC
            break
        if key == ord("d"):
            show_heatmap = not show_heatmap
        frame_idx += 1

    if detector is not None:
        detector.stop()
    reporter.flush()  # final report
    cv2.destroyAllWindows()
    print("[INFO] Monitoring stopped.")


if __name__ == "__main__":
    main()

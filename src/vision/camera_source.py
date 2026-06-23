"""
Camera / video source selection for the monitoring system.

Supports three input modes:
  1) Webcam        - local laptop camera (device index 0)
  2) HDMI capture  - an HDMI->USB capture card shows up as another video
                     device; you just pick its index (usually 1, 2, ...).
  3) Video folder  - point at a folder; every video file inside is processed
                     one by one (used for the "upload" workflow + per-video reports).

Usage:
    source = select_source()
    for frame, label, new_segment in source.frames():
        ...
"""

import os
import glob
import cv2

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg")


def list_video_devices(max_index=6):
    """Probe device indices 0..max_index and return the ones that open."""
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available.append(i)
            cap.release()
    return available


class FrameSource:
    """Unified iterator over webcam / HDMI / a folder of video files."""

    MODE_WEBCAM = "webcam"
    MODE_HDMI = "hdmi"
    MODE_FOLDER = "folder"

    def __init__(self, mode, device_index=0, video_dir=None,
                 width=1280, height=720):
        self.mode = mode
        self.device_index = device_index
        self.video_dir = video_dir
        self.width = width
        self.height = height
        self.is_live = mode in (self.MODE_WEBCAM, self.MODE_HDMI)

        self._video_files = []
        if mode == self.MODE_FOLDER:
            self._video_files = self._collect_videos(video_dir)
            if not self._video_files:
                raise FileNotFoundError(
                    f"No video files found in: {video_dir}")

    @staticmethod
    def _collect_videos(folder):
        files = []
        for ext in VIDEO_EXTS:
            files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
        return sorted(set(files))

    def _open_capture(self, target):
        cap = cv2.VideoCapture(target)
        if self.is_live:
            # MJPG only for HDMI capture cards (they often default to a raw
            # format that renders washed-out). Do NOT force it on the laptop
            # webcam -- that can break its colour output.
            if self.mode == self.MODE_HDMI:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def frames(self):
        """Yield (frame, label, new_segment).

        label        - human readable source name (e.g. "Webcam" or video filename)
        new_segment  - True on the first frame of a stream / each new video file.
                       The monitor uses this to start a fresh per-video report.
        """
        if self.is_live:
            label = "Webcam" if self.mode == self.MODE_WEBCAM else \
                f"HDMI (device {self.device_index})"
            cap = self._open_capture(self.device_index)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open {label}. Check the connection / index.")
            first = True
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame, label, first
                first = False
            cap.release()
        else:
            for path in self._video_files:
                label = os.path.basename(path)
                cap = self._open_capture(path)
                if not cap.isOpened():
                    print(f"[WARN] Skipping unreadable video: {label}")
                    continue
                print(f"[INFO] Processing video: {label}")
                first = True
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield frame, label, first
                    first = False
                cap.release()


def select_source():
    """Interactive menu -> configured FrameSource."""
    print("\n=== Select Input Source ===")
    print("  1 - Webcam (laptop camera)")
    print("  2 - HDMI capture device")
    print("  3 - Video folder (process files one by one)")
    choice = input("Enter 1 / 2 / 3: ").strip()

    if choice == "2":
        devices = list_video_devices()
        if devices:
            print(f"[INFO] Detected video device indices: {devices}")
        idx = input("Enter HDMI capture device index (e.g. 1 or 2): ").strip()
        idx = int(idx) if idx.isdigit() else 1
        return FrameSource(FrameSource.MODE_HDMI, device_index=idx)

    if choice == "3":
        folder = input("Enter the folder path containing videos: ").strip()
        folder = os.path.expanduser(folder.strip('"').strip("'"))
        return FrameSource(FrameSource.MODE_FOLDER, video_dir=folder)

    # default
    return FrameSource(FrameSource.MODE_WEBCAM, device_index=0)

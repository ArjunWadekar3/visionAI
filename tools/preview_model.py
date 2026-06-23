"""
Quick model previewer -- check whether a YOLO model actually detects people in
your footage BEFORE wiring it into the monitor.

Usage:
    python tools/preview_model.py <model.pt> <video_or_image> [conf]

Example:
    python tools/preview_model.py ~/models/head.pt ~/test_videos/drone1.mp4 0.30

Prints the model's class names and the per-frame detection count, and shows the
boxes in a window (ESC to quit). If counts stay 0 on overhead footage, that
model is not suitable -- try another head/aerial model.
"""

import sys

from ultralytics import YOLO
import cv2


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    model_path = sys.argv[1]
    source = sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.30

    model = YOLO(model_path)
    print(f"[INFO] Model classes: {model.names}")
    print(f"[INFO] Running at conf={conf}. Press ESC to quit.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {source}")
        sys.exit(1)

    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, conf=conf, verbose=False)
        n = 0
        if res and res[0].boxes is not None:
            n = len(res[0].boxes)
            frame = res[0].plot()  # draw all detections
        cv2.putText(frame, f"Detections: {n}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if frame_idx % 30 == 0:
            print(f"frame {frame_idx}: {n} detections")
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

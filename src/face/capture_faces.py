import cv2
import os

def capture_images(label, count=50, save_dir='data/faces'):
    os.makedirs(os.path.join(save_dir, label), exist_ok=True)
    cam = cv2.VideoCapture(0)

    print(f"[INFO] Capturing {count} images for label: {label}")
    captured = 0
    while captured < count:
        ret, frame = cam.read()
        if not ret:
            continue

        cv2.imshow("Capture - Press SPACE to save, ESC to exit", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC
            break
        elif key % 256 == 32:  # SPACE
            img_path = os.path.join(save_dir, label, f"{label}_{captured}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[+] Saved: {img_path}")
            captured += 1

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images("nonface")  # Replace with "nonface" for negatives

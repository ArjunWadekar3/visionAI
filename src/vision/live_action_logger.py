import cv2
import mediapipe as mp
import math
import face_recognition
import os
from deepface import DeepFace
from ultralytics import YOLO
import ctypes

# Init MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Helpers
def hand_side(landmarks):
    wrist_x = landmarks.landmark[0].x
    thumb_x = landmarks.landmark[4].x
    return "Right" if thumb_x < wrist_x else "Left"

def count_fingers(hand_landmarks, hand_label):
    # Tip landmarks for fingers
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    thumb_tip = 4

    fingers = []

    # Thumb (depends on hand side)
    if hand_label == "Right":
        fingers.append(hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[3].x)
    else:
        fingers.append(hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[3].x)

    # Other fingers
    for tip in finger_tips:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

    return fingers.count(True)

def face_direction(landmarks):
    nose = landmarks.landmark[1].x
    left_cheek = landmarks.landmark[234].x
    right_cheek = landmarks.landmark[454].x
    if nose < left_cheek:
        return "Looking Right"
    elif nose > right_cheek:
        return "Looking Left"
    else:
        return "Looking Forward"

# Load known faces
known_face_encodings = []
known_face_names = []
faces_dir = "data/faces/face/"
for filename in os.listdir(faces_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image = face_recognition.load_image_file(os.path.join(faces_dir, filename))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(filename)

# Load YOLOv8 model for object detection
object_model = YOLO('yolov8n.pt')  # Use the nano model for speed; you can use yolov8s.pt for more accuracy

# Spatial Understanding Helpers
def calculate_distance(object_width_pixels, known_width_cm=10):
    """Estimate distance to object using known width and pixel width"""
    # Focal length approximation (adjust based on your camera)
    focal_length = 1000  # pixels
    distance_cm = (known_width_cm * focal_length) / object_width_pixels
    return distance_cm

def get_spatial_position(x1, y1, x2, y2, frame_width, frame_height):
    """Determine spatial position of object in frame"""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Horizontal position
    if center_x < frame_width * 0.33:
        h_pos = "Left"
    elif center_x > frame_width * 0.67:
        h_pos = "Right"
    else:
        h_pos = "Center"
    
    # Vertical position
    if center_y < frame_height * 0.33:
        v_pos = "Top"
    elif center_y > frame_height * 0.67:
        v_pos = "Bottom"
    else:
        v_pos = "Middle"
    
    return f"{h_pos}-{v_pos}"

def check_spatial_relationship(obj1_coords, obj2_coords, threshold=100):
    """Check if two objects are spatially near each other"""
    x1, y1, x2, y2 = obj1_coords
    x3, y3, x4, y4 = obj2_coords
    
    center1_x, center1_y = (x1 + x2) / 2, (y1 + y2) / 2
    center2_x, center2_y = (x3 + x4) / 2, (y3 + y4) / 2
    
    distance = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance < threshold, distance

def get_object_size_category(width, height):
    """Categorize object size for distance estimation"""
    area = width * height
    if area < 1000:
        return "small", 5  # 5cm
    elif area < 5000:
        return "medium", 15  # 15cm
    else:
        return "large", 30  # 30cm

# Known object sizes for distance estimation (in cm)
OBJECT_SIZES = {
    'person': 50,  # Average person width
    'cell phone': 7,  # Phone width
    'laptop': 35,  # Laptop width
    'cup': 8,  # Cup diameter
    'bottle': 7,  # Bottle width
    'book': 15,  # Book width
    'chair': 45,  # Chair width
    'table': 80,  # Table width
    'car': 180,  # Car width
    'bicycle': 70,  # Bicycle width
}

# Ask user for camera choice
print("Select Camera Source:")
print("1 - Laptop Webcam")
print("2 - DroidCam (Mobile Back Camera)")
choice = input("Enter 1 or 2: ")

# Determine camera index
if choice == "2":
    test_cap = cv2.VideoCapture(1)  # Try DroidCam (usually index 1)
    if test_cap.isOpened():
        cap = test_cap
        print("[INFO] Using DroidCam (back camera).")
    else:
        print("[WARN] DroidCam not detected, falling back to laptop webcam.")
        cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(0)
    print("[INFO] Using Laptop Webcam.")

# Get screen resolution for full screen display
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
     mp_face.FaceMesh(min_detection_confidence=0.5) as face_mesh:

    print("[INFO] Starting VisionGuard AI Action Logger... Press ESC to quit")

    frame_count = 0
    last_face_results = []  # Store last detected faces' info
    object_tracks = {}  # Store object positions for movement tracking
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]

        hand_results = hands.process(rgb)
        face_results = face_mesh.process(rgb)

        # Process hands
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                side = hand_side(hand_landmarks)
                fingers_up = count_fingers(hand_landmarks, side)

                # Basic gesture recognition
                if fingers_up == 0:
                    gesture = "Fist"
                elif fingers_up == 1:
                    gesture = "One"
                elif fingers_up == 2:
                    gesture = "Peace ✌️"
                elif fingers_up == 5:
                    gesture = "Open Palm 🖐️"
                else:
                    gesture = f"{fingers_up} fingers"

                print(f"✋ {side} hand detected — {gesture}")

        # Process face direction every 5th frame
        if frame_count % 5 == 0:
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    #mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_TESSELATION)
                    direction = face_direction(face_landmarks)
                    print(f"\U0001F464 Face direction: {direction}")

        # --- Improved Face Recognition and Emotion Detection ---
        frame_count += 1
        if frame_count % 2 == 0:  # Run every 2nd frame for higher reliability
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            new_face_results = []
            if face_locations:
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    matched_names = [name for match, name in zip(matches, known_face_names) if match]
                    # Scale coordinates back to original frame size
                    scale = 4
                    t, r, b, l = top*scale, right*scale, bottom*scale, left*scale
                    label = ', '.join(matched_names) if matched_names else 'Unknown'
                    emotion = None
                    try:
                        face_img = frame[t:b, l:r]
                        if face_img.size > 0:
                            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                            if isinstance(result, list):
                                result = result[0]
                            emotion = result['dominant_emotion']
                    except Exception as e:
                        print(f"[DeepFace] Emotion detection error: {e}")
                    # Print recognized file name and emotion together in console
                    print(f"Recognized Face: {label} | Emotion: {emotion if emotion else 'N/A'}")
                    new_face_results.append({'coords': (l, t, r, b), 'name': label, 'emotion': emotion})
                last_face_results = new_face_results
            else:
                # No face detected: show Unknown
                last_face_results = [{'coords': (40, 40, 200, 200), 'name': 'Unknown', 'emotion': 'N/A'}]
                print("Recognized Face: Unknown | Emotion: N/A")

        # --- Draw last detected faces' info on every frame ---
        for face in last_face_results:
            l, t, r, b = face['coords']
            is_known = face['name'] != 'Unknown'
            color = (0, 255, 0) if is_known else (0, 0, 255)  # Green for known, red for unknown
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, face['name'], (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, face['emotion'] if face['emotion'] else 'N/A', (l, t-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Object detection with YOLOv8 and Spatial Understanding
        person_count = 0
        detected_objects = []
        results = object_model(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = object_model.model.names[cls]
                
                # Spatial Analysis
                object_width = x2 - x1
                object_height = y2 - y1
                
                # Distance estimation
                known_width = OBJECT_SIZES.get(label, 10)  # Default 10cm if unknown
                distance_cm = calculate_distance(object_width, known_width)
                
                # Spatial position
                position = get_spatial_position(x1, y1, x2, y2, frame_width, frame_height)
                
                # Movement tracking
                object_id = f"{label}_{x1}_{y1}"
                if object_id in object_tracks:
                    prev_x, prev_y = object_tracks[object_id]
                    movement = math.sqrt((x1 - prev_x)**2 + (y1 - prev_y)**2)
                    if movement > 10:  # Significant movement threshold
                        movement_direction = "Moving"
                    else:
                        movement_direction = "Static"
                else:
                    movement_direction = "New"
                
                object_tracks[object_id] = (x1, y1)
                
                # Store object info for relationship analysis
                detected_objects.append({
                    'label': label,
                    'coords': (x1, y1, x2, y2),
                    'distance': distance_cm,
                    'position': position,
                    'movement': movement_direction
                })
                
                # Draw enhanced bounding box with spatial info
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Display spatial information on frame
                distance_text = f"{distance_cm:.1f}cm"
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Dist: {distance_text}", (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Pos: {position}", (x1, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Move: {movement_direction}", (x1, y1+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if label == 'person':
                    person_count += 1
                
                # Print spatial information in console
                print(f"Object: {label} | Distance: {distance_cm:.1f}cm | Position: {position} | Movement: {movement_direction}")

        # Spatial Relationship Analysis
        for i, obj1 in enumerate(detected_objects):
            for j, obj2 in enumerate(detected_objects[i+1:], i+1):
                is_near, distance = check_spatial_relationship(obj1['coords'], obj2['coords'])
                if is_near:
                    relationship_text = f"{obj1['label']} near {obj2['label']}"
                    cv2.putText(frame, relationship_text, (20, 80 + len(detected_objects)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    print(f"Spatial Relationship: {relationship_text} (Distance: {distance:.1f}px)")

        # Display enhanced person count and spatial summary
        cv2.putText(frame, f'Person count: {person_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f'Objects detected: {len(detected_objects)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show output
        cv2.imshow("NeuralStream Vision", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import math
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import ctypes
from collections import defaultdict, deque
import time
import json

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Color schemes for better visualization
COLORS = {
    'person': (0, 255, 0),      # Green
    'vehicle': (255, 0, 0),     # Blue
    'furniture': (0, 255, 255), # Yellow
    'electronics': (255, 0, 255), # Magenta
    'unknown': (128, 128, 128), # Gray  
    'known_face': (0, 255, 0),  # Green
    'unknown_face': (0, 0, 255), # Red
    'warning': (0, 165, 255),   # Orange
    'alert': (0, 0, 255)        # Red
}

# Object categories for color coding
OBJECT_CATEGORIES = {
    'person': 'person',
    'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle', 'bicycle': 'vehicle',
    'chair': 'furniture', 'couch': 'furniture', 'bed': 'furniture', 'dining table': 'furniture',
    'laptop': 'electronics', 'cell phone': 'electronics', 'tv': 'electronics', 'keyboard': 'electronics',
    'mouse': 'electronics', 'remote': 'electronics'
}

class EmotionSmoother:
    """Smooth emotion detection across frames to avoid jittery results"""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.emotion_history = defaultdict(lambda: deque(maxlen=window_size))
    
    def smooth_emotion(self, face_id, emotion):
        self.emotion_history[face_id].append(emotion)
        # Return most common emotion in the window
        emotions = list(self.emotion_history[face_id])
        return max(set(emotions), key=emotions.count) if emotions else emotion

class ActivityAnalyzer:
    """Analyze activities and behaviors from object positions and poses"""
    def __init__(self):
        self.activity_rules = {
            'sitting': self._detect_sitting,
            'using_phone': self._detect_phone_usage,
            'at_computer': self._detect_computer_work,
            'loitering': self._detect_loitering,
            'running': self._detect_running
        }
        self.person_positions = defaultdict(lambda: deque(maxlen=30))  # Track 30 frames
        self.last_activity_time = defaultdict(float)
    
    def _detect_sitting(self, person_coords, objects, pose_landmarks=None):
        """Detect if person is sitting based on nearby chairs and pose"""
        person_center = ((person_coords[0] + person_coords[2]) / 2, 
                        (person_coords[1] + person_coords[3]) / 2)
        
        # Check for nearby chairs
        for obj in objects:
            if obj['label'] in ['chair', 'couch', 'bed']:
                obj_center = ((obj['coords'][0] + obj['coords'][2]) / 2,
                             (obj['coords'][1] + obj['coords'][3]) / 2)
                distance = math.sqrt((person_center[0] - obj_center[0])**2 + 
                                   (person_center[1] - obj_center[1])**2)
                if distance < 100:  # Close to furniture
                    return True, f"Sitting on {obj['label']}"
        return False, ""
    
    def _detect_phone_usage(self, person_coords, objects):
        """Detect phone usage by proximity"""
        person_center = ((person_coords[0] + person_coords[2]) / 2, 
                        (person_coords[1] + person_coords[3]) / 2)
        
        for obj in objects:
            if obj['label'] == 'cell phone':
                obj_center = ((obj['coords'][0] + obj['coords'][2]) / 2,
                             (obj['coords'][1] + obj['coords'][3]) / 2)
                distance = math.sqrt((person_center[0] - obj_center[0])**2 + 
                                   (person_center[1] - obj_center[1])**2)
                if distance < 80:
                    return True, "Using phone"
        return False, ""
    
    def _detect_computer_work(self, person_coords, objects):
        """Detect computer work"""
        person_center = ((person_coords[0] + person_coords[2]) / 2, 
                        (person_coords[1] + person_coords[3]) / 2)
        
        for obj in objects:
            if obj['label'] in ['laptop', 'keyboard', 'mouse']:
                obj_center = ((obj['coords'][0] + obj['coords'][2]) / 2,
                             (obj['coords'][1] + obj['coords'][3]) / 2)
                distance = math.sqrt((person_center[0] - obj_center[0])**2 + 
                                   (person_center[1] - obj_center[1])**2)
                if distance < 100:
                    return True, f"Working on {obj['label']}"
        return False, ""
    
    def _detect_loitering(self, person_id, person_coords):
        """Detect if person is loitering (staying in same area)"""
        center = ((person_coords[0] + person_coords[2]) / 2, 
                 (person_coords[1] + person_coords[3]) / 2)
        self.person_positions[person_id].append(center)
        
        if len(self.person_positions[person_id]) >= 20:  # Check last 20 positions
            positions = list(self.person_positions[person_id])
            # Calculate variance in positions
            x_positions = [pos[0] for pos in positions]
            y_positions = [pos[1] for pos in positions]
            x_var = np.var(x_positions)
            y_var = np.var(y_positions)
            
            # If low variance, person is loitering
            if x_var < 500 and y_var < 500:
                return True, "Loitering detected"
        return False, ""
    
    def _detect_running(self, person_id, person_coords):
        """Detect rapid movement (running)"""
        center = ((person_coords[0] + person_coords[2]) / 2, 
                 (person_coords[1] + person_coords[3]) / 2)
        
        if person_id in self.person_positions and len(self.person_positions[person_id]) > 0:
            last_pos = self.person_positions[person_id][-1]
            distance = math.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
            if distance > 50:  # Large movement between frames
                return True, "Running/Fast movement"
        
        self.person_positions[person_id].append(center)
        return False, ""
    
    def analyze_person_activity(self, person_id, person_coords, objects, pose_landmarks=None):
        """Analyze all activities for a person"""
        activities = []
        
        for activity_name, detector in self.activity_rules.items():
            try:
                if activity_name in ['loitering', 'running']:
                    detected, description = detector(person_id, person_coords)
                else:
                    detected, description = detector(person_coords, objects, pose_landmarks)
                
                if detected:
                    activities.append(description)
                    self.last_activity_time[f"{person_id}_{activity_name}"] = time.time()
            except Exception as e:
                continue
        
        return activities

class EnhancedFaceRecognizer:
    """Enhanced face recognition using DeepFace embeddings"""
    def __init__(self, faces_dir="data/faces/face/"):
        self.known_embeddings = []
        self.known_names = []
        self.faces_dir = faces_dir
        self.load_known_faces()
        self.emotion_smoother = EmotionSmoother()
    
    def load_known_faces(self):
        """Load known faces using DeepFace for better accuracy"""
        import os
        if not os.path.exists(self.faces_dir):
            print(f"[WARN] Faces directory {self.faces_dir} not found")
            return
            
        for filename in os.listdir(self.faces_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    img_path = os.path.join(self.faces_dir, filename)
                    # Use DeepFace to get embeddings
                    embedding = DeepFace.represent(img_path, model_name='Facenet')[0]['embedding']
                    self.known_embeddings.append(embedding)
                    self.known_names.append(filename.split('.')[0])  # Remove extension
                    print(f"[INFO] Loaded face: {filename}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {filename}: {e}")
    
    def recognize_face(self, face_img, face_id):
        """Recognize face using DeepFace with improved accuracy"""
        try:
            if face_img.size == 0:
                return "Unknown", "neutral"
            
            # Get face embedding
            embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]['embedding']
            
            # Compare with known faces
            min_distance = float('inf')
            best_match = "Unknown"
            
            for known_embedding, name in zip(self.known_embeddings, self.known_names):
                # Calculate cosine similarity
                distance = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
                if distance < min_distance and distance < 0.6:  # Threshold for recognition
                    min_distance = distance
                    best_match = name
            
            # Get emotion with smoothing
            try:
                emotion_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                if isinstance(emotion_result, list):
                    emotion_result = emotion_result[0]
                raw_emotion = emotion_result['dominant_emotion']
                smoothed_emotion = self.emotion_smoother.smooth_emotion(face_id, raw_emotion)
            except:
                smoothed_emotion = "neutral"
            
            return best_match, smoothed_emotion
            
        except Exception as e:
            return "Unknown", "neutral"

def draw_enhanced_bbox(frame, coords, label, confidence, color, additional_info=None):
    """Draw enhanced bounding boxes with better styling"""
    x1, y1, x2, y2 = coords
    
    # Draw main rectangle with rounded corners effect
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    
    # Draw corner accents
    corner_length = 15
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 1)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 1)
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 1)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 1)
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 1)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 1)
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 1)
    
    # Label background
    label_text = f"{label} {confidence:.2f}" if confidence else label
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
    cv2.putText(frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Additional info
    if additional_info:
        y_offset = 10
        for info in additional_info:
            cv2.putText(frame, info, (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1)
            y_offset += 10

# Removed info panel function - fullscreen feed only

# Hand gesture helpers (improved)
def hand_side(landmarks):
    wrist_x = landmarks.landmark[0].x
    thumb_x = landmarks.landmark[4].x
    return "Right" if thumb_x < wrist_x else "Left"

def count_fingers(hand_landmarks, hand_label):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    fingers = []

    if hand_label == "Right":
        fingers.append(hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[3].x)
    else:
        fingers.append(hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[3].x)

    for tip in finger_tips:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

    return fingers.count(True)

def recognize_gesture(fingers_up, hand_landmarks):
    """Enhanced gesture recognition"""
    if fingers_up == 0:
        return "✊ Fist"
    elif fingers_up == 1:
        # Check which finger is up
        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
            return "👆 Pointing"
        else:
            return "☝️ One"
    elif fingers_up == 2:
        # Check for peace sign or other two-finger gestures
        index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
        middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
        if index_up and middle_up:
            return "✌️ Peace"
        else:
            return "✌️ Two"
    elif fingers_up == 5:
        return "🖐️ Open Palm"
    else:
        return f"🖐️ {fingers_up} fingers"

# Enhanced spatial analysis
def calculate_distance(object_width_pixels, known_width_cm=10):
    focal_length = 1000
    distance_cm = (known_width_cm * focal_length) / object_width_pixels
    return min(distance_cm, 1000)  # Cap at 10 meters

def get_spatial_position(x1, y1, x2, y2, frame_width, frame_height):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    if center_x < frame_width * 0.33:
        h_pos = "Left"
    elif center_x > frame_width * 0.67:
        h_pos = "Right"
    else:
        h_pos = "Center"
    
    if center_y < frame_height * 0.33:
        v_pos = "Top"
    elif center_y > frame_height * 0.67:
        v_pos = "Bottom"
    else:
        v_pos = "Middle"
    
    return f"{h_pos}-{v_pos}"

# Enhanced object sizes for better distance estimation
OBJECT_SIZES = {
    'person': 50, 'cell phone': 7, 'laptop': 35, 'cup': 8, 'bottle': 7,
    'book': 15, 'chair': 45, 'dining table': 80, 'car': 180, 'bicycle': 70,
    'tv': 100, 'keyboard': 30, 'mouse': 8, 'remote': 15, 'scissors': 12,
    'teddy bear': 25, 'hair drier': 20, 'toothbrush': 15
}

def main():
    # Initialize components
    face_recognizer = EnhancedFaceRecognizer()
    activity_analyzer = ActivityAnalyzer()
    
    # Camera setup with fullscreen configuration
    print("Select Camera Source:")
    print("1 - Laptop Webcam")
    print("2 - DroidCam (Mobile Back Camera)")
    choice = input("Enter 1 or 2: ")
    
    if choice == "2":
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[WARN] DroidCam not detected, using laptop webcam.")
            cap = cv2.VideoCapture(0)
        else:
            print("[INFO] Using DroidCam (back camera).")
    else:
        cap = cv2.VideoCapture(0)
        print("[INFO] Using Laptop Webcam.")
    
    # Set camera to high resolution for better fullscreen display
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get screen dimensions for fullscreen
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    
    # Load YOLOv8 model (use yolov8s.pt for better accuracy)
    try:
        object_model = YOLO('yolov8s.pt')  # Upgraded from yolov8n.pt
        print("[INFO] Loaded YOLOv8s model for enhanced accuracy")
    except:
        object_model = YOLO('yolov8n.pt')
        print("[INFO] Loaded YOLOv8n model (fallback)")
    
    # Initialize MediaPipe
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
         mp_face.FaceMesh(min_detection_confidence=0.5) as face_mesh, \
         mp_pose.Pose(min_detection_confidence=0.5) as pose:
        
        print("[INFO] Enhanced VisionGuard AI started... Press ESC to quit")
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]
            
            # FPS calculation
            fps_counter += 1
            if fps_counter % 30 == 0:
                current_fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Process hands with enhanced gestures
            hand_results = hands.process(rgb)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    side = hand_side(hand_landmarks)
                    fingers_up = count_fingers(hand_landmarks, side)
                    gesture = recognize_gesture(fingers_up, hand_landmarks)
                    
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    print(f"✋ {side} hand: {gesture}")
            
            # Process pose
            pose_results = pose.process(rgb)
            pose_landmarks = None
            if pose_results.pose_landmarks:
                pose_landmarks = pose_results.pose_landmarks
                # Draw pose landmarks (optional, can be disabled for cleaner look)
                # mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Enhanced object detection
            detected_objects = []
            person_count = 0
            results = object_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = object_model.model.names[cls]
                    
                    if conf < 0.5:  # Higher confidence threshold
                        continue
                    
                    # Enhanced spatial analysis
                    object_width = x2 - x1
                    object_height = y2 - y1
                    known_width = OBJECT_SIZES.get(label, 10)
                    distance_cm = calculate_distance(object_width, known_width)
                    position = get_spatial_position(x1, y1, x2, y2, frame_width, frame_height)
                    
                    # Color coding by category
                    category = OBJECT_CATEGORIES.get(label, 'unknown')
                    color = COLORS.get(category, COLORS['unknown'])
                    
                    # Additional info for display
                    additional_info = [
                        f"Dist: {distance_cm:.1f}cm",
                        f"Pos: {position}",
                        f"Size: {object_width}x{object_height}"
                    ]
                    
                    # Enhanced bounding box
                    draw_enhanced_bbox(frame, (x1, y1, x2, y2), label, conf, color, additional_info)
                    
                    detected_objects.append({
                        'label': label,
                        'coords': (x1, y1, x2, y2),
                        'distance': distance_cm,
                        'position': position,
                        'confidence': conf
                    })
                    
                    if label == 'person':
                        person_count += 1
                    
                    print(f"🎯 {label} | Conf: {conf:.2f} | Dist: {distance_cm:.1f}cm | Pos: {position}")
            
            # Enhanced face recognition
            face_results = []
            if frame_count % 3 == 0:  # Process every 3rd frame for balance of speed/accuracy
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
                    cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
                
                for i, (x, y, w, h) in enumerate(faces):
                    # Scale back to original frame
                    x, y, w, h = x*2, y*2, w*2, h*2
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size > 0:
                        name, emotion = face_recognizer.recognize_face(face_img, f"face_{i}")
                        
                        face_results.append({
                            'coords': (x, y, x+w, y+h),
                            'name': name,
                            'emotion': emotion
                        })
                        
                        print(f"👤 {name} | Emotion: {emotion}")
            
            # Draw face results
            for face in face_results:
                x1, y1, x2, y2 = face['coords']
                color = COLORS['known_face'] if face['name'] != 'Unknown' else COLORS['unknown_face']
                
                draw_enhanced_bbox(frame, (x1, y1, x2, y2), face['name'], None, color, [f"Emotion: {face['emotion']}"])
            
            # Activity analysis
            all_activities = []
            for i, obj in enumerate(detected_objects):
                if obj['label'] == 'person':
                    activities = activity_analyzer.analyze_person_activity(
                        f"person_{i}", obj['coords'], detected_objects, pose_landmarks)
                    all_activities.extend(activities)
                    
                    if activities:
                        print(f"🎬 Person {i}: {', '.join(activities)}")
            
            # Display enhanced status (top-left corner only)
           # cv2.putText(frame, f'FPS: {current_fps:.1f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f'Persons: {person_count}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f'Objects: {len(detected_objects)}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Resize frame to screen size for fullscreen display
            frame_resized = cv2.resize(frame, (screen_width, screen_height))
            
            # Show fullscreen frame
            cv2.namedWindow("Enhanced VisionGuard AI - Fullscreen", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Enhanced VisionGuard AI - Fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Enhanced VisionGuard AI - Fullscreen", frame_resized)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
            
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Enhanced VisionGuard AI stopped.")

if __name__ == "__main__":
    main()
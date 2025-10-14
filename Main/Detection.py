import cv2
import mediapipe as mp
import numpy as np


MP_HANDS_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 2,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.5
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(**MP_HANDS_CONFIG)

def count_extended_fingers(landmarks, h, w):
    """
    Đếm ngón tay duỗi (bỏ thumb).
    Returns: int (số ngón duỗi)
    """
    tip_ids = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    pip_ids = [6, 10, 14, 18]  # Corresponding PIP
    extended = 0
    for tip, pip in zip(tip_ids, pip_ids):
        tip_y = landmarks.landmark[tip].y * h
        pip_y = landmarks.landmark[pip].y * h
        if tip_y < pip_y:  # Tip cao hơn PIP → duỗi
            extended += 1
    return extended

def extract_keypoints_from_frame(frame_rgb, multi_landmarks):
    """
    Extract keypoints từ frame, tính center, num_fingers.
    Returns: keypoints (array), hand_centers (list), hand_fingers (list)
    """
    hand_centers = []
    hand_fingers = []
    if not multi_landmarks:
        return np.zeros(84), hand_centers, hand_fingers  # FEATURES=84 từ model.py
    h, w, _ = frame_rgb.shape
    tay_features = []
    for hand_landmarks in multi_landmarks:
        keypoints = []
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for lm in hand_landmarks.landmark:
            x, y = lm.x * w, lm.y * h
            keypoints.extend([x, y])
            x_min = min(x_min, x); y_min = min(y_min, y); x_max = max(x_max, x); y_max = max(y_max, y)
        bbox_width = max(x_max - x_min, 1)
        bbox_height = max(y_max - y_min, 1)
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        center_x_norm = (center_x / w * 2) - 1
        center_y_norm = (center_y / h * 2) - 1
        hand_centers.append((center_x_norm, center_y_norm))
        num_fingers = count_extended_fingers(hand_landmarks, h, w)
        hand_fingers.append(num_fingers)
        normalized = []
        for i in range(0, len(keypoints), 2):
            x_norm = (keypoints[i] - center_x) / bbox_width
            y_norm = (keypoints[i + 1] - center_y) / bbox_height
            normalized.extend([x_norm, y_norm])
        tay_features.extend(normalized)
    # Pad nếu <2 tay
    while len(hand_centers) < 2:
        hand_centers.append((0, 0))
        hand_fingers.append(0)
        tay_features.extend(np.zeros(42).tolist())
    return np.array(tay_features), hand_centers, hand_fingers

def draw_hand_landmarks(frame, results, hand_centers, hand_fingers, previous_centers, previous_mouse_pos, gesture_label, confidence, mapped_action):
    """
    Vẽ landmarks, bbox, text, arrow, circle trên frame.
    """
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            color = (0, 255, 0) if hand_idx == 0 else (0, 0, 255)
            h, w, _ = frame.shape
            endpoint_landmarks = [4, 8, 12, 16, 20]
            for lm_idx in endpoint_landmarks:
                lm = hand_landmarks.landmark[lm_idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=color, thickness=2))
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x); y_min = min(y_min, y); x_max = max(x_max, x); y_max = max(y_max, y)
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), color, 2)
            
            if hand_idx == 0:
                cv2.putText(frame, f"{gesture_label} ({confidence:.2f})", (x_min-20, y_min-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Action: {mapped_action}", (x_min-20, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Fingers: {hand_fingers[hand_idx]}", (x_min-20, y_min-70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                if hand_centers[hand_idx] != (0, 0):
                    cx, cy = hand_centers[hand_idx]
                    cx_px = int(cx * w / 2 + w/2)
                    cy_px = int(cy * h / 2 + h/2)
                    cv2.circle(frame, (cx_px, cy_px), 8, color, -1)
                    
                    if len(previous_centers) > hand_idx:
                        prev_cx = previous_centers[hand_idx][0]
                        prev_cy = previous_centers[hand_idx][1]
                        prev_cx_px = int(prev_cx * w / 2 + w/2)
                        prev_cy_px = int(prev_cy * h / 2 + h/2)
                        cv2.arrowedLine(frame, (prev_cx_px, prev_cy_px), (cx_px, cy_px), color, 2)
                    
                    if previous_mouse_pos and len(previous_mouse_pos) > hand_idx and previous_mouse_pos[hand_idx]:
                        mouse_x, mouse_y = previous_mouse_pos[hand_idx]
                        cv2.putText(frame, f"Mouse: ({mouse_x}, {mouse_y})", (x_min-20, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return frame

def display_frame(frame, sequence_buffer, mapped_action):
    """
    Hiển thị frame với text overlay và imshow.
    """
    cv2.putText(frame, f"Buffer: {len(sequence_buffer)}/30 | Action: {mapped_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Gesture Recognition', frame)
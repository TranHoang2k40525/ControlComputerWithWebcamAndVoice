# main.py - File chạy chương trình chính (không cần config.py)

import cv2
import time
from collections import deque
import numpy as np

# Import modules (không cần config)
from Model import load_gesture_model, predict_gesture
from Detection import hands, extract_keypoints_from_frame, draw_hand_landmarks, display_frame
from Actions import execute_mouse_action, get_action_func, execute_action

# Load model
Model, label_encoder = load_gesture_model()

# Buffer & states
sequence_buffer = deque(maxlen=30)  # N_FRAMES từ model.py
previous_centers = [(0, 0), (0, 0)]
previous_mouse_pos = [None, None]
last_discrete_time = 0
last_action = "No action"
last_log_time = 0
should_stop = False

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam!")
    exit(1)
print("Mở webcam thành công!")

fps_start_time = time.time()
fps_counter = 0

while cap.isOpened() and not should_stop:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    keypoints, hand_centers, hand_fingers = extract_keypoints_from_frame(frame_rgb, results.multi_hand_landmarks)
    sequence_buffer.append(keypoints)
    
    # Predict
    gesture_label, confidence, pred_label, gesture_type = predict_gesture(Model, label_encoder, sequence_buffer)
    current_time = time.time()
    mapped_action = "N/A"
    
    execute_func = get_action_func(pred_label)
    
    if execute_func or pred_label == 'dichuyenchuot':
        mapped_action = pred_label
        if gesture_type == 'continuous':
            if results.multi_hand_landmarks is not None and pred_label == last_action:
                hand_idx = 1 if len(results.multi_hand_landmarks) > 1 else 0
                if pred_label == 'dichuyenchuot':
                    if hand_centers[hand_idx] != (0, 0):
                        curr_x_norm, curr_y_norm = hand_centers[hand_idx]
                        execute_mouse_action(curr_x_norm, curr_y_norm, previous_mouse_pos, hand_idx)
                    else:
                        previous_mouse_pos = [None, None]
                else:  # Scroll/tab
                    curr_x, curr_y = hand_centers[hand_idx]
                    prev_x, prev_y = previous_centers[hand_idx]
                    delta_x = curr_x - prev_x
                    delta_y = curr_y - prev_y
                    num_fingers = hand_fingers[hand_idx]
                    should_stop = execute_action(execute_func, pred_label, delta_x, delta_y, num_fingers, last_discrete_time, current_time)
                    previous_centers[hand_idx] = (curr_x, curr_y)
            else:
                previous_centers = hand_centers[:]
                while len(previous_centers) < 2:
                    previous_centers.append((0, 0))
                previous_mouse_pos = [None, None]
        else:
            should_stop = execute_action(execute_func, pred_label, 0, 0, 0, last_discrete_time, current_time)
            last_discrete_time = current_time
        
        if current_time - last_log_time >= 1.0 and pred_label != last_action:
            print(f"*** DETECTED: {pred_label} (Conf: {confidence:.2f}) | Type: {gesture_type} ***")
            last_log_time = current_time
        last_action = pred_label
    
    if should_stop:
        break
    
    # Draw và display
    frame = draw_hand_landmarks(frame, results, hand_centers, hand_fingers, previous_centers, previous_mouse_pos, gesture_label, confidence, mapped_action)
    display_frame(frame, sequence_buffer, mapped_action)
    
    # FPS
    fps_counter += 1
    if fps_counter % 30 == 0:
        fps_elapsed = time.time() - fps_start_time
        fps = fps_counter / fps_elapsed
        print(f"FPS: {fps:.1f}")
        fps_start_time = time.time()
        fps_counter = 0
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Đóng webcam! Chương trình kết thúc.")
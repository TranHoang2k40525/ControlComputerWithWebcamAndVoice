# gesture_real_time_step6_touchpad_fixed.py - Fixed: 2-finger scroll & zero-delay mouse + IndexError fix + Tab arg fix
# Chạy: python gesture_real_time_step6_touchpad_fixed.py
# Sửa: Vuốt: Chỉ khi >=2 ngón duỗi, delta_y từ tay. Mouse: Map direct, execute mỗi frame mượt.
# Nhấn 'q' hoặc di chuột ra góc để dừng.

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
from collections import deque
import pyautogui

# Cảnh báo an toàn PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01  # Gần zero-delay cho mượt (0.00001s ~ real-time)

# Tham số
MODEL_PATH = 'gesture_lstm_model.h5'
N_FRAMES = 30
FEATURES = 84
CONF_THRESHOLD = 0.5
DISCRETE_DELAY = 0.2
SCROLL_SENSITIVITY = 3.0  # Tăng cho mượt touchpad-like
SMOOTH_ALPHA = 0.5  # 0.5: 50% smooth; set 1.0 cho zero smooth (raw tay pos)
TAB_THRESHOLD = 0.05  # Thêm: Threshold cho delta_x ở tab (tránh nhiễu)

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"Không tìm model tại {MODEL_PATH}!")
    exit(1)
model = load_model(MODEL_PATH)
print("Model loaded thành công!")

# Label encoder
label_encoder = np.array([
    'clickchuotphai', 'clickchuottrai', 'dichuyenchuot', 'dungchuongtrinh',
    'mochorme', 'phongto', 'thunho', 'vuotlen', 'vuotphai', 'vuottrai', 'vuotxuong'
])
print(f"Labels: {label_encoder}")

# Phân loại gesture
GESTURE_TYPES = {
    'dichuyenchuot': 'continuous',
    'vuotlen': 'continuous',
    'vuotxuong': 'continuous',
    'vuotphai': 'continuous',
    'vuottrai': 'continuous',
    'clickchuotphai': 'discrete',
    'clickchuottrai': 'discrete',
    'dungchuongtrinh': 'discrete',
    'mochorme': 'discrete',
    'phongto': 'discrete',
    'thunho': 'discrete'
}

# Action functions (giữ nguyên cho non-scroll)
def execute_right_click():
    pyautogui.rightClick()
    print("Executed: Right click!")

def execute_left_click():
    pyautogui.leftClick()
    print("Executed: Left click!")

def execute_stop_program():
    print("Executed: Dừng chương trình! (Thoát)")
    return True

def execute_open_chrome():
    pyautogui.hotkey('win', 'r')
    time.sleep(0.3)
    pyautogui.write('chrome')
    time.sleep(0.1)
    pyautogui.press('enter')
    print("Executed: Mở Chrome!")

def execute_zoom_in():
    pyautogui.hotkey('ctrl', '+')
    print("Executed: Phóng to!")

def execute_zoom_out():
    pyautogui.hotkey('ctrl', '-')
    print("Executed: Thu nhỏ!")

def execute_tab_next(delta_x):
    if abs(delta_x) > TAB_THRESHOLD:  # Thêm threshold để tránh nhiễu
        pyautogui.hotkey('ctrl', 'tab')
        print(f"Executed: Tab next (delta_x: {delta_x:.2f})")
    else:
        print("Skip tab next: delta_x too small (threshold).")

def execute_tab_prev(delta_x):
    if abs(delta_x) > TAB_THRESHOLD:  # Thêm threshold để tránh nhiễu
        pyautogui.hotkey('ctrl', 'shift', 'tab')
        print(f"Executed: Tab prev (delta_x: {delta_x:.2f})")
    else:
        print("Skip tab prev: delta_x too small (threshold).")

# SỬA: Scroll functions (dùng delta_y, nhưng chỉ khi 2+ fingers)
def execute_scroll_up(delta_y, num_fingers):
    if num_fingers >= 2:
        scroll_amount = int(-delta_y * SCROLL_SENSITIVITY)
        pyautogui.scroll(scroll_amount)
        print(f"Executed: 2-finger scroll up {scroll_amount} (delta_y: {delta_y:.2f}, fingers: {num_fingers})")
    else:
        print("Skip scroll: <2 fingers detected.")

def execute_scroll_down(delta_y, num_fingers):
    if num_fingers >= 2:
        scroll_amount = int(-delta_y * SCROLL_SENSITIVITY)
        pyautogui.scroll(scroll_amount)
        print(f"Executed: 2-finger scroll down {scroll_amount} (delta_y: {delta_y:.2f}, fingers: {num_fingers})")
    else:
        print("Skip scroll: <2 fingers detected.")

ACTION_MAP = {
    'clickchuotphai': execute_right_click,
    'clickchuottrai': execute_left_click,
    'dichuyenchuot': None,  # Xử lý riêng
    'dungchuongtrinh': execute_stop_program,
    'mochorme': execute_open_chrome,
    'phongto': execute_zoom_in,
    'thunho': execute_zoom_out,
    'vuotlen': execute_scroll_up,
    'vuotxuong': execute_scroll_down,
    'vuotphai': execute_tab_next,
    'vuottrai': execute_tab_prev
}
print("Action Mapping touchpad-fixed ready! 2-finger scroll & zero-delay mouse.")

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# SỬA: Hàm đếm ngón tay duỗi (cho 2-finger scroll)
def count_extended_fingers(landmarks, h, w):
    # Index của tips và PIP (proximal) cho 4 ngón (thumb bỏ qua cho scroll)
    tip_ids = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    pip_ids = [6, 10, 14, 18]  # Corresponding PIP
    extended = 0
    for tip, pip in zip(tip_ids, pip_ids):
        tip_y = landmarks.landmark[tip].y * h
        pip_y = landmarks.landmark[pip].y * h
        if tip_y < pip_y:  # Tip cao hơn PIP → duỗi
            extended += 1
    return extended  # >=2 cho 2-finger

# Hàm extract (thêm num_fingers cho mỗi tay)
def extract_keypoints_from_frame(frame_rgb, multi_landmarks):
    hand_centers = []
    hand_fingers = []  # Thêm: Số ngón duỗi cho mỗi tay
    if not multi_landmarks:
        return np.zeros(FEATURES), hand_centers, hand_fingers
    h, w, _ = frame_rgb.shape
    tay_features = []
    for hand_landmarks in multi_landmarks:  # Loop qua tay detect (không fixed 2)
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

# Buffer & states
sequence_buffer = deque(maxlen=N_FRAMES)
previous_centers = [(0, 0), (0, 0)]
previous_mouse_pos = [None, None]
last_discrete_time = 0
last_action = "No action"
last_log_time = 0

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam!")
    exit(1)
print("Mở webcam touchpad-fixed! Duỗi 2 ngón di lên → Scroll up mượt.")

fps_start_time = time.time()
fps_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    keypoints, hand_centers, hand_fingers = extract_keypoints_from_frame(frame_rgb, results.multi_hand_landmarks)
    sequence_buffer.append(keypoints)
    
    # Predict
    gesture_label = "Chờ buffer..."
    confidence = 0.0
    current_action = "No action"
    mapped_action = "N/A"
    should_stop = False
    current_time = time.time()
    
    if len(sequence_buffer) == N_FRAMES:
        input_seq = np.array(sequence_buffer).reshape(1, N_FRAMES, FEATURES)
        pred = model.predict(input_seq, verbose=0)[0]
        pred_idx = np.argmax(pred)
        pred_label = label_encoder[pred_idx]
        confidence = pred[pred_idx]
        if confidence > CONF_THRESHOLD:
            current_action = pred_label
            gesture_label = pred_label
            execute_func = ACTION_MAP.get(pred_label)
            gesture_type = GESTURE_TYPES.get(pred_label, 'discrete')
            if execute_func or pred_label == 'dichuyenchuot':
                mapped_action = pred_label
                if gesture_type == 'continuous':
                    # FIX: Thêm kiểm tra None để tránh lỗi len()
                    if results.multi_hand_landmarks is not None and current_action == last_action:
                        hand_idx = 1 if len(results.multi_hand_landmarks) > 1 else 0  # Ưu tiên tay phải
                        if pred_label == 'dichuyenchuot':
                            if hand_centers[hand_idx] != (0, 0):  # Detect tay
                                curr_x_norm, curr_y_norm = hand_centers[hand_idx]
                                screen_w, screen_h = pyautogui.size()
                                curr_x_screen = int((curr_x_norm + 1) * screen_w / 2)
                                curr_y_screen = int((curr_y_norm + 1) * screen_h / 2)
                                # SỬA: Zero-delay smooth (alpha blend với prev)
                                if previous_mouse_pos[hand_idx] is not None:
                                    prev_x, prev_y = previous_mouse_pos[hand_idx]
                                    smooth_x = int(prev_x + SMOOTH_ALPHA * (curr_x_screen - prev_x))
                                    smooth_y = int(prev_y + SMOOTH_ALPHA * (curr_y_screen - prev_y))
                                    pyautogui.moveTo(smooth_x, smooth_y)
                                else:
                                    pyautogui.moveTo(curr_x_screen, curr_y_screen)
                                previous_mouse_pos[hand_idx] = (curr_x_screen, curr_y_screen)  # Update prev
                                print(f"Mouse zero-delay to ({curr_x_screen}, {curr_y_screen}) | Conf: {confidence:.2f}")
                            else:
                                previous_mouse_pos = [None, None]
                        else:  # Scroll/tab
                            curr_x, curr_y = hand_centers[hand_idx]
                            prev_x, prev_y = previous_centers[hand_idx]
                            delta_x = curr_x - prev_x
                            delta_y = curr_y - prev_y
                            num_fingers = hand_fingers[hand_idx]
                            # FIX: Phân biệt arg cho scroll (2 arg) và tab (1 arg)
                            if pred_label in ['vuotlen', 'vuotxuong']:
                                execute_func(delta_y, num_fingers)
                            else:  # vuotphai, vuottrai
                                execute_func(delta_x)
                            previous_centers[hand_idx] = (curr_x, curr_y)
                    else:
                        # FIX TRIỆT ĐỂ: Pad previous_centers lên 2 phần tử khi reset (tránh len=0 khi sau detect 2 tay)
                        previous_centers = hand_centers[:]
                        while len(previous_centers) < 2:
                            previous_centers.append((0, 0))
                        previous_mouse_pos = [None, None]
                else:
                    if current_action != last_action and (current_time - last_discrete_time >= DISCRETE_DELAY):
                        should_stop = execute_func()
                        last_discrete_time = current_time
                if current_time - last_log_time >= 1.0 and current_action != last_action:
                    print(f"*** DETECTED: {current_action} (Conf: {confidence:.2f}) | Type: {gesture_type} ***")
                    last_log_time = current_time
            last_action = current_action
        if should_stop:
            break
    
    # Vẽ (thêm hiển thị num_fingers)
    # FIX: Đã có kiểm tra, nhưng đảm bảo an toàn
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
            if hand_idx == 0 and len(sequence_buffer) == N_FRAMES:
                cv2.putText(frame, f"{gesture_label} ({confidence:.2f})", (x_min-20, y_min-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Action: {mapped_action}", (x_min-20, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Fingers: {hand_fingers[hand_idx]}", (x_min-20, y_min-70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                if hand_centers[hand_idx] != (0, 0):
                    cx, cy = hand_centers[hand_idx]
                    cx_px = int(cx * w / 2 + w/2)
                    cy_px = int(cy * h / 2 + h/2)
                    cv2.circle(frame, (cx_px, cy_px), 8, color, -1)
                    # FIX TRIỆT ĐỂ: Kiểm tra len trước khi truy cập previous_centers[hand_idx] (an toàn kép)
                    if len(previous_centers) > hand_idx:
                        prev_cx = previous_centers[hand_idx][0]
                        prev_cy = previous_centers[hand_idx][1]
                        prev_cx_px = int(prev_cx * w / 2 + w/2)
                        prev_cy_px = int(prev_cy * h / 2 + h/2)
                        cv2.arrowedLine(frame, (prev_cx_px, prev_cy_px), (cx_px, cy_px), color, 2)
                    if previous_mouse_pos[hand_idx]:
                        mouse_x, mouse_y = previous_mouse_pos[hand_idx]
                        cv2.putText(frame, f"Mouse: ({mouse_x}, {mouse_y})", (x_min-20, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # FPS
    fps_counter += 1
    if fps_counter % 30 == 0:
        fps_elapsed = time.time() - fps_start_time
        fps = fps_counter / fps_elapsed
        print(f"FPS: {fps:.1f}")
        fps_start_time = time.time()
        fps_counter = 0
    
    cv2.putText(frame, f"Buffer: {len(sequence_buffer)}/30 | Action: {mapped_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Gesture Recognition - Touchpad Fixed: 2-Finger Scroll & Zero-Delay Mouse + IndexError & Tab Fix', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Đóng webcam! Touchpad fixed hoàn tất - Test 2 ngón vuốt & di tay mượt! Không còn IndexError & Tab arg error! 🎉")
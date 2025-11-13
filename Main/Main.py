# main.py - File chạy chương trình chính (không cần config.py)

import cv2
import time
from collections import deque
import numpy as np

# Import modules
from Model import load_gesture_model, predict_gesture, N_FRAMES
from Detection import hands, extract_keypoints_from_frame, draw_hand_landmarks, display_frame, detect_aligned_fingers, stabilize_results_landmarks
from Actions import execute_mouse_to_point, get_action_func, execute_action, clear_actuator_target, pause_actuator_for
import pyautogui

# Tải model
Model, label_encoder = load_gesture_model()

# Bộ đệm và trạng thái
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

# Lấy kích thước màn hình để ánh xạ đầu ngón tay sang con trỏ và tính tỉ lệ hiển thị 16:9
screen_w, screen_h = pyautogui.size()

SCALE_FACTOR = 3  # Thu nhỏ khung hiển thị xuống 1/SCALE_FACTOR lần so với kích thước mục tiêu

def compute_display_size(screen_w, screen_h):
    # Chọn chiều rộng lớn nhất không vượt quá màn hình và giữ tỉ lệ 16:9
    target_w = min(screen_w, int(screen_h * 16 / 9))
    target_h = int(target_w * 9 / 16)
    return target_w, target_h

# Tính kích thước hiển thị tỉ lệ 16:9 trên màn hình, sau đó thu nhỏ SCALE_FACTOR lần
target_w, target_h = compute_display_size(screen_w, screen_h)
target_w_scaled = max(1, target_w // SCALE_FACTOR)
target_h_scaled = max(1, target_h // SCALE_FACTOR)

# Tạo cửa sổ có thể resize với kích thước đã scale để hiển thị nhỏ hơn
cv2.namedWindow('Gesture Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gesture Recognition', target_w_scaled, target_h_scaled)

while cap.isOpened() and not should_stop:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    # Làm mượt/ổn định các landmark thô của MediaPipe ngay trên kết quả để giảm rung
    try:
        results = stabilize_results_landmarks(results, frame.shape)
    except Exception as e:
        # Defensive: if smoothing fails, continue with raw results
        print(f"Lỗi khi làm mượt landmark: {e}")

    # Nếu mediapipe không phát hiện tay hoặc landmark không đầy đủ -> không dùng model
    use_model = False
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        all_full = True
        for hland in results.multi_hand_landmarks:
            if not hasattr(hland, 'landmark') or len(hland.landmark) != 21:
                all_full = False
                break
        use_model = all_full

    keypoints, hand_centers, hand_fingers = extract_keypoints_from_frame(frame_rgb, results.multi_hand_landmarks)
    # Kiểm tra trường hợp ghi đè bằng ngón tay thẳng hàng của MediaPipe (2 ngón -> click trái, 3 ngón -> click phải)
    current_time = time.time()
    aligned_action = detect_aligned_fingers(results, frame.shape)

    # Nếu người dùng giơ 4 ngón trở lên trên tay chính, ưu tiên hành động do model dự đoán
    # và KHÔNG áp dụng ghi đè click bằng ngón thẳng hàng. Dừng actuator tạm thời để an toàn.
    try:
        primary_fingers = hand_fingers[0] if len(hand_fingers) > 0 else 0
    except Exception:
        primary_fingers = 0

    if primary_fingers >= 4:
        # ưu tiên model; tạm dừng mọi chuyển động actuator cho tay chính
        clear_actuator_target(0)
        pause_actuator_for(0, timeout=0.8)
        aligned_action = None

    # Yêu cầu nghiêm ngặt hơn cho click dựa trên số ngón:
    # - Click trái (aligned 2 ngón) chỉ khi đúng 2 ngón duỗi trên tay chính
    # - Click phải (aligned 3 ngón) chỉ khi đúng 3 ngón duỗi trên tay chính
    # Nếu số ngón không khớp -> bỏ qua aligned click
    try:
        if aligned_action == 'clickchuottrai' and primary_fingers != 2:
            aligned_action = None
        if aligned_action == 'clickchuotphai' and primary_fingers != 3:
            aligned_action = None
    except Exception:
        pass

    if aligned_action is not None:
        # Nếu có aligned_action: xóa buffer model để tránh xung đột dự đoán
        sequence_buffer.clear()
        mapped_action = aligned_action
        execute_func = get_action_func(aligned_action)
        if execute_func:
            # Thực thi hành động rời rạc (discrete) theo cơ chế cooldown hiện có
            should_stop = execute_action(execute_func, aligned_action, current_time)
        # Bỏ qua việc thêm keypoints cho model trong lúc ghi đè
        use_model = False

    if use_model:
        sequence_buffer.append(keypoints)
    else:
        # Reset buffer để tránh dự đoán cũ; dừng mọi hành động liên tục ngay lập tức
        sequence_buffer.clear()
        # Reset in-place để không phá reference mà các hàm khác (actuator) đang giữ
        try:
            previous_mouse_pos[0] = None
            previous_mouse_pos[1] = None
        except Exception:
            previous_mouse_pos = [None, None]
        # Cố gắng hủy mọi kéo/thả chuột đang diễn ra
        try:
            pyautogui.mouseUp()
        except Exception:
            pass

    # Dự đoán (chỉ khi buffer đầy và mediapipe có landmarks)
    if use_model:
        gesture_label, confidence, pred_label, gesture_type = predict_gesture(Model, label_encoder, sequence_buffer)
    else:
        gesture_label, confidence, pred_label, gesture_type = "No action", 0.0, "No action", "discrete"
    current_time = time.time()
    mapped_action = "N/A"
    
    execute_func = get_action_func(pred_label)
    
    # Chỉ chấp nhận dự đoán có độ tin cậy cao (Model.CONF_THRESHOLD mặc định ~0.7)
    if pred_label != 'No action' and (execute_func or pred_label == 'dichuyenchuot'):
        mapped_action = pred_label
        if gesture_type == 'continuous':
            # Hành động liên tục (ví dụ: di chuyển chuột): thực thi ngay bằng tọa độ đầu ngón
            if pred_label == 'dichuyenchuot':
                if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0:
                    # Sử dụng landmark 8 (đầu ngón trỏ) của tay chính (tay 0) để điều khiển chuột
                    hand_idx = 0  # Tay chính (primary hand)
                    hand_landmarks = results.multi_hand_landmarks[hand_idx]
                    try:
                        # Landmark 8 là đầu ngón trỏ
                        lm = hand_landmarks.landmark[8]
                        # lm.x/lm.y được chuẩn hóa trong [0,1] tương ứng với frame
                        # Chuyển đổi sang tọa độ màn hình
                        screen_x = lm.x * screen_w
                        screen_y = lm.y * screen_h
                        execute_mouse_to_point(screen_x, screen_y, previous_mouse_pos, hand_idx)
                    except Exception as e:
                        # Lỗi khi lấy landmark hoặc di chuột -> clear vị trí cho tay này
                        print(f"Lỗi lấy vị trí đầu ngón: {e}")
                        try:
                            previous_mouse_pos[hand_idx] = None
                        except Exception:
                            pass
                else:
                    # Không phát hiện tay - reset trạng thái (in-place để không phá ref)
                    try:
                        previous_mouse_pos[0] = None
                        previous_mouse_pos[1] = None
                    except Exception:
                        previous_mouse_pos = [None, None]
            else:
                # For other continuous actions (scroll/tab), keep using the previous center delta logic
                if results.multi_hand_landmarks is not None and pred_label == last_action:
                    hand_idx = 1 if len(results.multi_hand_landmarks) > 1 else 0
                    curr_x, curr_y = hand_centers[hand_idx]
                    prev_x, prev_y = previous_centers[hand_idx]
                    delta_x = curr_x - prev_x
                    delta_y = curr_y - prev_y
                    num_fingers = hand_fingers[hand_idx]
                    # For continuous gestures (e.g., vertical scrolls now marked continuous), allow repeated execution
                    should_stop = execute_action(execute_func, pred_label, current_time, is_continuous=True)
                    previous_centers[hand_idx] = (curr_x, curr_y)
                else:
                    previous_centers = hand_centers[:]
                    while len(previous_centers) < 2:
                        previous_centers.append((0, 0))
                    try:
                        previous_mouse_pos[0] = None
                        previous_mouse_pos[1] = None
                    except Exception:
                        previous_mouse_pos = [None, None]
        else:
            # Hành động rời rạc: thực thi qua hàm execute_action thông thường
            should_stop = execute_action(execute_func, pred_label, current_time)
            last_discrete_time = current_time
        
        if current_time - last_log_time >= 1.0 and pred_label != last_action:
            print(f"*** PHÁT HIỆN: {pred_label} (Conf: {confidence:.2f}) | Kiểu: {gesture_type} ***")
            last_log_time = current_time
        last_action = pred_label
    
    if should_stop:
        break
    
    # Vẽ và hiển thị
    frame = draw_hand_landmarks(frame, results, hand_centers, hand_fingers, previous_centers, previous_mouse_pos, gesture_label, confidence, mapped_action)

    # Thay đổi kích thước frame về tỉ lệ 16:9 đã scale và hiển thị
    disp = cv2.resize(frame, (target_w_scaled, target_h_scaled))
    display_frame(disp, sequence_buffer, mapped_action)
    
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
# main.py - File chạy chương trình chính hợp nhất webcam gesture và voice control

# Standard library imports
import sys
import os
import time
import threading
import argparse
import codecs
from collections import deque

# Third-party imports
import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageDraw, ImageFont

# Local imports - webcam gesture control
from Model import load_gesture_model, predict_gesture, N_FRAMES
from Detection import hands, extract_keypoints_from_frame, draw_hand_landmarks, display_frame, detect_aligned_fingers, stabilize_results_landmarks
from Actions import execute_mouse_to_point, get_action_func, execute_action, clear_actuator_target, pause_actuator_for
import Actions

# Local imports - voice control
try:
    import speech_recognition as sr
    from google_listen import create_recognizer, listen_phrase, adjust_for_ambient_noise
    from google_model import VoiceModel
    import command_dispatcher as dispatcher
    VOICE_AVAILABLE = True
except ImportError as e:
    print(f"[!] Voice control không khả dụng: {e}")
    VOICE_AVAILABLE = False

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Biến toàn cục để điều khiển cả 2 thread
should_stop = False
stop_lock = threading.Lock()

# Voice Control GUI Log
voice_log_messages = deque(maxlen=500)  # Lưu tối đa 500 dòng log (tăng từ 100)
voice_log_lock = threading.Lock()
voice_scroll_offset = 0  # Vị trí scroll (0 = hiển thị messages mới nhất)

# REMOVED smart caching - render mỗi frame để real-time 100%

# Cache fonts để tránh load lại liên tục (TỐI ƯU HIỆU NĂNG)
_cached_fonts = None

def get_cached_fonts():
    """Load và cache fonts một lần duy nhất."""
    global _cached_fonts
    if _cached_fonts is None:
        try:
            _cached_fonts = {
                'title': ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22),
                'text': ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14),  # Giảm từ 16 → 14
                'footer': ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
            }
        except:
            _cached_fonts = {
                'title': ImageFont.load_default(),
                'text': ImageFont.load_default(),
                'footer': ImageFont.load_default()
            }
    return _cached_fonts

# Tải model cho webcam gesture
Model, label_encoder = load_gesture_model()

# Bộ đệm và trạng thái cho webcam
sequence_buffer = deque(maxlen=15)  # Giảm từ 30 → 15 frames để phản hồi nhanh hơn
previous_centers = [(0, 0), (0, 0)]
previous_mouse_pos = [None, None]
last_discrete_time = 0
last_action = "No action"
last_log_time = 0

# Tối ưu rendering voice panel
_voice_panel_cache = None
_voice_panel_frame_counter = 0
_VOICE_PANEL_RENDER_INTERVAL = 5  # Tăng từ 3 → 5 frames để giảm tải CPU

# ==================== VOICE CONTROL GUI ====================
def add_voice_log(msg):
    """Thêm message vào voice log GUI (thread-safe) và print ngay."""
    with voice_log_lock:
        timestamp = time.strftime('%H:%M:%S')
        formatted_msg = f"[{timestamp}] {msg}"
        voice_log_messages.append(formatted_msg)
    # Print ngay ra terminal (REAL-TIME)
    print(formatted_msg, flush=True)

# Đăng ký callback cho Actions.py để log vào GUI
try:
    Actions.set_voice_log_callback(add_voice_log)
except:
    pass

def create_voice_console_window(width=800, height=600, scroll_offset=0):
    """Tạo cửa sổ console cho voice control với nền trắng và chữ đen."""
    # NO CACHING - render mỗi frame để real-time
    panel_pil = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel_pil)
    
    # Sử dụng cached fonts thay vì load lại
    fonts = get_cached_fonts()
    font_title = fonts['title']
    font_text = fonts['text']
    font_footer = fonts['footer']
    
    # Vẽ header (nền xám nhạt)
    draw.rectangle([(0, 0), (width, 50)], fill=(220, 220, 220))
    draw.text((20, 15), "Hệ Thống Điều Khiển Giọng Nói", font=font_title, fill=(0, 0, 0))
    
    # Vẽ border
    draw.rectangle([(0, 0), (width-1, height-1)], outline=(180, 180, 180), width=2)
    
    # Hiển thị messages
    with voice_log_lock:
        messages = list(voice_log_messages)
    
    y_offset = 70
    line_height = 22  # Giảm từ 25 → 22 (font nhỏ hơn)
    
    # KHÔNG GIỚI HẠN max_lines - Hiển thị tất cả messages có thể fit vào panel
    # Tính số dòng tối đa có thể hiển thị dựa trên height
    available_height = height - 90  # 70 (header) + 20 (footer)
    max_lines_fit = available_height // line_height
    
    # Tính toán phạm vi hiển thị dựa trên scroll_offset
    # scroll_offset = 0 → hiển thị messages MỚI NHẤT (cuối deque)
    # scroll_offset > 0 → scroll lên (xem messages cũ hơn)
    total_messages = len(messages)
    
    if scroll_offset == 0:
        # Hiển thị tất cả messages mới nhất có thể fit
        display_messages = messages[-max_lines_fit:] if total_messages > max_lines_fit else messages
    else:
        # Scroll lên - hiển thị messages cũ hơn
        if total_messages > max_lines_fit:
            end_index = total_messages - scroll_offset
            start_index = max(0, end_index - max_lines_fit)
            display_messages = messages[start_index:end_index]
        else:
            display_messages = messages
    
    for msg in display_messages:
        # Wrap text nếu quá dài
        if len(msg) > 95:
            msg = msg[:92] + "..."
        
        draw.text((15, y_offset), msg, font=font_text, fill=(0, 0, 0))
        y_offset += line_height
        
        # Dừng khi hết chỗ
        if y_offset > height - 40:
            break
    
    # Hiển thị scroll indicator và tổng số messages
    if scroll_offset == 0:
        # Hiển thị range mới nhất
        if total_messages > max_lines_fit:
            scroll_info = f"[{total_messages - max_lines_fit + 1}-{total_messages}/{total_messages}] LIVE"
        else:
            scroll_info = f"[1-{total_messages}/{total_messages}] LIVE"
    else:
        # Hiển thị range đang scroll
        end_idx = total_messages - scroll_offset
        start_idx = max(1, end_idx - len(display_messages) + 1)
        scroll_info = f"[{start_idx}-{end_idx}/{total_messages}]"
    
    draw.text((width - 180, height - 20), scroll_info, font=font_footer, fill=(100, 100, 100))
    
    # Footer
    footer_text = "Nhấn 'q' để đóng | Mũi tên/Scroll chuột để xem lịch sử"
    draw.text((15, height - 20), footer_text, font=font_footer, fill=(100, 100, 100))
    
    # Chuyển PIL image sang OpenCV format
    panel = cv2.cvtColor(np.array(panel_pil), cv2.COLOR_RGB2BGR)
    
    return panel

def voice_gui_thread():
    """Thread này không còn được sử dụng - voice đã tích hợp vào webcam_gesture_thread."""
    pass

# ==================== VOICE CONTROL THREAD ====================
WAKE_WORDS = ["ok google", "hey google", "xin chào google"]
EXIT_WORDS = ["kết thúc", "dừng lại", "thoát"]


def discover_and_load_model(model_path, tokenizer_path, label_path, train_dir='Train'):
    """Nếu có path cụ thể thì dùng, nếu không thì tự động tìm kiếm."""
    if model_path and tokenizer_path and label_path:
        return VoiceModel(model_path, tokenizer_path, label_path)

    # Tự động tìm kiếm
    add_voice_log(f'[Voice] Đang tìm kiếm các file model trong thư mục "{train_dir}"...')
    m, t, l = VoiceModel.discover_from_train_dir(train_dir)
    add_voice_log(f'[Voice] Đã phát hiện: model={m}, tokenizer={t}, label_encoder={l}')
    return VoiceModel(m, t, l)


def voice_control_thread(use_model=True, model_path=None, tokenizer_path=None, label_path=None):
    """Thread điều khiển giọng nói qua LSTM."""
    global should_stop
    
    if not VOICE_AVAILABLE:
        add_voice_log('[Voice] Voice control không khả dụng do thiếu thư viện.')
        return
    
    add_voice_log('[Voice] === HỆ THỐNG ĐIỀU KHIỂN BẰNG GIỌNG NÓI ===')
    add_voice_log(f'[Voice] Từ khóa đánh thức: {WAKE_WORDS}')
    
    model = None
    if use_model:
        try:
            add_voice_log('[Voice] Đang tải mô hình LSTM AI...')
            model = discover_and_load_model(model_path, tokenizer_path, label_path)
            add_voice_log('[Voice] ✓ LSTM Model đã sẵn sàng!')
        except Exception as e:
            add_voice_log(f'[Voice] ! Lỗi khi load model: {e}')
            add_voice_log('[Voice] ! Không thể xử lý lệnh mà không có LSTM model.')
            model = None
    else:
        add_voice_log('[Voice] ! LSTM model bị tắt - Voice control sẽ không hoạt động.')

    r = create_recognizer()
    try:
        with sr.Microphone() as source:
            adjust_for_ambient_noise(r, source, duration=1.0)
            add_voice_log('[Voice] OK - Microphone sẵn sàng. Đang chờ từ khóa đánh thức...')

            while True:
                with stop_lock:
                    if should_stop:
                        break
                
                text = listen_phrase(r, source, timeout=None)
                if not text:
                    continue
                
                add_voice_log(f'[Voice] Nghe được: {text}')

                if any(w in text for w in WAKE_WORDS):
                    add_voice_log('[Voice] KÍCH HOẠT - Đã phát hiện từ khóa. Đang nghe lệnh...')
                    cmd = listen_phrase(r, source, timeout=5, time_limit=12)
                    if not cmd:
                        add_voice_log('[Voice] ! Không nghe thấy lệnh (timeout).')
                        continue

                    # Kiểm tra lệnh thoát
                    if any(w in cmd for w in EXIT_WORDS):
                        add_voice_log('[Voice] ! Nhận lệnh thoát. Đang tắt hệ thống.')
                        with stop_lock:
                            should_stop = True
                        break

                    # ============ XỬ LÝ LỆNH QUA LSTM ============
                    
                    if model is None:
                        add_voice_log('[Voice] ✗ LỖI: LSTM model chưa load - không thể xử lý lệnh!')
                        continue
                    
                    try:
                        # Gọi LSTM model để nhận diện hành động
                        add_voice_log(f'[Voice] >> LSTM Processing: "{cmd}"')
                        pred_label, confidence, _ = model.predict_action_from_text(cmd)
                        add_voice_log(f'[Voice] << LSTM Output: {pred_label} ({confidence:.1f}%)')
                        
                        # Xử lý hành động dựa trên kết quả LSTM
                        if pred_label == 'moapp':
                            add_voice_log(f'[Voice] ✓ Hành động: MỞ ỨNG DỤNG')
                            add_voice_log(f'[Voice] >> Đang mở ứng dụng...')
                            # Chạy trong thread riêng để không block webcam
                            threading.Thread(
                                target=lambda: Actions.execute_action(
                                    lambda: Actions.execute_open_app(cmd), 
                                    pred_label, 
                                    time.perf_counter(), 
                                    is_continuous=False
                                ),
                                daemon=True
                            ).start()
                            add_voice_log(f'[Voice] ✓✓ HOÀN TẤT: Mở ứng dụng')
                        
                        elif pred_label == 'nhapvanban':
                            add_voice_log(f'[Voice] ✓ Hành động: NHẬP VĂN BẢN')
                            add_voice_log(f'[Voice] >> Hãy nói nội dung cần nhập...')
                            
                            content = listen_phrase(r, source, timeout=5, time_limit=15)
                            
                            if content:
                                add_voice_log(f'[Voice] >> Đã thu: "{content}"')
                                # Chạy trong thread riêng để không block webcam
                                threading.Thread(
                                    target=lambda: Actions.execute_type_text(content),
                                    daemon=True
                                ).start()
                                add_voice_log(f'[Voice] ✓✓ HOÀN TẤT: Nhập văn bản')
                            else:
                                add_voice_log(f'[Voice] ! Không nghe thấy nội dung.')
                        
                        else:
                            # Các hành động khác: click, scroll, zoom, tab...
                            func = Actions.get_action_func(pred_label)
                            if func:
                                add_voice_log(f'[Voice] ✓ Hành động: {pred_label.upper()}')
                                # Chạy trong thread riêng để không block webcam
                                threading.Thread(
                                    target=lambda: Actions.execute_action(func, pred_label, time.perf_counter(), is_continuous=False),
                                    daemon=True
                                ).start()
                                add_voice_log(f'[Voice] ✓✓ HOÀN TẤT: {pred_label}')
                            else:
                                add_voice_log(f'[Voice] ✗ LỖI: Không tìm thấy hàm xử lý cho "{pred_label}"!')
                    
                    except Exception as e:
                        add_voice_log(f'[Voice] ✗ LỖI: {e}')
                        add_voice_log(f'[Voice] ! Không thể xử lý lệnh "{cmd}"')

    except KeyboardInterrupt:
        add_voice_log('[Voice] ! Dừng bởi người dùng')
        with stop_lock:
            should_stop = True
    except Exception as e:
        add_voice_log(f'[Voice] ! Lỗi: {e}')
        with stop_lock:
            should_stop = True


# ==================== WEBCAM GESTURE CONTROL ====================
def webcam_gesture_thread():
    """Thread điều khiển cử chỉ webcam."""
    global should_stop, sequence_buffer, previous_centers, previous_mouse_pos
    global last_discrete_time, last_action, last_log_time
    global voice_scroll_offset
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Webcam] ! Không mở được webcam!")
        with stop_lock:
            should_stop = True
        return
    print("[Webcam] Mở webcam thành công!")

    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0.0
    last_action_execute_time = None  # Tracking thời gian action cuối cùng được execute

    # Lấy kích thước màn hình
    screen_w, screen_h = pyautogui.size()

    SCALE_FACTOR = 3  # Thu nhỏ khung hiển thị

    def compute_display_size(screen_w, screen_h):
        target_w = min(screen_w, int(screen_h * 16 / 9))
        target_h = int(target_w * 9 / 16)
        return target_w, target_h

    # Tính kích thước hiển thị
    target_w, target_h = compute_display_size(screen_w, screen_h)
    target_w_scaled = max(1, target_w // SCALE_FACTOR)
    target_h_scaled = max(1, target_h // SCALE_FACTOR)

    # Kích thước voice panel
    voice_width = 800
    voice_height = target_h_scaled
    
    # Tạo cửa sổ hợp nhất (webcam + voice)
    window_name = 'Control PC with Webcam + Voice'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, target_w_scaled + voice_width, target_h_scaled)
    
    # Callback cho mouse wheel (scroll voice panel)
    def mouse_callback(event, x, y, flags, param):
        global voice_scroll_offset
        if event == cv2.EVENT_MOUSEWHEEL and x >= target_w_scaled:
            max_lines = (voice_height - 110) // 25
            with voice_log_lock:
                total_messages = len(voice_log_messages)
                if flags > 0:
                    voice_scroll_offset = min(voice_scroll_offset + 3, total_messages - max_lines)
                else:
                    voice_scroll_offset = max(0, voice_scroll_offset - 3)
    
    cv2.setMouseCallback(window_name, mouse_callback)

    while cap.isOpened():
        with stop_lock:
            if should_stop:
                break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        # TẮT stabilize_results_landmarks để tăng FPS từ 10-12 lên 25-30
        # MediaPipe đã có tracking built-in, không cần smoothing thêm
        # try:
        #     results = stabilize_results_landmarks(results, frame.shape)
        # except Exception as e:
        #     print(f"[Webcam] Lỗi khi làm mượt landmark: {e}")

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
        # Hàm trả về tuple: (action_name, num_extended_fingers)
        aligned_action, aligned_fingers = detect_aligned_fingers(results, frame.shape)

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

        # Yêu cầu CHẶT CHẼ cho click dựa trên số ngón:
        # - Click trái: cả aligned_fingers và primary_fingers đều phải = 2
        # - Click phải: cả aligned_fingers và primary_fingers đều phải = 3 (KHÔNG >= 3)
        # Điều này tránh nhầm lẫn với vuốt (4-5 ngón duỗi)
        try:
            if aligned_action == 'clickchuottrai':
                # Yêu cầu: cả 2 phương pháp đếm đều = 2
                if aligned_fingers != 2 or primary_fingers != 2:
                    aligned_action = None
            elif aligned_action == 'clickchuotphai':
                # Yêu cầu: cả 2 phương pháp đếm đều = 3 (không chấp nhận >3)
                if aligned_fingers != 3 or primary_fingers != 3:
                    aligned_action = None
        except Exception:
            aligned_action = None

        if aligned_action is not None:
            # Nếu có aligned_action: xóa buffer model để tránh xung đột dự đoán
            sequence_buffer.clear()
            mapped_action = aligned_action
            execute_func = get_action_func(aligned_action)
            if execute_func:
                # Thực thi hành động rời rạc (discrete) theo cơ chế cooldown hiện có
                local_stop = execute_action(execute_func, aligned_action, current_time)
                if local_stop:
                    with stop_lock:
                        should_stop = True
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
                            print(f"[Webcam] Lỗi lấy vị trí đầu ngón: {e}")
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
                        local_stop = execute_action(execute_func, pred_label, current_time, is_continuous=True)
                        if local_stop:
                            with stop_lock:
                                should_stop = True
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
                local_stop = execute_action(execute_func, pred_label, current_time)
                if local_stop:
                    with stop_lock:
                        should_stop = True
                last_discrete_time = current_time
            
            # Cập nhật thời gian action được execute
            last_action_execute_time = current_time
            
            if current_time - last_log_time >= 1.0 and pred_label != last_action:
                print(f"[Webcam] *** PHÁT HIỆN: {pred_label} (Conf: {confidence:.2f}) | Kiểu: {gesture_type} ***")
                last_log_time = current_time
            last_action = pred_label
        
        with stop_lock:
            if should_stop:
                break
        
        # Vẽ và hiển thị với buffer frames và action time
        frame = draw_hand_landmarks(frame, results, hand_centers, hand_fingers, previous_centers, previous_mouse_pos, gesture_label, confidence, mapped_action, buffer_frames=len(sequence_buffer), last_action_time=last_action_execute_time)

        # Thay đổi kích thước webcam
        disp = cv2.resize(frame, (target_w_scaled, target_h_scaled))
        
        # Tối ưu: Render voice panel chỉ mỗi 3 frames thay vì mỗi frame
        global _voice_panel_cache, _voice_panel_frame_counter
        _voice_panel_frame_counter += 1
        if _voice_panel_cache is None or _voice_panel_frame_counter >= _VOICE_PANEL_RENDER_INTERVAL:
            _voice_panel_cache = create_voice_console_window(width=voice_width, height=voice_height, scroll_offset=voice_scroll_offset)
            _voice_panel_frame_counter = 0
        voice_panel = _voice_panel_cache
        
        # Tạo canvas kết hợp: webcam (trái) + voice (phải)
        combined = np.zeros((target_h_scaled, target_w_scaled + voice_width, 3), dtype=np.uint8)
        combined[0:target_h_scaled, 0:target_w_scaled] = disp
        combined[0:voice_height, target_w_scaled:target_w_scaled+voice_width] = voice_panel
        
        cv2.imshow(window_name, combined)
        
        # Tính FPS real-time mỗi frame
        fps_counter += 1
        fps_elapsed = time.time() - fps_start_time
        if fps_elapsed > 0:
            current_fps = fps_counter / fps_elapsed
        
        # Reset FPS counter mỗi giây để có số liệu chính xác
        if fps_elapsed >= 1.0:
            print(f"[Webcam] FPS: {current_fps:.1f}")
            fps_start_time = time.time()
            fps_counter = 0
        
        # Xử lý phím - Tối ưu: 16ms (60 FPS) thay vì 10ms để giảm tải CPU
        key = cv2.waitKey(16)
        if key & 0xFF == ord('q'):
            print("[Webcam] Nhấn 'q' để thoát")
            with stop_lock:
                should_stop = True
            break
        elif key == 82 or key == 0:  # Up arrow - scroll voice
            max_lines = (voice_height - 110) // 25
            with voice_log_lock:
                total_messages = len(voice_log_messages)
                voice_scroll_offset = min(voice_scroll_offset + 1, total_messages - max_lines)
        elif key == 84 or key == 1:  # Down arrow
            voice_scroll_offset = max(0, voice_scroll_offset - 1)

    cap.release()
    cv2.destroyAllWindows()
    print("[Webcam] Đóng webcam! Thread kết thúc.")


# ==================== MAIN PROGRAM ====================
def parse_args():
    p = argparse.ArgumentParser(description='Chương trình điều khiển kết hợp Webcam Gesture + Voice Control')
    p.add_argument('--no-voice', action='store_true', help='Tắt điều khiển giọng nói')
    p.add_argument('--no-webcam', action='store_true', help='Tắt điều khiển webcam')
    p.add_argument('--use-model', action='store_true', help='Sử dụng model AI cho voice control')
    p.add_argument('--model-path', help='Path to .h5 model file cho voice')
    p.add_argument('--tokenizer-path', help='Path to tokenizer .pkl cho voice')
    p.add_argument('--label-encoder-path', help='Path to label_encoder .pkl cho voice')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Tự động phát hiện model files cho voice control nếu không chỉ định
    if VOICE_AVAILABLE and not args.no_voice and not args.use_model:
        try:
            print("[Main] Đang tự động tìm kiếm model files cho voice control...")
            m, t, l = VoiceModel.discover_anywhere()
            print(f"[Main] ✓ Đã tìm thấy model files:")
            print(f"    • Model: {m}")
            print(f"    • Tokenizer: {t}")
            print(f"    • Label Encoder: {l}")
            args.use_model = True
            args.model_path = args.model_path or m
            args.tokenizer_path = args.tokenizer_path or t
            args.label_encoder_path = args.label_encoder_path or l
        except Exception as e:
            print(f"[Main] ! Không tìm thấy model cho voice: {e}")
            print("[Main] ! Voice control sẽ chỉ dùng từ khóa")
    
    print("\n" + "="*60)
    print("  CHƯƠNG TRÌNH ĐIỀU KHIỂN KẾT HỢP")
    print("  Webcam Gesture Control + Voice Control")
    print("="*60)
    
    threads = []
    
    # Khởi động Voice Control Thread
    if not args.no_voice and VOICE_AVAILABLE:
        voice_thread = threading.Thread(
            target=voice_control_thread,
            args=(args.use_model, args.model_path, 
                  args.tokenizer_path, args.label_encoder_path),
            daemon=True
        )
        voice_thread.start()
        threads.append(('Voice Control', voice_thread))
        print("[Main] ✓ Voice Control thread đã khởi động (GUI tích hợp trong webcam)")
    else:
        if args.no_voice:
            print("[Main] Voice Control bị tắt (--no-voice)")
        else:
            print("[Main] Voice Control không khả dụng")
    
    # Khởi động Webcam Gesture Thread
    if not args.no_webcam:
        webcam_thread = threading.Thread(
            target=webcam_gesture_thread,
            daemon=True
        )
        webcam_thread.start()
        threads.append(('Webcam Gesture', webcam_thread))
        print("[Main] ✓ Webcam Gesture thread đã khởi động")
    else:
        print("[Main] Webcam Gesture bị tắt (--no-webcam)")
    
    print("\n[Main] Hệ thống đang hoạt động...")
    print("[Main] Nhấn Ctrl+C hoặc nói 'kết thúc' để thoát")
    print("[Main] Hoặc nhấn 'q' trong cửa sổ webcam hoặc voice console để thoát\n")
    
    try:
        # Chờ các threads hoàn thành
        for name, thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n[Main] ! Nhận Ctrl+C, đang thoát...")
        with stop_lock:
            should_stop = True
    
    # Đợi tất cả threads kết thúc
    for name, thread in threads:
        if thread.is_alive():
            print(f"[Main] Đang đợi {name} thread kết thúc...")
            thread.join(timeout=2.0)
    
    print("[Main] Chương trình đã kết thúc hoàn toàn!")
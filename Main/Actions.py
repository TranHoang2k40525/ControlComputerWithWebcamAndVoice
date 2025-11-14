# actions.py - Các hàm thực thi hành động (gộp cấu hình: SCROLL_AMOUNT, SMOOTH_ALPHA, v.v.)

import pyautogui
import time
import math
import cv2
import numpy as np
import datetime
import threading
# Gộp config actions
SMOOTH_ALPHA = 0.6  # Tăng lên để phản hồi nhanh hơn (40% mới, 60% cũ)
DISCRETE_COOLDOWN = 1.0
# Cấu hình scroll mượt khi gesture được duy trì (per-call nhỏ, được gọi mỗi frame)
CONTINUOUS_SCROLL_STEP = 30  # số lượng scroll (px) mỗi lần gọi liên tiếp
CONTINUOUS_SCROLL_MAX_STEP = 900  # giới hạn tối đa số scroll trong 1 lần tính
# Cấu hình cho di chuyển chuột mượt mà
MOUSE_DEAD_ZONE = 0.02  # Vùng chết để lọc nhiễu (2% màn hình) - giảm để nhạy hơn
MOUSE_SPEED_MULTIPLIER =  4 # Hệ số tốc độ di chuyển - tăng lên để chuột di nhanh hơn
MOUSE_MAX_MOVE = 100  # Giới hạn di chuyển tối đa mỗi frame (pixels) - tăng lên để cho phép di xa hơn
MOUSE_MAX_SPEED_PX_PER_SEC = 600.0  # Giới hạn tốc độ theo thời gian (px / second)
ACTUATION_HZ = 60  # Tần số actuator sẽ di chuyển con trỏ (Hz)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

# Tracking last execution time cho từng gesture (cooldown)
last_execution_times = {}

def execute_right_click():
    pyautogui.rightClick()
    print("Executed: Right click!")
    return False  # Không dừng chương trình

def execute_left_click():
    pyautogui.leftClick()
    print("Executed: Left click!")
    return False  # Không dừng chương trình

def execute_stop_program():
    print("Executed: Dừng chương trình! (Thoát)")
    return False  # Dừng chương trình

def execute_open_app():
    """Mở app, không dừng chương trình."""
    
    pyautogui.hotkey('win', 'r')
    # 2. Chờ hộp thoại Run xuất hiện (rất quan trọng)
    time.sleep(0.5)
    pyautogui.write('"C:\Users\Public\Desktop\Cốc Cốc.lnk"')
    pyautogui.press('enter')
    print("Executed: Mở Coc Coc!")
    return False  # Không dừng chương trình

def execute_zoom_in():
    pyautogui.hotkey('ctrl', '+')
    print("Executed: Phóng to!")
    return False

def execute_zoom_out():
    pyautogui.hotkey('ctrl', '-')
    print("Executed: Thu nhỏ!")
    return False

def execute_tab_next():
    """Chuyển tab tiếp theo (Ctrl+Tab)."""
    pyautogui.hotkey('ctrl', 'tab')
    print("Executed: Tab next (Ctrl+Tab)")
    return False

def execute_tab_prev():
    """Chuyển tab trước đó (Ctrl+Shift+Tab)."""
    pyautogui.hotkey('ctrl', 'shift', 'tab')
    print("Executed: Tab prev (Ctrl+Shift+Tab)")
    return False

def execute_scroll_up():
    """Scroll LÊN.

    Chuyển sang chế độ 'mượt' khi được gọi lặp (continuous):
    - Mỗi lần gọi chỉ scroll một bước nhỏ (CONTINUOUS_SCROLL_STEP).
    - Bước scroll có thể được phóng đại nhẹ dựa trên khoảng thời gian giữa hai lần gọi liên tiếp
      để giữ tốc độ tương đối theo thời gian.
    - Không dùng sleep() trong hàm này để tránh block main loop.
    """
    now = time.perf_counter()
    prev = _last_scroll_time.get('up', 0.0)
    dt = now - prev if prev else 0.0
    # Cập nhật thời điểm gọi
    _last_scroll_time['up'] = now

    # Tỷ lệ tăng step theo dt (clamp để tránh nhảy lớn khi frame drop)
    multiplier = min(3.0, max(1.0, dt * 30.0))  # dt ~0.03 -> ~<~1 normal, frame drops -> tăng
    step = int(min(CONTINUOUS_SCROLL_MAX_STEP, CONTINUOUS_SCROLL_STEP * multiplier))

    try:
        pyautogui.scroll(step)
    except Exception:
        # Nếu pyautogui thất bại (failsafe), bỏ qua
        pass

    print(f"Executed: Scroll UP step={step} (dt={dt:.3f}s)")
    return False

def execute_scroll_down():
    """Scroll XUỐNG.

    Hoạt động tương tự `execute_scroll_up` nhưng với giá trị âm.
    """
    
    now = time.perf_counter()
    prev = _last_scroll_time.get('down', 0.0)
    dt = now - prev if prev else 0.0
    _last_scroll_time['down'] = now

    multiplier = min(3.0, max(1.0, dt * 30.0))
    step = int(min(CONTINUOUS_SCROLL_MAX_STEP, CONTINUOUS_SCROLL_STEP * multiplier))

    try:
        pyautogui.scroll(-step)
    except Exception:
        pass

    print(f"Executed: Scroll DOWN step={step} (dt={dt:.3f}s)")
    return False



# Lưu trữ lịch sử di chuyển để phát hiện rung lắc
_movement_history = {}

# Timing helpers để debug độ trễ
_last_call_time = {}   # hand_idx -> perf_counter của lần gọi trước
_last_move_time = {}   # hand_idx -> perf_counter của lần di chuyển chuột trước
# Thời gian gọi scroll gần nhất (dùng để scale step cho scroll liên tục)
_last_scroll_time = {'up': 0.0, 'down': 0.0}

# Actuator: vòng lặp riêng để di chuột mượt ngay cả khi fps thấp
_actuator_targets = {}  # hand_idx -> (x, y) mục tiêu (float)
_actuator_lock = threading.Lock()
_actuator_thread = None
_actuator_running = False

# Manual-override / actuator bookkeeping
_last_target_update = {}        # hand_idx -> perf_counter của lần set target gần nhất
_manual_override_until = {}     # hand_idx -> perf_counter until which actuator is paused
_last_actuator_pos = {}         # hand_idx -> (x,y) last position actuator moved to

# Manual override params
MANUAL_OVERRIDE_THRESHOLD_PX = 40  # px, nếu cursor di chuyển > threshold so với last_actuator_pos và không phải actuator -> pause
MANUAL_OVERRIDE_TIMEOUT = 1.0      # giây pause khi phát hiện manual override

# Smoothed finger positions (px) để tránh di chuyển do nhiễu nhỏ
_finger_smoothed = {}  # hand_idx -> (x_px, y_px)
FINGER_SMOOTH_ALPHA = 0.7  # EMA alpha cho vị trí ngón tay (tách khỏi SMOOTH_ALPHA chuột)
MIN_MOVE_PIXELS = 6  # Nếu di chuyển sau lọc < MIN_MOVE_PIXELS -> coi như không di chuyển

# Time-aware pointer smoothing (bù gap do fps/latency)
# (Các biến time-aware pointer filter đã bị loại bỏ vì không sử dụng)

# Per-hand Kalman filters (2D constant velocity)
_kalman_filters = {}


class KalmanFilter2D:
    """Lớp bọc Kalman 2D đơn giản (mô hình vận tốc không đổi) dùng OpenCV KalmanFilter.

    Trạng thái: [x, y, vx, vy]
    Quan sát/đo lường: [x, y]
    """
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        #ma trận chuyển trạng thái
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],#xnew= xold +vxold(1*x + 0*y + 1*vxold + 0*vyold)
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)
        # ma tran quan sát z
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def initialize(self, x, y):
        self.kf.statePost = np.array([[float(x)], [float(y)], [0.0], [0.0]], dtype=np.float32)
        self.initialized = True

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def update(self, x, y):
        meas = np.array([[float(x)], [float(y)]], dtype=np.float32)
        post = self.kf.correct(meas)
        return float(post[0]), float(post[1])


def _actuator_loop():
    global _actuator_running
    period = 1.0 / max(1, ACTUATION_HZ)
    while _actuator_running:
        with _actuator_lock:
            items = list(_actuator_targets.items())
        # Nếu không có target nào, ngủ một vòng để tránh busy-loop
        if not items:
            time.sleep(period)
            continue

        for hand_idx, (tx, ty) in items:
            try:
                cx, cy = pyautogui.position()
                now = time.perf_counter()

                # Tôn trọng manual override: nếu actuator đang bị tạm dừng cho tay này thì bỏ qua
                if _manual_override_until.get(hand_idx, 0) > now:
                    # update last_actuator_pos to current cursor to stay in sync
                    _last_actuator_pos[hand_idx] = (int(cx), int(cy))
                    continue

                # Phát hiện di chuyển tay/thao tác chuột thủ công: nếu con trỏ di chuyển nhiều so với
                # last_actuator_pos và không có cập nhật target gần đây thì coi là di chuyển bằng chuột thật
                prev_act_pos = _last_actuator_pos.get(hand_idx)
                last_target_time = _last_target_update.get(hand_idx, 0)
                if prev_act_pos is not None:
                    delta_user = ((cx - prev_act_pos[0])**2 + (cy - prev_act_pos[1])**2) ** 0.5
                    if delta_user > MANUAL_OVERRIDE_THRESHOLD_PX and (now - last_target_time) > 0.05:
                        # Assume user moved the physical mouse -> pause actuator for a bit and clear target
                        _manual_override_until[hand_idx] = now + MANUAL_OVERRIDE_TIMEOUT
                        with _actuator_lock:
                            if hand_idx in _actuator_targets:
                                del _actuator_targets[hand_idx]
                        _last_actuator_pos[hand_idx] = (int(cx), int(cy))
                        continue
                dx = tx - cx
                dy = ty - cy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 1.0:
                    # update last actuator pos
                    _last_actuator_pos[hand_idx] = (int(cx), int(cy))
                    continue

                # Bước di chuyển tối đa cho mỗi vòng lặp dựa trên giới hạn tốc độ
                allowed = MOUSE_MAX_SPEED_PX_PER_SEC * period
                step = min(dist, allowed)
                nx = cx + (dx / dist) * step
                ny = cy + (dy / dist) * step
                pyautogui.moveTo(int(nx), int(ny))
                # record actuator move
                _last_actuator_pos[hand_idx] = (int(nx), int(ny))
            except Exception:
                # Bỏ qua lỗi actuator (ví dụ failsafe của pyautogui)
                pass

        time.sleep(period)


def start_actuator():
    global _actuator_thread, _actuator_running
    if _actuator_thread is not None:
        return
    _actuator_running = True
    _actuator_thread = threading.Thread(target=_actuator_loop, daemon=True)
    _actuator_thread.start()


# Khởi động actuator khi module được import
try:
    start_actuator()
except Exception:
    pass

def execute_mouse_to_point(screen_x, screen_y, previous_mouse_pos, hand_idx, smooth_alpha=None):
    """
    Di chuyển chuột theo vị trí tương đối.
    Vùng chết chỉ áp dụng cho rung lắc (thay đổi hướng liên tục), 
    không áp dụng cho di chuyển chậm có hướng.
    """
    if smooth_alpha is None:
        smooth_alpha = SMOOTH_ALPHA
    
    # Bắt đầu đo thời gian gọi để debug độ trễ
    call_perf = time.perf_counter()
    call_wall = time.time()
    prev_call = _last_call_time.get(hand_idx)
    dt_since_last_call = (call_perf - prev_call) if (prev_call is not None) else 0.0
    _last_call_time[hand_idx] = call_perf

    # Lấy vị trí hiện tại của chuột
    current_mouse_x, current_mouse_y = pyautogui.position()
    
    # Apply per-hand finger EMA smoothing to measured screen_x/screen_y
    try:
        prev_fs = _finger_smoothed.get(hand_idx)
    except Exception:
        prev_fs = None
    # EMA (Exponential Moving Average)(trung bình trượt động mũ)
    if prev_fs is None:
        _finger_smoothed[hand_idx] = (float(screen_x), float(screen_y))
        meas_x, meas_y = float(screen_x), float(screen_y)
    else:
        meas_x = FINGER_SMOOTH_ALPHA * float(screen_x) + (1.0 - FINGER_SMOOTH_ALPHA) * prev_fs[0]
        meas_y = FINGER_SMOOTH_ALPHA * float(screen_y) + (1.0 - FINGER_SMOOTH_ALPHA) * prev_fs[1]
        _finger_smoothed[hand_idx] = (meas_x, meas_y)

    # Áp dụng Kalman filter cho từng tay (nếu có) để loại bỏ jitter và dự đoán vị trí
    try:
        kf = _kalman_filters.get(hand_idx)
    except Exception:
        kf = None

    if kf is None:
        # Khởi tạo Kalman cho tay này
        kf = KalmanFilter2D()
        kf.initialize(meas_x, meas_y)
        _kalman_filters[hand_idx] = kf
    else:
        # Dự đoán và cập nhật với measurement hiện tại
        kf.predict()
        meas_x, meas_y = kf.update(meas_x, meas_y)

    # Nếu đã có vị trí trước đó của ngón tay
    if previous_mouse_pos[hand_idx] is not None:
        prev_finger_x, prev_finger_y = previous_mouse_pos[hand_idx]

        # Dùng measurement đã làm mượt để tính delta
        delta_x = meas_x - prev_finger_x
        delta_y = meas_y - prev_finger_y

        # Tính khoảng cách thay đổi
        distance = (delta_x ** 2 + delta_y ** 2) ** 0.5

        # Khởi tạo lịch sử nếu chưa có
        if hand_idx not in _movement_history:
            _movement_history[hand_idx] = []

        # Lưu delta vào lịch sử (giữ 5 frame gần nhất)
        _movement_history[hand_idx].append((delta_x, delta_y))
        if len(_movement_history[hand_idx]) > 5:
            _movement_history[hand_idx].pop(0)

        # Kiểm tra xem có phải rung lắc không (thay đổi hướng liên tục)
        is_jittering = False
        if len(_movement_history[hand_idx]) >= 3:
            # Tính tổng vector của 3 frame gần nhất
            sum_x = sum(h[0] for h in _movement_history[hand_idx][-3:])
            sum_y = sum(h[1] for h in _movement_history[hand_idx][-3:])
            sum_distance = (sum_x ** 2 + sum_y ** 2) ** 0.5

            # Tính tổng khoảng cách tuyệt đối
            total_distance = sum((h[0]**2 + h[1]**2)**0.5 for h in _movement_history[hand_idx][-3:])

            # Nếu tổng vector nhỏ hơn nhiều so với tổng khoảng cách → rung lắc
            if total_distance > 0 and sum_distance / total_distance < 0.3:
                is_jittering = True

        # Chỉ áp dụng vùng chết nếu đang rung lắc
        screen_w, screen_h = pyautogui.size()
        distance_normalized = distance / ((screen_w ** 2 + screen_h ** 2) ** 0.5)

        should_move = True
        if is_jittering and distance_normalized < MOUSE_DEAD_ZONE:
            should_move = False

        # Bỏ qua các di chuyển rất nhỏ để tránh drift chuột
        if should_move and distance > max(0.5, MIN_MOVE_PIXELS):
            # Áp dụng hệ số tốc độ
            move_x = delta_x * MOUSE_SPEED_MULTIPLIER
            move_y = delta_y * MOUSE_SPEED_MULTIPLIER

            # Giới hạn tốc độ di chuyển tối đa
            move_distance = (move_x ** 2 + move_y ** 2) ** 0.5
            if move_distance > MOUSE_MAX_MOVE:
                scale = MOUSE_MAX_MOVE / move_distance
                move_x *= scale
                move_y *= scale

            # Tính vị trí mới của chuột (tương đối)
            new_mouse_x = current_mouse_x + move_x
            new_mouse_y = current_mouse_y + move_y

            # Áp dụng smoothing
            smooth_x = int(current_mouse_x + smooth_alpha * (new_mouse_x - current_mouse_x))
            smooth_y = int(current_mouse_y + smooth_alpha * (new_mouse_y - current_mouse_y))

            # Đảm bảo không ra ngoài màn hình
            smooth_x = max(0, min(screen_w - 1, smooth_x))
            smooth_y = max(0, min(screen_h - 1, smooth_y))

            # Thay vì di chuyển trực tiếp, cập nhật target cho actuator để actuation mượt
            with _actuator_lock:
                _actuator_targets[hand_idx] = (float(smooth_x), float(smooth_y))
            # ghi nhận thời điểm set target và xóa override nếu có
            _last_target_update[hand_idx] = call_perf
            if hand_idx in _manual_override_until:
                _manual_override_until.pop(hand_idx, None)
            last_move = _last_move_time.get(hand_idx)
            dt_since_last_move = (call_perf - last_move) if (last_move is not None) else 0.0
            _last_move_time[hand_idx] = call_perf

            status = "JITTER" if is_jittering else "MOVE"
            ts = datetime.datetime.fromtimestamp(call_wall).strftime('%H:%M:%S.%f')
            print(f"[{ts}] [{status}] dt_call={dt_since_last_call:.3f}s dt_last_move={dt_since_last_move:.3f}s queued_target=({smooth_x},{smooth_y})")
        else:
            # Trong vùng chết hoặc di chuyển quá nhỏ
            # Log dead zone cùng timestamp/latency để debug
            ts = datetime.datetime.fromtimestamp(call_wall).strftime('%H:%M:%S.%f')
            proc_time = time.perf_counter() - call_perf
            last_move = _last_move_time.get(hand_idx)
            dt_since_last_move = (call_perf - last_move) if (last_move is not None) else 0.0
            if is_jittering:
                print(f"[{ts}] [DEAD ZONE] dt_call={dt_since_last_call:.3f}s dt_last_move={dt_since_last_move:.3f}s proc={proc_time:.4f}s Jittering detected, no move (dist: {distance_normalized:.4f})")
    else:
        # Lần đầu tiên - chỉ lưu vị trí, không di chuyển
        ts = datetime.datetime.fromtimestamp(call_wall).strftime('%H:%M:%S.%f')
        proc_time = time.perf_counter() - call_perf
        print(f"[{ts}] [INIT] dt_call={dt_since_last_call:.3f}s proc={proc_time:.4f}s Mouse tracking initialized at finger pos ({int(screen_x)}, {int(screen_y)})")
        # Reset lịch sử
        _movement_history[hand_idx] = []
    
    # Lưu vị trí ngón tay hiện tại (dùng measurement đã làm mượt)
    previous_mouse_pos[hand_idx] = (meas_x, meas_y)

def get_action_func(pred_label):
    """
    Trả về function tương ứng từ pred_label.
    """
    action_map = {
        'clickchuotphai': execute_right_click,
        'clickchuottrai': execute_left_click,
        'dungchuongtrinh': execute_stop_program,
        'moapp': execute_open_app,
        'phongto': execute_zoom_in,
        'thunho': execute_zoom_out,
        'vuotlen': execute_scroll_up,
        'vuotxuong': execute_scroll_down,
        'vuotphai': execute_tab_next,
        'vuottrai': execute_tab_prev
    }
    return action_map.get(pred_label)

def execute_action(execute_func, pred_label, current_time, is_continuous: bool = False):
    """
    Execute action với cooldown. Nếu is_continuous=True thì bypass cooldown so the action
    can be executed repeatedly while the gesture persists (used for continuous scrolls).
    Returns: should_stop (bool)
    """
    global last_execution_times

    if not is_continuous:
        if pred_label in last_execution_times:
            time_since_last = current_time - last_execution_times[pred_label]
            if time_since_last < DISCRETE_COOLDOWN:
                return False

    # Execute action
    should_stop = execute_func()

    # Cập nhật thời gian execute cuối
    last_execution_times[pred_label] = current_time

    # Return True nếu cần dừng chương trình
    return should_stop if should_stop else False


def clear_actuator_target(hand_idx):
    """Xóa target actuator cho tay hand_idx (dừng actuator điều khiển con trỏ cho tay đó)."""
    try:
        with _actuator_lock:
            if hand_idx in _actuator_targets:
                del _actuator_targets[hand_idx]
    except Exception:
        pass


def pause_actuator_for(hand_idx, timeout=MANUAL_OVERRIDE_TIMEOUT):
    """Tạm dừng actuator cho hand_idx trong `timeout` giây."""
    try:
        _manual_override_until[hand_idx] = time.perf_counter() + float(timeout)
        # Also clear any existing target
        clear_actuator_target(hand_idx)
    except Exception:
        pass
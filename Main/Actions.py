# actions.py - Các hàm execute actions (gộp config: SCROLL_SENSITIVITY, SMOOTH_ALPHA, etc.)

import pyautogui
import time
from Model import GESTURE_TYPES  # Import GESTURE_TYPES từ model.py
# Gộp config actions
SCROLL_SENSITIVITY = 3.0  # Tăng cho mượt touchpad-like
SMOOTH_ALPHA = 0.5  # 0.5: 50% smooth; set 1.0 cho zero smooth (raw tay pos)
TAB_THRESHOLD = 0.05  # Threshold cho delta_x ở tab (tránh nhiễu)
DISCRETE_DELAY = 0.2

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

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
    if abs(delta_x) > TAB_THRESHOLD:
        pyautogui.hotkey('ctrl', 'tab')
        print(f"Executed: Tab next (delta_x: {delta_x:.2f})")
    else:
        print("Skip tab next: delta_x too small.")

def execute_tab_prev(delta_x):
    if abs(delta_x) > TAB_THRESHOLD:
        pyautogui.hotkey('ctrl', 'shift', 'tab')
        print(f"Executed: Tab prev (delta_x: {delta_x:.2f})")
    else:
        print("Skip tab prev: delta_x too small.")

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

def execute_mouse_action(curr_x_norm, curr_y_norm, previous_mouse_pos, hand_idx):
    """
    Xử lý di chuyển mouse (zero-delay smooth).
    """
    screen_w, screen_h = pyautogui.size()
    curr_x_screen = int((curr_x_norm + 1) * screen_w / 2)
    curr_y_screen = int((curr_y_norm + 1) * screen_h / 2)
    if previous_mouse_pos[hand_idx] is not None:
        prev_x, prev_y = previous_mouse_pos[hand_idx]
        smooth_x = int(prev_x + SMOOTH_ALPHA * (curr_x_screen - prev_x))
        smooth_y = int(prev_y + SMOOTH_ALPHA * (curr_y_screen - prev_y))
        pyautogui.moveTo(smooth_x, smooth_y)
    else:
        pyautogui.moveTo(curr_x_screen, curr_y_screen)
    previous_mouse_pos[hand_idx] = (curr_x_screen, curr_y_screen)
    print(f"Mouse zero-delay to ({curr_x_screen}, {curr_y_screen})")

def get_action_func(pred_label):
    """
    Trả về function tương ứng từ pred_label.
    """
    action_map = {
        'clickchuotphai': execute_right_click,
        'clickchuottrai': execute_left_click,
        'dungchuongtrinh': execute_stop_program,
        'mochorme': execute_open_chrome,
        'phongto': execute_zoom_in,
        'thunho': execute_zoom_out,
        'vuotlen': execute_scroll_up,
        'vuotxuong': execute_scroll_down,
        'vuotphai': execute_tab_next,
        'vuottrai': execute_tab_prev
    }
    return action_map.get(pred_label)

def execute_action(execute_func, pred_label, delta_x, delta_y, num_fingers, last_discrete_time, current_time):
    """
    Execute action dựa trên loại gesture.
    Returns: should_stop (bool)
    """
    gesture_type = GESTURE_TYPES.get(pred_label, 'discrete')
    if pred_label == 'dichuyenchuot':
        # Mouse xử lý riêng, không dùng execute_func
        return False
    elif gesture_type == 'continuous':
        if pred_label in ['vuotlen', 'vuotxuong']:
            execute_func(delta_y, num_fingers)
        elif pred_label in ['vuotphai', 'vuottrai']:
            execute_func(delta_x)
    else:  # Discrete
        if current_time - last_discrete_time >= DISCRETE_DELAY:
            should_stop = execute_func()
            return should_stop
    return False
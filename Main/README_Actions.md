# 📘 PHÂN TÍCH CHI TIẾT FILE `Actions.py`

## 🎯 TỔNG QUAN

**File:** `Actions.py`  
**Dòng code:** 730 dòng  
**Ngôn ngữ:** Python 3.x  
**Mục đích:** Module thực thi **TẤT CẢ** các hành động điều khiển máy tính (click chuột, scroll, mở app, nhập text, zoom, tab, di chuyển chuột)

### **Vai trò trong hệ thống:**
```
Main.py (Thread webcam/voice)
    ↓
Model.py (LSTM prediction) → "clickchuottrai", "dichuyenchuot", etc.
    ↓
Actions.py ← ĐÂY - Thực thi hành động
    ↓
pyautogui / subprocess → Điều khiển OS
```

---

## 📦 CẤU TRÚC FILE

### **1. IMPORTS (Dòng 1-15)**

```python
import os              # Kiểm tra file path
import time            # Timing, cooldown, performance
import math            # Tính toán (không dùng nhiều)
import datetime        # Format timestamp cho log
import threading       # Actuator thread cho di chuột mượt
import subprocess      # Mở ứng dụng/website
import getpass         # Lấy username hiện tại

import cv2             # KalmanFilter (OpenCV)
import numpy as np     # Ma trận, vector
import pyautogui       # Điều khiển chuột/bàn phím
import pyperclip       # Copy/paste text
```

**Ý nghĩa từng thư viện:**
- **pyautogui**: Core library để điều khiển chuột (click, scroll, moveTo) và bàn phím (hotkey, write)
- **cv2.KalmanFilter**: Làm mượt tọa độ chuột bằng thuật toán Kalman Filter
- **threading**: Tạo thread riêng cho actuator (di chuột 60 FPS)
- **subprocess**: Mở app/website bằng shell command
- **pyperclip**: Paste text nhanh hơn pyautogui.write()

---

## 🔧 CẤU HÌNH TOÀN CỤC (Dòng 17-43)

### **A. Voice Log Callback**

```python
_voice_log_callback = None  # Hàm callback để log vào GUI

def set_voice_log_callback(callback):
    global _voice_log_callback
    _voice_log_callback = callback
```

**Mục đích:** Gửi log từ Actions.py → Main.py → Voice Console GUI

**Cách hoạt động:**
```python
# Main.py gọi khi khởi động
Actions.set_voice_log_callback(add_voice_log)

# Trong Actions.py
log_action("✓ Click chuột trái")
    ↓
if _voice_log_callback:
    _voice_log_callback("✓ Click chuột trái")
    ↓
Voice Console GUI hiển thị
```

### **B. Cấu hình SMOOTH & COOLDOWN (Dòng 30-44)**

```python
SMOOTH_ALPHA = 0.7              # EMA smoothing cho cursor (70% mới, 30% cũ)
DISCRETE_COOLDOWN = 1.0         # Cooldown 1 giây cho hành động rời rạc
CONTINUOUS_SCROLL_STEP = 30     # Scroll 30px mỗi frame
CONTINUOUS_SCROLL_MAX_STEP = 900 # Giới hạn scroll tối đa 900px
```

**Ý nghĩa:**
- **SMOOTH_ALPHA = 0.7**: Khi di chuột, vị trí mới = 70% target + 30% vị trí hiện tại
- **DISCRETE_COOLDOWN = 1.0**: Sau khi click, phải đợi 1 giây mới click lại được (tránh spam)
- **CONTINUOUS_SCROLL_STEP = 30**: Mỗi frame scroll 30 pixels → mượt mà

### **C. Cấu hình DI CHUYỂN CHUỘT (Dòng 37-43)**

```python
MOUSE_DEAD_ZONE = 0.03           # Vùng chết 3% màn hình
MOUSE_SPEED_MULTIPLIER = 6       # Tốc độ chuột x6
MOUSE_MAX_MOVE = 100             # Di chuyển tối đa 100px/frame
MOUSE_MAX_SPEED_PX_PER_SEC = 600.0  # Tốc độ tối đa 600 px/s
ACTUATION_HZ = 60                # Actuator chạy 60 FPS
```

**Công thức tính toán:**

#### **1. Dead Zone (Vùng chết - Chống rung)**
```python
distance_normalized = distance / sqrt(screen_w² + screen_h²)

if distance_normalized < MOUSE_DEAD_ZONE:  # < 3%
    # KHÔNG DI CHUYỂN - Chặn rung nhỏ
```

**Ví dụ:**
- Màn hình 1920×1080 → đường chéo = √(1920² + 1080²) = 2203 px
- Dead zone = 2203 × 0.03 = **66 pixels**
- Di chuyển < 66px → BỊ CHẶN (tránh rung)

#### **2. Speed Multiplier**
```python
move_x = delta_x * MOUSE_SPEED_MULTIPLIER
move_y = delta_y * MOUSE_SPEED_MULTIPLIER

# Nếu ngón tay di 10px → cursor di 10 × 6 = 60px
```

#### **3. Max Move (Giới hạn nhảy)**
```python
move_distance = sqrt(move_x² + move_y²)
if move_distance > MOUSE_MAX_MOVE:  # > 100px
    scale = MOUSE_MAX_MOVE / move_distance
    move_x *= scale
    move_y *= scale
```

**Mục đích:** Tránh cursor nhảy quá xa khi tay cử động nhanh

#### **4. Actuator Speed Limit**
```python
allowed = MOUSE_MAX_SPEED_PX_PER_SEC * period  # 600 × (1/60) = 10 px/tick
step = min(dist, allowed)
```

**Mục đích:** Cursor di chuyển mượt mà, không nhảy cóc

### **D. PyAutoGUI Settings (Dòng 44-45)**

```python
pyautogui.FAILSAFE = True   # Di chuột vào góc (0,0) → Dừng chương trình
pyautogui.PAUSE = 0.0       # Không delay giữa các lệnh
```

---

## 🎮 CÁC HÀM EXECUTE_* (DISCRETE ACTIONS)

### **1. execute_right_click() (Dòng 50-53)**

```python
def execute_right_click():
    pyautogui.rightClick()
    log_action("✓ Thực thi: Click chuột phải")
    return False
```

**Input:** Không có  
**Output:** `False` (không dừng chương trình)  
**Hành động:** Click chuột phải tại vị trí hiện tại  
**Công dụng:** Mở context menu (right-click menu)

**Cách gọi:**
```python
# Từ Main.py
execute_func = get_action_func('clickchuotphai')
execute_func()  # → execute_right_click()
```

### **2. execute_left_click() (Dòng 55-58)**

```python
def execute_left_click():
    pyautogui.leftClick()
    log_action("✓ Thực thi: Click chuột trái")
    return False
```

**Tương tự execute_right_click()** nhưng cho click trái

### **3. execute_stop_program() (Dòng 60-62)**

```python
def execute_stop_program():
    log_action("! Dừng chương trình - Thoát hệ thống")
    return True  # ← QUAN TRỌNG!
```

**Output:** `True` (dừng chương trình)  
**Đặc biệt:** Đây là **HÀM DUY NHẤT** trả về `True`

**Luồng xử lý:**
```python
# Main.py
should_stop = execute_action(execute_stop_program, ...)
if should_stop:  # True
    with stop_lock:
        should_stop = True  # Dừng cả 2 threads
```

### **4. APP_DATABASE (Dòng 64-145)**

**Cấu trúc database:**
```python
APP_DATABASE = {
    'app_id': {
        'display_name': str,           # Tên hiển thị
        'paths': [str, ...],           # Đường dẫn .exe/.lnk
        'keywords': [str, ...],        # Từ khóa tìm kiếm
        'url': str (optional)          # URL cho website
    }
}
```

**Ví dụ cụ thể:**
```python
'chrome': {
    'display_name': 'Google Chrome',
    'paths': [
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
    ],
    'keywords': ['chrome', 'trình duyệt chrome', 'google chrome']
}
```

**Cách hoạt động:**
1. User nói: **"mở chrome"**
2. `find_app_by_keyword("mở chrome")` → tìm thấy `'chrome'`
3. Duyệt qua `paths` → tìm path tồn tại
4. `subprocess.Popen([path])` → Mở Chrome

**Website entry:**
```python
'facebook': {
    'display_name': 'Facebook',
    'url': 'https://www.facebook.com',
    'keywords': ['facebook', 'face', 'fb']
}
```

**Cách mở website:**
```python
subprocess.Popen(['cmd', '/c', 'start', url], shell=True)
# Windows sẽ mở URL bằng browser mặc định
```

### **5. find_app_by_keyword() (Dòng 147-160)**

```python
def find_app_by_keyword(text):
    if not text:
        return None, None
    
    text_lower = text.lower()  # "MỞ CHROME" → "mở chrome"
    
    for app_name, config in APP_DATABASE.items():
        keywords = config.get('keywords', [])
        for keyword in keywords:
            if keyword in text_lower:  # "chrome" in "mở chrome"
                return app_name, config
    
    return None, None
```

**Input:** `"mở chrome"` (string)  
**Output:** `('chrome', {...config...})` hoặc `(None, None)`  
**Thuật toán:** Linear search qua keywords

**Ví dụ:**
```python
find_app_by_keyword("mở chrome")
# Loop:
#   'chrome' keywords = ['chrome', 'trình duyệt chrome', ...]
#   'chrome' in 'mở chrome' → TRUE
# Return: ('chrome', {...})
```

### **6. execute_open_app() (Dòng 168-242)**

**Signature:**
```python
def execute_open_app(app_name=None) -> bool
```

**Parameters:**
- `app_name`: Text từ voice command hoặc gesture prediction
  - Nếu `None` → Trả về `False` (cần input thêm)
  - Nếu có → Tìm và mở app

**Luồng xử lý chi tiết:**

#### **Step 1: Cooldown Check**
```python
_last_open_app_time = {}  # Tracking thời gian mở app
_OPEN_APP_COOLDOWN = 2.0  # 2 giây

if found_app:
    current_time = time.perf_counter()
    last_time = _last_open_app_time.get(found_app, 0)
    if current_time - last_time < 2.0:
        # BỎ QUA - Mới mở app này rồi
        return False
    _last_open_app_time[found_app] = current_time
```

**Mục đích:** Tránh mở Chrome 5 lần khi voice recognition nhận "mở chrome" nhiều lần

#### **Step 2: Find App**
```python
found_app, config = find_app_by_keyword(app_name)
if not found_app:
    log_action(f"✗ Không tìm thấy '{app_name}'")
    return False
```

#### **Step 3: Open Website**
```python
if 'url' in config:
    url = config['url']
    log_action(f">> Đang mở website: {display_name}...")
    subprocess.Popen(['cmd', '/c', 'start', url], shell=True)
    log_action(f"✓ Đã mở website: {display_name}")
    return False
```

**Giải thích command:**
- `cmd /c start URL` → Windows shell mở URL bằng browser mặc định
- `shell=True` → Chạy qua shell để interpret command

#### **Step 4: Open Application**
```python
paths = config.get('paths', [])
username = getpass.getuser()  # Lấy username: "hoang"
paths = [p.format(username) if '{}' in p else p for p in paths]
```

**Ví dụ path substitution:**
```python
# Trước:
r'C:\Users\{}\AppData\Local\Programs\...'

# Sau:
r'C:\Users\hoang\AppData\Local\Programs\...'
```

**Tìm path tồn tại:**
```python
app_path = None
for path in paths:
    if os.path.exists(path):
        app_path = path
        break
```

**Mở app:**
```python
if app_path.lower().endswith('.lnk'):
    # Shortcut file
    subprocess.Popen(['cmd', '/c', 'start', '', app_path], shell=True)
else:
    # .exe file
    subprocess.Popen([app_path], shell=False)
```

**Phân biệt .lnk và .exe:**
- **.lnk** (shortcut): Phải dùng `start` command
- **.exe**: Chạy trực tiếp

### **7. execute_zoom_in/out() (Dòng 244-252)**

```python
def execute_zoom_in():
    pyautogui.hotkey('ctrl', '+')
    log_action("✓ Thực thi: Phóng to (Ctrl +)")
    return False

def execute_zoom_out():
    pyautogui.hotkey('ctrl', '-')
    log_action("✓ Thực thi: Thu nhỏ (Ctrl -)")
    return False
```

**Công dụng:**
- Browser: Phóng to/thu nhỏ trang web
- PDF viewer: Zoom in/out document
- Image viewer: Phóng to/thu nhỏ ảnh

### **8. execute_tab_next/prev() (Dòng 254-264)**

```python
def execute_tab_next():
    pyautogui.hotkey('ctrl', 'tab')  # Next tab
    
def execute_tab_prev():
    pyautogui.hotkey('ctrl', 'shift', 'tab')  # Previous tab
```

**Hotkeys:**
- `Ctrl+Tab` → Chuyển sang tab bên phải
- `Ctrl+Shift+Tab` → Chuyển sang tab bên trái

### **9. execute_type_text() (Dòng 266-299)**

```python
def execute_type_text(text_content=None):
    if text_content is None:
        return False  # Cần input thêm
    
    if not text_content:
        log_action("✗ Không có nội dung")
        return False
    
    # Format: Viết hoa chữ đầu
    formatted_text = text_content[0].upper() + text_content[1:]
    
    try:
        pyperclip.copy(formatted_text)  # Copy vào clipboard
        pyautogui.hotkey('ctrl', 'v')   # Paste
        log_action(f"✓ Đã nhập văn bản: '{formatted_text}'")
    except ImportError:
        # Fallback: type trực tiếp
        pyautogui.write(formatted_text)
```

**Tại sao dùng pyperclip thay vì pyautogui.write()?**
- **pyperclip + Ctrl+V**: **NHANH** (~0.01s cho 100 ký tự)
- **pyautogui.write()**: **CHẬM** (~2s cho 100 ký tự, type từng ký tự)

**Ví dụ:**
```python
# Voice: "nhập văn bản" → "xin chào"
execute_type_text("xin chào")
    ↓
formatted_text = "Xin chào"  # Viết hoa X
    ↓
pyperclip.copy("Xin chào")
    ↓
pyautogui.hotkey('ctrl', 'v')
    ↓
Text xuất hiện tại cursor position
```

---

## 🔄 CONTINUOUS ACTIONS (SCROLL)

### **10. execute_scroll_up/down() (Dòng 301-348)**

**Đặc điểm:** Hàm này được gọi **MỖI FRAME** khi gesture duy trì

#### **A. execute_scroll_up() (Dòng 301-326)**

```python
_last_scroll_time = {'up': 0.0, 'down': 0.0}  # Global tracking

def execute_scroll_up():
    now = time.perf_counter()
    prev = _last_scroll_time.get('up', 0.0)
    dt = now - prev if prev else 0.0  # Delta time
    _last_scroll_time['up'] = now
    
    # Tính multiplier dựa trên dt
    multiplier = min(3.0, max(1.0, dt * 30.0))
    step = int(min(CONTINUOUS_SCROLL_MAX_STEP, 
                   CONTINUOUS_SCROLL_STEP * multiplier))
    
    pyautogui.scroll(step)  # Scroll LÊN (dương)
    return False
```

**Công thức multiplier:**
```
multiplier = clamp(dt × 30, 1.0, 3.0)
```

**Ý nghĩa:**
- `dt = 0.033s` (30 FPS) → `multiplier = 0.033 × 30 = 0.99 ≈ 1.0`
- `dt = 0.1s` (10 FPS - lag) → `multiplier = 0.1 × 30 = 3.0` (MAX)

**Mục đích:** Khi FPS thấp, tăng step để tốc độ scroll không bị chậm

**Ví dụ thực tế:**
```
Frame 1: dt=0.033s → step = 30 × 1.0 = 30px
Frame 2: dt=0.033s → step = 30 × 1.0 = 30px
Frame 3: dt=0.1s (lag!) → step = 30 × 3.0 = 90px  ← Bù lag
Frame 4: dt=0.033s → step = 30 × 1.0 = 30px
```

#### **B. execute_scroll_down() (Dòng 328-348)**

**Giống execute_scroll_up()** nhưng `pyautogui.scroll(-step)` (âm = xuống)

---

## 🎛️ SMOOTHING MECHANISMS

### **11. BIẾN TOÀN CỤC CHO SMOOTHING (Dòng 350-395)**

```python
# Lịch sử di chuyển
_movement_history = {}  # hand_idx -> [(dx,dy), ...]

# Timing
_last_call_time = {}    # hand_idx -> time của lần gọi trước
_last_move_time = {}    # hand_idx -> time của lần move trước
_last_scroll_time = {'up': 0.0, 'down': 0.0}

# Actuator
_actuator_targets = {}  # hand_idx -> (target_x, target_y)
_actuator_lock = threading.Lock()
_actuator_thread = None
_actuator_running = False

# Manual override detection
_last_target_update = {}      # hand_idx -> time của lần set target
_manual_override_until = {}   # hand_idx -> time tạm dừng đến
_last_actuator_pos = {}       # hand_idx -> (x,y) vị trí actuator cuối

# EMA smoothing
_finger_smoothed = {}         # hand_idx -> (x_px, y_px) đã smooth
FINGER_SMOOTH_ALPHA = 0.92    # 92% mới, 8% cũ

# Kalman filters
_kalman_filters = {}          # hand_idx -> KalmanFilter2D instance
```

**Giải thích từng biến:**

#### **_movement_history**
```python
_movement_history[0] = [
    (dx1, dy1),  # Frame 1
    (dx2, dy2),  # Frame 2
    (dx3, dy3),  # Frame 3
    (dx4, dy4),  # Frame 4
    (dx5, dy5)   # Frame 5 (mới nhất)
]
```
**Mục đích:** Phát hiện rung lắc (jittering) bằng cách xem delta có đổi hướng liên tục không

#### **_actuator_targets**
```python
_actuator_targets[0] = (1250.5, 680.3)  # Target cursor position
```
**Mục đích:** Actuator thread di chuyển cursor từ vị trí hiện tại → target với tốc độ 600px/s

#### **_finger_smoothed**
```python
_finger_smoothed[0] = (1248.7, 678.9)  # EMA smoothed finger position
```
**Mục đích:** Làm mượt tọa độ ngón tay trước khi cho vào Kalman

---

## 🧮 KALMAN FILTER (Dòng 399-430)

### **Class KalmanFilter2D**

```python
class KalmanFilter2D:
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 states, 2 measurements
```

#### **State Vector (4D):**
```
x = [x, y, vx, vy]ᵀ
```
- `x, y`: Vị trí cursor
- `vx, vy`: Vận tốc (pixel/frame)

#### **Transition Matrix (Predict):**
```python
self.kf.transitionMatrix = np.array([
    [1, 0, 1, 0],  # x_new = x_old + vx_old
    [0, 1, 0, 1],  # y_new = y_old + vy_old
    [0, 0, 1, 0],  # vx_new = vx_old
    [0, 0, 0, 1]   # vy_new = vy_old
], dtype=np.float32)
```

**Mô hình:** **Constant Velocity** (vận tốc không đổi)

#### **Measurement Matrix (Update):**
```python
self.kf.measurementMatrix = np.array([
    [1, 0, 0, 0],  # Chỉ đo x (không đo vận tốc)
    [0, 1, 0, 0]   # Chỉ đo y
], dtype=np.float32)
```

#### **Noise Covariances:**
```python
self.kf.processNoiseCov = np.eye(4) * 1e-2  # Model không hoàn hảo
self.kf.measurementNoiseCov = np.eye(2) * 1e-1  # Sensor có noise
```

**Ý nghĩa:**
- **Process noise = 0.01**: Model vận tốc không đổi sai ~1%
- **Measurement noise = 0.1**: MediaPipe tracking sai ~10%

#### **Các phương thức:**

```python
def initialize(self, x, y):
    # Khởi tạo state = [x, y, 0, 0]
    self.kf.statePost = np.array([[x], [y], [0.0], [0.0]])
    self.initialized = True

def predict(self):
    # Dự đoán vị trí tiếp theo dựa trên vận tốc
    pred = self.kf.predict()
    return float(pred[0]), float(pred[1])

def update(self, x, y):
    # Cập nhật với measurement mới
    meas = np.array([[x], [y]])
    post = self.kf.correct(meas)
    return float(post[0]), float(post[1])
```

**Luồng hoạt động:**
```
Frame 1: initialize(500, 300)
         state = [500, 300, 0, 0]

Frame 2: predict() → [500, 300] (vx=0, vy=0)
         Đo được: (510, 305)
         update(510, 305) → [508, 303, 8, 3]
         ↑ Kalman smoothing + ước lượng vận tốc

Frame 3: predict() → [516, 306] (dự đoán từ vận tốc)
         Đo được: (520, 310)
         update(520, 310) → [518, 308, 10, 5]
```

---

## 🤖 ACTUATOR THREAD (Dòng 432-492)

### **Mục đích:**
Di chuyển cursor **mượt mà 60 FPS** ngay cả khi Main thread chạy chậm (15-25 FPS)

### **_actuator_loop() (Dòng 432-492)**

```python
def _actuator_loop():
    global _actuator_running
    period = 1.0 / ACTUATION_HZ  # 1/60 = 0.0167s
    
    while _actuator_running:
        with _actuator_lock:
            items = list(_actuator_targets.items())
        
        if not items:
            time.sleep(period)
            continue
```

**Bước 1: Lấy targets (thread-safe)**
```python
items = [(hand_idx, (target_x, target_y)), ...]
```

**Bước 2: Kiểm tra manual override**
```python
if _manual_override_until.get(hand_idx, 0) > now:
    # Actuator bị tạm dừng (user dùng chuột thật)
    _last_actuator_pos[hand_idx] = (cx, cy)
    continue
```

**Bước 3: Phát hiện di chuyển bằng chuột thật**
```python
prev_act_pos = _last_actuator_pos.get(hand_idx)
delta_user = sqrt((cx - prev_x)² + (cy - prev_y)²)

if delta_user > 40 and (now - last_target_time) > 0.05:
    # User di chuột thật → Pause actuator 1 giây
    _manual_override_until[hand_idx] = now + 1.0
    del _actuator_targets[hand_idx]
    continue
```

**Mục đích:** Khi user cầm chuột thật di chuyển → Tạm ngừng actuator để không tranh nhau

**Bước 4: Di chuyển từng bước**
```python
dx = target_x - current_x
dy = target_y - current_y
dist = sqrt(dx² + dy²)

if dist < 1.0:
    continue  # Đã đến target

# Giới hạn tốc độ
allowed = 600 * (1/60) = 10 pixels/tick
step = min(dist, allowed)

# Di chuyển
nx = cx + (dx/dist) * step
ny = cy + (dy/dist) * step
pyautogui.moveTo(int(nx), int(ny))
```

**Ví dụ:**
```
Target: (1000, 500)
Current: (800, 400)

Tick 1: dx=200, dy=100, dist=224
        step = min(224, 10) = 10
        nx = 800 + (200/224)*10 = 808.9
        ny = 400 + (100/224)*10 = 404.5
        moveTo(809, 405)

Tick 2: dx=191, dy=95, dist=213
        step = min(213, 10) = 10
        nx = 809 + (191/213)*10 = 817.9
        moveTo(818, 409)

... (tiếp tục đến khi dist < 1)
```

**Bước 5: Sleep để duy trì 60 Hz**
```python
time.sleep(period)  # 0.0167s
```

### **start_actuator() (Dòng 495-502)**

```python
def start_actuator():
    global _actuator_thread, _actuator_running
    if _actuator_thread is not None:
        return
    _actuator_running = True
    _actuator_thread = threading.Thread(target=_actuator_loop, daemon=True)
    _actuator_thread.start()

# Auto-start khi import
start_actuator()
```

**daemon=True:** Thread tự động tắt khi main program thoát

---

## 🖱️ EXECUTE_MOUSE_TO_POINT (Dòng 509-656)

### **Signature:**
```python
def execute_mouse_to_point(screen_x, screen_y, previous_mouse_pos, hand_idx, smooth_alpha=None)
```

**Parameters:**
- `screen_x, screen_y`: Tọa độ ngón trỏ (pixel màn hình)
- `previous_mouse_pos`: List lưu vị trí trước đó `[hand0_pos, hand1_pos]`
- `hand_idx`: 0 = tay chính, 1 = tay phụ
- `smooth_alpha`: Hệ số EMA (default = 0.7)

**Return:** None (cập nhật `previous_mouse_pos` in-place)

### **LUỒNG XỬ LÝ CHI TIẾT:**

#### **Step 1: Timing (Dòng 520-524)**
```python
call_perf = time.perf_counter()
prev_call = _last_call_time.get(hand_idx)
dt_since_last_call = call_perf - prev_call  # Delta time
_last_call_time[hand_idx] = call_perf
```

**Mục đích:** Đo thời gian giữa 2 lần gọi để debug độ trễ

#### **Step 2: Lấy vị trí cursor hiện tại (Dòng 527)**
```python
current_mouse_x, current_mouse_y = pyautogui.position()
```

#### **Step 3: EMA Smoothing cho ngón tay (Dòng 529-541)**
```python
prev_fs = _finger_smoothed.get(hand_idx)

if prev_fs is None:
    # Lần đầu
    _finger_smoothed[hand_idx] = (screen_x, screen_y)
    meas_x, meas_y = screen_x, screen_y
else:
    # EMA: α = 0.92
    meas_x = 0.92 * screen_x + 0.08 * prev_fs[0]
    meas_y = 0.92 * screen_y + 0.08 * prev_fs[1]
    _finger_smoothed[hand_idx] = (meas_x, meas_y)
```

**Công thức EMA:**
$$\text{smoothed}_t = \alpha \cdot \text{measured}_t + (1-\alpha) \cdot \text{smoothed}_{t-1}$$

**Với α = 0.92:**
$$\text{smoothed}_t = 0.92 \cdot \text{measured}_t + 0.08 \cdot \text{smoothed}_{t-1}$$

**Ví dụ:**
```
Frame 1: measured = (500, 300)
         smoothed = (500, 300)

Frame 2: measured = (510, 305)
         smoothed = 0.92×(510,305) + 0.08×(500,300)
                  = (469.2+40, 280.6+24)
                  = (509.2, 304.6)
```

**Đặc điểm α = 0.92:**
- **Responsive cao** (92% mới)
- **Loại bỏ jitter** nhỏ (8% cũ)

#### **Step 4: Kalman Filter (Dòng 543-558)**
```python
kf = _kalman_filters.get(hand_idx)

if kf is None:
    # Khởi tạo
    kf = KalmanFilter2D()
    kf.initialize(meas_x, meas_y)
    _kalman_filters[hand_idx] = kf
else:
    # Dự đoán và cập nhật
    kf.predict()
    meas_x, meas_y = kf.update(meas_x, meas_y)
```

**Kết quả:** `(meas_x, meas_y)` đã qua **2 lớp smoothing**: EMA → Kalman

#### **Step 5: Tính Delta (Dòng 560-568)**
```python
if previous_mouse_pos[hand_idx] is not None:
    prev_finger_x, prev_finger_y = previous_mouse_pos[hand_idx]
    
    delta_x = meas_x - prev_finger_x
    delta_y = meas_y - prev_finger_y
    distance = sqrt(delta_x² + delta_y²)
```

#### **Step 6: Lưu lịch sử di chuyển (Dòng 570-576)**
```python
if hand_idx not in _movement_history:
    _movement_history[hand_idx] = []

_movement_history[hand_idx].append((delta_x, delta_y))
if len(_movement_history[hand_idx]) > 5:
    _movement_history[hand_idx].pop(0)  # Giữ 5 frame mới nhất
```

#### **Step 7: Jitter Detection (Dòng 578-593)**

**Thuật toán phát hiện rung lắc:**
```python
if len(_movement_history[hand_idx]) >= 5:
    # Tính tổng vector
    sum_x = sum(h[0] for h in history[-5:])
    sum_y = sum(h[1] for h in history[-5:])
    sum_distance = sqrt(sum_x² + sum_y²)
    
    # Tính tổng khoảng cách tuyệt đối
    total_distance = sum(sqrt(h[0]² + h[1]²) for h in history[-5:])
    
    # Nếu tổng vector << tổng khoảng cách → Rung lắc!
    if total_distance > 0 and sum_distance / total_distance < 0.5:
        is_jittering = True
```

**Ví dụ rung lắc:**
```
Frame 1: delta = (+10, +5)
Frame 2: delta = (-8, -4)   ← Đổi hướng!
Frame 3: delta = (+9, +6)   ← Đổi hướng!
Frame 4: delta = (-10, -5)  ← Đổi hướng!
Frame 5: delta = (+8, +4)   ← Đổi hướng!

sum_vector = (+10-8+9-10+8, +5-4+6-5+4) = (9, 6)
sum_distance = sqrt(9² + 6²) = 10.8

total_distance = 11.2 + 8.9 + 10.8 + 11.2 + 8.9 = 51
Ratio = 10.8 / 51 = 0.21 < 0.5 → RUNG LẮC!
```

**Kiểm tra thêm:**
```python
if not is_jittering and total_distance < 8.0:
    is_jittering = True  # Di chuyển nhỏ liên tục cũng là rung
```

#### **Step 8: Dead Zone Check (Dòng 595-606)**
```python
distance_normalized = distance / sqrt(screen_w² + screen_h²)

should_move = True

if is_jittering and distance_normalized < 0.03:  # 3%
    should_move = False  # Chặn rung trong dead zone
elif distance < 4.0:  # < 4 pixels
    should_move = False  # Chặn di chuyển quá nhỏ
```

**Logic:**
- Nếu **RUNG** và di chuyển < 3% màn hình → **KHÔNG DI**
- Nếu di chuyển < 4 pixels (dù không rung) → **KHÔNG DI**

#### **Step 9: Apply Speed & Smoothing (Dòng 608-634)**
```python
if should_move and distance > 1:
    # Speed multiplier
    move_x = delta_x * 6
    move_y = delta_y * 6
    
    # Giới hạn max move
    move_distance = sqrt(move_x² + move_y²)
    if move_distance > 100:
        scale = 100 / move_distance
        move_x *= scale
        move_y *= scale
    
    # Tính vị trí mới
    new_mouse_x = current_mouse_x + move_x
    new_mouse_y = current_mouse_y + move_y
    
    # EMA smoothing
    smooth_x = int(current_x + 0.7 * (new_x - current_x))
    smooth_y = int(current_y + 0.7 * (new_y - current_y))
    
    # Clamp vào màn hình
    smooth_x = max(0, min(screen_w - 1, smooth_x))
    smooth_y = max(0, min(screen_h - 1, smooth_y))
```

#### **Step 10: Cập nhật Actuator Target (Dòng 636-644)**
```python
with _actuator_lock:
    _actuator_targets[hand_idx] = (float(smooth_x), float(smooth_y))

_last_target_update[hand_idx] = call_perf
if hand_idx in _manual_override_until:
    _manual_override_until.pop(hand_idx)
```

**Không di chuyển trực tiếp** mà **set target** cho actuator thread

#### **Step 11: Lưu vị trí ngón tay (Dòng 654-656)**
```python
previous_mouse_pos[hand_idx] = (meas_x, meas_y)
```

---

## 🔗 GET_ACTION_FUNC & EXECUTE_ACTION (Dòng 658-697)

### **get_action_func() (Dòng 658-672)**
```python
def get_action_func(pred_label):
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
        'vuottrai': execute_tab_prev,
        'nhapvanban': execute_type_text
    }
    return action_map.get(pred_label)
```

**Input:** `"clickchuottrai"` (string)  
**Output:** `execute_left_click` (function object)

### **execute_action() (Dòng 674-697)**
```python
def execute_action(execute_func, pred_label, current_time, is_continuous=False):
    global last_execution_times
    
    # Cooldown check (chỉ cho discrete)
    if not is_continuous:
        if pred_label in last_execution_times:
            time_since_last = current_time - last_execution_times[pred_label]
            if time_since_last < 1.0:
                return False  # Bỏ qua - Cooldown active
    
    # Execute
    should_stop = execute_func()
    
    # Update time
    last_execution_times[pred_label] = current_time
    
    return should_stop
```

**Parameters:**
- `execute_func`: Function cần gọi
- `pred_label`: Tên action (để tracking cooldown)
- `current_time`: `time.perf_counter()`
- `is_continuous`: True = không có cooldown (scroll, di chuột)

**Ví dụ:**
```python
# Discrete action
execute_action(execute_left_click, 'clickchuottrai', time.perf_counter())
# → Click 1 lần
# → Phải đợi 1 giây mới click lại

# Continuous action
execute_action(execute_scroll_up, 'vuotlen', time.perf_counter(), is_continuous=True)
# → Scroll mỗi frame, không cooldown
```

---

## 🛠️ UTILITY FUNCTIONS (Dòng 700-730)

### **clear_actuator_target() (Dòng 700-707)**
```python
def clear_actuator_target(hand_idx):
    try:
        with _actuator_lock:
            if hand_idx in _actuator_targets:
                del _actuator_targets[hand_idx]
    except Exception:
        pass
```

**Mục đích:** Dừng actuator di chuyển cursor cho tay `hand_idx`

**Khi nào dùng:**
- User giơ 4-5 ngón (vuốt gesture) → Clear target tay chính
- Phát hiện click gesture → Clear để tránh di chuyển cursor

### **pause_actuator_for() (Dòng 710-718)**
```python
def pause_actuator_for(hand_idx, timeout=1.0):
    try:
        _manual_override_until[hand_idx] = time.perf_counter() + timeout
        clear_actuator_target(hand_idx)
    except Exception:
        pass
```

**Mục đích:** Tạm dừng actuator trong `timeout` giây

**Ví dụ:**
```python
# Main.py
if primary_fingers >= 4:  # Vuốt 4 ngón
    pause_actuator_for(0, timeout=0.8)  # Pause 0.8s
```

---

## 📊 BẢNG TỔNG HỢP CÔNG THỨC

| Công thức | Biểu thức | Mục đích |
|-----------|-----------|----------|
| **EMA (ngón tay)** | $s_t = 0.92 m_t + 0.08 s_{t-1}$ | Làm mượt tọa độ ngón trỏ |
| **EMA (cursor)** | $c_t = c_{t-1} + 0.7(n_t - c_{t-1})$ | Làm mượt di chuyển cursor |
| **Dead Zone** | $\frac{d}{\sqrt{w^2+h^2}} < 0.03$ | Chặn rung nhỏ |
| **Speed Multiplier** | $\vec{m} = 6 \times \vec{d}$ | Tăng tốc cursor |
| **Max Move** | $\vec{m}' = \min(\|\vec{m}\|, 100) \frac{\vec{m}}{\|\vec{m}\|}$ | Giới hạn nhảy |
| **Actuator Speed** | $step = \min(dist, 600 \times \frac{1}{60})$ | Giới hạn tốc độ |
| **Jitter Ratio** | $\frac{\|\sum \vec{d_i}\|}{\sum \|\vec{d_i}\|} < 0.5$ | Phát hiện rung |
| **Scroll Multiplier** | $m = \text{clamp}(dt \times 30, 1, 3)$ | Bù lag scroll |
| **Kalman Predict** | $\vec{x}_t = \mathbf{F}\vec{x}_{t-1}$ | Dự đoán vị trí |
| **Kalman Update** | $\vec{x}_t = \vec{x}_t + \mathbf{K}(\vec{z}_t - \mathbf{H}\vec{x}_t)$ | Cập nhật với đo |

---

## 🔄 FLOWCHART TỔNG QUÁT

```
┌─────────────────────────────────────────────┐
│     Main.py - LSTM Predict Action           │
│     → "clickchuottrai" / "dichuyenchuot"    │
└──────────────┬──────────────────────────────┘
               ↓
       ┌───────────────┐
       │ Action Type?  │
       └───┬───────┬───┘
           │       │
    DISCRETE    CONTINUOUS
       ↓           ↓
┌──────────────┐ ┌─────────────────┐
│ Cooldown?    │ │ execute_mouse_  │
│ > 1.0s       │ │ to_point()      │
└──┬───────────┘ └─────┬───────────┘
   │                   │
   ↓                   ↓
┌──────────────┐ ┌─────────────────┐
│ execute_     │ │ EMA → Kalman    │
│ func()       │ │ → Jitter Check  │
│ • click()    │ │ → Dead Zone     │
│ • hotkey()   │ │ → Speed×6       │
│ • Popen()    │ │ → Set Target    │
└──────────────┘ └─────┬───────────┘
                       ↓
                 ┌─────────────────┐
                 │ Actuator Thread │
                 │ (60 FPS)        │
                 │ → moveTo()      │
                 └─────────────────┘
```

---

## 📝 VÍ DỤ SỬ DỤNG THỰC TẾ

### **Ví dụ 1: Click chuột trái**
```python
# Main.py
pred_label = "clickchuottrai"
execute_func = get_action_func(pred_label)
# → execute_func = execute_left_click

should_stop = execute_action(execute_func, pred_label, time.perf_counter())
# → Gọi execute_left_click()
#   → pyautogui.leftClick()
#   → log_action("✓ Click trái")
#   → return False

# Nếu click lại trong 1 giây:
should_stop = execute_action(execute_func, pred_label, time.perf_counter())
# → Cooldown active → return False (không click)
```

### **Ví dụ 2: Di chuyển chuột**
```python
# Main.py - Mỗi frame
if pred_label == 'dichuyenchuot':
    lm8 = hand_landmarks.landmark[8]  # Ngón trỏ
    screen_x = lm8.x * screen_w
    screen_y = lm8.y * screen_h
    
    execute_mouse_to_point(screen_x, screen_y, previous_mouse_pos, hand_idx=0)
    # ↓
    # EMA: (1000, 500) → (1002, 503) [smoothed]
    # ↓
    # Kalman: predict(1002, 503) → update() → (1001, 502) [filtered]
    # ↓
    # Delta: (1001-prev_x, 502-prev_y) = (5, 3)
    # ↓
    # Jitter check: OK (không rung)
    # ↓
    # Dead zone: 5px > 4px → OK
    # ↓
    # Speed: (5×6, 3×6) = (30, 18)
    # ↓
    # Max move: 30² + 18² = 36² < 100 → OK
    # ↓
    # New pos: current + (30, 18) = (850, 450)
    # ↓
    # EMA smooth: (850 + 0.7×30, 450 + 0.7×18) = (871, 463)
    # ↓
    # Set actuator target: (871, 463)
    # ↓
    # Actuator thread: moveTo(871, 463) with 600px/s speed
```

### **Ví dụ 3: Scroll liên tục**
```python
# Main.py - Frame 1
execute_scroll_up()
# → dt = 0 (lần đầu)
# → multiplier = 1.0
# → step = 30 × 1.0 = 30px
# → pyautogui.scroll(30)

# Frame 2 (sau 0.033s)
execute_scroll_up()
# → dt = 0.033s
# → multiplier = 0.033 × 30 = 0.99 ≈ 1.0
# → step = 30px
# → scroll(30)

# Frame 3 (lag - sau 0.1s)
execute_scroll_up()
# → dt = 0.1s
# → multiplier = 0.1 × 30 = 3.0 (max)
# → step = 30 × 3.0 = 90px ← Bù lag!
# → scroll(90)
```

### **Ví dụ 4: Mở Chrome**
```python
# Main.py - Voice command: "mở chrome"
execute_open_app("mở chrome")
# ↓
# find_app_by_keyword("mở chrome")
#   → 'chrome' in "mở chrome" → found!
#   → return ('chrome', {...config...})
# ↓
# Cooldown check: OK (chưa mở Chrome gần đây)
# ↓
# paths = [r'C:\Program Files\Google\Chrome\...\chrome.exe', ...]
# ↓
# os.path.exists(paths[0]) → True
# ↓
# subprocess.Popen([r'C:\...\chrome.exe'])
# ↓
# Chrome mở!
```

---

## 🎯 TÓM TẮT CHỨC NĂNG

| Function | Type | Input | Output | Cooldown | Mục đích |
|----------|------|-------|--------|----------|----------|
| `execute_left_click()` | Discrete | - | False | 1.0s | Click trái |
| `execute_right_click()` | Discrete | - | False | 1.0s | Click phải |
| `execute_stop_program()` | Discrete | - | True | - | Thoát |
| `execute_open_app()` | Discrete | app_name | False | 2.0s | Mở app/web |
| `execute_zoom_in/out()` | Discrete | - | False | 1.0s | Zoom |
| `execute_tab_next/prev()` | Discrete | - | False | 1.0s | Chuyển tab |
| `execute_type_text()` | Discrete | text | False | 1.0s | Nhập text |
| `execute_scroll_up/down()` | Continuous | - | False | None | Scroll |
| `execute_mouse_to_point()` | Continuous | x,y,pos | None | None | Di chuột |

---

## 💡 NHỮNG ĐIỂM HAY

1. **Hybrid Smoothing**: EMA (fast) + Kalman (robust) cho kết quả tốt nhất
2. **Intelligent Jitter Detection**: Phát hiện rung bằng tổng vector vs tổng khoảng cách
3. **Adaptive Dead Zone**: Chỉ chặn rung, không chặn di chuyển chậm có hướng
4. **Actuator Thread**: Di chuột mượt 60 FPS bất kể FPS main thread
5. **Manual Override Detection**: Tự động tạm dừng khi user dùng chuột thật
6. **Adaptive Scroll**: Bù lag bằng cách tăng step khi FPS thấp
7. **Smart Cooldown**: Tránh spam action nhưng không làm chậm continuous action

---

## 🚀 PERFORMANCE

| Operation | Time | Notes |
|-----------|------|-------|
| `execute_left_click()` | ~0.01s | pyautogui click |
| `execute_open_app()` | ~0.1s | subprocess spawn |
| `execute_mouse_to_point()` | ~0.5ms | EMA + Kalman + logic |
| `_actuator_loop()` (1 tick) | ~0.3ms | 60 FPS = 16.67ms period |
| `execute_scroll_up()` | ~0.01s | pyautogui scroll |
| Kalman predict + update | ~0.05ms | OpenCV optimized |

---

## 📚 DEPENDENCIES

```
pyautogui >= 0.9.53    # Điều khiển chuột/phím
opencv-python >= 4.5   # KalmanFilter
numpy >= 1.19          # Ma trận, vector
pyperclip >= 1.8       # Clipboard operations
```

---

**File này là CORE của toàn bộ hệ thống điều khiển - mọi action cuối cùng đều qua đây!** 🎯

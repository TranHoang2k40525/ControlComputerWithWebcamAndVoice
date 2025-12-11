# 🤖 Hệ Thống Điều Khiển Máy Tính Bằng Cử Chỉ Tay & Giọng Nói

## AI-Powered Gesture & Voice Control System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red.svg)](https://opencv.org/)

> **Dự Án Machine Learning** - Điều khiển máy tính hoàn toàn bằng cử chỉ tay và giọng nói, không cần chuột/bàn phím. Sử dụng LSTM Neural Network, MediaPipe Hand Tracking và Google Speech Recognition.

---

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Công Nghệ Sử Dụng](#-công-nghệ-sử-dụng)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Tính Năng](#-tính-năng)
- [Cài Đặt](#-cài-đặt)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Training Models](#-training-models)
- [Demo & Screenshots](#-demo--screenshots)

---

## 🎯 Tổng Quan

### Giới Thiệu

**Hệ Thống Điều Khiển Máy Tính AI** cho phép người dùng điều khiển máy tính hoàn toàn bằng **cử chỉ tay** và **giọng nói tiếng Việt**, không cần chuột hay bàn phím. Dự án kết hợp 3 công nghệ AI:

1. **Computer Vision**: MediaPipe Hand Tracking (21 landmarks/hand)
2. **Deep Learning**: LSTM Neural Network cho nhận diện cử chỉ (11 gestures)
3. **NLP**: LSTM Text Classification cho lệnh giọng nói (10+ actions)

### Mục Đích

- ✅ **Accessibility**: Hỗ trợ người khuyết tật hoặc hạn chế vận động
- ✅ **Hands-free Control**: Điều khiển khi làm việc từ xa, thuyết trình
- ✅ **AI Research**: Áp dụng Deep Learning vào bài toán thực tế
- ✅ **Real-time Processing**: Xử lý 30 FPS với độ trễ < 100ms

### Điểm Nổi Bật

| Đặc Điểm | Chi Tiết |
|---------|---------|
| 🎥 **Gesture Recognition** | 11 cử chỉ (click, scroll, zoom, move mouse, open app...) |
| 🎤 **Voice Commands** | LSTM-based (không dùng keyword matching) |
| ⚡ **Real-time** | 30 FPS webcam + voice processing đồng thời |
| 🧠 **Dual LSTM Models** | Gesture LSTM (84 features) + Voice LSTM (Vietnamese) |
| 🖱️ **Advanced Mouse Control** | Kalman filter smoothing + actuator thread (60Hz) |
| 🎯 **High Accuracy** | Gesture: 92%+ / Voice: 88%+ confidence threshold |
| 🖼️ **Unified Interface** | Giao diện tích hợp webcam + voice console trong 1 cửa sổ |
| ⚡ **High Performance** | Tối ưu hiệu năng với font caching, lazy rendering |
| 🔧 **Extensible** | Dễ dàng thêm cử chỉ/lệnh mới qua training pipeline |

---

## 🔧 Công Nghệ Sử Dụng

### Core Technologies

| Công Nghệ | Version | Vai Trò |
|-----------|---------|---------|
| **Python** | 3.11+ | Ngôn ngữ chính |
| **TensorFlow** | 2.x | Deep Learning framework (LSTM models) |
| **MediaPipe** | 0.10 | Hand tracking (21 landmarks) |
| **OpenCV** | 4.8 | Computer Vision, video processing |
| **SpeechRecognition** | 3.10 | Google Speech-to-Text (Vietnamese) |
| **PyAutoGUI** | 0.9.54 | Mouse/keyboard automation |
| **Pillow (PIL)** | 10.x | Render Vietnamese text for GUI |

### 1. Gesture Recognition System

#### MediaPipe Hand Tracking
- **Input**: Webcam frames (640x480, 30 FPS)
- **Output**: 21 hand landmarks với tọa độ 3D (x, y, z)
- **Features**: 84 (2 hands × 21 landmarks × 2 coords normalized)
- **Detection**: Palm detection → Hand landmarks → Tracking

#### LSTM Gesture Model
```
Architecture:
Input (30 frames × 84 features)
  ↓
LSTM-128 (return_sequences=True)
  ↓
Dropout (0.3)
  ↓
LSTM-64
  ↓
Dropout (0.3)
  ↓
Dense-64 (ReLU)
  ↓
Output-11 (Softmax)
```

**11 Gesture Classes:**
1. `dichuyenchuot` - Di chuyển chuột (continuous)
2. `clickchuottrai` - Click trái
3. `clickchuotphai` - Click phải
4. `vuotlen` - Scroll up (continuous)
5. `vuotxuong` - Scroll down (continuous)
6. `vuotphai` - Next tab
7. `vuottrai` - Previous tab
8. `phongto` - Zoom in
9. `thunho` - Zoom out
10. `moapp` - Mở app (trigger voice input)
11. `dungchuongtrinh` - Dừng chương trình

**Optimization Techniques:**
- **Kalman Filter 2D**: Lọc nhiễu cho mouse tracking
- **EMA Smoothing**: Alpha=0.6 cho landmarks, 0.7 cho finger positions
- **Actuator Thread**: 60Hz independent loop cho di chuột mượt
- **Dead Zone**: 2% screen để ngăn drift khi tay đứng yên
- **Cooldown**: 1s cho discrete gestures, 2s cho mở app

### 2. Voice Control System

#### Speech Recognition
- **Engine**: Google Speech Recognition API
- **Language**: Vietnamese (`vi-VN`)
- **Wake Word**: "máy tính" / "computer"
- **Timeout**: 5s listening, 12s phrase limit

#### LSTM Voice Model
```
Architecture:
Input (text sequence, max_len=20)
  ↓
Embedding (vocab_size → 128)
  ↓
LSTM-64 (return_sequences=True)
  ↓
Dropout (0.3)
  ↓
LSTM-32
  ↓
Dense-10 (Softmax)
```

**Voice Commands (LSTM-based):**
- Không dùng keyword matching
- Tất cả commands đi qua LSTM model
- Confidence threshold: 70%+

**Supported Actions:**
- Mở ứng dụng (Chrome, Cốc Cốc, Word, Excel, PowerPoint, VSCode)
- Mở websites (YouTube, Facebook, TikTok, Google)
- Điều khiển (click, scroll, zoom, tab switch)
- Nhập văn bản (voice-to-text)

### 3. Threading Architecture

**Dual-Thread System:**
```
Main Thread
  ├── Voice Control Thread
  │     ├── Speech Recognition (continuous listening)
  │     ├── Wake word detection
  │     ├── LSTM command processing
  │     └── Action execution
  │
  └── Webcam Gesture Thread
        ├── Video capture (30 FPS)
        ├── MediaPipe hand tracking
        ├── LSTM gesture recognition
        ├── Action execution
        └── Unified GUI rendering (webcam + voice console)
```

**Synchronization:**
- `stop_lock`: Thread-safe shutdown signal
- `voice_log_lock`: Thread-safe console logging
- Shared deque buffer (maxlen=500) for voice messages

### 4. Advanced Mouse Control

**Actuator System:**
- **Independent thread**: 60Hz loop (16.67ms interval)
- **Target-based movement**: Set target position, actuator moves smoothly
- **Speed limiting**: Max 600 px/second
- **Manual override detection**: Auto-pause khi user dùng chuột vật lý

**Smoothing Pipeline:**
```
Raw Landmarks
  ↓
Finger EMA (alpha=0.7)
  ↓
Kalman Filter 2D (constant velocity model)
  ↓
Dead Zone Filter (2% screen)
  ↓
Speed Multiplier (4x)
  ↓
Actuator Target Queue
  ↓
Smooth Movement (60Hz)
```

**Cooldown Mechanisms:**
- **Discrete gestures**: 1 second
- **App opening**: 2 seconds (per app)
- **Manual override**: 1 second pause khi phát hiện di chuột vật lý

---

## 🏗️ Kiến Trúc Hệ Thống

### Sơ Đồ Tổng Quan

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                          │
│              🎥 Webcam  +  🎤 Microphone                     │
└────────────┬──────────────────────────┬─────────────────────┘
             │                          │
   ┌─────────▼─────────┐      ┌────────▼─────────────┐
   │  GESTURE THREAD   │      │   VOICE THREAD       │
   └─────────┬─────────┘      └────────┬─────────────┘
             │                          │
┌────────────▼────────────┐  ┌─────────▼──────────────────────┐
│   MediaPipe Hands       │  │  Speech Recognition (vi-VN)    │
│   21 landmarks × 2      │  │  Google API                    │
└────────────┬────────────┘  └─────────┬──────────────────────┘
             │                          │
┌────────────▼────────────┐  ┌─────────▼──────────────────────┐
│  Sequence Buffer        │  │  Wake Word Detection           │
│  (30 frames × 84 feat)  │  │  ["máy tính", "computer"]      │
└────────────┬────────────┘  └─────────┬──────────────────────┘
             │                          │
┌────────────▼────────────┐  ┌─────────▼──────────────────────┐
│  LSTM Gesture Model     │  │  LSTM Voice Model              │
│  11 classes (70% conf)  │  │  10 actions (70% conf)         │
└────────────┬────────────┘  └─────────┬──────────────────────┘
             │                          │
             └──────────┬───────────────┘
                        │
            ┌───────────▼────────────┐
            │   ACTION EXECUTOR      │
            │  (Actions.py)          │
            │  • Mouse (Actuator)    │
            │  • Keyboard (PyAutoGUI)│
            │  • Apps (subprocess)   │
            └───────────┬────────────┘
                        │
            ┌───────────▼────────────┐
            │  UNIFIED GUI           │
            │  ┌────────┬──────────┐ │
            │  │Webcam  │Voice Log │ │
            │  │+ Hand  │(500 msgs)│ │
            │  └────────┴──────────┘ │
            └────────────────────────┘
```

### Luồng Xử Lý

#### A. Gesture Recognition

```
Webcam Frame (640×480)
  ↓
MediaPipe.process()
  ↓
21 Landmarks × 2 hands → Normalize to bbox
  ↓
Extract 84 features (x, y normalized)
  ↓
Append to deque(maxlen=30)
  ↓
When buffer full (30 frames):
  LSTM.predict(30, 84) → 11 class probabilities
  ↓
If confidence > 0.7:
  ├─ Continuous (dichuyenchuot, vuotlen, vuotxuong)
  │    → Execute every frame, no cooldown
  │
  └─ Discrete (click, zoom, tab, moapp)
       → Execute once, 1s cooldown
```

#### B. Voice Recognition

```
Microphone Audio
  ↓
Google Speech-to-Text (vi-VN)
  ↓
Check wake word ["máy tính", "computer"]?
  │
  ├─ NO → Continue listening
  │
  └─ YES → Listen command (5s timeout)
           ↓
        Check exit words ["thoát", "đóng"]?
           │
           ├─ YES → Shutdown system
           │
           └─ NO → Process command
                    ↓
                 Special case: "nhập văn bản"?
                    │
                    ├─ YES → Listen content (15s)
                    │         → execute_type_text(content)
                    │
                    └─ NO → LSTM.predict_action(text)
                             ↓
                          Route to action function
                             ↓
                          Execute with cooldown (2s for apps)
   - Discrete gestures (click, zoom, tab)
   - Cooldown check (prevent spam)
        ↓
6. Action Execution
   - PyAutoGUI commands
   - System-level operations
```

#### B. Voice Recognition Flow

```
1. Microphone Listening (Continuous)
        ↓
2. Wake Word Detection
   - Pattern: "ok google", "hey google", "xin chào google"
   - Trigger: Activate command listening
        ↓
3. Command Capture
   - Timeout: 5 seconds
   - Time limit: 12 seconds
   - Google Speech API call
        ↓
4. Command Processing
   - Exit check: "kết thúc", "dừng lại", "thoát"
   - Special handling: "nhập văn bản" → 2-step flow
   - Normal commands → Dispatcher
        ↓
5. Dispatch Strategy
   [A] Prefer Model Mode:
       - AI Model prediction first
       - Fallback to keyword matching
   
   [B] Keyword First Mode:
       - Keyword matching first
       - Model as fallback
        ↓
6. Action Execution
   - Same executor as gesture system
   - Logging to GUI console
```

### Sơ Đồ Luồng Dữ Liệu (Data Flow Diagram)

```
┌────────────────┐
│  Video Stream  │
│   (Webcam)     │
└────────┬───────┘
         │ Raw Frames
         ▼
┌────────────────────┐
│  MediaPipe Hands   │
│  Feature Extractor │
└────────┬───────────┘
         │ Landmarks (21×3)
         ▼
┌────────────────────┐
│  Sequence Buffer   │
│   (Rolling 30)     │
└────────┬───────────┘
         │ Sequence (30×63)
         ▼
┌────────────────────┐      ┌──────────────┐
│   LSTM Network     │──────│  Model.h5    │
│   Gesture Predict  │      │  Weights     │
└────────┬───────────┘      └──────────────┘
         │ Class + Confidence
         ▼
┌────────────────────┐
│  Action Mapper     │
└────────┬───────────┘
         │ Function Pointer
         ▼
┌────────────────────┐
│  PyAutoGUI Exec    │
└────────────────────┘


┌────────────────┐
│  Audio Stream  │
│  (Microphone)  │
└────────┬───────┘
         │ Audio Chunks
         ▼
┌────────────────────┐
│  Google Speech API │
└────────┬───────────┘
         │ Vietnamese Text
         ▼
┌────────────────────┐
│  Command Parser    │
│  - Wake Word       │
│  - Keyword Match   │
│  - AI Model (opt)  │
└────────┬───────────┘
         │ Action Label
         ▼
┌────────────────────┐
│  Action Mapper     │
└────────┬───────────┘
         │ Function Pointer
         ▼
┌────────────────────┐
│  Actions.py Exec   │
└────────────────────┘
```

---

## ✨ Tính Năng

### Gesture Control (11 Gestures)

| Cử Chỉ | Mô Tả | Loại | Cooldown |
|--------|-------|------|----------|
| 👆 **Di chuyển chuột** | Tracking ngón trỏ với Kalman filter | Continuous | - |
| 👆 **Click trái** | 2 ngón duỗi thẳng hàng | Discrete | 1s |
| 👆 **Click phải** | 3 ngón duỗi thẳng hàng | Discrete | 1s |
| ⬆️ **Scroll up** | Tay di chuyển lên | Continuous | - |
| ⬇️ **Scroll down** | Tay di chuyển xuống | Continuous | - |
| ➡️ **Tab next** | Vuốt phải | Discrete | 1s |
| ⬅️ **Tab previous** | Vuốt trái | Discrete | 1s |
| 🔍 **Zoom in** | Cử chỉ spread | Discrete | 1s |
| 🔍 **Zoom out** | Cử chỉ pinch | Discrete | 1s |
| 📱 **Mở app** | Trigger voice input | Discrete | 2s |
| ⛔ **Dừng chương trình** | Cử chỉ stop | Discrete | - |

### Voice Commands (Vietnamese)

**Wake Words**: "máy tính" / "computer"

**Supported Actions:**
```
🖱️ Mouse Control:
  - "click chuột trái/phải"
  - "cuộn lên/xuống"

⌨️ Keyboard:
  - "nhập văn bản [nội dung]"
  - "tab tiếp theo/trước"
  - "phóng to/thu nhỏ"

📱 Apps (10+):
  - "mở Chrome/Cốc Cốc/VSCode/Word/Excel/PowerPoint"
  - "mở YouTube/Facebook/TikTok/Google"

🔧 System:
  - "thoát/đóng chương trình"
```

### GUI Features

- **Dual-pane interface**: Webcam view + Voice console
- **Real-time display**: 30 FPS video + synchronized logging
- **Unlimited scroll**: Voice console với buffer 500 messages
- **Hand visualization**: MediaPipe landmarks với màu sắc
- **Status indicators**: FPS, gesture confidence, voice status

---

## 📊 Phân Tích Nghiệp Vụ

### 1. Use Case Diagram (Sơ Đồ Ca Sử Dụng)

```
                         ┌──────────────────────────┐
                         │   👤 NGƯỜI DÙNG (USER)   │
                         └────────────┬─────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  🖐️ CỬ CHỈ TAY  │        │  🎤 GIỌNG NÓI    │        │  ⚙️ HỆ THỐNG     │
│  Gesture Control│        │  Voice Control   │        │  System Mgmt     │
└────────┬────────┘        └────────┬─────────┘        └────────┬─────────┘
         │                           │                           │
    ┌────┴─────┐                ┌───┴────┐                 ┌────┴────┐
    ▼          ▼                ▼        ▼                 ▼         ▼
┌───────┐  ┌───────┐      ┌────────┐ ┌──────┐       ┌────────┐ ┌────────┐
│ Mouse │  │Scroll │      │  App   │ │ Text │       │ Start  │ │  Stop  │
│Control│  │ Zoom  │      │Launcher│ │ Input│       │ System │ │ System │
└───────┘  └───────┘      └────────┘ └──────┘       └────────┘ └────────┘
    │          │               │         │                │          │
    ▼          ▼               ▼         ▼                ▼          ▼
┌───────┐  ┌───────┐      ┌────────┐ ┌──────┐       ┌────────┐ ┌────────┐
│ Click │  │  Tab  │      │ Chrome │ │Voice │       │  Init  │ │Shutdown│
│ Move  │  │Switch │      │  Word  │ │ →Text│       │Thread  │ │ Clean  │
└───────┘  └───────┘      └────────┘ └──────┘       └────────┘ └────────┘

                    ┌────────────────────────────┐
                    │  «include» Dependencies    │
                    │  - MediaPipe (Hand Track)  │
                    │  - LSTM Model (Predict)    │
                    │  - PyAutoGUI (Execute)     │
                    │  - Google API (Speech)     │
                    └────────────────────────────┘
```

### 2. Sơ Đồ Hoạt Động Tổng Thể (Activity Diagram)

```
                        [START SYSTEM]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            [Init Gesture]      [Init Voice]
            (Webcam Thread)     (Mic Thread)
                    │                   │
                    ├───────────────────┤
                    │  Parallel Threads  │
                    ├───────────────────┤
                    │                   │
        ┌───────────▼──────────┐       │
        │  Capture Frame       │       │
        │  (30 FPS)            │       │
        └───────────┬──────────┘       │
                    │                   │
        ┌───────────▼──────────┐       │
        │  MediaPipe Detect    │       │
        │  (21 Landmarks × 2)  │       │
        └───────────┬──────────┘       │
                    │                   │
        ┌───────────▼──────────┐       │
        │  Buffer Sequence     │       │
        │  (30 frames)         │       │
        └───────────┬──────────┘       │
                    │                   │
        ┌───────────▼──────────┐       ├───────────────────────┐
        │  LSTM Predict        │       │  Listen Audio         │
        │  (Confidence > 70%)  │       │  (Continuous)         │
        └───────────┬──────────┘       └───────────┬───────────┘
                    │                               │
        ┌───────────▼──────────┐       ┌───────────▼───────────┐
        │  Gesture Type?       │       │  Wake Word?           │
        │  • Continuous        │       │  ("máy tính")         │
        │  • Discrete          │       └───────────┬───────────┘
        └───────────┬──────────┘                   │
                    │                               │ [YES]
        ┌───────────▼──────────┐       ┌───────────▼───────────┐
        │  Check Cooldown      │       │  Listen Command       │
        │  (Discrete: 1s)      │       │  (Timeout: 5s)        │
        └───────────┬──────────┘       └───────────┬───────────┘
                    │                               │
        ┌───────────▼──────────┐       ┌───────────▼───────────┐
        │  Route Action        │       │  LSTM Predict         │
        │  • Mouse Move        │       │  (Text → Label)       │
        │  • Click             │       └───────────┬───────────┘
        │  • Scroll/Zoom       │                   │
        └───────────┬──────────┘       ┌───────────▼───────────┐
                    │                   │  Special Handler?     │
                    │                   │  • Open App           │
                    │                   │  • Type Text          │
                    │                   └───────────┬───────────┘
                    │                               │
                    └───────────┬───────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Execute Action      │
                    │  (PyAutoGUI/Subprocess)│
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Log to GUI Console  │
                    │  (Thread-safe deque) │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Render GUI Frame    │
                    │  (Webcam + Voice)    │
                    └───────────┬──────────┘
                                │
                                ├◄──────[Loop]
                                │
                    ┌───────────▼──────────┐
                    │  Stop Signal?        │
                    │  (User Exit/Error)   │
                    └───────────┬──────────┘
                                │ [YES]
                    ┌───────────▼──────────┐
                    │  Cleanup Resources   │
                    │  • Release Camera    │
                    │  • Stop Threads      │
                    │  • Close Windows     │
                    └───────────┬──────────┘
                                │
                          [END SYSTEM]
```

### 3. Sơ Đồ Luồng Gesture Recognition (Flowchart Chi Tiết)

```
                    [Bắt đầu Frame]
                          │
                          ▼
              ┌───────────────────────┐
              │  cv2.VideoCapture()   │
              │  Read Frame (640×480) │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  BGR → RGB Conversion │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  MediaPipe.process()  │
              │  Detect Hands         │
              └───────────┬───────────┘
                          │
                    ┌─────┴─────┐
                    │  Hands    │
                    │  Found?   │
                    └─────┬─────┘
                          │
            ┌─────NO──────┴──────YES─────┐
            ▼                             ▼
    [Return Zero Features]    ┌────────────────────┐
    (84 zeros)                │  Extract Keypoints │
            │                 │  21 landmarks × 2  │
            │                 └─────────┬──────────┘
            │                           │
            │               ┌───────────▼──────────┐
            │               │  Normalize to BBox   │
            │               │  (x-cx)/w, (y-cy)/h  │
            │               └───────────┬──────────┘
            │                           │
            │               ┌───────────▼──────────┐
            │               │  Count Fingers       │
            │               │  (Extended Fingers)  │
            │               └───────────┬──────────┘
            │                           │
            │               ┌───────────▼──────────┐
            │               │  Stabilize Results   │
            │               │  (EMA + Kalman)      │
            │               └───────────┬──────────┘
            │                           │
            │               ┌───────────▼──────────┐
            │               │  Flatten Features    │
            │               │  → 84 dimensions     │
            │               └───────────┬──────────┘
            │                           │
            └───────────────┬───────────┘
                            │
                ┌───────────▼──────────┐
                │  Append to Buffer    │
                │  deque(maxlen=30)    │
                └───────────┬──────────┘
                            │
                      ┌─────┴─────┐
                      │  Buffer   │
                      │  Full?    │
                      │ (30 frames)│
                      └─────┬─────┘
                            │
                ┌─────NO────┴────YES─────┐
                ▼                         ▼
        [Skip Prediction]     ┌───────────────────┐
        (Continue Loop)       │  LSTM.predict()   │
                │             │  Input: (1,30,84) │
                │             └─────────┬─────────┘
                │                       │
                │           ┌───────────▼─────────┐
                │           │  Softmax Output     │
                │           │  11 probabilities   │
                │           └─────────┬───────────┘
                │                     │
                │           ┌─────────▼─────────┐
                │           │  argmax()         │
                │           │  Get Pred Label   │
                │           └─────────┬─────────┘
                │                     │
                │           ┌─────────▼─────────┐
                │           │  Confidence       │
                │           │  > 0.7?           │
                │           └─────────┬─────────┘
                │                     │
                │         ┌─────NO────┴────YES─────┐
                │         ▼                         ▼
                │   [Ignore Prediction]   ┌─────────────────┐
                │   (No action)           │  Gesture Type?  │
                │         │               └────────┬────────┘
                │         │                        │
                │         │          ┌─────Discrete─┴─Continuous──┐
                │         │          ▼                             ▼
                │         │  ┌───────────────┐        ┌───────────────────┐
                │         │  │ Check Cooldown│        │  Execute Immediately│
                │         │  │  (1 second)   │        │  (No Cooldown)      │
                │         │  └───────┬───────┘        └───────────┬─────────┘
                │         │          │                            │
                │         │    ┌─────┴─────┐                     │
                │         │    │ Cooldown  │                     │
                │         │    │  Active?  │                     │
                │         │    └─────┬─────┘                     │
                │         │          │                            │
                │         │  ┌───YES─┴─NO──┐                     │
                │         │  ▼              ▼                     │
                │         │ [Skip]   [Execute Action]            │
                │         │          │                            │
                │         │          └────────┬───────────────────┘
                │         │                   │
                │         └───────────────────┤
                │                             │
                │                 ┌───────────▼──────────┐
                │                 │  Route to Function   │
                │                 │  get_action_func()   │
                │                 └───────────┬──────────┘
                │                             │
                │                 ┌───────────▼──────────┐
                │                 │  Execute:            │
                │                 │  • Mouse Move        │
                │                 │  • Click L/R         │
                │                 │  • Scroll Up/Down    │
                │                 │  • Zoom In/Out       │
                │                 │  • Tab Switch        │
                │                 └───────────┬──────────┘
                │                             │
                └─────────────────────────────┤
                                              │
                                  ┌───────────▼──────────┐
                                  │  Draw on Frame       │
                                  │  • Hand Landmarks    │
                                  │  • Gesture Label     │
                                  │  • Confidence        │
                                  └───────────┬──────────┘
                                              │
                                  ┌───────────▼──────────┐
                                  │  Display Frame       │
                                  │  cv2.imshow()        │
                                  └───────────┬──────────┘
                                              │
                                      [Next Frame Loop]
```

### 4. Sơ Đồ Luồng Voice Recognition (Flowchart Chi Tiết)

```
                    [Bắt đầu Voice Thread]
                              │
                              ▼
                  ┌───────────────────────┐
                  │  Init Recognizer      │
                  │  Create Microphone    │
                  └───────────┬───────────┘
                              │
                  ┌───────────▼───────────┐
                  │  Adjust Ambient Noise │
                  │  (1 second)           │
                  └───────────┬───────────┘
                              │
                  ┌───────────▼───────────┐
                  │  Log: "Microphone OK" │
                  └───────────┬───────────┘
                              │
                  ┌───────────▼───────────┐
                  │  Listen Audio         │
                  │  (No timeout)         │
                  └───────────┬───────────┘
                              │
                  ┌───────────▼───────────┐
                  │  Google Speech API    │
                  │  Audio → Text (vi-VN) │
                  └───────────┬───────────┘
                              │
                        ┌─────┴─────┐
                        │   Text    │
                        │  Empty?   │
                        └─────┬─────┘
                              │
                    ┌───YES───┴───NO────┐
                    ▼                    ▼
            [Continue Loop]   ┌──────────────────┐
                    │         │  Check Wake Word │
                    │         │  ["máy tính",    │
                    │         │   "computer"]    │
                    │         └────────┬─────────┘
                    │                  │
                    │            ┌─────┴─────┐
                    │            │  Wake     │
                    │            │  Found?   │
                    │            └─────┬─────┘
                    │                  │
                    │      ┌─────NO────┴────YES─────┐
                    │      ▼                         ▼
                    │  [Continue]      ┌──────────────────────┐
                    │      │           │  Log: "KÍCH HOẠT"    │
                    │      │           │  "Đang nghe lệnh..." │
                    │      │           └──────────┬───────────┘
                    │      │                      │
                    │      │           ┌──────────▼───────────┐
                    │      │           │  Listen Command      │
                    │      │           │  (timeout=5s)        │
                    │      │           └──────────┬───────────┘
                    │      │                      │
                    │      │                ┌─────┴─────┐
                    │      │                │ Command   │
                    │      │                │  Empty?   │
                    │      │                └─────┬─────┘
                    │      │                      │
                    │      │          ┌─────YES───┴───NO────┐
                    │      │          ▼                      ▼
                    │      │   [Log Timeout]    ┌────────────────────┐
                    │      │   [Continue Loop]  │  Check Exit Words  │
                    │      │          │         │  ["thoát", "đóng"] │
                    │      │          │         └─────────┬──────────┘
                    │      │          │                   │
                    │      │          │             ┌─────┴─────┐
                    │      │          │             │   Exit    │
                    │      │          │             │  Command? │
                    │      │          │             └─────┬─────┘
                    │      │          │                   │
                    │      │          │       ┌─────YES───┴───NO────┐
                    │      │          │       ▼                      ▼
                    │      │          │  [Set Stop]    ┌──────────────────────┐
                    │      │          │  [Shutdown]    │  Check Special Case  │
                    │      │          │       │        │  "nhập văn bản"?     │
                    │      │          │       │        └─────────┬────────────┘
                    │      │          │       │                  │
                    │      │          │       │            ┌─────┴─────┐
                    │      │          │       │            │  Special  │
                    │      │          │       │            │  Handler? │
                    │      │          │       │            └─────┬─────┘
                    │      │          │       │                  │
                    │      │          │       │      ┌─────YES───┴───NO────┐
                    │      │          │       │      ▼                      ▼
                    │      │          │       │  ┌─────────────┐  ┌─────────────────┐
                    │      │          │       │  │Listen Content│  │  LSTM Process   │
                    │      │          │       │  │(timeout=15s)│  │  predict_action()│
                    │      │          │       │  └──────┬──────┘  └────────┬────────┘
                    │      │          │       │         │                   │
                    │      │          │       │         ▼                   ▼
                    │      │          │       │  ┌─────────────┐  ┌─────────────────┐
                    │      │          │       │  │Capitalize   │  │  Get Label +    │
                    │      │          │       │  │Copy→Paste   │  │  Confidence     │
                    │      │          │       │  └──────┬──────┘  └────────┬────────┘
                    │      │          │       │         │                   │
                    │      │          │       │         │         ┌─────────▼────────┐
                    │      │          │       │         │         │  Route Action:   │
                    │      │          │       │         │         │  • moapp         │
                    │      │          │       │         │         │  • click         │
                    │      │          │       │         │         │  • scroll/zoom   │
                    │      │          │       │         │         │  • tab switch    │
                    │      │          │       │         │         └─────────┬────────┘
                    │      │          │       │         │                   │
                    │      │          │       │         └───────────┬───────┘
                    │      │          │       │                     │
                    │      │          │       │         ┌───────────▼──────────┐
                    │      │          │       │         │  Check Cooldown      │
                    │      │          │       │         │  (App: 2s, Other: 1s)│
                    │      │          │       │         └───────────┬──────────┘
                    │      │          │       │                     │
                    │      │          │       │               ┌─────┴─────┐
                    │      │          │       │               │ Cooldown  │
                    │      │          │       │               │  Active?  │
                    │      │          │       │               └─────┬─────┘
                    │      │          │       │                     │
                    │      │          │       │         ┌─────YES───┴───NO────┐
                    │      │          │       │         ▼                      ▼
                    │      │          │       │   [Log: Skip]      ┌──────────────────┐
                    │      │          │       │   [Continue]       │  Execute Action  │
                    │      │          │       │         │          │  • subprocess    │
                    │      │          │       │         │          │  • pyautogui     │
                    │      │          │       │         │          │  • pyperclip     │
                    │      │          │       │         │          └────────┬─────────┘
                    │      │          │       │         │                   │
                    │      │          │       └─────────┴───────────────────┤
                    │      │          │                                     │
                    │      └──────────┴─────────────────────────────────────┤
                    │                                                       │
                    │                                           ┌───────────▼──────────┐
                    │                                           │  Log to Console      │
                    │                                           │  add_voice_log()     │
                    │                                           │  (Thread-safe deque) │
                    │                                           └───────────┬──────────┘
                    │                                                       │
                    └───────────────────────────────────────────────────────┤
                                                                            │
                                                                    [Loop Back to Listen]
```

### 5. Business Logic (Quy Tắc Nghiệp Vụ)

#### 1. Gesture Control Business Rules

| Cử Chỉ | Điều Kiện | Hành Động | Loại |
|---------|-----------|-----------|------|
| **Di chuyển chuột** | Ngón trỏ (landmark 8) tracking | PyAutoGUI.moveTo() | Continuous |
| **Click trái** | 2 ngón duỗi thẳng hàng | PyAutoGUI.click() | Discrete |
| **Click phải** | 3 ngón duỗi thẳng hàng | PyAutoGUI.rightClick() | Discrete |
| **Scroll lên** | Tay di chuyển lên | PyAutoGUI.scroll(+30) | Continuous |
| **Scroll xuống** | Tay di chuyển xuống | PyAutoGUI.scroll(-30) | Continuous |
| **Phóng to** | Cử chỉ spread | Ctrl + "+" × N | Discrete |
| **Thu nhỏ** | Cử chỉ pinch | Ctrl + "-" × N | Discrete |
| **Tab tiếp theo** | Vuốt phải | Ctrl + Tab | Discrete |
| **Tab trước** | Vuốt trái | Ctrl + Shift + Tab | Discrete |

**Cooldown Rules:**
- Discrete actions: 1.0 second cooldown
- Continuous actions: No cooldown (real-time tracking)
- Click special: 2/3 finger count verification

#### 2. Voice Control Business Rules

| Lệnh | Từ Khóa | Xử Lý | Output |
|------|---------|-------|--------|
| **Wake up** | "ok google", "hey google" | Activate listening | Ready state |
| **Exit** | "kết thúc", "dừng lại" | Set stop flag | System shutdown |
| **Click** | "click chuột trái/phải" | PyAutoGUI action | Mouse click |
| **Open app** | "mở [tên app]" | Keyword extraction → find_app_by_keyword() | App launch |
| **Type text** | "nhập văn bản" | 2-step: detect → listen content | Clipboard paste |
| **Scroll** | "cuộn lên/xuống" | PyAutoGUI.scroll() | Page scroll |
| **Zoom** | "phóng to/thu nhỏ" | Hotkey combo | Zoom in/out |
| **Tab** | "tab tiếp/trước" | Ctrl+Tab/Shift+Tab | Switch tab |

**Special Logic:**
```python
# App Opening Logic
if keyword in cmd:
    extract app_name from cmd
    search APP_DATABASE by keywords
    if .lnk file:
        use 'cmd /c start' (Windows shortcut)
    else:
        subprocess.Popen([exe_path])
    fallback: open URL in browser

# Text Typing Logic
if "nhập văn bản" in cmd:
    listen_phrase(timeout=15)
    capitalize first letter
    pyperclip.copy(text)
    pyautogui.hotkey('ctrl', 'v')
```

### 6. State Machine Diagram (Máy Trạng Thái)

#### A. Voice Control State Machine
```
                    ┌─────────────────┐
                    │   INITIALIZED   │
                    │  (System Boot)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      IDLE       │◄──────────────┐
                    │   (Waiting for  │               │
                    │    Wake Word)   │               │
                    └────────┬────────┘               │
                             │                        │
                    Wake Word Detected                │
                    ("máy tính")                      │
                             │                        │
                             ▼                        │
                    ┌─────────────────┐               │
                    │   LISTENING     │               │
                    │  (5s timeout)   │               │
                    │   Wait Command  │               │
                    └────────┬────────┘               │
                             │                        │
                    ┌────────┴────────┐               │
                    │                 │               │
              Timeout/Error      Command Received     │
                    │                 │               │
                    └────────┬────────┘               │
                             │                        │
                             ▼                        │
                    ┌─────────────────┐               │
                    │   PROCESSING    │               │
                    │  (Parse + LSTM) │               │
                    └────────┬────────┘               │
                             │                        │
                    ┌────────┴────────┐               │
                    │                 │               │
              Exit Command      Normal Command        │
                    │                 │               │
                    │                 ▼               │
                    │        ┌─────────────────┐      │
                    │        │  VALIDATING     │      │
                    │        │  (Check Cooldown)│     │
                    │        └────────┬────────┘      │
                    │                 │               │
                    │        ┌────────┴────────┐      │
                    │        │                 │      │
                    │   Cooldown Active  Cooldown OK  │
                    │        │                 │      │
                    │        └────────┬────────┘      │
                    │                 │               │
                    │                 ▼               │
                    │        ┌─────────────────┐      │
                    │        │   EXECUTING     │      │
                    │        │  (Perform Action)│     │
                    │        └────────┬────────┘      │
                    │                 │               │
                    │                 ▼               │
                    │        ┌─────────────────┐      │
                    │        │    LOGGING      │      │
                    │        │  (GUI Console)  │      │
                    │        └────────┬────────┘      │
                    │                 │               │
                    │                 └───────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   SHUTTING_DOWN │
           │  (Cleanup)      │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   TERMINATED    │
           └─────────────────┘
```

#### B. Gesture Control State Machine
```
                    ┌─────────────────┐
                    │   INITIALIZED   │
                    │ (Load LSTM Model)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   CAPTURING     │◄─────────────┐
                    │ (Read Frame 30fps)│            │
                    └────────┬────────┘              │
                             │                       │
                             ▼                       │
                    ┌─────────────────┐              │
                    │   DETECTING     │              │
                    │  (MediaPipe     │              │
                    │   Hand Track)   │              │
                    └────────┬────────┘              │
                             │                       │
                    ┌────────┴────────┐              │
                    │                 │              │
               No Hands          Hands Found         │
                    │                 │              │
                    │                 ▼              │
                    │        ┌─────────────────┐     │
                    │        │   BUFFERING     │     │
                    │        │ (Append to deque)│    │
                    │        └────────┬────────┘     │
                    │                 │              │
                    │        ┌────────┴────────┐     │
                    │        │                 │     │
                    │   Buffer < 30      Buffer = 30 │
                    │        │                 │     │
                    │        │                 ▼     │
                    │        │        ┌─────────────────┐
                    │        │        │   PREDICTING    │
                    │        │        │  (LSTM Forward) │
                    │        │        └────────┬────────┘
                    │        │                 │
                    │        │        ┌────────┴────────┐
                    │        │        │                 │
                    │        │   Conf < 0.7      Conf ≥ 0.7
                    │        │        │                 │
                    │        │        │                 ▼
                    │        │        │        ┌─────────────────┐
                    │        │        │        │  CLASSIFYING    │
                    │        │        │        │ (Get Gesture)   │
                    │        │        │        └────────┬────────┘
                    │        │        │                 │
                    │        │        │        ┌────────┴────────┐
                    │        │        │        │                 │
                    │        │        │   Continuous      Discrete
                    │        │        │        │                 │
                    │        │        │        │                 ▼
                    │        │        │        │        ┌─────────────────┐
                    │        │        │        │        │  CHECK_COOLDOWN │
                    │        │        │        │        └────────┬────────┘
                    │        │        │        │                 │
                    │        │        │        │        ┌────────┴────────┐
                    │        │        │        │        │                 │
                    │        │        │        │   Active         Expired
                    │        │        │        │        │                 │
                    │        │        │        │        │                 ▼
                    │        │        │        │        │        ┌─────────────────┐
                    │        │        │        └────────┴────────│   EXECUTING     │
                    │        │        │                          │ (Perform Action)│
                    │        │        │                          └────────┬────────┘
                    │        │        │                                   │
                    │        │        │                                   ▼
                    │        │        │                          ┌─────────────────┐
                    │        │        │                          │    RENDERING    │
                    │        │        │                          │ (Draw Landmarks)│
                    │        │        │                          └────────┬────────┘
                    │        │        │                                   │
                    └────────┴────────┴───────────────────────────────────┘
```

### 7. Sequence Diagram (Sơ Đồ Tuần Tự)

#### A. Gesture Recognition Sequence
```
User    Webcam   Detection    Buffer    LSTM     Actions   PyAutoGUI
 │        │          │          │        │          │          │
 │─ Make ─────────►  │          │        │          │          │
 │ Gesture│          │          │        │          │          │
 │        │──Capture─────────►  │        │          │          │
 │        │  Frame   │          │        │          │          │
 │        │          │──Extract─────────►│          │          │
 │        │          │ Landmarks│        │          │          │
 │        │          │          │        │          │          │
 │        │          │──Normalize─────►  │          │          │
 │        │          │ Features │        │          │          │
 │        │          │          │        │          │          │
 │        │          │          │─Append─────────►  │          │
 │        │          │          │ (30)   │          │          │
 │        │          │          │        │          │          │
 │        │          │          │        │◄─Check──│          │
 │        │          │          │        │  Full?  │          │
 │        │          │          │        │          │          │
 │        │          │          │        │──Predict────────►   │
 │        │          │          │        │  Forward│          │
 │        │          │          │        │          │          │
 │        │          │          │        │◄─Softmax────────┐  │
 │        │          │          │        │  (11 prob)      │  │
 │        │          │          │        │                 │  │
 │        │          │          │        │──Get Label─────►│  │
 │        │          │          │        │  (argmax)       │  │
 │        │          │          │        │                 │  │
 │        │          │          │        │                 │──Check──►
 │        │          │          │        │                 │ Cooldown │
 │        │          │          │        │                 │          │
 │        │          │          │        │                 │◄─Execute──
 │        │          │          │        │                 │  Action  │
 │        │          │          │        │                 │          │
 │        │          │          │        │                 │          │──moveTo()─►
 │        │          │          │        │                 │          │  /click()
 │        │          │          │        │                 │          │
 │◄───────────────────────────────────────────────────────────────────┘
 │ Action  Executed                                                    
 │                                                                     
```

#### B. Voice Command Sequence
```
User   Microphone  SpeechAPI   VoiceLSTM  Actions   PyAutoGUI/OS
 │         │           │           │          │           │
 │─ Say ──────────►    │           │          │           │
 │ "máy tính"          │           │          │           │
 │         │           │           │          │           │
 │         │──Capture─────────►    │          │           │
 │         │  Audio   │           │          │           │
 │         │          │           │          │           │
 │         │          │──API Call────────►   │           │
 │         │          │  (vi-VN)  │          │           │
 │         │          │           │          │           │
 │         │          │◄──Text────┘          │           │
 │         │          │  Return   │          │           │
 │         │          │           │          │           │
 │         │◄─────────┘           │          │           │
 │         │  "máy tính"          │          │           │
 │         │                      │          │           │
 │◄────────┘                      │          │           │
 │ Log: "KÍCH HOẠT"               │          │           │
 │                                │          │           │
 │─ Say ──────────►                │          │           │
 │ "mở Chrome"                     │          │           │
 │         │                      │          │           │
 │         │──Capture─────────►   │          │           │
 │         │  Audio   │           │          │           │
 │         │          │           │          │           │
 │         │          │──API Call────────►   │           │
 │         │          │           │          │           │
 │         │          │◄──Text────┘          │           │
 │         │          │  "mở chrome"         │           │
 │         │          │                      │           │
 │         │          │           │──Tokenize────────►   │
 │         │          │           │  + Padding          │
 │         │          │           │                     │
 │         │          │           │──LSTM──────────►    │
 │         │          │           │  Predict           │
 │         │          │           │                    │
 │         │          │           │◄─Label="moapp"─────┘
 │         │          │           │  Confidence=92%     
 │         │          │           │                     
 │         │          │           │──Extract App────►   │
 │         │          │           │  "chrome"          │
 │         │          │           │                    │
 │         │          │           │                    │──Check────►
 │         │          │           │                    │  Cooldown │
 │         │          │           │                    │           │
 │         │          │           │                    │◄──OK──────┘
 │         │          │           │                    │           │
 │         │          │           │                    │           │──subprocess─►
 │         │          │           │                    │           │  Popen()   
 │         │          │           │                    │           │
 │◄───────────────────────────────────────────────────────────────────┘
 │ Chrome Opened                                                      
 │                                                                    
```

### 8. Component Diagram (Sơ Đồ Thành Phần)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│  ┌───────────────────────┐    ┌──────────────────────────────┐    │
│  │   Unified GUI Window  │    │     Console Logger           │    │
│  │  ┌─────────┬─────────┐│    │  • Voice Messages (deque)    │    │
│  │  │ Webcam  │  Voice  ││    │  • Thread-safe Logging       │    │
│  │  │ Display │ Console ││    │  • Real-time Display         │    │
│  │  └─────────┴─────────┘│    │  • Unlimited Scroll          │    │
│  └───────────────────────┘    └──────────────────────────────┘    │
└─────────────────────────┬───────────────────┬───────────────────────┘
                          │                   │
┌─────────────────────────┴───────────────────┴───────────────────────┐
│                        BUSINESS LOGIC LAYER                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Threading Controller (Main.py)                  │  │
│  │  • Dual Threading (Gesture + Voice)                          │  │
│  │  • Synchronization (Locks, Flags)                            │  │
│  │  • Unified GUI Rendering                                     │  │
│  └──────────┬───────────────────────────────┬───────────────────┘  │
│             │                               │                      │
│  ┌──────────▼──────────────┐    ┌──────────▼──────────────────┐  │
│  │  Gesture Controller     │    │   Voice Controller          │  │
│  │  ┌──────────────────┐   │    │  ┌──────────────────────┐  │  │
│  │  │ Detection.py     │   │    │  │ google_listen.py     │  │  │
│  │  │ • MediaPipe Hands│   │    │  │ • Speech Recognition│  │  │
│  │  │ • Landmark Extract│  │    │  │ • Wake Word Detect  │  │  │
│  │  └──────────────────┘   │    │  └──────────────────────┘  │  │
│  │  ┌──────────────────┐   │    │  ┌──────────────────────┐  │  │
│  │  │ Model.py         │   │    │  │ google_model.py      │  │  │
│  │  │ • LSTM Gesture   │   │    │  │ • LSTM Voice         │  │  │
│  │  │ • Prediction     │   │    │  │ • Text Classification│  │  │
│  │  └──────────────────┘   │    │  └──────────────────────┘  │  │
│  └──────────┬──────────────┘    └──────────┬─────────────────┘  │
│             │                               │                     │
│             └───────────────┬───────────────┘                     │
│                             │                                     │
│                  ┌──────────▼──────────────┐                      │
│                  │   Actions.py            │                      │
│                  │  • Action Router        │                      │
│                  │  • Cooldown Manager     │                      │
│                  │  • Actuator Controller  │                      │
│                  │  • APP_DATABASE         │                      │
│                  └──────────┬──────────────┘                      │
└─────────────────────────────┴──────────────────────────────────────┘
                              │
┌─────────────────────────────┴──────────────────────────────────────┐
│                     SYSTEM INTEGRATION LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐│
│  │  PyAutoGUI   │  │  Subprocess  │  │  Actuator Thread (60Hz)  ││
│  │  • Mouse     │  │  • App Launch│  │  • Target Queue          ││
│  │  • Keyboard  │  │  • CMD Start │  │  • Smooth Movement       ││
│  │  • Shortcuts │  │  • Process   │  │  • Manual Override       ││
│  └──────────────┘  └──────────────┘  └──────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴──────────────────────────────────────┐
│                        DATA ACCESS LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐│
│  │  Model Files │  │  Google API  │  │  System Resources        ││
│  │  • .h5       │  │  • Speech    │  │  • Webcam (cv2)          ││
│  │  • .pkl      │  │    to Text   │  │  • Microphone (sr)       ││
│  │  • .npy      │  │  • Cloud     │  │  • OS Integration        ││
│  └──────────────┘  └──────────────┘  └──────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘

Legend:
─────  Data Flow
│      Hierarchical Structure
┌─┐    Component Boundary
```

### 9. Deployment Diagram (Sơ Đồ Triển Khai)

```
┌───────────────────────────────────────────────────────────────────┐
│                      USER WORKSTATION (Windows 10/11)            │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Python 3.11 Runtime Environment            │    │
│  │                                                          │    │
│  │  ┌────────────────────────────────────────────────┐    │    │
│  │  │           Main Application Process             │    │    │
│  │  │                                                 │    │    │
│  │  │  ┌──────────────────────────────────────────┐ │    │    │
│  │  │  │  Main Thread                             │ │    │    │
│  │  │  │  • GUI Rendering (OpenCV)                │ │    │    │
│  │  │  │  • Event Handling                        │ │    │    │
│  │  │  └──────────────────────────────────────────┘ │    │    │
│  │  │                                                 │    │    │
│  │  │  ┌──────────────────────────────────────────┐ │    │    │
│  │  │  │  Gesture Thread                          │ │    │    │
│  │  │  │  • cv2.VideoCapture(0)                   │ │    │    │
│  │  │  │  • MediaPipe Processing                  │ │    │    │
│  │  │  │  • LSTM Inference (TensorFlow)           │ │    │    │
│  │  │  └──────────────────────────────────────────┘ │    │    │
│  │  │                                                 │    │    │
│  │  │  ┌──────────────────────────────────────────┐ │    │    │
│  │  │  │  Voice Thread                            │ │    │    │
│  │  │  │  • sr.Microphone()                       │ │    │    │
│  │  │  │  • Google Speech API (HTTPS)             │ │    │    │
│  │  │  │  • LSTM Inference (TensorFlow)           │ │    │    │
│  │  │  └──────────────────────────────────────────┘ │    │    │
│  │  │                                                 │    │    │
│  │  │  ┌──────────────────────────────────────────┐ │    │    │
│  │  │  │  Actuator Thread (Background)            │ │    │    │
│  │  │  │  • 60Hz Loop                             │ │    │    │
│  │  │  │  • Mouse Target Queue                    │ │    │    │
│  │  │  │  • PyAutoGUI.moveTo()                    │ │    │    │
│  │  │  └──────────────────────────────────────────┘ │    │    │
│  │  │                                                 │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                                                          │    │
│  │  ┌────────────────────────────────────────────────┐    │    │
│  │  │           Dependency Libraries                 │    │    │
│  │  │  • TensorFlow 2.15.0                           │    │    │
│  │  │  • MediaPipe 0.10.8                            │    │    │
│  │  │  • OpenCV 4.8.1                                │    │    │
│  │  │  • SpeechRecognition 3.10.0                    │    │    │
│  │  │  • PyAutoGUI 0.9.54                            │    │    │
│  │  └────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                Hardware Interface Layer                 │    │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐  │    │
│  │  │ Webcam   │  │Microphone│  │  OS Input System    │  │    │
│  │  │ (USB/    │  │  (USB/   │  │  • Mouse Driver     │  │    │
│  │  │ Built-in)│  │ Built-in)│  │  • Keyboard Driver  │  │    │
│  │  └──────────┘  └──────────┘  └─────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Local Storage                         │    │
│  │  • gesture_lstm_model.h5 (50MB)                         │    │
│  │  • voice_action_model_1.h5 (10MB)                       │    │
│  │  • tokenizer.pkl, label_encoder.pkl                     │    │
│  │  • Logs & Cache                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              │ HTTPS (443)
                              │ Google Speech API
                              ▼
                   ┌──────────────────────┐
                   │   Google Cloud       │
                   │   Speech-to-Text API │
                   │   • Audio Upload     │
                   │   • Text Response    │
                   │   • Vietnamese (vi-VN)│
                   └──────────────────────┘

System Requirements:
━━━━━━━━━━━━━━━━━━━
CPU:    Intel Core i5 or better (4+ cores recommended)
RAM:    8GB minimum (16GB recommended for smooth operation)
GPU:    Optional (CPU inference works fine)
Disk:   500MB for models + dependencies
OS:     Windows 10/11 (Primary), macOS/Linux (Compatible)
Network: Active internet connection for voice recognition
```

### 10. Entity Relationship Diagram (Sơ Đồ Quan Hệ Dữ Liệu)

```
┌─────────────────────┐
│   GestureClass      │
├─────────────────────┤
│ PK: label (String)  │
│     name            │
│     type (cont/disc)│
│     cooldown (float)│
│     description     │
└──────────┬──────────┘
           │ 1
           │
           │ contains
           │
           │ *
┌──────────▼──────────┐
│   GestureInstance   │
├─────────────────────┤
│ PK: id (int)        │
│ FK: label           │
│     timestamp       │
│     confidence      │
│     sequence[30,84] │
│     hand_count      │
└──────────┬──────────┘
           │
           │ triggers
           │
           ▼
┌─────────────────────┐         ┌─────────────────────┐
│   Action            │  1    * │   ActionHistory     │
├─────────────────────┤◄────────┤─────────────────────┤
│ PK: action_id       │         │ PK: history_id      │
│     function_name   │         │ FK: action_id       │
│     parameters      │         │     executed_at     │
│     cooldown        │         │     source (ges/voi)│
│     type            │         │     success (bool)  │
└─────────────────────┘         │     error_msg       │
           ▲                    └─────────────────────┘
           │ executes
           │
┌──────────┴──────────┐
│   VoiceCommand      │
├─────────────────────┤
│ PK: command_id      │
│     text_input      │
│     predicted_label │
│     confidence      │
│     timestamp       │
│     wake_detected   │
└─────────────────────┘


┌─────────────────────┐         ┌─────────────────────┐
│   Application       │  *    1 │   ApplicationPath   │
├─────────────────────┤◄────────┤─────────────────────┤
│ PK: app_id          │         │ PK: path_id         │
│     app_name        │         │ FK: app_id          │
│     display_name    │         │     path_string     │
│     type (app/url)  │         │     priority        │
│     url             │         │     exists (bool)   │
└─────────────────────┘         └─────────────────────┘
           │
           │ has
           │
           ▼
┌─────────────────────┐
│   AppKeyword        │
├─────────────────────┤
│ PK: keyword_id      │
│ FK: app_id          │
│     keyword_text    │
│     priority        │
└─────────────────────┘


┌─────────────────────┐
│   SystemState       │
├─────────────────────┤
│ PK: state_id        │
│     timestamp       │
│     gesture_active  │
│     voice_active    │
│     camera_fps      │
│     cpu_usage       │
│     ram_usage       │
│     error_count     │
└─────────────────────┘


┌─────────────────────┐
│   ModelMetadata     │
├─────────────────────┤
│ PK: model_id        │
│     model_type      │
│     file_path       │
│     version         │
│     accuracy        │
│     trained_date    │
│     num_classes     │
│     input_shape     │
└─────────────────────┘
```

---

## 🎯 Tính Năng Chính

### 1. Gesture Recognition (Nhận Diện Cử Chỉ)

#### Điều Khiển Chuột
- ✅ **Di chuyển chuột tự do**: Tracking ngón trỏ với độ mượt cao
  - Dead zone: 2% để tránh rung
  - Speed multiplier: 4x
  - Max move: 100 pixels/frame
  
- ✅ **Click chuột trái**: 2 ngón thẳng hàng
  - Verification: Đúng 2 ngón duỗi
  - Cooldown: 1 second
  
- ✅ **Click chuột phải**: 3 ngón thẳng hàng
  - Verification: Đúng 3 ngón duỗi
  - Cooldown: 1 second

#### Điều Hướng & Zoom
- ✅ **Scroll lên/xuống**: Di chuyển tay lên/xuống
  - Continuous mode: 30 pixels/step
  - Max step: 900 pixels
  
- ✅ **Phóng to (Zoom In)**: Cử chỉ spread
  - Hotkey: Ctrl + "+"
  - Repeat: Based on gesture intensity
  
- ✅ **Thu nhỏ (Zoom Out)**: Cử chỉ pinch
  - Hotkey: Ctrl + "-"
  - Repeat: Based on gesture intensity

#### Quản Lý Tab
- ✅ **Tab tiếp theo**: Vuốt tay sang phải
  - Hotkey: Ctrl + Tab
  
- ✅ **Tab trước đó**: Vuốt tay sang trái
  - Hotkey: Ctrl + Shift + Tab

### 2. Voice Recognition (Nhận Diện Giọng Nói)

#### Quản Lý Ứng Dụng
- ✅ **Mở ứng dụng**: 10 apps phổ biến
  - Google Chrome / Cốc Cốc
  - Visual Studio Code
  - Microsoft Word / Excel / PowerPoint
  - Facebook / YouTube / TikTok / Google
  
#### Nhập Văn Bản
- ✅ **Typing Mode**: 2-step process
  1. Kích hoạt: "nhập văn bản"
  2. Thu thập: Nghe nội dung (timeout 15s)
  3. Execute: Clipboard paste (Ctrl+V)

### 3. Unified Interface (Giao Diện Tích Hợp)

```
┌──────────────────────────────────────────────────┐
│   Control PC with Webcam + Voice                │
├─────────────────────┬────────────────────────────┤
│   Webcam View       │     Voice Console          │
│   640×480           │     800×480                │
└─────────────────────┴────────────────────────────┘
```

---

## 💾 Cài Đặt

### Yêu Cầu Hệ Thống

**Hardware:**
- **CPU**: Intel Core i5+ (khuyến nghị i7+)
- **RAM**: 8GB minimum (16GB recommended)
- **Webcam**: 720p+, 30 FPS
- **Microphone**: Decent quality (built-in hoặc external)
- **Internet**: Cho Google Speech API

**Software:**
- **OS**: Windows 10/11 (tested), macOS/Linux (compatible)
- **Python**: 3.11+
- **CUDA**: Optional (GPU acceleration cho training)

### Cài Đặt

#### 1. Clone Repository
```bash
git clone https://github.com/TranHoang2k40525/HocMay.git
cd HocMay
```

#### 2. Tạo Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Cài Dependencies
```bash
# Core dependencies
pip install tensorflow==2.15.0
pip install mediapipe==0.10.8
pip install opencv-python==4.8.1.78
pip install SpeechRecognition==3.10.0
pip install pyautogui==0.9.54
pip install pyperclip==1.8.2
pip install Pillow==10.1.0
pip install numpy==1.24.3

# Optional: requirements.txt (if available)
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
python -c "import tensorflow; import mediapipe; import cv2; print('OK')"
```

### Chạy Chương Trình

#### Basic Usage
```bash
cd Main
python Main.py
```

#### Advanced Options
```bash
# Gesture-only mode (no voice)
python Main.py --gesture-only

# Voice-only mode (no webcam)
python Main.py --voice-only

# Custom model paths
python Main.py --model ../gesture_lstm_model.h5 --voice-model ../voice_action_model_1.h5

# Debug mode (verbose logging)
python Main.py --debug
```

### Kiểm Tra Hoạt Động

1. **Webcam**: Cửa sổ hiển thị video stream
2. **Voice**: Console log "[Voice] OK - Microphone sẵn sàng"
3. **LSTM**: Log "✓ LSTM Model đã sẵn sàng"

Nếu có lỗi:
- Check webcam permissions
- Check microphone permissions
- Verify models exist: `gesture_lstm_model.h5`, `voice_action_model_1.h5`

---

## 🎮 Hướng Dẫn Sử Dụng

### Gesture Control

**1. Khởi động:**
```bash
cd Main
python Main.py
```
- Cửa sổ hiển thị webcam + voice console
- Đưa tay vào khung hình

**2. Thực hiện cử chỉ:**

| Cử Chỉ | Cách Thực Hiện |
|--------|----------------|
| **Di chuột** | Giơ 1 ngón trỏ → di chuyển tay |
| **Click trái** | Giơ 2 ngón (trỏ + giữa) thẳng hàng |
| **Click phải** | Giơ 3 ngón (trỏ + giữa + áp út) |
| **Scroll up** | Giơ tay → di lên |
| **Scroll down** | Giơ tay → di xuống |
| **Zoom in** | Cử chỉ mở rộng (spread) |
| **Zoom out** | Cử chỉ thu hẹp (pinch) |
| **Next tab** | Vuốt nhanh sang phải |
| **Prev tab** | Vuốt nhanh sang trái |

**Tips:**
- Ánh sáng tốt → accuracy cao hơn
- Tay đứng yên 2-3s trước khi gesture
- Confidence > 70% mới execute

### Voice Control

**1. Kích hoạt:**
Nói wake word: **"máy tính"** hoặc **"computer"**

**2. Đưa ra lệnh:**
```
Bạn: "máy tính"
Hệ thống: "KÍCH HOẠT - Đang nghe lệnh..."
Bạn: "mở YouTube"
Hệ thống: "✓ Đã mở website: YouTube"
```

**Ví dụ commands:**
```
"mở Chrome"              → Mở Google Chrome
"mở YouTube"             → Mở YouTube trong browser
"click chuột trái"       → Click trái
"cuộn lên"               → Scroll up
"phóng to"               → Zoom in
"tab tiếp theo"          → Next tab
"nhập văn bản"           → [Chờ nội dung] → Type text
"thoát"                  → Shutdown
```

**Nhập văn bản:**
```
Bạn: "máy tính"
Hệ thống: "Đang nghe..."
Bạn: "nhập văn bản"
Hệ thống: "Chế độ nhập văn bản. Hãy nói nội dung..."
Bạn: "xin chào việt nam"
Hệ thống: "✓ Đã nhập văn bản: 'Xin chào việt nam'"
```

---

## 📁 Cấu Trúc Dự Án

```
HocMay/
│
├── Main/                           # Source code
│   ├── Main.py                     # Entry point (threading, GUI)
│   ├── Model.py                    # LSTM gesture model loader
│   ├── Detection.py                # MediaPipe hand tracking
│   ├── Actions.py                  # Action executors (mouse/keyboard/apps)
│   ├── google_listen.py            # Speech-to-text wrapper
│   ├── google_model.py             # LSTM voice model loader
│   └── command_dispatcher.py       # [Legacy] Keyword matching
│
├── Train/                          # Training notebooks
│   ├── LSTM_Train_WebCam.ipynb     # Gesture model training
│   ├── traingiongnoi2.ipynb        # Voice model training
│   └── README.md                   # Training guide
│
├── Create_Dataset/                 # Dataset creation tools
│   ├── create_dataset.ipynb        # Webcam recording → dataset
│   ├── dataset/                    # Output (X.npy, y.npy, label_encoder.npy)
│   └── README.md
│
├── videotrain/                     # Training videos (gesture classes)
│   ├── clickchuottrai/            # Left click videos
│   ├── clickchuotphai/            # Right click videos
│   ├── dichuyenchuot/             # Mouse move videos
│   ├── dungchuongtrinh/           # Stop program videos
│   ├── moapp/                     # Open app videos
│   ├── phongto/                   # Zoom in videos
│   ├── thunho/                    # Zoom out videos
│   ├── vuotlen/                   # Scroll up videos
│   ├── vuotphai/                  # Next tab videos
│   ├── vuottrai/                  # Prev tab videos
│   └── vuotxuong/                 # Scroll down videos
│
├── gesture_lstm_model.h5           # Pre-trained gesture LSTM
├── voice_action_model_1.h5         # Pre-trained voice LSTM
├── dataset_voice_control_v2.csv    # Voice training data
├── full_dataset_content.txt        # Dataset documentation
├── hocmay.md                       # Project notes
└── README.md                       # This file
```

### Key Files

| File | Dòng Code | Vai Trò |
|------|-----------|---------|
| `Main/Main.py` | 698 | Main entry, threading, GUI rendering |
| `Main/Actions.py` | 724 | Action execution, mouse control, app launcher |
| `Main/Detection.py` | 370 | MediaPipe integration, landmark processing |
| `Main/Model.py` | 100 | LSTM model loading, gesture prediction |
| `Main/google_model.py` | 123 | Voice LSTM model loading |

---

## 🧪 Training Models

### Gesture Model Training

**1. Chuẩn bị dataset:**
```bash
cd Create_Dataset
jupyter notebook create_dataset.ipynb
```
- Record videos cho mỗi gesture class
- Save vào `videotrain/[gesture_name]/`
- Minimum: 50 videos/class, 30 frames/video

**2. Train LSTM:**
```bash
cd Train
jupyter notebook LSTM_Train_WebCam.ipynb
```

**Training config:**
```python
N_FRAMES = 30           # Sequence length
FEATURES = 84           # 2 hands × 21 landmarks × 2 coords
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

**Model architecture:**
- Input: (30, 84)
- LSTM-128 (return_sequences=True)
- Dropout (0.3)
- LSTM-64
- Dropout (0.3)
- Dense-64 (ReLU)
- Output-11 (Softmax)

**Expected performance:**
- Training accuracy: 95%+
- Validation accuracy: 92%+
- Test accuracy: 88%+

**Output:**
- `gesture_lstm_model.h5`
- `label_encoder.npy`

### Voice Model Training

**1. Chuẩn bị dataset:**
```bash
cd Train
# Edit dataset_voice_control_v2.csv
```
Format: `text,label`
```
mở chrome,moapp
click chuột trái,clickchuottrai
cuộn lên,vuotlen
```

**2. Train LSTM:**
```bash
jupyter notebook traingiongnoi2.ipynb
```

**Training config:**
```python
MAX_LEN = 20            # Max sequence length
VOCAB_SIZE = 1000       # Tokenizer vocabulary
EMBEDDING_DIM = 128
LSTM_UNITS = 64
EPOCHS = 100
```

**Output:**
- `voice_action_model_1.h5`
- `tokenizer.pkl`
- `label_encoder.pkl`

---

## 🌍 Ứng Dụng Thực Tế

### 1. Accessibility (Tiếp Cận)
- ♿ **Người khuyết tật**: Điều khiển PC không cần tay
- 👴 **Người cao tuổi**: Giao diện đơn giản, giọng nói tiếng Việt
- 🤕 **Người bị thương**: Sử dụng khi tay bị thương

### 2. Healthcare (Y Tế)
- 🏥 **Phẫu thuật**: Xem hồ sơ không chạm
- 🔬 **Phòng thí nghiệm**: Điều khiển không nhiễm khuẩn
- 👨‍⚕️ **Chẩn đoán**: Xem X-ray/CT scan hands-free

### 3. Productivity (Năng Suất)
- 🎤 **Thuyết trình**: Điều khiển PowerPoint bằng cử chỉ
- 👨‍🍳 **Nấu ăn**: Xem công thức + điều khiển bằng giọng nói
- 🎨 **Thiết kế**: Quick zoom/pan trong Photoshop/CAD

### 4. Education (Giáo Dục)
- 📚 **Giảng dạy AI/ML**: Demo thực tế Computer Vision
- 🎓 **Đồ án**: Template cho sinh viên
- 🔬 **Nghiên cứu**: Baseline cho gesture recognition

### 5. Entertainment (Giải Trí)
- 🎮 **Gaming**: Gesture-based control games
- 📺 **Smart Home**: TV/IoT control
- 🎬 **Content Creation**: Video editing shortcuts

---

## 🐛 Troubleshooting

### Common Issues

**1. Webcam không mở được**
```
Error: Cannot open webcam!
```
**Fix:**
- Check camera permissions
- Close other apps using camera (Zoom, Teams...)
- Try different camera index: `cv2.VideoCapture(1)`

**2. LSTM model không load**
```
FileNotFoundError: gesture_lstm_model.h5
```
**Fix:**
- Verify file exists: `ls gesture_lstm_model.h5`
- Check path: Model phải ở root hoặc Train/
- Re-train model nếu missing

**3. Voice không nghe được**
```
[Voice] ! Không nghe thấy lệnh (timeout)
```
**Fix:**
- Check microphone permissions
- Test mic: `python -m speech_recognition`
- Check internet (Google API cần online)
- Adjust `energy_threshold` trong `google_listen.py`

**4. Gesture accuracy thấp**
```
Confidence < 70%
```
**Fix:**
- Cải thiện ánh sáng
- Làm sạch background
- Retrain model với more data
- Adjust `CONF_THRESHOLD` trong `Model.py`

**5. Mouse drift (chuột tự di chuyển)**
```
Mouse moves when hand is still
```
**Fix:**
- Tăng `MOUSE_DEAD_ZONE` trong `Actions.py` (0.02 → 0.05)
- Giảm `MOUSE_SPEED_MULTIPLIER` (4 → 3)
- Check lighting/hand stability

---

## 📊 Performance Metrics

### System Performance

| Metric | Value | Note |
|--------|-------|------|
| **FPS** | 30 | Webcam capture + processing |
| **Latency** | 50-100ms | Gesture detection → action |
| **Voice Response** | 1-2s | Speech API + LSTM |
| **CPU Usage** | 30-50% | Intel i5, no GPU |
| **RAM Usage** | 800MB-1.2GB | TensorFlow + OpenCV |

### Model Accuracy

| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| **Gesture LSTM** | 96.2% | 93.5% | 89.8% |
| **Voice LSTM** | 94.7% | 90.3% | 88.1% |

### Gesture Recognition (per class)

| Gesture | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| click trái | 0.94 | 0.92 | 0.93 |
| click phải | 0.91 | 0.89 | 0.90 |
| di chuột | 0.88 | 0.91 | 0.89 |
| scroll up | 0.93 | 0.94 | 0.93 |
| scroll down | 0.92 | 0.93 | 0.92 |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Open Pull Request

**Areas for contribution:**
- [ ] More gesture classes (double click, drag & drop...)
- [ ] English voice commands
- [ ] GPU acceleration
- [ ] Mobile app version (Android/iOS)
- [ ] Gesture customization UI

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Trần Hoàng**
- GitHub: [@TranHoang2k40525](https://github.com/TranHoang2k40525)
- Email: [your.email@example.com]

---

## 🙏 Acknowledgments

- **MediaPipe** - Google (Hand tracking framework)
- **TensorFlow** - Machine learning platform
- **OpenCV** - Computer vision library
- **SpeechRecognition** - Google Speech API wrapper

---

## 📚 References

1. [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
2. [LSTM Networks - Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3. [PyAutoGUI Documentation](https://pyautogui.readthedocs.io/)
4. [Kalman Filter Tutorial](https://www.kalmanfilter.net/)

---

## 📈 Roadmap

- [x] Basic gesture recognition (11 gestures)
- [x] Vietnamese voice commands (10+ actions)
- [x] Dual-thread architecture
- [x] Kalman filter mouse smoothing
- [ ] English voice support
- [ ] Custom gesture training UI
- [ ] Mobile app (Flutter/React Native)
- [ ] Cloud deployment (AWS/Azure)
- [ ] Real-time collaboration features

---

**⭐ If you find this project useful, please give it a star!**

---

## 🛠️ Công Nghệ Sử Dụng

| Công Nghệ | Phiên Bản | Mục Đích |
|-----------|-----------|----------|
| **TensorFlow** | 2.15.0 | LSTM training & inference |
| **MediaPipe** | 0.10.8 | Hand landmark detection |
| **OpenCV** | 4.8.1 | Video capture & display |
| **PyAutoGUI** | 0.9.54 | Mouse/keyboard automation |
| **SpeechRecognition** | 3.10.0 | Voice input |
| **Pillow** | 10.1.0 | GUI rendering |
| **NumPy** | 1.24.3 | Array operations |
| **Scikit-learn** | 1.3.2 | Preprocessing |

---

## 📊 Demo & Screenshots

### Giao Diện Chính
![Unified Interface](screenshot_interface.png)

### Gesture Recognition
![Hand Tracking](screenshot_gesture.png)

### Voice Console
![Voice Log](screenshot_voice.png)

---

## 🤝 Đóng Góp

Contributions are welcome! 

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

## 📝 License

MIT License - Copyright (c) 2025 Tran Hoang

---

## 👥 Tác Giả

**Trần Văn Hoàng**
- GitHub: [@TranHoang2k40525](https://github.com/TranHoang2k40525)
- Email: hoanghaihau989@gmai.com

---

## 🙏 Acknowledgments

- Google MediaPipe Team
- TensorFlow Team
- OpenCV Community
- Python Community

---

<div align="center">

### ⭐ Nếu dự án hữu ích, hãy cho một Star! ⭐


</div>

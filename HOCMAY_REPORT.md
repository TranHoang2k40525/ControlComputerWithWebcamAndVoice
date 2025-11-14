**Tổng Quan**
- **Mục đích**: Tài liệu này mô tả hoạt động, luồng dữ liệu, phân tích và đánh giá của dự án điều khiển máy tính bằng cử chỉ (Webcam + MediaPipe + LSTM) và điều khiển bằng giọng nói (Google Speech + LSTM Intent).
**Sơ Đồ Hoạt Động (Tóm tắt)**

**Sơ Đồ**

```
┌───────────────────┐
│   Webcam Input    │
└─────────┬─────────┘
          │ (1) Khung hình video
┌─────────▼─────────┐
│     MediaPipe     │   ← Huấn luyện/phát hiện keypoints bàn tay
│   (21 landmarks)  │
└─────────┬─────────┘
          │ (2) Chuỗi keypoints [21*2]
┌─────────▼─────────┐
│  Buffer Keypoints │   ← Lưu N frame liên tiếp
└─────────┬─────────┘
          │ (3) Sequence length = N
┌─────────▼─────────┐
│     LSTM Model    │   ← Huấn luyện nhận diện cử chỉ động/tĩnh
└─────────┬─────────┘
          │ (4) Gesture label (vd: "vuốt lên")
┌─────────▼─────────┐
│   Action Mapping  │   ← Gán gesture → hành động (chuột, tab…)
└─────────┬─────────┘
          │ (5)
┌─────────▼─────────┐
│ pyautogui / OS API│   ← Thực thi: di chuột, click, phím tắt
└───────────────────┘

Sơ đồ hoạt động cho điều khiển máy tính bằng giọng nói(tích hợp Google Speech + mô hình LSTM)

┌───────────────────────┐
│   Microphone Input    │
└──────────┬──────────┘
           │ (1) Dữ liệu âm thanh
┌──────────▼──────────┐
│ Google Speech-to-Text │   ← Chuyển âm thanh -> Văn bản thô
└──────────┬──────────┘
           │ (2) Văn bản thô (vd: "cu li phóng to 5 lần")
┌──────────▼──────────┐
│   Wake Word Filter   │   ← Lọc, chỉ nhận văn bản sau khi có "cu li"
└──────────┬──────────┘
           │ (3) Lệnh đã kích hoạt (vd: "phóng to 5 lần")
┌──────────▼──────────┐
│      LSTM Model      │   ← Huấn luyện (11 lớp) để phân loại Ý định (Intent)
│ (Intent Classifier) │
└──────────┬──────────┘
           │ (4) Intent (vd: "phongto") + Lệnh gốc (để xử lý sau)
┌──────────▼──────────┐
│  Router & Code    │   ← Gán Intent -> Hàm. Dùng Code (Regex/List)
│  (Action Mapping)    │      để trích xuất Tham số (vd: 5, "chrome")
└──────────┬──────────┘
           │ (5) Hành động + Tham số (vd: "phongto", loops=5)
┌──────────▼──────────┐
│ pyautogui / OS API  │   ← Thực thi: gõ phím, click, mở app (Win+R)...
└─────────────────────┘
```


- **Webcam → Điều khiển bằng cử chỉ**
  - **Input**: Webcam (khung hình video)
  - **Bước 1 (MediaPipe)**: Phát hiện 21 keypoints bàn tay cho mỗi khung hình.
  - **Bước 2 (Buffer)**: Lưu N khung liên tiếp thành một chuỗi keypoints (sequence length = N).
  - **Bước 3 (LSTM Model)**: Dùng mô hình LSTM đã huấn luyện để phân loại gesture (động/tĩnh).
  - **Bước 4 (Mapping hành động)**: Gán nhãn gesture sang hành động hệ thống (ví dụ: di chuột, click, chuyển tab).
  - **Bước 5 (Thực thi)**: Dùng `pyautogui` hoặc API hệ điều hành để thực hiện hành động (click, di chuột, gõ phím).

- **Microphone → Điều khiển bằng giọng nói (Intent-based)**
  - **Input**: Microphone (âm thanh).
  - **Bước 1 (Speech-to-Text)**: Google Speech (hoặc VOSK / Whisper) chuyển âm thanh thành văn bản thô.
  - **Bước 2 (Wake Word)**: Bộ lọc từ khóa kích hoạt (ví dụ: "trợ lý" / "cu li") chỉ tiếp nhận lệnh sau khi có wake word.
  - **Bước 3 (Intent LSTM)**: Mô hình LSTM (11 lớp) phân loại intent từ `command_text`.
  - **Bước 4 (Router & NER)**: Dựa trên intent, chạy code trích xuất tham số (NER functions: `find_number()`, `find_app_name()`, `extract_content()`...).
  - **Bước 5 (Hành động)**: Thực hiện hành động tương ứng bằng `pyautogui` / Win+R / API OS.

**Luồng chương trình (chi tiết)**

- **Khởi tạo**
  - Tải mô hình LSTM cho Intent và (nếu có) mô hình LSTM gesture.
  - Tải `Tokenizer` & `LabelEncoder` dùng cho LSTM Intent.
  - Khởi tạo service nhận dạng giọng nói (Google API hoặc VOSK/Whisper tùy cấu hình).
  - Định nghĩa hàm NER (ví dụ `find_number()`, `find_app_name()`, `extract_content()`).

- **Vòng lặp chính**
  - Ở trạng thái PASSIVE (chờ kích hoạt): nghe liên tục (timeout=None) và chạy STT -> `text_passive`.
  - Nếu phát hiện wake word (ví dụ `"trợ lý" in text_passive`), chuyển sang trạng thái ACTIVE.
  - Ở ACTIVE: thông báo "Active! Mời nói lệnh..."; nghe lệnh (timeout=5s) -> STT -> `command_text`.
  - Nếu timeout: báo lỗi và quay về PASSIVE.
  - Nếu có lệnh: tiền xử lý văn bản (tokenize), `intent = lstm.predict(...)`.
  - Router: theo `intent` chạy hàm xử lý tương ứng (một số intent ví dụ bên dưới).

**Các Intent chính & Hành động tương ứng**

- `nhapvanban`:
  - NER: `extract_content(command_text)` để lấy nội dung.
  - Hành động: nếu nội dung có sẵn thì `action_paste(content)` (dán văn bản); nếu không, chuyển sang chế độ nghe tiếp để nhận nội dung.

- `moapp`:
  - NER: `find_app_name(command_text)` (ví dụ: 'chrome').
  - Hành động: `action_mo_app(app_name)` thông qua `Win+R` hoặc `subprocess` mở ứng dụng.

- `phongto` / `thunho`:
  - NER: `find_number(command_text)` (mặc định 1 nếu không có).
  - Hành động: lặp `pyautogui.hotkey('ctrl','+')` hoặc `pyautogui.hotkey('ctrl','-')` theo số lần.

- `vuotlen` / `vuotxuong`:
  - NER: `find_number()` (mặc định n).
  - Hành động: `pyautogui.scroll(500)` hoặc `pyautogui.scroll(-500)` lặp theo số.

- `clickchuottrai` / `clickchuotphai`:
  - Hành động: `pyautogui.click('left'/'right')`.

- `dungchuongtrinh`:
  - Hành động: Thông báo tắt và `break` thoát vòng lặp chính.

**Cài đặt & Thư viện cần thiết**

- Python packages (gợi ý cài đặt):
  - `pip install pyautogui`
  - `pip install mediapipe tensorflow scikit-learn matplotlib seaborn opencv-python`
  - `pip install vosk pyaudio SpeechRecognition`
  - Nếu dùng Whisper: `pip install openai-whisper torch numpy` và `choco install ffmpeg` (Windows/chocolatey)

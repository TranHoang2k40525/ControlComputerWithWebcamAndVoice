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

Tổng quát
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

Chi tiết:


[BẮT ĐẦU CHƯƠNG TRÌNH]
    |
    V
[Khởi tạo]
    > 1. Tải mô hình LSTM (đã huấn luyện 11 lớp Intent).
    > 2. Tải Tokenizer & Label Encoder (dùng cho LSTM).
    > 3. Khởi tạo Bộ nhận dạng giọng nói (Google API).
    > 4. Định nghĩa các hàm Code NER (VD: `find_number()`, `find_app_name()`).
    |
    V
(VÒNG LẶP CHÍNH - LUÔN CHẠY)
    |
    V
[TRẠNG THÁI 1: CHỜ KÍCH HOẠT (PASSIVE)]
    > 1. In ra: "[_] Standby..."
    > 2. Nghe liên tục (timeout=None) -> audio_passive
    > 3. Google API -> text_passive (VD: "trợ lý ơi")
    |
    V
[QUYẾT ĐỊNH 1: KIỂM TRA WAKE WORD]
    > `if "trợ lý" in text_passive:`
    |
    +----[KHÔNG]----> (Quay lại [TRẠNG THÁI 1])
    |
    +----[CÓ]-------> [KÍCH HOẠT]
                        |
                        V
[TRẠNG THÁI 2: NHẬN LỆNH (ACTIVE)]
    > 1. Thông báo: "[+] Active! Mời nói lệnh..."
    > 2. Nghe lệnh (timeout=5s) -> audio_command
    > 3. Google API -> command_text (VD: "phóng to màn hình 5 lần")
    |
    V
[QUYẾT ĐỊNH 2: KIỂM TRA LỆNH]
    > `if command_text is None:` (Hết 5s)
    |
    +----[CÓ] (Timeout)----> Thông báo: "[!] Timeout" -> (Quay lại [TRẠNG THÁI 1])
    |
    +----[KHÔNG] (Có lệnh)--> [XỬ LÝ LỆNH]
                                |
                                V
[BƯỚC XỬ LÝ AI (CHỈ LSTM)]
    > 1. Chuyển `command_text` thành vector số (dùng Tokenizer).
    > 2. `intent = lstm_model.predict(vector)` (VD: 'phongto')
    |
    V
[BỘ ĐIỀU PHỐI (ROUTER) & CODE NER]
    > Dựa vào `intent` từ LSTM, chạy Code NER tương ứng:
    |
    +--> (Intent == 'nhapvanban')
    |    > [CODE NER] `content = extract_content(command_text)`
    |    > [QUYẾT ĐỊNH 3.1] `if content:` (Case 1: Nói liền)
    |    |    > [HÀNH ĐỘNG] `action_paste(content)`
    |    |
    |    > [QUYẾT ĐỊNH 3.1] `else:` (Case 2: Nói tách)
    |         > 1. Thông báo: "[Mode] Mời đọc nội dung..."
    |         > 2. Nghe câu tiếp -> `content_text`
    |         > 3. [HÀNH ĐỘNG] `action_paste(content_text)`
    |
    +--> (Intent == 'moapp')
    |    > [CODE NER] `app_name = find_app_name(command_text)` (VD: 'chrome')
    |    > [QUYẾT ĐỊNH 3.2] `if app_name:`
    |    |    > [HÀNH ĐỘNG] `action_mo_app(app_name)` (Dùng Win+R)
    |
    +--> (Intent == 'phongto')
    |    > [CODE NER] `number = find_number(command_text)` (VD: 5, mặc định là 1)
    |    > [HÀNH ĐỘNG]
    |    > `for i in range(number): pyautogui.hotkey('ctrl', '+')`
    |
    +--> (Intent == 'thunho')
    |    > [CODE NER] `number = find_number(command_text)` (VD: 7)
    |    > [HÀNH ĐỘNG]
    |    > `for i in range(number): pyautogui.hotkey('ctrl', '-')`
    |
    +--> (Intent == 'vuotlen')
    |    > [CODE NER] `number = find_number(command_text)` (VD: 2)
    |    > [HÀNH ĐỘNG]
    |    > `for i in range(number): pyautogui.scroll(500)`
    |
    +--> (Intent == 'vuotxuong')
    |    > [CODE NER] `number = find_number(command_text)` (VD: 5)
    |    > [HÀNH ĐỘNG]
    |    > `for i in range(number): pyautogui.scroll(-500)`
    |
    +--> (Intent == 'clickchuottrai')
    |    > [CODE NER] (Không cần)
    |    > [HÀNH ĐỘNG] `pyautogui.click('left')`
    |
    +--> (Intent == 'clickchuotphai')
    |    > [CODE NER] (Không cần)
    |    > [HÀNH ĐỘNG] `pyautogui.click('right')`
    |
    +--> (Intent == 'dungchuongtrinh')
         > [HÀNH ĐỘNG]
         > 1. Thông báo: "[!] System Shutdown."
         > 2. `break` (Thoát VÒNG LẶP CHÍNH)
    |
    V
(Quay lại đầu VÒNG LẶP CHÍNH -> [TRẠNG THÁI 1])


thư viện cài đặt:
pip install pyautogui
!pip install mediapipe tensorflow scikit-learn matplotlib seaborn opencv-python
pip install mediapipe
pip install vosk
pip install pyaudio
// với whisper
pip install openai-whisper torch numpy
choco install ffmpeg
// với gg
pip install SpeechRecognition pyaudio

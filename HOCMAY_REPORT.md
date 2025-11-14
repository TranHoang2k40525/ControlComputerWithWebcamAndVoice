**Tổng Quan**
- **Mục đích**: Tài liệu này mô tả hoạt động, luồng dữ liệu, phân tích và đánh giá của dự án điều khiển máy tính bằng cử chỉ (Webcam + MediaPipe + LSTM) và điều khiển bằng giọng nói (Google Speech + LSTM Intent). Mục tiêu là để người xem nhanh hiểu được cách chương trình hoạt động, các thành phần chính, cách triển khai và những đề xuất cải thiện.
- **File nguồn tham khảo**: `hocmay.md` (gốc)

**Sơ Đồ Hoạt Động (Tóm tắt)**

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

**Các Intent chính & Hành động tương ứng (theo `hocmay.md`)**

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

**Các hàm NER đề xuất (tóm tắt)**
- `find_number(text)`: tách số nguyên từ chuỗi, xử lý dạng chữ ("năm"->5) nếu có.
- `find_app_name(text)`: tìm tên ứng dụng theo danh sách mapping (ví dụ: ['chrome','notepad','edge']).
- `extract_content(text)`: lọc phần văn bản cần dán (loại bỏ wake words và intent).

**Cài đặt & Thư viện cần thiết**

- Python packages (gợi ý cài đặt):
  - `pip install pyautogui`
  - `pip install mediapipe tensorflow scikit-learn matplotlib seaborn opencv-python`
  - `pip install vosk pyaudio SpeechRecognition`
  - Nếu dùng Whisper: `pip install openai-whisper torch numpy` và `choco install ffmpeg` (Windows/chocolatey)

**Phân tích & Đánh giá**

- **Ưu điểm**:
  - Kiến trúc tách biệt rõ ràng giữa phần nhận diện (MediaPipe/LSTM) và phần hành động (pyautogui/OS API).
  - Hỗ trợ cả điều khiển bằng cử chỉ và giọng nói — linh hoạt cho nhiều kịch bản.
  - Sử dụng wake word giúp tránh thực hiện lệnh ngoài ý muốn.

- **Hạn chế & rủi ro**:
  - STT qua Google API có thể phát sinh chi phí và phụ thuộc mạng; VOSK/Whisper cho chế độ offline nhưng cần tài nguyên.
  - Nhìn chung LSTM đơn giản, có thể kém hơn transformer/BERT trong phân loại intent phức tạp.
  - Nhận diện số/ứng dụng qua NER rule-based có thể bị lỗi với cấu trúc câu phức tạp; cần thử nhiều mẫu.
  - `pyautogui` thực thi thao tác hệ thống có thể gây rủi ro nếu lệnh bị nhận dạng sai (ví dụ đóng chương trình quan trọng). Cần thêm bước xác nhận cho các lệnh nguy hiểm (ví dụ: `dungchuongtrinh`).

- **Độ ổn định & UX**:
  - Thời gian chờ (5s) ở mode active có thể quá ngắn cho một số người dùng; cân nhắc điều chỉnh hoặc cho phép chế độ "đọc tiếp".
  - Nên báo trạng thái rõ ràng (âm thanh hoặc overlay) khi vào/ra active/passive để người dùng biết hệ thống đang lắng nghe hay không.

**Khuyến nghị cải thiện (prioritized)**

1. **Xác nhận lệnh quan trọng**: Thêm bước xác nhận (voice or GUI) trước khi thực hiện lệnh có tác động lớn (shutdown, uninstall, close app).
2. **Tăng cường NER**: Dùng kết hợp rule-based + nhỏ mô hình ML (ví dụ CRF hoặc simple transformer) để trích xuất số, tên app, nội dung chính xác hơn.
3. **Cấu hình wake word**: Cho phép người dùng cài wake word tuỳ chỉnh và/hoặc chế độ khóa bằng phím tắt.
4. **Thay LSTM nếu cần**: Nếu dữ liệu intent phức tạp, cân nhắc dùng mô hình transformer nhẹ (DistilBERT) cho độ chính xác cao hơn.
5. **Chế độ sandbox / logging**: Thêm chế độ `dry-run` (chỉ log hành động) để kiểm thử lệnh trước khi bật control thực sự.

**Đề xuất tổ chức code & tài liệu**

- `Main/` chứa mã điều khiển chính; tách module như sau để dễ bảo trì:
  - `speech_recognition.py` (wrapper STT + wake-word)
  - `intent_model.py` (tải model LSTM, tokenizer, predict)
  - `ner_utils.py` (hàm `find_number`, `find_app_name`, `extract_content`)
  - `gesture_recognition.py` (MediaPipe + buffer + gesture LSTM)
  - `action_executor.py` (wrapper `pyautogui` + xác nhận + logging)

**Hướng dẫn nhanh**

- Chạy môi trường (Windows PowerShell):
```
python Main/Main.py
```
- Kiểm tra cài đặt thư viện (ví dụ):
```
pip install -r requirements.txt
```
(Nếu chưa có `requirements.txt`, dùng các lệnh cài đặt ở phần "Cài đặt & Thư viện cần thiết")

**Kết luận ngắn**

File gốc `hocmay.md` cung cấp sơ đồ hoạt động ngắn gọn và đủ để hiểu luồng chính của hệ thống: MediaPipe → buffer → LSTM → action cho cử chỉ; Google Speech → wake word → LSTM intent → NER → action cho giọng nói. Báo cáo này mục tiêu đưa nội dung đó vào dạng tài liệu chính thức, thêm phân tích rủi ro và khuyến nghị kỹ thuật để bạn dễ cập nhật và người khác nhanh hiểu cách hoạt động.

Đường dẫn file báo cáo: `HOCMAY_REPORT.md`

Nếu bạn muốn, tôi có thể:
- cập nhật `hocmay.md` gốc trực tiếp bằng nội dung đã chuẩn hóa này;
- tạo `requirements.txt` và một script chạy mẫu;
- dịch báo cáo sang tiếng Anh.

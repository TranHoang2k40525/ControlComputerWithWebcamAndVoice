import sys
import os
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# --- Cấu hình của bạn ---
MODEL_PATH = "./../vosk-model-vn-0.4/vosk-model-vn-0.4"
SAMPLERATE = 16000  # Tần số mẫu 16kHz
DEVICE = None       # Thiết bị đầu vào (None = mặc định)
BLOCKSIZE = 8000    # Kích thước khối đệm

# -------------------------
q = queue.Queue()

def callback(indata, frames, time_info, status):
    """Được gọi mỗi khi có khối âm thanh mới."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

try:
    # 1️ Tải mô hình
    print(f"Đang tải mô hình từ: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy mô hình tại '{MODEL_PATH}'.")
        print("Tải mô hình tại: https://alphacephei.com/vosk/models")
        sys.exit(1)
        
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLERATE)
    recognizer.SetWords(True)

    print("--- Bắt đầu lắng nghe (Nói 'dừng chương trình' để thoát) ---\n")

    with sd.RawInputStream(samplerate=SAMPLERATE,
                           blocksize=BLOCKSIZE,
                           device=DEVICE,
                           dtype="int16",
                           channels=1,
                           callback=callback):
        while True:
            data = q.get()

            if recognizer.AcceptWaveform(data):
                result_json = recognizer.Result()
                result_dict = json.loads(result_json)
                text_result = result_dict.get("text", "").strip()

                if text_result:
                    print(f"\nĐã nhận dạng: {text_result}\n")

                    #  Nếu phát hiện lệnh dừng
                    if ("dừng" in text_result or 
                        "stop" in text_result or 
                        "thoát" in text_result):
                        print("--- Đã dừng chương trình ---")
                        break

            else:
                partial_json = recognizer.PartialResult()
                partial_dict = json.loads(partial_json)
                partial_text = partial_dict.get("partial", "")
                print(f"\rĐang nghe... {partial_text}", end="")

except KeyboardInterrupt:
    print("\n--- Đã dừng chương trình bằng tay ---")
except Exception as e:
    print(f"\nĐã xảy ra lỗi: {type(e).__name__}: {e}")
    sys.exit(1)

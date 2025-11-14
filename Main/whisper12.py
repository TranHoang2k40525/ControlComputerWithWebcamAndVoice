import sys
import queue
import time
import numpy as np
import sounddevice as sd
import whisper
import torch

# --- Cấu hình Whisper ---
MODEL_NAME = "base"         # Tên mô hình Whisper bạn đã tải (small, base, medium...)
LANGUAGE = "vi"            # Chỉ định ngôn ngữ "vi" (Vietnamese)
SAMPLERATE = 16000           # Tần số mẫu (Whisper bắt buộc 16kHz)
SILENCE_DURATION_SEC = 2.0   # Số giây im lặng để coi là kết thúc câu
SILENCE_THRESHOLD = 0.01     # Ngưỡng âm lượng (dưới mức này coi là im lặng)
                               # (Dùng cho dữ liệu float32)
# -------------------------

# Hàng đợi (Queue) để nhận dữ liệu âm thanh
q = queue.Queue()

def callback(indata, frames, time_info, status):
    """Được gọi mỗi khi có khối âm thanh mới."""
    if status:
        print(status, file=sys.stderr)
    # Whisper làm việc với mảng float32
    q.put(indata.copy())

try:
    # 1️ Tải mô hình Whisper
    print(f"Đang tải mô hình Whisper '{MODEL_NAME}'...")
    # Tự động dùng GPU (cuda) nếu có, không thì dùng CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(MODEL_NAME, device=device)
    print(f"Đã tải mô hình, chạy trên: {device}")
    
    # Tùy chọn để tối ưu hóa (fp16 chỉ dùng được với GPU)
    fp16 = device == "cuda"

    print(f"\n--- Bắt đầu lắng nghe (Im lặng {SILENCE_DURATION_SEC} giây để xử lý) ---")
    print("--- Nói 'dừng chương trình' hoặc 'thoát' để kết thúc ---")

    # Biến để lưu trữ âm thanh và trạng thái
    audio_buffer = []       # List để lưu các khối âm thanh
    silence_start_time = None # Mốc thời gian bắt đầu im lặng
    is_speaking = False     # Trạng thái đang nói

    # Bắt đầu luồng âm thanh
    with sd.InputStream(samplerate=SAMPLERATE,
                            blocksize=int(SAMPLERATE * 0.1), # Lấy mẫu 100ms/lần
                            device=None,
                            dtype="float32",
                            channels=1,
                            callback=callback):
        while True:
            # Lấy dữ liệu từ hàng đợi
            data = q.get()
            
            # Tính biên độ trung bình (để phát hiện im lặng)
            amplitude = np.mean(np.abs(data))

            if amplitude > SILENCE_THRESHOLD:
                # --- Đang có tiếng nói ---
                if not is_speaking:
                    print(f"\rĐang nghe... ", end="")
                    is_speaking = True
                
                audio_buffer.append(data)
                silence_start_time = None # Reset mốc im lặng
            
            elif is_speaking:
                # --- Đã im lặng (sau khi nói) ---
                if silence_start_time is None:
                    # Bắt đầu đếm thời gian im lặng
                    silence_start_time = time.time()
                
                # Kiểm tra xem đã im lặng đủ lâu chưa
                elif time.time() - silence_start_time > SILENCE_DURATION_SEC:
                    # --- Coi là kết thúc câu, bắt đầu xử lý ---
                    print(f"\rĐang xử lý... (chờ {int(SILENCE_DURATION_SEC)}s)", end="")
                    
                    # Nối các khối âm thanh lại
                    full_audio = np.concatenate(audio_buffer)
                    audio_buffer = [] # Xóa buffer cho câu tiếp theo
                    is_speaking = False
                    silence_start_time = None

                    # 4. Chuyển đổi âm thanh (Phần chính của Whisper)
                    result = model.transcribe(full_audio, 
                                              language=LANGUAGE, 
                                              fp16=fp16)
                    
                    text_result = result.get("text", "").strip()

                    # Xóa dòng "Đang xử lý..."
                    print("\r" + " " * 80 + "\r", end="")

                    # 5. In kết quả
                    if text_result:
                        print(f"Đã nhận dạng: {text_result}\n")
                        
                        # Kiểm tra lệnh dừng
                        text_lower = text_result.lower()
                        if ("dừng" in text_lower or 
                            "stop" in text_lower or 
                            "thoát" in text_lower):
                            print("--- Đã dừng chương trình (theo lệnh) ---")
                            break
                    
                    # Quay lại chờ câu nói mới
                    print("--- Chờ câu nói tiếp theo... ---")

except KeyboardInterrupt:
    print("\n--- Đã dừng chương trình bằng tay ---")
except Exception as e:
    print(f"\nĐã xảy ra lỗi: {type(e).__name__}: {e}")
    if "Invalid input device" in str(e):
        print("Lỗi: Không tìm thấy micro. Vui lòng kiểm tra thiết bị đầu vào.")
    if "No such file or directory: 'ffmpeg'" in str(e):
         print("Lỗi: CHƯA CÀI ĐẶT FFMPEG. Vui lòng xem lại hướng dẫn ở trên.")
    sys.exit(1)
import time
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

# --- 1. Cấu hình mô hình (Giữ nguyên) ---
model_size = "small"
device_type = "cpu"
compute_type = "int8"

print(f"Đang tải mô hình {model_size} (chỉ lần đầu tiên)...")
model = WhisperModel(model_size, device=device_type, compute_type=compute_type)
print("Mô hình đã tải xong.")

# --- 2. Cấu hình Ghi âm ---
fs = 16000  # Sample rate (Tần số lấy mẫu)
seconds = 5  # Độ dài ghi âm (5 giây)
audio_file = "recording.wav" # Tên file sẽ lưu tạm

print("---------------------------------")
print(f"Bắt đầu ghi âm trong {seconds} giây... (Hãy nói gì đó!)")

# Ghi âm từ micro
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Chờ cho đến khi ghi âm xong

# Lưu file âm thanh
write(audio_file, fs, myrecording)

print(f"Đã ghi âm xong và lưu vào file: {audio_file}")
print("Bắt đầu xử lý (việc này sẽ mất một lúc trên CPU)...")

# --- 3. Chạy nhận diện (Giống như trước) ---
start_time = time.time()
segments, info = model.transcribe(audio_file, beam_size=5, language="vi")

# --- 4. In kết quả (Giống như trước) ---
full_text = ""
for segment in segments:
    # segment.text đã có dấu câu, không cần thêm " "
    full_text += segment.text 

end_time = time.time()

print("---------------------------------")
print(f"Văn bản nhận diện được: {full_text.strip()}")
print(f"Tổng thời gian xử lý: {end_time - start_time:.2f} giây")
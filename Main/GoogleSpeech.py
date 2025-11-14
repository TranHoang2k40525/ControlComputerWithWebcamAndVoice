import speech_recognition as sr
import pyautogui
import pyperclip
import time
import sys

# --- CẤU HÌNH TỪ KHÓA ---
WAKE_WORDS = ["trợ lý", "này trợ lý", "hey trợ lý"]
EXIT_WORDS = ["kết thúc", "dừng lại", "thoát"]
LANGUAGE = "vi-VN"

# 1. Nhóm Nhập liệu (Có hành động thực tế)
CMD_TYPE = ["nhập văn bản", "gõ văn bản", "gõ chữ", "viết chữ", "chế độ gõ"]

# 2. Nhóm Chuột (Chưa gán hành động - Chỉ nhận diện)
CMD_CLICK = ["click", "bấm chuột", "nhấn chuột", "chuột trái", "chuột phải"]

# 3. Nhóm Tab (Chưa gán hành động - Chỉ nhận diện)
CMD_TAB = ["chuyển tab", "tab mới", "đóng tab", "tab trước"]

# 4. Nhóm App (Chưa gán hành động - Chỉ nhận diện)
CMD_OPEN = ["mở", "bật", "khởi động"]

def print_status(text, end="\n"):
    """Hàm in trạng thái sạch sẽ, xóa dòng cũ"""
    sys.stdout.write("\r" + " " * 50 + "\r") # Xóa dòng hiện tại
    sys.stdout.write(text + end)
    sys.stdout.flush()

def listen_phrase(recognizer, source, timeout=None, time_limit=None):
    """Thu âm và trả về text"""
    try:
        print_status("[*] Đang nghe...", end="")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=time_limit)
        print_status("[.] Đang xử lý...", end="")
        text = recognizer.recognize_google(audio, language=LANGUAGE)
        return text.lower().strip()
    except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
        return None

def action_paste(text):
    """Thực thi: Paste văn bản"""
    if not text: return
    # Format nhẹ: Viết hoa chữ đầu
    formatted_text = text[0].upper() + text[1:]
    
    pyperclip.copy(formatted_text)
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "v")
    print_status(f"[V] Đã nhập: '{formatted_text}'")

def main():
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.8 

    print("\n=== VOICE CONTROL SYSTEM V1.0 ===")
    print(f"[Target] Wake Word: {WAKE_WORDS}")
    print("[Target] Commands: Type | Click | Tab | Open")
    print("-" * 40)

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1.0)
        print("[OK] System Ready.\n")

        while True:
            # --- 1. PASSIVE MODE ---
            print_status("[_] Standby...", end="")
            text = listen_phrase(r, source)

            if not text: continue

            # Check Wake Word
            if any(w in text for w in WAKE_WORDS):
                print_status(f"[+] Active (Heard: '{text}')")
                
                # --- 2. ACTIVE MODE ---
                # print_status("[?] Command waiting...") 
                cmd = listen_phrase(r, source, timeout=5)

                if cmd:
                    # A. Kiểm tra Lệnh Thoát
                    if any(w in cmd for w in EXIT_WORDS):
                        print_status("[!] System Shutdown.")
                        break

                    # B. Kiểm tra Lệnh Nhập liệu (CÓ HÀNH ĐỘNG)
                    elif any(w in cmd for w in CMD_TYPE):
                        print_status("[Mode] Typing initiated...")
                        # Nghe nội dung để nhập
                        content = listen_phrase(r, source, timeout=5, time_limit=15)
                        if content:
                            action_paste(content)
                        else:
                            print_status("[!] No content heard.")

                    # C. Kiểm tra Lệnh Chuột (Placeholder)
                    elif any(w in cmd for w in CMD_CLICK):
                        print_status(f"[Log] Detect CLICK command: '{cmd}' (No Action)")

                    # D. Kiểm tra Lệnh Tab (Placeholder)
                    elif any(w in cmd for w in CMD_TAB):
                        print_status(f"[Log] Detect TAB command: '{cmd}' (No Action)")

                    # E. Kiểm tra Lệnh App (Placeholder)
                    elif any(w in cmd for w in CMD_OPEN):
                        print_status(f"[Log] Detect OPEN APP command: '{cmd}' (No Action)")

                    # F. Lệnh không xác định
                    else:
                        print_status(f"[x] Unknown command: '{cmd}'")
                
                else:
                    print_status("[!] Timeout (No command).")
                
                print("") # Xuống dòng cho vòng lặp mới

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Force Stop.")
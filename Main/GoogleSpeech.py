import os
import sys
import time

# Fix Unicode encoding for Windows console và bật unbuffered output
if sys.platform == 'win32':
    import codecs
    # Unbuffered output để hiển thị real-time
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Make sure current directory (Main/) is importable when running from repo root
import os
import sys
import time
import argparse
from typing import Optional
import speech_recognition as sr
# Make sure current directory (Main/) is importable when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from google_listen import create_recognizer, listen_phrase, adjust_for_ambient_noise
from google_model import VoiceModel
import command_dispatcher as dispatcher
import Actions

WAKE_WORDS = ["ok google", "hey google", "xin chào google"]
EXIT_WORDS = ["kết thúc", "dừng lại", "thoát"]


def discover_and_load_model(model_path: Optional[str], tokenizer_path: Optional[str], label_path: Optional[str], train_dir: str = 'Train'):
    """Nếu có path cụ thể thì dùng, nếu không thì tự động tìm kiếm.
    Trả về VoiceModel instance hoặc raise lỗi.
    """
    if model_path and tokenizer_path and label_path:
        return VoiceModel(model_path, tokenizer_path, label_path)

    # Tự động tìm kiếm
    print(f'[i] Đang tìm kiếm các file model trong thư mục "{train_dir}"...')
    m, t, l = VoiceModel.discover_from_train_dir(train_dir)
    print(f'[i] Đã phát hiện: model={m}, tokenizer={t}, label_encoder={l}')
    return VoiceModel(m, t, l)


def run_orchestrator(use_model: bool, prefer_model: bool, model_path: Optional[str], tokenizer_path: Optional[str], label_path: Optional[str], debug: bool = False):
    print('\n=== HỆ THỐNG ĐIỀU KHIỂN BẰNG GIỌNG NÓI ===')
    print(f'Từ khóa đánh thức: {WAKE_WORDS}')
    model = None
    if use_model:
        try:
            print('[*] Đang tải mô hình AI...')
            model = discover_and_load_model(model_path, tokenizer_path, label_path)
            print('[✓] Mô hình AI đã sẵn sàng!')
        except Exception as e:
            print(f'[!] Lỗi khi load model: {e}')
            print('[!] Tiếp tục với chế độ chỉ dùng từ khóa.')
            model = None
    else:
        print('[i] Chế độ: Chỉ sử dụng từ khóa (không dùng AI model)')

    r = create_recognizer()
    try:
        
        with sr.Microphone() as source:
            adjust_for_ambient_noise(r, source, duration=1.0)
            print('[OK] Microphone sẵn sàng. Đang chờ từ khóa đánh thức...\n')

            while True:
                text = listen_phrase(r, source, timeout=None)
                if not text:
                    continue
                print(f'[Nghe được] {text}', flush=True)

                if any(w in text for w in WAKE_WORDS):
                    print('[KÍCH HOẠT] Đã phát hiện từ khóa. Đang nghe lệnh...', flush=True)
                    cmd = listen_phrase(r, source, timeout=5, time_limit=12)
                    if not cmd:
                        print('[!] Không nghe thấy lệnh (timeout).', flush=True)
                        continue

                    # Kiểm tra lệnh thoát
                    if any(w in cmd for w in EXIT_WORDS):
                        print('[!] Nhận lệnh thoát. Đang tắt hệ thống.', flush=True)
                        break

                    # Kiểm tra lệnh nhập văn bản (xử lý đặc biệt)
                    typing_keywords = ['nhập văn bản', 'gõ văn bản', 'gõ chữ', 'viết chữ', 'chế độ gõ']
                    if any(kw in cmd for kw in typing_keywords):
                        print('[Chế độ] Bắt đầu nhập văn bản. Hãy nói nội dung...', flush=True)
                        # Nghe nội dung cần nhập
                        content = listen_phrase(r, source, timeout=5, time_limit=15)
                        if content:
                            Actions.execute_type_text(content)
                        else:
                            print('[!] Không nghe thấy nội dung để nhập.', flush=True)
                        continue  # Bỏ qua xử lý lệnh thông thường

                    executed = False
                    exec_label = None
                    exec_conf = None

                    # Ưu tiên model: thử model trước, sau đó mới keyword mapping
                    if prefer_model and model is not None:
                        try:
                            pred_label, confidence, _ = model.predict_action_from_text(cmd)
                            func = Actions.get_action_func(pred_label)
                            exec_label = pred_label
                            exec_conf = confidence
                            if func:
                                Actions.execute_action(func, pred_label, time.perf_counter(), is_continuous=False)
                                executed = True
                        except Exception as e:
                            print(f'[!] Dự đoán model thất bại: {e}')

                    if not executed:
                        # Thử keyword hoặc model fallback (dispatcher xử lý keyword->action và model tùy chọn)
                        result = dispatcher.dispatch_command(cmd, model=(None if prefer_model else model), use_model_if_no_keyword=(not prefer_model and model is not None), debug=debug)
                        executed = bool(result.get('executed'))
                        exec_label = exec_label or result.get('label')
                        exec_conf = exec_conf or result.get('confidence')

                    if executed:
                        print(f"[OK] Đã thực thi: {exec_label} (độ tin cậy={exec_conf})", flush=True)
                    else:
                        print(f"[--] Không thực thi được. label={exec_label} confidence={exec_conf}", flush=True)

    except KeyboardInterrupt:
        print('\n[!] Dừng bởi người dùng')


def parse_args():
    p = argparse.ArgumentParser(description='Voice control orchestrator')
    p.add_argument('--use-model', action='store_true', help='Load and use trained model')
    p.add_argument('--prefer-model', action='store_true', help='Prefer model prediction over keyword mapping')
    p.add_argument('--debug', action='store_true', help='Enable debug logging for dispatcher')
    p.add_argument('--no-model', action='store_true', help='Do not auto-load or use model even if discovered')
    p.add_argument('--model-path', help='Path to .h5 model file')
    p.add_argument('--tokenizer-path', help='Path to tokenizer .pkl')
    p.add_argument('--label-encoder-path', help='Path to label_encoder .pkl')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Tự động phát hiện model files nếu người dùng không chỉ định
    if not args.use_model and not getattr(args, 'no_model', False):
        try:
            print("[i] Đang tự động tìm kiếm model files...")
            m, t, l = VoiceModel.discover_anywhere()
            print(f"[✓] Đã tìm thấy model files:")
            print(f"    • Model: {m}")
            print(f"    • Tokenizer: {t}")
            print(f"    • Label Encoder: {l}")
            args.use_model = True
            # Chỉ set path nếu chưa được chỉ định
            args.model_path = args.model_path or m
            args.tokenizer_path = args.tokenizer_path or t
            args.label_encoder_path = args.label_encoder_path or l
        except Exception as e:
            # Không tìm thấy model — tiếp tục với chế độ keyword
            print(f"[!] Không tìm thấy model: {e}")
            print("[!] Tiếp tục với chế độ chỉ dùng từ khóa")

    run_orchestrator(use_model=args.use_model, prefer_model=args.prefer_model, model_path=args.model_path, tokenizer_path=args.tokenizer_path, label_path=args.label_encoder_path, debug=bool(args.debug))
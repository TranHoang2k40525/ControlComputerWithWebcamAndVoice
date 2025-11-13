
import os
import json
import queue
import threading
import time
from typing import List, Optional
import sounddevice as sd
from vosk import Model, KaldiRecognizer
MODEL_VN_PATH = "vosk-model-vn-0.4/vosk-model-vn-0.4"
MODEL_EN_PATH = "vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"

class SpeechController:
    """Controller: wake-word + listening window.
    Cách hoạt động chính:
    - Nhiều mô hình (ví dụ VN + EN) được nạp và chạy song song.
    - Dispatcher nhân bản khung audio tới từng worker queue, mỗi worker
      xử lý bằng KaldiRecognizer riêng của nó.
    - Khi bất kỳ recognizer nào trả về kết quả hoàn chỉnh (Result),
      ta xử lý kết quả đó bằng `_process_recognized_text`.
    - Các bản dịch (transcript) được thu thập trong _collected_texts;
      cơ chế loại trùng sẽ tránh thêm bản sao y hệt nhiều lần.

    Ghi chú về sample_rate:
    - sample_rate (ví dụ 16000) là tần số lấy mẫu khi thu âm. Hầu hết model
      VOSK dùng 16 kHz; nếu bạn thu ở tần số khác mà không chuyển đổi, kết quả
      có thể kém.
    """

    def __init__(self, model_paths: Optional[List[str]] = None, wake_phrases: Optional[List[str]] = None,
                 sample_rate: int = 16000, silence_timeout: float = 3.0):
        # cấu hình cơ bản
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        self.wake_phrases = [p.lower() for p in (wake_phrases or ["chào computer", "hello computer"])]

        # queue nhận audio thô (bytes) từ callback; dispatcher sẽ phân phối
        self._input_q = queue.Queue()
        self._stop_event = threading.Event()
        self._dispatcher_thread = None
        self._worker_threads = []
        self._stream = None

        # queue nhận kết quả (model_path, text, timestamp) do các worker đẩy vào
        self._result_q = queue.Queue()
        self._combiner_thread = None

        # mỗi recognizer có 1 queue nhận bản sao của các khung audio để xử lý song song
        self._rec_queues = []

        # trạng thái: 'idle' hoặc 'listening'
        self.state = 'idle'
        self._last_speech_time = 0.0
        self._collected_texts = []
        self._last_appended_text = ''
        # buffer tạm thời cho combiner: list của (model_path, text, ts)
        self._pending_results = []

        # Lưu danh sách đường dẫn và nạp mô hình qua phương thức riêng
        if model_paths is None:
            model_paths = [MODEL_VN_PATH, MODEL_EN_PATH]
        self.model_paths = list(model_paths)
        self.recognizers = []  # list of (model_path, Model, KaldiRecognizer)
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.load_models()

    def load_models(self):
        """Tải các model từ self.model_paths vào self.recognizers.

        Gọi lại phương thức này nếu bạn cập nhật self.model_paths trong runtime.
        """
        self.recognizers = []
        self._rec_queues = []
        for raw_p in self.model_paths:
            if not raw_p:
                continue
            # giải quyết đường dẫn: nếu là tương đối theo repo_root, thử ghép
            p = raw_p
            if not os.path.isabs(p):
                candidate = os.path.join(self.repo_root, p)
                if os.path.isdir(candidate):
                    p = candidate
            # nếu vẫn chưa là thư mục hợp lệ, thử dùng p as-is
            if not os.path.isdir(p):
                print(f" Không tìm thấy thư mục mô hình: {raw_p} (đã thử {p})")
                continue

            try:
                m = Model(p)
                r = KaldiRecognizer(m, self.sample_rate)
                self.recognizers.append((p, m, r))
                # tạo queue cho worker tương ứng
                self._rec_queues.append(queue.Queue())
                print(f" Đã nạp mô hình: {p}")
            except Exception as e:
                print(f" Lỗi khi nạp mô hình {p}: {e}")

        if not self.recognizers:
            raise FileNotFoundError("Không có mô hình VOSK hợp lệ trong model_paths")

    def save_model_paths(self, filename: str):
        """Lưu danh sách đường dẫn mô hình ra file JSON (kiểu cổ điển)."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.model_paths, f, ensure_ascii=False, indent=2)
            print(f" Đã lưu model_paths vào: {filename}")
        except Exception as e:
            print(f" Lỗi khi lưu model_paths: {e}")

    @staticmethod
    def load_model_paths_from_file(filename: str) -> List[str]:
        """Đọc danh sách đường dẫn model từ file JSON và trả về list"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as e:
            print(f" Lỗi khi đọc model_paths từ file: {e}")
        return []

    def _audio_callback(self, indata, frames, time_info, status):
        # indata: numpy array (frames, channels)
        if status:
            pass
        try:
            # chuyển ngay sang bytes để tiết kiệm công việc cho dispatcher
            self._input_q.put(indata.tobytes())
        except Exception:
            pass

    def start(self):
        if self._dispatcher_thread is not None and self._dispatcher_thread.is_alive():
            return
        self._stop_event.clear()
        # khởi dispatcher
        self._dispatcher_thread = threading.Thread(target=self._dispatcher, daemon=True)
        self._dispatcher_thread.start()

        # khởi combiner (nhận kết quả từ workers và gom/merge)
        self._combiner_thread = threading.Thread(target=self._combiner, daemon=True)
        self._combiner_thread.start()

        # worker threads (1 per recognizer)
        self._worker_threads = []
        for i in range(len(self.recognizers)):
            t = threading.Thread(target=self._rec_worker, args=(i,), daemon=True)
            self._worker_threads.append(t)
            t.start()

        # khởi stream sau khi threads đã sẵn sàng
        blocksize = int(max(256, self.sample_rate // 10))  # ~100ms blocks
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', callback=self._audio_callback, blocksize=blocksize, latency='low')
        self._stream.start()
        print(" Bắt đầu thu âm (micro) và xử lý (dispatcher + workers)")

    def stop(self):
        self._stop_event.set()
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        # đợi các luồng dừng
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=1.0)
        if self._combiner_thread is not None:
            self._combiner_thread.join(timeout=1.0)
        for t in self._worker_threads:
            t.join(timeout=0.5)
        print(" Đã dừng")

    def _process_recognized_text(self, text: str):
        text = text.strip().lower()
        if not text:
            return
        print(f" KQ nhận dạng: '{text}'")

        if self.state == 'idle':
            # kiểm tra wake phrase
            for wp in self.wake_phrases:
                if wp in text:
                    self.state = 'listening'
                    self._collected_texts = []
                    self._last_appended_text = ''
                    self._last_speech_time = time.time()
                    print(" Phát hiện lệnh khởi động -> đang lắng nghe (đếm ngược 3s khi im lặng)")
                    return
            return

        if self.state == 'listening':
            if text != self._last_appended_text:
                self._collected_texts.append(text)
                self._last_appended_text = text
                print(f" Thêm vào buffer: '{text}'")
            self._last_speech_time = time.time()

    def _finalize_listening(self):
        self.state = 'idle'
        transcript = ' '.join(self._collected_texts).strip()
        if transcript:
            print(" Kết thúc lắng nghe. Nội dung lắng nghe:")
            print(transcript)
        else:
            print(" Kết thúc lắng nghe. Không có nội dung.")
        print(" Trạng thái: chờ wake-word 'chào computer' để kích hoạt lại")
        self._collected_texts = []
        self._last_appended_text = ''

    def _dispatcher(self):
        """Dispatcher: đọc audio bytes từ _input_q và phân phối cho các queue của worker.

        Việc phân phối này đảm bảo mỗi recognizer nhận được cùng một khung âm
        để xử lý song song (parallel)."""
        while not self._stop_event.is_set():
            try:
                b = self._input_q.get(timeout=0.2)
            except queue.Empty:
                # kiểm tra timeout
                if self.state == 'listening' and (time.time() - self._last_speech_time) > self.silence_timeout:
                    self._finalize_listening()
                continue

            # phân phối khung đến từng worker (mỗi worker có queue riêng)
            for q in self._rec_queues:
                try:
                    q.put_nowait(b)
                except Exception:
                    # nếu queue đầy, bỏ khung cũ nhất rồi thêm
                    try:
                        q.get_nowait()
                        q.put_nowait(b)
                    except Exception:
                        pass

    def _merge_texts(self, items):
        """Merge một list các text (theo thứ tự) thành 1 transcript duy nhất.

        items: list of (model_path, text, ts)
        Chiến lược đơn giản:
        - giữ thứ tự thời gian
        - nếu một text chứa text khác -> chọn text dài hơn
        - nếu tương tự (ratio > 0.6) -> chọn text dài hơn
        - khác thì nối các token mới (loại trùng token đã thấy)
        """
        from difflib import SequenceMatcher

        texts = [t for (_m, t, _ts) in items if t]
        if not texts:
            return ''
        merged = ''
        seen_tokens = set()
        for t in texts:
            t = t.strip()
            if not merged:
                merged = t
                for w in t.split():
                    seen_tokens.add(w)
                continue
            # nếu t chứa merged hoặc merged chứa t
            if t in merged:
                continue
            if merged in t:
                merged = t
                for w in t.split():
                    seen_tokens.add(w)
                continue
            # nếu rất giống -> chọn dài hơn
            r = SequenceMatcher(None, merged, t).ratio()
            if r > 0.6:
                merged = merged if len(merged) >= len(t) else t
                for w in merged.split():
                    seen_tokens.add(w)
                continue
            # khác: append only new tokens to avoid duplication
            parts = []
            for w in t.split():
                if w not in seen_tokens:
                    parts.append(w)
                    seen_tokens.add(w)
            if parts:
                merged = (merged + ' ' + ' '.join(parts)).strip()
        return merged

    def _combiner(self):
        """Read recognized outputs from workers and combine results.

        Behaviour:
        - If idle: check any result for wake_phrase -> set listening.
        - If listening: collect results until silence_timeout since last result,
          then merge and finalize.
        """
        while not self._stop_event.is_set():
            try:
                mpath, text, ts = self._result_q.get(timeout=0.2)
            except queue.Empty:
                # nếu đang listening và timeout
                if self.state == 'listening' and (time.time() - self._last_speech_time) > self.silence_timeout:
                    # finalize using pending results
                    merged = self._merge_texts(self._pending_results)
                    if merged:
                        print(" KQ kết hợp: ", merged)
                        # append to collected_texts once
                        if merged != self._last_appended_text:
                            self._collected_texts.append(merged)
                            self._last_appended_text = merged
                    self._pending_results = []
                    self._finalize_listening()
                continue

            # lưu result
            # Nếu đang ở trạng thái 'idle', kiểm tra wake-phrase trước khi lưu
            low = text.lower()
            if self.state == 'idle':
                matched_wp = None
                for wp in self.wake_phrases:
                    if wp in low:
                        matched_wp = wp
                        break
                if matched_wp:
                    # Khi phát hiện wake-word, chuyển sang listening
                    self.state = 'listening'
                    self._collected_texts = []
                    self._last_appended_text = ''
                    # Xóa mọi kết quả trước đó để không bao gồm pre-wake outputs
                    self._pending_results = []
                    self._last_speech_time = ts
                    print(f" Phát hiện wake-word (từ model): đang lắng nghe (wp='{matched_wp}')")
                    # Nếu phần text có nội dung sau wake-phrase, tách phần sau và lưu
                    try:
                        idx = low.index(matched_wp) + len(matched_wp)
                        tail = text[idx:].strip()
                        if tail:
                            # lưu phần sau wake phrase như bắt đầu của lượt lắng nghe
                            self._pending_results.append((mpath, tail, ts))
                            print(f" (sau wake) [{os.path.basename(mpath)}] -> {tail}")
                    except Exception:
                        pass
                    continue

            # Nếu đang listening thì lưu kết quả vào pending
            if self.state == 'listening':
                # lưu result (model_path, text, ts)
                self._pending_results.append((mpath, text, ts))
                # update last speech time
                self._last_speech_time = ts
                # debug print which model produced the text
                print(f"[{os.path.basename(mpath)}] -> {text}")
                continue

            # Nếu vẫn ở idle (không wake detected) thì bỏ qua (không lưu pre-wake outputs)
            # nhưng vẫn in debug để bạn thấy model hoạt động
            print(f"(pre-wake) [{os.path.basename(mpath)}] -> {text}")
            continue

    def _rec_worker(self, idx: int):
        """Worker xử lý audio cho recognizer thứ idx. Chạy song song cho mỗi model."""
        mpath, mobj, rec = self.recognizers[idx]
        q = self._rec_queues[idx]
        while not self._stop_event.is_set():
            try:
                b = q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if rec.AcceptWaveform(b):
                    res = rec.Result()
                    try:
                        j = json.loads(res)
                        text = j.get('text', '').strip()
                    except Exception:
                        text = ''
                    if text:
                        # đưa kết quả vào result queue để combiner xử lý
                        ts = time.time()
                        try:
                            self._result_q.put_nowait((mpath, text, ts))
                        except Exception:
                            # nếu queue đầy, bỏ qua
                            pass
                else:
                    # phần partial không dùng ở đây (đơn giản hóa)
                    pass
            except Exception as e:
                print(f" Lỗi recognizer ({mpath}): {e}")


if __name__ == '__main__':
    # Chế độ chạy độc lập (lối cổ điển): dùng hai thư mục mặc định ở repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # model paths mặc định (theo nested layout đã khai báo)
    model_paths = []
    for p in (MODEL_VN_PATH, MODEL_EN_PATH):
        if not p:
            continue
        if os.path.isabs(p) and os.path.isdir(p):
            model_paths.append(p)
            continue
        candidate = os.path.join(repo_root, p)
        if os.path.isdir(candidate):
            model_paths.append(candidate)

    if not model_paths:
        print("Không tìm thấy mô hình ở đường dẫn mặc định (đang tìm pattern nested):")
        print(' -', os.path.join(repo_root, MODEL_VN_PATH))
        print(' -', os.path.join(repo_root, MODEL_EN_PATH))
        print("Hãy giải nén mô hình vào các thư mục trên (ví dụ: vosk-model-vn-0.4/vosk-model-vn-0.4) hoặc sửa MODEL_*_PATH trong file này.")
        raise SystemExit(1)

    sc = SpeechController(model_paths=model_paths,
                          wake_phrases=["chào computer", "hello computer","xin chào", "alo"],
                          sample_rate=16000,
                          silence_timeout=3.0)
    try:
        sc.start()
        print("Đang chạy. Nói 'xin chào' để kích hoạt.")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('Người dùng dừng chương trình')
    finally:
        sc.stop()




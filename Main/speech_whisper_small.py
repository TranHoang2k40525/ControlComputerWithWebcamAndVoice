import os
import queue
import threading
import time
from typing import List, Optional

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# Tên model faster-whisper (mặc định dùng phiên bản multilingual 'small')
# Bạn có thể đặt 'small.en' nếu muốn chỉ tiếng Anh, hoặc đường dẫn tới model cache.
MODEL_NAME = "small"


class WhisperController:
    """Controller đơn giản dựa trên faster-whisper.

    Hoạt động tương tự `SpeechController` dùng VOSK:
    - Thu micro thành các khung nhỏ (chunk)
    - Gọi model.transcribe trên chunk âm thanh
    - Dò wake-phrases trong text để chuyển sang trạng thái listening
    - Trong listening window, gom các text thành transcript cuối cùng sau silence_timeout

    Lưu ý: faster-whisper chạy batch CPU/onnx; tuỳ cấu hình máy, bạn có thể muốn
    điều chỉnh model/device/compute_type khi tạo WhisperModel.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu",
                 compute_type: Optional[str] = None,
                 wake_phrases: Optional[List[str]] = None,
                 sample_rate: int = 16000, chunk_seconds: float = 1.0,
                 silence_timeout: float = 2.0):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.chunk_seconds = max(0.25, float(chunk_seconds))
        self.silence_timeout = float(silence_timeout)

        self.wake_phrases = [p.lower() for p in (wake_phrases or ["chào computer", "hello computer","hello", "xin chào", "alo"])]

        # audio queue (bytes) từ callback
        self._input_q = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._stream = None

        # trạng thái lắng nghe
        self.state = 'idle'
        self._last_speech_time = 0.0
        self._collected_texts: List[str] = []
        self._last_appended_text = ''

        # VAD setup (webrtcvad) - optional but recommended
        if webrtcvad is None:
            print("Warning: webrtcvad not installed. Install with `pip install webrtcvad` for better VAD.")
            self.vad = None
        else:
            self.vad = webrtcvad.Vad(2)  # aggressiveness 0-3
        # frame duration for VAD (ms)
        self.vad_frame_ms = 30
        self.vad_frame_bytes = int(self.sample_rate * (self.vad_frame_ms / 1000.0) * 2)  # 16-bit -> 2 bytes

        # nạp model
        if not HAVE_FASTER_WHISPER:
            raise ImportError("faster-whisper is not installed. Install it with `pip install faster-whisper` to enable transcription.")

        print(f"Loading WhisperModel('{self.model_name}') on {self.device} (compute_type={self.compute_type})...")
        try:
            if self.compute_type:
                self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
            else:
                self.model = WhisperModel(self.model_name, device=self.device)
        except Exception as e:
            print("Không thể nạp WhisperModel:", e)
            raise

    def _audio_callback(self, indata, frames, time_info, status):
        # indata: numpy int16
        if status:
            pass
        try:
            self._input_q.put(indata.copy())
        except Exception:
            pass

    def _bytes_from_queue(self, timeout: float = 0.2) -> Optional[bytes]:
        """Collect next available block of audio bytes from the input queue."""
        try:
            data = self._input_q.get(timeout=timeout)
        except Exception:
            return None
        arr = np.asarray(data)
        if arr.ndim > 1:
            arr = arr[:, 0]
        return arr.tobytes()

    def _transcribe_buffer(self, pcm_int16: bytes) -> str:
        """Transcribe raw PCM int16 bytes (mono, sample_rate) to text using faster-whisper."""
        try:
            # convert bytes -> numpy float32
            arr = np.frombuffer(pcm_int16, dtype=np.int16)
            if arr.size == 0:
                return ''
            audio_f32 = (arr.astype(np.float32) / 32768.0)
            # call model.transcribe; not conditioning on previous to keep segments independent
            try:
                segments, info = self.model.transcribe(audio_f32, language=None)
                segs = segments
            except Exception:
                segs = self.model.transcribe(audio_f32)

            texts = []
            for s in segs:
                t = getattr(s, 'text', None) or (s.get('text') if isinstance(s, dict) else str(s))
                if t:
                    texts.append(str(t).strip())
            return ' '.join(texts).strip()
        except Exception as e:
            print("Transcribe error:", e)
            return ''

    def start(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()

        # tạo worker thread xử lý accumulation -> transcribe
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        blocksize = int(max(256, self.sample_rate // 10))
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', callback=self._audio_callback, blocksize=blocksize, latency='low')
        self._stream.start()
        print("WhisperController: Bắt đầu thu âm và xử lý")

    def stop(self):
        self._stop_event.set()
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
        print("WhisperController: Đã dừng")

    def _worker(self):
        # Real-time incremental transcription using a sliding buffer.
        # Strategy:
        # - Keep a float32 audio buffer and a buffer_start_time (absolute time of first sample)
        # - Periodically call model.transcribe on the buffer with
        #   condition_on_previous_text=True so model can keep context
        # - Use segment start/end timestamps to only emit newly recognized segments
        # - Trim history to `max_history_seconds` to avoid unbounded growth

        chunk_frames = int(self.chunk_seconds * self.sample_rate)
        max_history_seconds = max(10.0, 30.0)
        max_history_frames = int(max_history_seconds * self.sample_rate)

        buf = np.zeros((0,), dtype=np.int16)
        buf_start_time = None  # absolute time of the first sample in buf
        last_transcribe_time = 0.0
        last_output_time = 0.0  # absolute time up to which we've emitted segments

        while not self._stop_event.is_set():
            try:
                data = self._input_q.get(timeout=0.2)
            except Exception:
                # no new audio; check for silence-based finalize
                if self.state == 'listening' and (time.time() - self._last_speech_time) > self.silence_timeout:
                    self._finalize_listening()
                continue

            # append incoming frames
            arr = np.asarray(data)
            if arr.ndim > 1:
                arr = arr[:, 0]

            # set buf_start_time when buffer was empty
            now = time.time()
            if buf.size == 0:
                # approximate: samples just arrived now, so first-sample time ~ now - len(arr)/sr
                buf_start_time = now - (len(arr) / float(self.sample_rate))

            buf = np.concatenate([buf, arr])

            # trim history if too long
            if len(buf) > max_history_frames:
                # drop the oldest samples
                drop = len(buf) - max_history_frames
                buf = buf[drop:]
                if buf_start_time is not None:
                    buf_start_time += (drop / float(self.sample_rate))

            # Decide whether to transcribe now: either enough frames or interval elapsed
            now = time.time()
            if len(buf) >= chunk_frames or (now - last_transcribe_time) > (self.chunk_seconds / 2.0):
                last_transcribe_time = now
                # convert to float32 in [-1,1]
                audio_f32 = (buf.astype(np.float32) / 32768.0)
                try:
                    # call transcribe with condition_on_previous_text to keep continuity
                    segments_iter = None
                    try:
                        segments, info = self.model.transcribe(audio_f32, language=None, condition_on_previous_text=True)
                        segments_iter = segments
                    except Exception:
                        segments_iter = self.model.transcribe(audio_f32, condition_on_previous_text=True)

                    # collect segments into list to inspect timestamps
                    segs = []
                    for seg in segments_iter:
                        if hasattr(seg, 'start') and hasattr(seg, 'end'):
                            text = getattr(seg, 'text', '')
                            segs.append({'start': float(seg.start), 'end': float(seg.end), 'text': str(text).strip()})
                        else:
                            # fallback: treat whole returned text as one segment starting at 0
                            t = getattr(seg, 'text', None) or (seg.get('text') if isinstance(seg, dict) else str(seg))
                            segs.append({'start': 0.0, 'end': len(buf) / float(self.sample_rate), 'text': str(t).strip()})

                    # compute absolute times for segments and only handle new ones
                    if buf_start_time is None:
                        buf_start_time = now - (len(buf) / float(self.sample_rate))

                    new_segments = []
                    for s in segs:
                        abs_start = buf_start_time + s['start']
                        abs_end = buf_start_time + s['end']
                        # consider new if its end is after last_output_time + small epsilon
                        if abs_end > last_output_time + 0.01:
                            new_segments.append({'start': abs_start, 'end': abs_end, 'text': s['text']})

                    # if there are new segments, process them in chronological order
                    if new_segments:
                        for s in new_segments:
                            ts = s['end']
                            text = s['text']
                            if text:
                                # handle as incremental transcription result
                                self._handle_transcribed_text(text, ts)
                                last_output_time = max(last_output_time, s['end'] + buf_start_time)

                except Exception as e:
                    print("Whisper transcribe error:", e)

    def _handle_transcribed_text(self, text: str, ts: float):
        low = text.lower().strip()
        print(f"[whisper:{self.model_name}] -> {text}")

        if self.state == 'idle':
            # check wake phrases
            for wp in self.wake_phrases:
                if wp in low:
                    # wake detected; set listening and capture tail
                    self.state = 'listening'
                    self._collected_texts = []
                    self._last_appended_text = ''
                    self._last_speech_time = ts
                    print(f"Whisper: phát hiện wake-word '{wp}', đang lắng nghe")
                    # capture tail after wake
                    try:
                        idx = low.index(wp) + len(wp)
                        tail = text[idx:].strip()
                        if tail:
                            self._collected_texts.append(tail)
                            self._last_appended_text = tail
                            print(f"Whisper (sau wake) -> {tail}")
                    except Exception:
                        pass
                    return
            # pre-wake: ignore for final transcript
            return

        if self.state == 'listening':
            # append if not duplicate
            if text != self._last_appended_text:
                self._collected_texts.append(text)
                self._last_appended_text = text
            self._last_speech_time = ts

    def _finalize_listening(self):
        self.state = 'idle'
        transcript = ' '.join(self._collected_texts).strip()
        if transcript:
            print("Whisper: Kết thúc lắng nghe. Nội dung lắng nghe:")
            print(transcript)
        else:
            print("Whisper: Kết thúc lắng nghe. Không có nội dung.")
        print("Whisper: chờ wake-word để kích hoạt lại")
        self._collected_texts = []
        self._last_appended_text = ''

    def run_forever(self, activation_timeout: float = None):
        """Main loop implementing the state machine you requested.

        Behavior:
        - Wait for user to speak (VAD). When a speech segment ends (silence > 5s), transcribe it.
        - If transcription contains a wake-phrase -> enter listening mode.
        - In listening mode: print 'Lắng nghe' and wait up to 5s for the user to start speaking. If no speech -> exit listening with empty text.
          If speech occurs, record until silence >5s, then transcribe and print text. If the text equals a termination keyword ("kết thúc", "thoát"), break out of listening and return to idle.
        - Loop forever until KeyboardInterrupt.
        """
        print("Whisper VAD-run loop: bắt đầu. Nói wake-phrase để kích hoạt.")
        silence_seconds = 5.0
        if activation_timeout is not None:
            activation_timeout = float(activation_timeout)

        # Start audio stream
        if self._stream is None:
            blocksize = int(max(256, self.sample_rate // 20))
            self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', callback=self._audio_callback, blocksize=blocksize, latency='low')
            self._stream.start()

        # circular buffer of bytes to assemble frames
        pending = bytearray()

        try:
            while True:
                # Wait for initial speech (idle state)
                # Build frames and use VAD to detect speech start
                if not HAVE_WEBRTC_VAD:
                    # fallback: energy-based VAD implemented in pure Python
                    # We'll process frames of size `vad_frame_bytes` and compute short-term RMS
                    def energy_is_speech(frame_bytes: bytes, thresh: float = 0.02) -> bool:
                        # frame_bytes are int16 pcm little-endian
                        if not frame_bytes:
                            return False
                        arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
                        if arr.size == 0:
                            return False
                        rms = np.sqrt(np.mean(arr * arr)) / 32768.0
                        return float(rms) >= float(thresh)

                    speech_seen = False
                    speech_buffer = bytearray()
                    last_voice_time = None
                    # loop until we have a full speech segment ended by silence > silence_seconds
                    while True:
                        b = self._bytes_from_queue(timeout=0.5)
                        if b:
                            pending.extend(b)
                        # while we have enough for a VAD frame, test
                        while len(pending) >= self.vad_frame_bytes:
                            frame = bytes(pending[:self.vad_frame_bytes])
                            del pending[:self.vad_frame_bytes]
                            is_speech = False
                            try:
                                is_speech = energy_is_speech(frame)
                            except Exception:
                                is_speech = False
                            if is_speech:
                                speech_seen = True
                                last_voice_time = time.time()
                                speech_buffer.extend(frame)
                            else:
                                # append non-speech frames as well to retain context
                                speech_buffer.extend(frame)

                        # if we've seen speech and now silence for > silence_seconds, break
                        if speech_seen and last_voice_time is not None and (time.time() - last_voice_time) > silence_seconds:
                            break
                        # if no speech yet, keep waiting
                        if not speech_seen and len(speech_buffer) > (self.sample_rate * 2):
                            excess = len(speech_buffer) - int(self.sample_rate * 2)
                            if excess > 0:
                                del speech_buffer[:excess]

                    if not speech_buffer:
                        continue
                    text = self._transcribe_buffer(bytes(speech_buffer))
                else:
                    # accumulate until we detect a speech end (voice then silence)
                    speech_seen = False
                    speech_buffer = bytearray()
                    last_voice_time = None
                    # loop until we have a full speech segment ended by silence > silence_seconds
                    while True:
                        b = self._bytes_from_queue(timeout=0.5)
                        if b:
                            pending.extend(b)
                        # while we have enough for a VAD frame, test
                        while len(pending) >= self.vad_frame_bytes:
                            frame = bytes(pending[:self.vad_frame_bytes])
                            del pending[:self.vad_frame_bytes]
                            is_speech = False
                            try:
                                is_speech = self.vad.is_speech(frame, sample_rate=self.sample_rate)
                            except Exception:
                                is_speech = False
                            if is_speech:
                                speech_seen = True
                                last_voice_time = time.time()
                                speech_buffer.extend(frame)
                            else:
                                # treat non-speech frames as potential silence; append them too (so we keep full audio)
                                speech_buffer.extend(frame)

                        # if we've seen speech and now silence for > silence_seconds, break
                        if speech_seen and last_voice_time is not None and (time.time() - last_voice_time) > silence_seconds:
                            break
                        # if no speech yet, keep waiting
                        if not speech_seen and len(speech_buffer) > (self.sample_rate * 2):
                            # prevent unbounded growth, trim older data
                            # keep last 2s
                            excess = len(speech_buffer) - int(self.sample_rate * 2)
                            if excess > 0:
                                del speech_buffer[:excess]

                    if not speech_buffer:
                        continue
                    text = self._transcribe_buffer(bytes(speech_buffer))

                low = text.lower().strip()
                print(f"[Detected] -> {text}")

                # check for wake phrase
                is_wake = any(wp in low for wp in self.wake_phrases)
                if not is_wake:
                    # not a wake; continue waiting
                    continue

                # wake detected -> enter listening loop
                print("=== Lắng nghe (activated) ===")
                # now wait up to silence_seconds for user to start speaking
                # gather frames; if no speech in silence_seconds -> end with empty transcript
                # reuse pending buffer
                listen_start = time.time()
                speech_started = False
                listen_buffer = bytearray()
                last_voice_time = None

                # wait for speech to start within silence_seconds
                start_wait_deadline = time.time() + silence_seconds
                while time.time() < start_wait_deadline:
                    b = self._bytes_from_queue(timeout=0.5)
                    if b:
                        pending.extend(b)
                    # test frames
                    while len(pending) >= self.vad_frame_bytes:
                        frame = bytes(pending[:self.vad_frame_bytes])
                        del pending[:self.vad_frame_bytes]
                        is_speech = False
                        try:
                            is_speech = self.vad.is_speech(frame, sample_rate=self.sample_rate)
                        except Exception:
                            is_speech = False
                        listen_buffer.extend(frame)
                        if is_speech:
                            speech_started = True
                            last_voice_time = time.time()
                            break
                    if speech_started:
                        break

                if not speech_started:
                    print("Kết thúc lắng nghe (không có giọng): transcript trống")
                    print("=== End listening ===")
                    continue

                # record until silence > silence_seconds
                while True:
                    b = self._bytes_from_queue(timeout=0.5)
                    if b:
                        pending.extend(b)
                    while len(pending) >= self.vad_frame_bytes:
                        frame = bytes(pending[:self.vad_frame_bytes])
                        del pending[:self.vad_frame_bytes]
                        listen_buffer.extend(frame)
                        is_speech = False
                        try:
                            is_speech = self.vad.is_speech(frame, sample_rate=self.sample_rate)
                        except Exception:
                            is_speech = False
                        if is_speech:
                            last_voice_time = time.time()
                    # check silence end
                    if last_voice_time is not None and (time.time() - last_voice_time) > silence_seconds:
                        break

                # transcribe listened buffer
                final_text = self._transcribe_buffer(bytes(listen_buffer))
                print("Kết thúc lắng nghe. Văn bản:", final_text)
                print("=== End listening ===")

                low_final = final_text.lower().strip()
                # termination keywords
                if any(k in low_final for k in ['kết thúc', 'thoát', 'dừng']):
                    print("Nghe lệnh kết thúc hội thoại - trở về chế độ chờ wake-word")
                    continue
                # otherwise loop back and wait for next wake
        except KeyboardInterrupt:
            print("Run loop stopped by user")
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass


if __name__ == '__main__':
    # Runner đơn giản
    wc = WhisperController(model_name=MODEL_NAME, device='cpu', compute_type=None,
                            sample_rate=16000, chunk_seconds=1.0, silence_timeout=2.0)
    try:
        # Use the VAD-driven run loop which waits for wake-phrases and then listens
        wc.run_forever()
    except KeyboardInterrupt:
        print('Dừng bởi người dùng')
    finally:
        wc.stop()

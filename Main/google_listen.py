import speech_recognition as sr
import sys

LANGUAGE_DEFAULT = "vi-VN"


def create_recognizer(energy_threshold=300, dynamic_energy_threshold=True, pause_threshold=0.8):
    r = sr.Recognizer()
    r.energy_threshold = energy_threshold
    r.dynamic_energy_threshold = dynamic_energy_threshold
    r.pause_threshold = pause_threshold
    return r


def listen_phrase(recognizer: sr.Recognizer, source, timeout=None, time_limit=None, language: str = LANGUAGE_DEFAULT):
    """Listen once and return recognized lower-case text or None on failure."""
    try:
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=time_limit)
        text = recognizer.recognize_google(audio, language=language)
        return text.lower().strip()
    except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
        return None


def adjust_for_ambient_noise(recognizer: sr.Recognizer, source, duration: float = 1.0):
    try:
        recognizer.adjust_for_ambient_noise(source, duration=duration)
    except Exception:
        # best-effort; microphone may not be available during unit tests
        pass

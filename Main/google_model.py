import os
import pickle
import numpy as np
import glob
from typing import Optional, Tuple, List

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    # allow importing file even if tensorflow isn't installed in this environment
    load_model = None
    pad_sequences = None


class VoiceModel:
    """Load Keras model + tokenizer + label encoder để dự đoán hành động.

    Tokenizer phải là Keras Tokenizer đã pickle và label encoder phải là
    sklearn LabelEncoder đã pickle (có thuộc tính `.classes_`).
    """

    def __init__(self, model_path: str, tokenizer_path: str, label_encoder_path: str, max_len: int = 20):
        if load_model is None:
            raise RuntimeError("TensorFlow không có sẵn trong môi trường này")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Không tìm thấy file tokenizer: {tokenizer_path}")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Không tìm thấy file label encoder: {label_encoder_path}")

        print(f"[*] Đang load model từ: {model_path}")
        self.model = load_model(model_path)
        print(f"[*] Đang load tokenizer từ: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"[*] Đang load label encoder từ: {label_encoder_path}")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.max_len = max_len
        print(f"[✓] Model đã load thành công! Số lớp hành động: {len(self.label_encoder.classes_)}")

    @classmethod
    def discover_from_train_dir(cls, train_dir: str = 'Train') -> Tuple[str, str, str]:
        """Tìm kiếm model/tokenizer/label encoder files trong thư mục `train_dir`.

        Trả về tuple (model_path, tokenizer_path, label_encoder_path) hoặc raise FileNotFoundError.
        Sử dụng pattern: voice_action_model*.h5, tokenizer*.pkl, label_encoder*.pkl.
        """
        model_candidates = glob.glob(os.path.join(train_dir, 'voice_action_model*.h5'))
        if not model_candidates:
            model_candidates = glob.glob(os.path.join(train_dir, '*.h5'))
        tokenizer_candidates = glob.glob(os.path.join(train_dir, 'tokenizer*.pkl'))
        label_candidates = glob.glob(os.path.join(train_dir, 'label_encoder*.pkl'))

        if not model_candidates:
            raise FileNotFoundError(f"Không tìm thấy file .h5 model trong '{train_dir}'")
        if not tokenizer_candidates:
            raise FileNotFoundError(f"Không tìm thấy file tokenizer .pkl trong '{train_dir}'")
        if not label_candidates:
            raise FileNotFoundError(f"Không tìm thấy file label_encoder .pkl trong '{train_dir}'")

        # Chọn file đầu tiên tìm được
        model_path = model_candidates[0]
        tokenizer_path = tokenizer_candidates[0]
        label_path = label_candidates[0]
        return model_path, tokenizer_path, label_path

    @classmethod
    def discover_anywhere(cls, search_dirs: Optional[List[str]] = None) -> Tuple[str, str, str]:
        """Tìm kiếm model/tokenizer/label files trong các thư mục phổ biến (Train, thư mục gốc, thư mục cha).

        Trả về tuple (model_path, tokenizer_path, label_encoder_path) hoặc raise FileNotFoundError.
        """
        if search_dirs is None:
            # Tìm kiếm theo thứ tự: thư mục cha (..), thư mục hiện tại (.), Train/, ../Train/
            search_dirs = ['..', '.', 'Train', os.path.join('..', 'Train')]
        
        for d in search_dirs:
            try:
                result = cls.discover_from_train_dir(d)
                # Chuyển sang absolute path để tránh nhầm lẫn
                abs_model = os.path.abspath(result[0])
                abs_tokenizer = os.path.abspath(result[1])
                abs_label = os.path.abspath(result[2])
                return abs_model, abs_tokenizer, abs_label
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"Không tìm thấy model/tokenizer/label trong các thư mục: {search_dirs}")

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Return tokenizer sequences (list of token id lists) without padding."""
        return self.tokenizer.texts_to_sequences(texts)

    def save_token_sequences(self, texts: List[str], out_path: str):
        """Convert texts to token id sequences and save as a numpy object array at `out_path`.

        This mirrors the user's request to keep a tokenized array for the model to read.
        """
        seqs = self.texts_to_sequences(texts)
        # Save as object array to preserve variable-length sequences
        np.save(out_path, np.array(seqs, dtype=object))
        return out_path

    def texts_to_padded(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=self.max_len)

    def predict_action_from_text(self, input_text: str):
        """Return a tuple (pred_label, confidence, probs_array).

        pred_label is the string label (from label_encoder.classes_), confidence is float (0..100)
        and probs_array is the raw prediction array.
        """
        x = self.texts_to_padded([input_text])
        probs = self.model.predict(x)[0]
        idx = int(np.argmax(probs))
        label = self.label_encoder.classes_[idx]
        confidence = float(probs[idx]) * 100.0
        return label, confidence, probs

# model.py - Load model và predict gesture (gộp config: LABEL_ENCODER, N_FRAMES, etc.)

import os
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = '../gesture_lstm_model.h5'
N_FRAMES = 30
FEATURES = 84
CONF_THRESHOLD = 0.5

LABEL_ENCODER = np.array([
    'clickchuotphai', 'clickchuottrai', 'dichuyenchuot', 'dungchuongtrinh',
    'mochorme', 'phongto', 'thunho', 'vuotlen', 'vuotphai', 'vuottrai', 'vuotxuong'
])

GESTURE_TYPES = {
    'dichuyenchuot': 'continuous',
    'vuotlen': 'continuous',
    'vuotxuong': 'continuous',
    'vuotphai': 'continuous',
    'vuottrai': 'continuous',
    'clickchuotphai': 'discrete',
    'clickchuottrai': 'discrete',
    'dungchuongtrinh': 'discrete',
    'mochorme': 'discrete',
    'phongto': 'discrete',
    'thunho': 'discrete'
}

def load_gesture_model():
    """
    Load model và label encoder.
    Returns: model, label_encoder
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Không tìm model tại {MODEL_PATH}!")
        exit(1)
    model = load_model(MODEL_PATH)
    print("Model loaded thành công!")
    print(f"Labels: {LABEL_ENCODER}")
    return model, LABEL_ENCODER

def predict_gesture(model, label_encoder, sequence_buffer):
    """
    Predict gesture từ sequence buffer.
    Returns: gesture_label (str), confidence (float), pred_label (str), gesture_type (str)
    """
    if len(sequence_buffer) != N_FRAMES:
        return "Chờ buffer...", 0.0, "No action", "discrete"
    
    input_seq = np.array(sequence_buffer).reshape(1, N_FRAMES, FEATURES)
    pred = model.predict(input_seq, verbose=0)[0]
    pred_idx = np.argmax(pred)
    pred_label = label_encoder[pred_idx]
    confidence = pred[pred_idx]
    gesture_type = GESTURE_TYPES.get(pred_label, 'discrete')
    
    if confidence > CONF_THRESHOLD:
        gesture_label = pred_label
    else:
        gesture_label = "Chờ buffer..."
        confidence = 0.0
        pred_label = "No action"
        gesture_type = "discrete"
    
    return gesture_label, confidence, pred_label, gesture_type
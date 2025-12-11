import time
import Actions

# Map Vietnamese phrases to action labels used by Actions.get_action_func()
KEYWORD_ACTION_MAP = {
    'chuột phải': 'clickchuotphai',
    'chuột trái': 'clickchuottrai',
    'click phải': 'clickchuotphai',
    'click trái': 'clickchuottrai',
    'dừng chương trình': 'dungchuongtrinh',
    'dừng lại': 'dungchuongtrinh',
    'mở app': 'moapp',
    'mở ứng dụng': 'moapp',
    'mở coc coc': 'moapp',
    'mở trình duyệt': 'moapp',
    'mở chrome': 'moapp',
    'mở vscode': 'moapp',
    'mở visual studio code': 'moapp',
    'mở word': 'moapp',
    'mở facebook': 'moapp',
    'mở youtube': 'moapp',
    'mở tiktok': 'moapp',
    'phóng to': 'phongto',
    'thu nhỏ': 'thunho',
    'lên': 'vuotlen',
    'xuống': 'vuotxuong',
    'tab tiếp': 'vuotphai',
    'tab trước': 'vuottrai',
    'nhập văn bản': 'nhapvanban',
    'gõ văn bản': 'nhapvanban',
    'gõ chữ': 'nhapvanban',
    'viết chữ': 'nhapvanban',
    'chế độ gõ': 'nhapvanban',
}


def find_keyword_label(cmd_text: str):
    if not cmd_text:
        return None
    for phrase, label in KEYWORD_ACTION_MAP.items():
        if phrase in cmd_text:
            return label
    return None


def dispatch_command(cmd_text: str, model=None, use_model_if_no_keyword: bool = True, debug: bool = False):
    """Dispatch a spoken/recognized command.

    - If a keyword maps to an action label, execute the mapped action.
    - Otherwise, if model is provided and use_model_if_no_keyword is True, run model prediction
      and execute the predicted action label if an action exists.

    Returns: dict with keys: executed (bool), label (str|None), confidence (float|None)
    """
    now = time.perf_counter()
    result = {'executed': False, 'label': None, 'confidence': None}

    # 1) Keyword match
    label = find_keyword_label(cmd_text)
    if label:
        if debug:
            print(f"[debug] Keyword matched phrase -> label='{label}' for cmd='{cmd_text}'")
        
        # Nếu là moapp, trích xuất tên app từ câu lệnh
        if label == 'moapp':
            func = lambda: Actions.execute_open_app(cmd_text)
            Actions.execute_action(func, label, now, is_continuous=False)
            result.update({'executed': True, 'label': label, 'confidence': None})
            return result
        
        # Các action khác xử lý bình thường
        func = Actions.get_action_func(label)
        if func:
            Actions.execute_action(func, label, now, is_continuous=False)
            result.update({'executed': True, 'label': label, 'confidence': None})
            return result

    # 2) No keyword matched — try model if available and allowed
    if model is not None and use_model_if_no_keyword:
        try:
            pred_label, confidence, probs = model.predict_action_from_text(cmd_text)
            if debug:
                print(f"[debug] Model predicted label='{pred_label}' confidence={confidence:.2f}% for cmd='{cmd_text}'")
            
            result.update({'label': pred_label, 'confidence': confidence})
            
            # Nếu model dự đoán là moapp, trích xuất tên app
            if pred_label == 'moapp':
                func = lambda: Actions.execute_open_app(cmd_text)
                Actions.execute_action(func, pred_label, now, is_continuous=False)
                result['executed'] = True
                return result
            
            # Các action khác
            func = Actions.get_action_func(pred_label)
            if func:
                Actions.execute_action(func, pred_label, now, is_continuous=False)
                result['executed'] = True
            return result
        except Exception as e:
            if debug:
                print(f"[debug] Model prediction error: {e}")
            return result

    # nothing executed
    if debug:
        print(f"[debug] No keyword matched and model not used/available for cmd='{cmd_text}'")
    return result

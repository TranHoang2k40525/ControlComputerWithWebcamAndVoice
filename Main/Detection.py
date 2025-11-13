import cv2
import mediapipe as mp
import numpy as np


MP_HANDS_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 2,
    'min_detection_confidence': 0.8,
    'min_tracking_confidence': 0.5
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(**MP_HANDS_CONFIG)

# Trạng thái dùng cho làm mượt theo thời gian các landmark giữa các frame
_prev_smoothed = []        # danh sách theo tay: mỗi phần tử là 21 điểm (x_px, y_px, z)
_prev_centers_px = []      # danh sách tâm tay trước đó (cx, cy) theo pixel
_prev_age = []             # bộ đếm số frame bị mất (missed-frame) cho mỗi tay theo dõi
_SMOOTH_ALPHA = 0.65       # hệ số EMA: lớn hơn -> nhạy hơn, nhỏ hơn -> mượt hơn
_MAX_MATCH_DIST_PX = 200   # khoảng cách pixel tối đa để ghép tay giữa các frame
_MAX_MISSED_FRAMES = 6     # số frame tối đa trước khi quên một tay đã theo dõi

def count_extended_fingers(landmarks, h, w):
    """
    Đếm ngón tay duỗi (bỏ thumb).
    Returns: int (số ngón duỗi)
    """
    tip_ids = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    pip_ids = [6, 10, 14, 18]  # Corresponding PIP
    extended = 0
    for tip, pip in zip(tip_ids, pip_ids):
        tip_y = landmarks.landmark[tip].y * h
        pip_y = landmarks.landmark[pip].y * h
        if tip_y < pip_y:  # Tip cao hơn PIP → duỗi
            extended += 1
    return extended

def extract_keypoints_from_frame(frame_rgb, multi_landmarks):
    """
    Extract keypoints từ frame, tính center, num_fingers.
    Returns: keypoints (array), hand_centers (list), hand_fingers (list)
    """
    hand_centers = []
    hand_fingers = []
    if not multi_landmarks:
        return np.zeros(84), hand_centers, hand_fingers  # FEATURES=84 từ model.py
    h, w, _ = frame_rgb.shape
    tay_features = []
    for hand_landmarks in multi_landmarks:
        # Ensure mediapipe provided full 21 landmarks for this hand
        # (Mediapipe returns full set when detection is successful, this guard is defensive)
        if not hasattr(hand_landmarks, 'landmark') or len(hand_landmarks.landmark) != 21:
            continue
        keypoints = []
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for lm in hand_landmarks.landmark:
            x, y = lm.x * w, lm.y * h
            keypoints.extend([x, y])
            x_min = min(x_min, x); y_min = min(y_min, y); x_max = max(x_max, x); y_max = max(y_max, y)
        bbox_width = max(x_max - x_min, 1)
        bbox_height = max(y_max - y_min, 1)
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        center_x_norm = (center_x / w * 2) - 1
        center_y_norm = (center_y / h * 2) - 1
        hand_centers.append((center_x_norm, center_y_norm))
        num_fingers = count_extended_fingers(hand_landmarks, h, w)
        hand_fingers.append(num_fingers)
        normalized = []
        for i in range(0, len(keypoints), 2):
            x_norm = (keypoints[i] - center_x) / bbox_width
            y_norm = (keypoints[i + 1] - center_y) / bbox_height
            normalized.extend([x_norm, y_norm])
        tay_features.extend(normalized)
    # Pad nếu <2 tay
    while len(hand_centers) < 2:
        hand_centers.append((0, 0))
        hand_fingers.append(0)
        tay_features.extend(np.zeros(42).tolist())
    return np.array(tay_features), hand_centers, hand_fingers


def stabilize_results_landmarks(results, frame_shape, alpha=_SMOOTH_ALPHA, max_match_dist=_MAX_MATCH_DIST_PX, max_age=_MAX_MISSED_FRAMES):
    """Làm mượt / ổn định các landmark Mediapipe ngay trên đối tượng results.

    - Dùng trung bình động mũ (EMA) cho từng landmark trong không gian pixel (x_px, y_px, z)
    - Ghép tay mới phát hiện với tay đã theo dõi trước đó bằng khoảng cách tâm (center)
    - Ghi lại toạ độ đã làm mượt (đã chuẩn hoá) trở lại `results.multi_hand_landmarks` để
      pipeline phía sau vẫn hoạt động như trước.

    Tham số:
        results: kết quả trả về từ hands.process(frame_rgb)
        frame_shape: frame.shape (h, w, ...)
    Trả về: cùng đối tượng results, với landmark.x/landmark.y đã được ghi bằng giá trị mượt
    """
    global _prev_smoothed, _prev_centers_px, _prev_age

    if results is None or results.multi_hand_landmarks is None:
        # Nothing detected: age previous tracked hands and forget old ones
        for i in range(len(_prev_age)):
            _prev_age[i] += 1
        # remove stale entries
        keep = []
        for i, a in enumerate(_prev_age):
            if a <= max_age:
                keep.append(i)
        if len(keep) != len(_prev_age):
            _prev_smoothed = [ _prev_smoothed[i] for i in keep ]
            _prev_centers_px = [ _prev_centers_px[i] for i in keep ]
            _prev_age = [ _prev_age[i] for i in keep ]
        return results

    h, w = frame_shape[0], frame_shape[1]

    # Tạo danh sách đo được: landmarks và tâm tay cho các detections mới
    measured_list = []  # danh sách dict: { 'landmarks_px': [(x,y,z)...], 'center': (cx, cy) }
    for hand_landmarks in results.multi_hand_landmarks:
        pts_px = []
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for lm in hand_landmarks.landmark:
            mx = lm.x * w
            my = lm.y * h
            mz = getattr(lm, 'z', 0.0)
            # Giới hạn giá trị đo vào trong khung để tránh các phép chiếu bất thường
            mx = max(0.0, min(w - 1.0, mx))
            my = max(0.0, min(h - 1.0, my))
            pts_px.append((mx, my, mz))
            x_min = min(x_min, mx); y_min = min(y_min, my); x_max = max(x_max, mx); y_max = max(y_max, my)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        measured_list.append({'landmarks_px': pts_px, 'center': (cx, cy)})

    # Chuẩn bị ghép: với mỗi tay đo được tìm chỉ số tay đã theo dõi phù hợp nhất
    used_prev = set()
    new_prev_smoothed = []
    new_prev_centers = []
    new_prev_age = []

    for meas in measured_list:
        m_cx, m_cy = meas['center']
        best_idx = None
        best_dist = None
        for i, prev_c in enumerate(_prev_centers_px):
            if i in used_prev:
                continue
            px, py = prev_c
            d = ((px - m_cx) ** 2 + (py - m_cy) ** 2) ** 0.5
            if best_idx is None or d < best_dist:
                best_idx = i; best_dist = d

        if best_idx is not None and best_dist is not None and best_dist <= max_match_dist and best_idx not in used_prev:
            # Ghép được với tay đã theo dõi trước đó
            used_prev.add(best_idx)
            prev_landmarks = _prev_smoothed[best_idx]
            smoothed = []
            for j, (mx, my, mz) in enumerate(meas['landmarks_px']):
                px_prev, py_prev, pz_prev = prev_landmarks[j]
                sx = alpha * mx + (1.0 - alpha) * px_prev
                sy = alpha * my + (1.0 - alpha) * py_prev
                sz = alpha * mz + (1.0 - alpha) * pz_prev
                # clamp again
                sx = max(0.0, min(w - 1.0, sx))
                sy = max(0.0, min(h - 1.0, sy))
                smoothed.append((sx, sy, sz))
            new_prev_smoothed.append(smoothed)
            new_prev_centers.append(((prev_landmarks[0][0] + prev_landmarks[9][0]) / 2.0, (prev_landmarks[0][1] + prev_landmarks[9][1]) / 2.0) if prev_landmarks else meas['center'])
            new_prev_age.append(0)
        else:
            # Tay mới (không ghép): khởi tạo trạng thái mượt bằng giá trị đo
            smoothed = []
            for (mx, my, mz) in meas['landmarks_px']:
                smoothed.append((mx, my, mz))
            new_prev_smoothed.append(smoothed)
            new_prev_centers.append(meas['center'])
            new_prev_age.append(0)

    # Những tay trước đó không được ghép sẽ bị tăng age và giữ lại nếu còn mới
    for i in range(len(_prev_smoothed)):
        if i not in used_prev:
            age = _prev_age[i] + 1
            if age <= max_age:
                # keep old smoothed state to allow re-match
                new_prev_smoothed.append(_prev_smoothed[i])
                new_prev_centers.append(_prev_centers_px[i])
                new_prev_age.append(age)

    # Ghi đè biến toàn cục với trạng thái mới
    _prev_smoothed = new_prev_smoothed
    _prev_centers_px = new_prev_centers
    _prev_age = new_prev_age

    # Ghi lại các giá trị đã làm mượt trở lại vào results (chuẩn hóa theo w/h)
    # Lưu ý: thứ tự trong results.multi_hand_landmarks tương ứng với measured_list
    for hand_i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if hand_i < len(new_prev_smoothed):
            smoothed = new_prev_smoothed[hand_i]
            for j, lm in enumerate(hand_landmarks.landmark):
                sx, sy, sz = smoothed[j]
                lm.x = float(sx / w)
                lm.y = float(sy / h)
                try:
                    lm.z = float(sz)
                except Exception:
                    pass
        else:
            # Trường hợp dự phòng: đảm bảo landmark được giới hạn trong [0,1]
            for lm in hand_landmarks.landmark:
                lm.x = float(max(0.0, min(1.0, lm.x)))
                lm.y = float(max(0.0, min(1.0, lm.y)))

    return results

def draw_hand_landmarks(frame, results, hand_centers, hand_fingers, previous_centers, previous_mouse_pos, gesture_label, confidence, mapped_action):
    """
    Vẽ landmarks, bbox, text, arrow, circle trên frame.
    """
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            color = (0, 255, 0) if hand_idx == 0 else (0, 0, 255)
            h, w, _ = frame.shape
            endpoint_landmarks = [4, 8, 12, 16, 20]
            for lm_idx in endpoint_landmarks:
                lm = hand_landmarks.landmark[lm_idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=color, thickness=2))
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x); y_min = min(y_min, y); x_max = max(x_max, x); y_max = max(y_max, y)
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), color, 2)
            
            # show status only on primary hand for readability
            if hand_idx == 0:
                cv2.putText(frame, f"{gesture_label} ({confidence:.2f})", (x_min-20, y_min-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Action: {mapped_action}", (x_min-20, y_min-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Fingers: {hand_fingers[hand_idx]}", (x_min-20, y_min-70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                if hand_centers[hand_idx] != (0, 0):
                    cx, cy = hand_centers[hand_idx]
                    cx_px = int(cx * w / 2 + w/2)
                    cy_px = int(cy * h / 2 + h/2)
                    cv2.circle(frame, (cx_px, cy_px), 8, color, -1)
                    
                    if len(previous_centers) > hand_idx:
                        prev_cx = previous_centers[hand_idx][0]
                        prev_cy = previous_centers[hand_idx][1]
                        prev_cx_px = int(prev_cx * w / 2 + w/2)
                        prev_cy_px = int(prev_cy * h / 2 + h/2)
                        cv2.arrowedLine(frame, (prev_cx_px, prev_cy_px), (cx_px, cy_px), color, 2)
                    
                    if previous_mouse_pos and len(previous_mouse_pos) > hand_idx and previous_mouse_pos[hand_idx]:
                        mouse_x, mouse_y = previous_mouse_pos[hand_idx]
                        cv2.putText(frame, f"Finger: ({mouse_x}, {mouse_y})", (x_min-20, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Highlight ngón trỏ (landmark 8) của tay chính với vòng tròn lớn màu xanh lá
            if hand_idx == 0:
                try:
                    lm8 = hand_landmarks.landmark[8]
                    x8, y8 = int(lm8.x * w), int(lm8.y * h)
                    # Vẽ vòng tròn lớn màu xanh lá cho ngón trỏ
                    cv2.circle(frame, (x8, y8), 12, (0, 255, 0), 3)
                    cv2.circle(frame, (x8, y8), 8, (0, 255, 0), -1)
                    # Thêm text "CURSOR" để dễ nhận biết
                    cv2.putText(frame, "CURSOR", (x8 + 15, y8 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                except Exception:
                    pass
    
    return frame


def _angle_between(v1, v2):
    """Return angle in degrees between two vectors."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos = np.dot(v1, v2) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return np.degrees(np.arccos(cos))


def _is_finger_straight(hand_landmarks, finger_idx_list, h, w, angle_thresh=15.0):
    """Check whether a finger (list of 4 landmark indices) is approximately straight.

    Uses angles between consecutive bone vectors and compares to threshold (degrees).
    """
    pts = []
    for idx in finger_idx_list:
        lm = hand_landmarks.landmark[idx]
        pts.append((lm.x * w, lm.y * h))
    # vectors between joints: v0 = p0->p1, v1 = p1->p2, v2 = p2->p3
    v0 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    v1 = (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
    v2 = (pts[3][0] - pts[2][0], pts[3][1] - pts[2][1])
    a1 = _angle_between(v0, v1)
    a2 = _angle_between(v1, v2)
    return (a1 < angle_thresh) and (a2 < angle_thresh)


def _fingers_parallel(hand_landmarks, finger_a_idxs, finger_b_idxs, h, w, angle_thresh=15.0):
    """Check whether two fingers are pointing in (nearly) the same direction.

    Uses vector from MCP (first index) to TIP (last index) for each finger.
    """
    a0 = hand_landmarks.landmark[finger_a_idxs[0]]
    a3 = hand_landmarks.landmark[finger_a_idxs[-1]]
    b0 = hand_landmarks.landmark[finger_b_idxs[0]]
    b3 = hand_landmarks.landmark[finger_b_idxs[-1]]
    va = (a3.x * w - a0.x * w, a3.y * h - a0.y * h)
    vb = (b3.x * w - b0.x * w, b3.y * h - b0.y * h)
    ang = _angle_between(va, vb)
    return ang < angle_thresh


def detect_aligned_fingers(results, frame_shape):
    """Detect special aligned-finger gestures directly from Mediapipe landmarks.

    Rules implemented (primary hand index 0):
      - If Index (5,6,7,8) and Middle (9,10,11,12) are straight and parallel -> left click
      - If Index+Middle+Ring (13,14,15,16) are all straight and mutually parallel (approx) -> right click

    Returns: 'clickchuottrai' or 'clickchuotphai' or None
    """
    if results is None or results.multi_hand_landmarks is None or len(results.multi_hand_landmarks) == 0:
        return None

    h, w = frame_shape[0], frame_shape[1]
    # Use primary hand (first detected)
    try:
        hand_landmarks = results.multi_hand_landmarks[0]
    except Exception:
        return None

    index_idxs = [5, 6, 7, 8]
    middle_idxs = [9, 10, 11, 12]
    ring_idxs = [13, 14, 15, 16]

    try:
        idx_straight = _is_finger_straight(hand_landmarks, index_idxs, h, w)
        mid_straight = _is_finger_straight(hand_landmarks, middle_idxs, h, w)
        ring_straight = _is_finger_straight(hand_landmarks, ring_idxs, h, w)

        idx_mid_parallel = _fingers_parallel(hand_landmarks, index_idxs, middle_idxs, h, w)
        mid_ring_parallel = _fingers_parallel(hand_landmarks, middle_idxs, ring_idxs, h, w)
        idx_ring_parallel = _fingers_parallel(hand_landmarks, index_idxs, ring_idxs, h, w)
    except Exception:
        return None

    # 3-finger case: index+middle+ring straight and roughly parallel pairwise
    if idx_straight and mid_straight and ring_straight and idx_mid_parallel and mid_ring_parallel and idx_ring_parallel:
        return 'clickchuotphai'  # right click for 3 fingers

    # 2-finger case: index+middle straight and parallel
    if idx_straight and mid_straight and idx_mid_parallel:
        return 'clickchuottrai'  # left click for 2 fingers

    return None

def display_frame(frame, sequence_buffer, mapped_action):
    """
    Hiển thị frame với text overlay và imshow.
    """
    cv2.putText(frame, f"Buffer: {len(sequence_buffer)}/30 | Action: {mapped_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Gesture Recognition', frame)
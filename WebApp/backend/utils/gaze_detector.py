import cv2
import numpy as np
import mediapipe as mp
import time
import base64

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33, 160, 158, 133, 153, 144]
GAZE_THRESHOLD   = 0.30
DISTRACTION_TIME = 2.0
MIRRORED_FEED    = True

last_center_time = time.time()
alerted          = False

def _eye_center(pts: np.ndarray) -> np.ndarray:
    return np.array([pts[:, 0].mean(), pts[:, 1].mean()])

def _gaze_ratio(pts: np.ndarray, gray: np.ndarray) -> float:
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [pts], 255)

    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()

    _, th = cv2.threshold(eye[min_y:max_y, min_x:max_x], 70, 255,
                          cv2.THRESH_BINARY_INV)

    cnt, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)

    if cnt:
        M = cv2.moments(max(cnt, key=cv2.contourArea))
        if M['m00']:
            cx = int(M['m10'] / M['m00']) + min_x
            cy = int(M['m01'] / M['m00']) + min_y
            pupil = np.array([cx, cy])
        else:
            pupil = _eye_center(pts)
    else:
        pupil = _eye_center(pts)

    w = max_x - min_x or 1
    return (pupil[0] - min_x) / w


def process_frame(frame):
    """Return an encoded JPEG with green landmarks, gaze direction,
    and landmark coordinates for optional extra drawing."""
    global last_center_time, alerted

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    results   = face_mesh.process(rgb)
    direction = "None"
    dots      = []

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        le = np.array([[int(lm[i].x * W), int(lm[i].y * H)]
                       for i in LEFT_EYE], dtype=np.int32)
        re = np.array([[int(lm[i].x * W), int(lm[i].y * H)]
                       for i in RIGHT_EYE], dtype=np.int32)

        lg = _gaze_ratio(le, gray)
        rg = _gaze_ratio(re, gray)
        avg = (lg + rg) / 2.0

        if avg <= 0.5 - GAZE_THRESHOLD:
            direction = "Right" if MIRRORED_FEED else "Left"
        elif avg >= 0.5 + GAZE_THRESHOLD:
            direction = "Left"  if MIRRORED_FEED else "Right"
        else:
            direction = "Center"

        if direction != "Center":
            if time.time() - last_center_time > DISTRACTION_TIME:
                alerted = True
        else:
            last_center_time = time.time()
            alerted = False

        for (x, y) in np.vstack((le, re)):
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            dots.append({'x': int(x), 'y': int(y)})

    ok, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("JPEG encode failed")

    return jpeg.tobytes(), direction, dots

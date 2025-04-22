import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python  # import tasks API
from mediapipe.tasks.python import vision  # import vision module

# gpu delegate
base_options = python.BaseOptions(
    delegate=python.BaseOptions.Delegate.GPU,  # enable gpu delegate :contentReference[oaicite:0]{index=0}
    model_asset_path="face_landmarker.task",
)
# init face landmarker
options = vision.FaceLandmarkerOptions(
    base_options=base_options,  # pass gpu delegate :contentReference[oaicite:1]{index=1}
    running_mode=vision.FaceLandmarkerOptions.RUNNING_MODE_VIDEO,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)  # create gpu-enabled landmarker

cap = cv2.VideoCapture(0)  # init camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
THRESH = 0.3
TIMEOUT = 2.0

last_time = 0
alerted = False


def get_center(pts, img):
    x = pts[:, 0].mean()
    y = pts[:, 1].mean()
    return np.array([x, y])


def gaze_ratio(pts, img):
    region = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(region, [pts], 255)
    eye = cv2.bitwise_and(img, img, mask=region)
    minx, maxx = pts[:, 0].min(), pts[:, 0].max()
    miny, maxy = pts[:, 1].min(), pts[:, 1].max()
    _, th = cv2.threshold(eye[miny:maxy, minx:maxx], 70, 255, cv2.THRESH_BINARY_INV)
    cnt, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnt:
        M = cv2.moments(max(cnt, key=cv2.contourArea))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + minx
            pupil = np.array([cx, int(M["m01"] / M["m00"]) + miny])
        else:
            pupil = get_center(pts, img)
    else:
        pupil = get_center(pts, img)
    w = maxx - minx or 1
    return (pupil[0] - minx) / w


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # to gray
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = face_landmarker.detect_for_video(mp_img, int(time.time() * 1000))
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            h, w, _ = frame.shape
            le = np.array([[int(p.x * w), int(p.y * h)] for i, p in enumerate(lm.landmarks) if i in LEFT_EYE])
            re = np.array([[int(p.x * w), int(p.y * h)] for i, p in enumerate(lm.landmarks) if i in RIGHT_EYE])
            lg = (gaze_ratio(le, gray) + gaze_ratio(re, gray)) / 2
            dir = "Left" if lg <= 0.5 - THRESH else "Right" if lg >= 0.5 + THRESH else "Center"
            for pt in np.vstack((le, re)):
                cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)
            cv2.putText(frame, dir, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if dir != "Center" and time.time() - last_time > TIMEOUT and not alerted:
                cv2.putText(frame, "ALERT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alerted = True
            if dir == "Center":
                last_time = time.time()
                alerted = False
        cv2.imshow("gaze", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    face_landmarker.close()
    cv2.destroyAllWindows()

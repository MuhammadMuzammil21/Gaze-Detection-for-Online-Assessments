#!/usr/bin/env python3
# gaze_detector.py
#
# Lightweight real-time gaze detector (CPU-only).
#   • MediaPipe FaceMesh + Iris → 478 landmarks incl. 4-point pupils
#   • One-key (“c”) centre calibration
#   • Smoothed gaze vector + distraction flag
#   • Runs as a webcam test or headless JSON streamer for Flask back-end
#
# Install:
#   pip install mediapipe opencv-python numpy
#
# Run:
#   python gaze_detector.py          # GUI overlay, Esc to quit
#   python gaze_detector.py --no-gui # headless, prints one JSON per frame
#
# Author: (your name)
# ----------------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import json

# ──────────────────────────────────────────────────────────────────────
# MediaPipe initialisation
# ──────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,            # enable iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices we need
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE   = [33, 133]               # [outer, inner] corners
RIGHT_EYE  = [362, 263]              # [inner, outer] corners

# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
def iris_center(lms, idxs, w, h):
    """Mean of the four iris points (pixel coords)."""
    pts = np.array([(lms[i].x * w, lms[i].y * h) for i in idxs])
    return np.mean(pts, axis=0)

def eye_corners(lms, idxs, w, h):
    return np.array([(lms[i].x * w, lms[i].y * h) for i in idxs])

def gaze_from_landmarks(lms, w, h):
    """
    Returns (horizontal, vertical) ∈ [0,1] × [0,1]
        0   = extreme left / top
        0.5 = centre
        1   = extreme right / bottom
    """
    # centres
    left_c   = iris_center(lms, LEFT_IRIS,  w, h)
    right_c  = iris_center(lms, RIGHT_IRIS, w, h)
    left_e   = eye_corners(lms, LEFT_EYE,  w, h)
    right_e  = eye_corners(lms, RIGHT_EYE, w, h)

    # horizontal ratio (iris between eye corners)
    l_hr = (left_c[0]  - left_e[0][0])  / (left_e[1][0]  - left_e[0][0]  + 1e-6)
    r_hr = (right_c[0] - right_e[1][0]) / (right_e[0][0] - right_e[1][0] + 1e-6)
    horizontal = np.clip(np.mean([l_hr, r_hr]), 0.0, 1.0)

    # vertical ratio (iris vs. eyelid line)
    l_eye_h = abs(left_e[0][1]  - left_e[1][1])  + 1e-6
    r_eye_h = abs(right_e[0][1] - right_e[1][1]) + 1e-6
    l_vr = (left_c[1]  - min(left_e[:,1]))  / l_eye_h
    r_vr = (right_c[1] - min(right_e[:,1])) / r_eye_h
    vertical = np.clip(np.mean([l_vr, r_vr]), 0.0, 1.0)

    return horizontal, vertical

def distraction_flag(hor, ver, centre=(0.5, 0.5), thresh=0.35):
    """True when gaze deviates > thresh from calibrated centre."""
    return abs(hor - centre[0]) > thresh or abs(ver - centre[1]) > thresh

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main(show_gui=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    alpha        = 0.15        # exponential smoothing factor
    smoothed     = None
    centre       = (0.5, 0.5)  # default gaze centre
    centre_buf   = []          # rolling buffer for calibration
    CAL_SAMPLES  = 10

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame capture failed")
            break

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            if show_gui:
                cv2.putText(frame, "No face detected", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Gaze detector", frame)
            else:
                print(json.dumps({"error": "no_face", "ts": time.time()}))
            if show_gui and (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        lms = res.multi_face_landmarks[0].landmark
        hor, ver = gaze_from_landmarks(lms, w, h)

        # EMA smoothing
        smoothed = np.array([hor, ver]) if smoothed is None \
                   else alpha * np.array([hor, ver]) + (1-alpha) * smoothed

        distracted = distraction_flag(smoothed[0], smoothed[1], centre)

        # ── GUI overlay ───────────────────────────────────────────────
        key = 0
        if show_gui:
            # iris dots
            cv2.circle(frame, tuple(iris_center(lms, LEFT_IRIS, w, h).astype(int)),  3, (255,0,255), -1)
            cv2.circle(frame, tuple(iris_center(lms, RIGHT_IRIS, w, h).astype(int)), 3, (255,0,255), -1)

            # arrow from nose tip
            origin = (int(lms[1].x * w), int(lms[1].y * h))
            scale  = 0.4
            end_pt = (int(origin[0] + (smoothed[0] - centre[0]) * w * scale),
                      int(origin[1] + (smoothed[1] - centre[1]) * h * scale))
            cv2.arrowedLine(frame, origin, end_pt, (0,255,0), 2, tipLength=0.2)

            # HUD text
            cv2.putText(frame, f"Gaze: ({smoothed[0]:.2f}, {smoothed[1]:.2f})",
                        (30, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if distracted:
                cv2.putText(frame, "DISTRACTED", (30, h-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

            cv2.imshow("Gaze detector", frame)
            key = cv2.waitKey(1) & 0xFF

        # ── calibration: hold “c” while looking at screen centre ──────
        if key == ord('c'):
            centre_buf.append(smoothed.copy())
            if len(centre_buf) == CAL_SAMPLES:
                centre = tuple(np.median(centre_buf, axis=0))
                print(f"[CAL] centre = {centre}")
                centre_buf.clear()
        else:
            centre_buf.clear()  # only collect consecutive frames

        # headless JSON output
        if not show_gui:
            print(json.dumps({
                "gaze": [float(smoothed[0]), float(smoothed[1])],
                "distracted": bool(distracted),
                "ts": time.time()
            }))

        # exit on Esc
        if key == 27:
            break

    # ─────────────────────────────────────────────────────────────────
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time gaze detector")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable OpenCV overlay and print JSON only")
    main(show_gui=not parser.parse_args().no_gui)

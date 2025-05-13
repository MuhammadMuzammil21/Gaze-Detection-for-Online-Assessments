# utils/gaze_detector.py
import cv2
from .gaze_tracking.gaze_tracking import GazeTracking

gaze_tracker = GazeTracking()  # Initialize once

def process_frame(frame):
    gaze_tracker.refresh(frame)
    new_frame = gaze_tracker.annotated_frame()

    if gaze_tracker.is_blinking():
        gaze = "blinking"
    if gaze_tracker.is_right():
        gaze = "looking away"
    elif gaze_tracker.is_left():
        gaze = "looking away"
    elif gaze_tracker.is_center():
        gaze = "On screen"
    elif gaze_tracker.is_looking_up():
        gaze = "looking away"
    else:
        gaze = "undetected"

    # Check for distraction (e.g., not center)
    distracted = gaze != "center"

    # Encode annotated frame to JPEG
    ret, jpeg = cv2.imencode(".jpg", new_frame)
    return jpeg.tobytes(), gaze, distracted

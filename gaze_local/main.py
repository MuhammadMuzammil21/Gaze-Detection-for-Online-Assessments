import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Better eye landmark accuracy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize OpenCV Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Constants for gaze detection
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Mediapipe landmarks for left eye
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Mediapipe landmarks for right eye
GAZE_THRESHOLD = 0.3  # Threshold for gaze direction
DISTRACTION_TIME = 2.0  # Seconds before distraction alert

# Variables for distraction detection
last_gaze_time = time.time()
distraction_detected = False

def get_eye_center(eye_points, gray_frame):
    """Calculate the center of the eye based on landmarks."""
    x = eye_points[:, 0].mean()
    y = eye_points[:, 1].mean()
    return np.array([x, y])

def get_gaze_ratio(eye_points, gray_frame):
    """Calculate gaze ratio based on pupil position."""
    eye_region = np.array(eye_points, dtype=np.int32)
    mask = np.zeros(gray_frame.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

    # Get bounding box of eye
    min_x = np.min(eye_points[:, 0])
    max_x = np.max(eye_points[:, 0])
    min_y = np.min(eye_points[:, 1])
    max_y = np.max(eye_points[:, 1])

    # Threshold to detect pupil (darkest region)
    _, thresh = cv2.threshold(eye[min_y:max_y, min_x:max_x], 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + min_x
            cy = int(M["m01"] / M["m00"]) + min_y
            pupil = np.array([cx, cy])
        else:
            pupil = get_eye_center(eye_points, gray_frame)
    else:
        pupil = get_eye_center(eye_points, gray_frame)

    # Calculate gaze ratio
    eye_center = get_eye_center(eye_points, gray_frame)
    eye_width = max_x - min_x
    if eye_width == 0:
        return 1.0
    gaze_ratio = (pupil[0] - min_x) / eye_width
    return gaze_ratio

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for efficiency
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using Haar cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Process face region with Mediapipe
            face_roi = rgb_frame[y:y+h, x:x+w]
            results = face_mesh.process(face_roi)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get image dimensions for scaling landmarks
                    h_full, w_full, _ = frame.shape
                    h_roi, w_roi, _ = face_roi.shape

                    # Extract left and right eye landmarks
                    left_eye_points = []
                    right_eye_points = []
                    for idx in LEFT_EYE_INDICES:
                        lm = face_landmarks.landmark[idx]
                        left_eye_points.append([lm.x * w_roi + x, lm.y * h_roi + y])
                    for idx in RIGHT_EYE_INDICES:
                        lm = face_landmarks.landmark[idx]
                        right_eye_points.append([lm.x * w_roi + x, lm.y * h_roi + y])

                    left_eye_points = np.array(left_eye_points, dtype=np.int32)
                    right_eye_points = np.array(right_eye_points, dtype=np.int32)

                    # Calculate gaze ratios
                    left_gaze_ratio = get_gaze_ratio(left_eye_points, gray)
                    right_gaze_ratio = get_gaze_ratio(right_eye_points, gray)
                    gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2

                    # Determine gaze direction
                    if gaze_ratio <= 0.5 - GAZE_THRESHOLD:
                        gaze_direction = "Left"
                    elif gaze_ratio >= 0.5 + GAZE_THRESHOLD:
                        gaze_direction = "Right"
                    else:
                        gaze_direction = "Center"

                    # Draw eye landmarks for visualization
                    for (ex, ey) in left_eye_points:
                        cv2.circle(frame, (ex, ey), 2, (0, 255, 0), -1)
                    for (ex, ey) in right_eye_points:
                        cv2.circle(frame, (ex, ey), 2, (0, 255, 0), -1)

                    # Display gaze direction
                    cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Distraction detection for driving
                    if gaze_direction != "Center":
                        if time.time() - last_gaze_time > DISTRACTION_TIME and not distraction_detected:
                            cv2.putText(frame, "ALERT: Distraction Detected!", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            distraction_detected = True
                    else:
                        last_gaze_time = time.time()
                        distraction_detected = False

        # Show the frame
        cv2.imshow("Gaze Detection", frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program terminated by user.")

finally:
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
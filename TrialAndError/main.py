import cv2
import numpy as np
from gaze_tracking import GazeTracking
import mediapipe as mp

# 3D model points of facial landmarks in mm
MODEL_POINTS = np.array([
    (0.0,   0.0,     0.0),       # Nose tip
    (0.0,  -330.0,  -65.0),      # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

def estimate_head_pose(frame, face_landmarks):
    h, w = frame.shape[:2]
    # 2D image points from MediaPipe FaceMesh
    image_points = np.array([
        [face_landmarks.landmark[1].x * w,   face_landmarks.landmark[1].y * h],   # Nose tip
        [face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h], # Chin
        [face_landmarks.landmark[33].x * w,  face_landmarks.landmark[33].y * h],  # Left eye left corner
        [face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h], # Right eye right corner
        [face_landmarks.landmark[61].x * w,  face_landmarks.landmark[61].y * h],  # Left mouth corner
        [face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h]  # Right mouth corner
    ], dtype=np.float64)

    # Camera internals
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,             center[0]],
        [0,            focal_length,  center[1]],
        [0,            0,             1         ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))  # assume no lens distortion

    # Solve for pose
    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Project a 3D point (0,0,1000mm) onto image plane to draw nose direction
    (nose_end_point2D, _) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]),
        rotation_vec,
        translation_vec,
        camera_matrix,
        dist_coeffs
    )
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return p1, p2

def main():
    # Initialize detectors
    gaze = GazeTracking()
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                max_num_faces=1,
                                                refine_landmarks=True,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror
        # Gaze detection
        gaze.refresh(frame)
        annotated = gaze.annotated_frame()
        text = ""
        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"
        cv2.putText(annotated, text, (30, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

        # Head pose estimation
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            p1, p2 = estimate_head_pose(frame, landmarks)
            cv2.line(annotated, p1, p2, (255, 0, 0), 2)
            cv2.circle(annotated, p1, 5, (0,0,255), -1)

        cv2.imshow("Gaze + Head Pose", annotated)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

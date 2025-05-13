import cv2
import dlib
import numpy as np
from imutils import face_utils

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Camera parameters (approximate for a typical webcam)
focal_length = 640  # in pixels, depends on camera
camera_matrix = np.array([[focal_length, 0, 320],
                         [0, focal_length, 240],
                         [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# 3D model points for a generic face (in world coordinates)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),# Left eye left corner
    (225.0, 170.0, -135.0), # Right eye right corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0) # Right mouth corner
], dtype="double")

# Function to estimate gaze direction
def get_gaze_direction(eye_points, image_points, camera_matrix, dist_coeffs):
    # Calculate eye center
    eye_center = np.mean(eye_points, axis=0)
    
    # 2D image points for pose estimation
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    # Project a 3D point to get gaze direction
    point_3d = np.array([(0.0, 0.0, 500.0)], dtype="double")
    (point_2d, _) = cv2.projectPoints(point_3d, rotation_vector, translation_vector,
                                     camera_matrix, dist_coeffs)
    
    # Gaze vector in 3D
    gaze_vector = point_2d[0][0] - eye_center
    gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
    
    return eye_center, gaze_vector

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extract eye points (left: 36-41, right: 42-47)
        left_eye_points = shape[36:42]
        right_eye_points = shape[42:48]
        
        # 2D image points for pose estimation
        image_points = np.array([
            shape[30],  # Nose tip
            shape[8],   # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corner
            shape[48],  # Left mouth corner
            shape[54]   # Right mouth corner
        ], dtype="double")
        
        # Get gaze direction for both eyes
        left_eye_center, left_gaze = get_gaze_direction(left_eye_points, image_points, camera_matrix, dist_coeffs)
        right_eye_center, right_gaze = get_gaze_direction(right_eye_points, image_points, camera_matrix, dist_coeffs)
        
        # Draw gaze vectors
        vector_length = 50
        left_gaze_end = left_eye_center + vector_length * left_gaze
        right_gaze_end = right_eye_center + vector_length * right_gaze
        
        # Convert to integer for drawing
        left_eye_center = tuple(left_eye_center.astype(int))
        left_gaze_end = tuple(left_gaze_end.astype(int))
        right_eye_center = tuple(right_eye_center.astype(int))
        right_gaze_end = tuple(right_gaze_end.astype(int))
        
        # Draw circles at pupil centers
        cv2.circle(frame, left_eye_center, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_eye_center, 5, (0, 255, 0), -1)
        
        # Draw gaze vectors
        cv2.arrowedLine(frame, left_eye_center, left_gaze_end, (0, 0, 255), 2)
        cv2.arrowedLine(frame, right_eye_center, right_gaze_end, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("Gaze Detection", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
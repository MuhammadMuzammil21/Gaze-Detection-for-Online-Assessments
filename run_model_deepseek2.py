"""
run_model.py
Real-time gaze estimation with calibration, distraction detection, and MediaPipe face processing.
"""

import os
import cv2
import torch
import numpy as np
import scipy.io as sio
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

# Constants
CHECKPOINT_PATH = "checkpoints/final_10.pth"  # Trained model
CALIBRATION_PARTICIPANT = "p00"  # Use calibration data from this participant
DATA_ROOT = "data/MPIIFaceGaze"  # Match training data path
FACE_IMAGE_SIZE = (224, 224)  # Must match training size
DISTRACTION_THRESHOLDS = {
    'pitch': 25,    # Degrees
    'yaw': 35,      # Degrees
    'roll': 25,      # Degrees
    'gaze_dist': 0.4 # Normalized screen distance
}

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 3D Face Model for Pose Estimation (based on MPII's 6-point model)
FACE_3D_MODEL = np.array([
    [-152.0, 51.0, 16.0],    # Left eye left corner
    [-52.0, 51.0, 16.0],     # Left eye right corner
    [52.0, 51.0, 16.0],      # Right eye left corner
    [152.0, 51.0, 16.0],     # Right eye right corner
    [-100.0, -77.0, -5.0],   # Mouth left corner
    [100.0, -77.0, -5.0]    # Mouth right corner
], dtype=np.float64) / 1000  # Convert mm to meters

# MediaPipe landmark indices matching FACE_3D_MODEL
LANDMARK_INDICES = [33, 133, 362, 263, 61, 291]

class GazeRunner:
    def __init__(self):
        # Load calibration data
        self._load_calibration()
        
        # Initialize model
        self.model, self.scalers = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize(FACE_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Camera properties (from dataset)
        self.camera_matrix = None
        self.dist_coeffs = None
        self._load_camera_params()

    def _load_calibration(self):
        """Load screen and monitor calibration data"""
        calib_path = os.path.join(DATA_ROOT, CALIBRATION_PARTICIPANT, "Calibration")
        
        # Screen properties
        screen_size = sio.loadmat(os.path.join(calib_path, "screenSize.mat"))
        self.screen_width = int(screen_size['width_pixel'][0,0])
        self.screen_height = int(screen_size['height_pixel'][0,0])
        
        # Monitor pose in camera coordinates
        monitor_pose = sio.loadmat(os.path.join(calib_path, "monitorPose.mat"))
        self.monitor_rot = cv2.Rodrigues(monitor_pose['rvecs'])[0]
        self.monitor_trans = monitor_pose['tvecs']

    def _load_camera_params(self):
        """Load camera intrinsic parameters from dataset"""
        calib_path = os.path.join(DATA_ROOT, CALIBRATION_PARTICIPANT, "Calibration")
        camera_params = sio.loadmat(os.path.join(calib_path, "Camera.mat"))
        self.camera_matrix = camera_params['cameraMatrix']
        self.dist_coeffs = camera_params['distCoeffs']

    def _load_model(self):
        """Load trained model with scalers"""
        checkpoint = torch.load(CHECKPOINT_PATH)
        
        # Initialize model architecture
        model = torch.load('model_arch.pth')  # Save architecture separately during training
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        # Initialize scalers (should be saved during training)
        scalers = {
            'head_pose': StandardScaler(),
            'eye_landmarks': StandardScaler()
        }
        scalers['head_pose'].mean_ = checkpoint['head_pose_mean']
        scalers['head_pose'].scale_ = checkpoint['head_pose_scale']
        scalers['eye_landmarks'].mean_ = checkpoint['eye_mean']
        scalers['eye_landmarks'].scale_ = checkpoint['eye_scale']
        
        return model.to('cuda' if torch.cuda.is_available() else 'cpu'), scalers

    def _preprocess_face(self, frame, landmarks):
        """Crop and align face similar to dataset format"""
        # Get face bounding box with padding
        x_coords = [lm.x * frame.shape[1] for lm in landmarks.landmark]
        y_coords = [lm.y * frame.shape[0] for lm in landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add 20% padding
        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(frame.shape[1], x_max + pad_x)
        y_max = min(frame.shape[0], y_max + pad_y)
        
        # Crop and resize
        face_img = frame[y_min:y_max, x_min:x_max]
        return cv2.resize(face_img, FACE_IMAGE_SIZE)

    def _get_head_pose(self, landmarks):
        """Estimate head pose using 3D-2D correspondence"""
        # Extract 2D landmarks
        image_points = np.array([
            [landmarks.landmark[i].x * self.frame_width,
             landmarks.landmark[i].y * self.frame_height]
            for i in LANDMARK_INDICES
        ], dtype=np.float64)
        
        # Solve PnP
        _, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        # Convert rotation vector to angles
        rotation_mat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = np.degrees(cv2.RQDecomp3x3(rotation_mat)[0])
        return np.array([pitch, yaw, roll, *tvec.flatten()])

    def _check_distraction(self, head_pose, gaze_screen):
        """Determine if user is distracted"""
        # Head orientation check
        head_distracted = any([
            abs(head_pose[0]) > DISTRACTION_THRESHOLDS['pitch'],
            abs(head_pose[1]) > DISTRACTION_THRESHOLDS['yaw'],
            abs(head_pose[2]) > DISTRACTION_THRESHOLDS['roll']
        ])
        
        # Gaze distance from screen center
        screen_center = np.array([self.screen_width/2, self.screen_height/2])
        gaze_dist = np.linalg.norm(gaze_screen - screen_center) / max(self.screen_width, self.screen_height)
        
        return head_distracted or (gaze_dist > DISTRACTION_THRESHOLDS['gaze_dist'])

    def run(self):
        cap = cv2.VideoCapture(0)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            # Face detection
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue
                
            face_landmarks = results.multi_face_landmarks[0]
            
            try:
                # Preprocess face image
                face_img = self._preprocess_face(frame, face_landmarks)
                
                # Get head pose
                head_pose = self._get_head_pose(face_landmarks)
                
                # Calculate bounding box coordinates for face landmarks
                x_coords = [lm.x * self.frame_width for lm in face_landmarks.landmark]
                y_coords = [lm.y * self.frame_height for lm in face_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                # Get eye landmarks (normalized to cropped face)
                eye_landmarks = []
                for idx in LANDMARK_INDICES:
                    x = (face_landmarks.landmark[idx].x * self.frame_width - x_min) / face_img.shape[1]
                    y = (face_landmarks.landmark[idx].y * self.frame_height - y_min) / face_img.shape[0]
                    eye_landmarks.extend([x, y])
                eye_landmarks = np.array(eye_landmarks)
                
                # Normalize features
                head_pose_norm = self.scalers['head_pose'].transform([head_pose])[0]
                eye_norm = self.scalers['eye_landmarks'].transform([eye_landmarks])[0]
                
                # Predict gaze vector
                with torch.no_grad():
                    gaze_3d = self.model(
                        self.transform(Image.fromarray(face_img)).unsqueeze(0).cuda(),
                        torch.FloatTensor(head_pose_norm).unsqueeze(0).cuda(),
                        torch.FloatTensor(eye_norm).unsqueeze(0).cuda()
                    ).cpu().numpy()[0]
                
                # Convert to screen coordinates
                # (Implementation depends on your coordinate system - see note below)
                gaze_screen = self._convert_to_screen(gaze_3d)
                
                # Distraction check
                if self._check_distraction(head_pose, gaze_screen):
                    cv2.putText(frame, "DISTRACTED!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # Visualization
                cv2.circle(frame, tuple(gaze_screen.astype(int)), 10, (0,255,0), -1)
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
                
            cv2.imshow('Gaze Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    runner = GazeRunner()
    runner.run()
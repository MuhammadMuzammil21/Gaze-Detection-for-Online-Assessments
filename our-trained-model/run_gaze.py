import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import pickle
import math
import torch.nn as nn
from torchvision.models import resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "gaze_model_final.pth"
SCALERS_PATH = "scalers.pkl"
IMAGE_SIZE = (224, 224)
HEAD_POSE_YAW_THRESHOLD = 25
HEAD_POSE_PITCH_THRESHOLD = 20
GAZE_VECTOR_XY_COMPONENT_THRESHOLD = 0.6


class GazeEstimationModel(nn.Module):
    def __init__(self, num_landmarks=12, num_head_pose=6, num_gaze_outputs=3):
        super(GazeEstimationModel, self).__init__()
        self.resnet = resnet18(weights=None)
        num_ftrs_image = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.landmark_fc = nn.Sequential(nn.Linear(num_landmarks, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.head_pose_fc = nn.Sequential(nn.Linear(num_head_pose, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        combined_feature_size = num_ftrs_image + 32 + 32
        self.gaze_fc = nn.Sequential(nn.Linear(combined_feature_size, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_gaze_outputs))

    def forward(self, image, landmarks, head_pose):
        image_features = self.resnet(image)
        landmark_features = self.landmark_fc(landmarks)
        head_pose_features = self.head_pose_fc(head_pose)
        combined_features = torch.cat((image_features, landmark_features, head_pose_features), dim=1)
        gaze = self.gaze_fc(combined_features)
        return gaze


model = GazeEstimationModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with open(SCALERS_PATH, "rb") as f:
    scalers = pickle.load(f)
    landmark_scaler = scalers["landmark_scaler"]
    pose_scaler = scalers["pose_scaler"]

transform_inference = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

landmark_indices_for_model = [33, 133, 263, 362, 61, 291]
pnp_landmark_indices = [1, 152, 33, 263, 61, 291]

model_points = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

calibration_gaze_offset = np.array([0.0, 0.0, 0.0])
is_calibrated = False
calibration_active = False
calibration_samples = []
CALIBRATION_SAMPLE_COUNT = 30

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    frame_height, frame_width = frame.shape[:2]
    distracted_reason = ""

    if not is_calibrated:
        cv2.putText(frame, "Press 'c' to start calibration", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Calibrated (Press 'r' to reset)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    if calibration_active:
        samples_remaining = CALIBRATION_SAMPLE_COUNT - len(calibration_samples)
        cv2.putText(
            frame,
            f"CALIBRATING: Look at center and stay still ({samples_remaining})",
            (frame_width // 2 - 250, frame_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(frame, (frame_width // 2, frame_height // 2), 10, (0, 0, 255), -1)
        cv2.circle(frame, (frame_width // 2, frame_height // 2), 20, (0, 0, 255), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_coords = [landmark.x for landmark in face_landmarks.landmark]
            y_coords = [landmark.y for landmark in face_landmarks.landmark]
            x_min_f, x_max_f = min(x_coords), max(x_coords)
            y_min_f, y_max_f = min(y_coords), max(y_coords)
            x_min = int(x_min_f * frame_width)
            y_min = int(y_min_f * frame_height)
            x_max = int(x_max_f * frame_width)
            y_max = int(y_max_f * frame_height)
            padding_x = int((x_max - x_min) * 0.2)
            padding_y = int((y_max - y_min) * 0.2)
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(frame_width, x_max + padding_x)
            y_max = min(frame_height, y_max + padding_y)

            if x_max <= x_min or y_max <= y_min:
                continue

            face_crop_bgr = frame[y_min:y_max, x_min:x_max]
            if face_crop_bgr.size == 0:
                continue

            face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB))
            image_tensor = transform_inference(face_crop_pil).unsqueeze(0).to(DEVICE)

            image_points = np.array(
                [(face_landmarks.landmark[i].x * frame_width, face_landmarks.landmark[i].y * frame_height) for i in pnp_landmark_indices],
                dtype=np.float64,
            )

            focal_length = frame_width
            cam_center = (frame_width / 2, frame_height / 2)
            camera_matrix = np.array([[focal_length, 0, cam_center[0]], [0, focal_length, cam_center[1]], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success_pnp:
                continue

            head_pose_features_raw = np.concatenate((rotation_vector.flatten(), translation_vector.flatten()))
            head_pose_tensor = (
                torch.tensor(pose_scaler.transform(head_pose_features_raw.reshape(1, -1)).flatten(), dtype=torch.float32)
                .unsqueeze(0)
                .to(DEVICE)
            )

            rmat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rmat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            pitch, yaw, roll = euler_angles.flatten()[:3]
            pitch = -pitch
            pitch = (pitch + 180) % 360

            if abs(yaw) > HEAD_POSE_YAW_THRESHOLD:
                distracted_reason += f"Yaw: {yaw:.1f} "
            if abs(pitch) > HEAD_POSE_PITCH_THRESHOLD:
                distracted_reason += f"Pitch: {pitch:.1f} "

            landmark_features_for_model_raw = []
            for idx in landmark_indices_for_model:
                lm_x_frame = face_landmarks.landmark[idx].x * frame_width
                lm_y_frame = face_landmarks.landmark[idx].y * frame_height
                lm_x_crop = lm_x_frame - x_min
                lm_y_crop = lm_y_frame - y_min
                landmark_features_for_model_raw.extend([lm_x_crop, lm_y_crop])

            landmark_features_for_model_raw = np.array(landmark_features_for_model_raw, dtype=np.float32) / 450.0
            landmark_tensor = (
                torch.tensor(landmark_scaler.transform(landmark_features_for_model_raw.reshape(1, -1)).flatten(), dtype=torch.float32)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():
                gaze_3d_pred = model(image_tensor, landmark_tensor, head_pose_tensor).cpu().numpy().flatten()

            if calibration_active and len(calibration_samples) < CALIBRATION_SAMPLE_COUNT:
                calibration_samples.append(gaze_3d_pred)
                if len(calibration_samples) >= CALIBRATION_SAMPLE_COUNT:
                    avg_gaze = np.mean(np.array(calibration_samples), axis=0)
                    calibration_gaze_offset = np.array([avg_gaze[0], avg_gaze[1], 0])
                    calibration_active = False
                    is_calibrated = True

            original_gaze = gaze_3d_pred.copy()
            if is_calibrated:
                gaze_3d_pred = gaze_3d_pred - calibration_gaze_offset
                norm = np.linalg.norm(gaze_3d_pred)
                gaze_3d_pred = gaze_3d_pred / (norm + 1e-8)

            if np.any(np.isnan(gaze_3d_pred)) or np.any(np.isinf(gaze_3d_pred)):
                continue

            if abs(gaze_3d_pred[0]) > GAZE_VECTOR_XY_COMPONENT_THRESHOLD or abs(gaze_3d_pred[1]) > GAZE_VECTOR_XY_COMPONENT_THRESHOLD:
                distracted_reason += f"Gaze Out "

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            nose_tip_2d = (int(face_landmarks.landmark[1].x * frame_width), int(face_landmarks.landmark[1].y * frame_height))

            gaze_line_length = 150
            try:
                gaze_end_x = int(nose_tip_2d[0] + gaze_3d_pred[0] * gaze_line_length)
                gaze_end_y = int(nose_tip_2d[1] + gaze_3d_pred[1] * gaze_line_length)
            except ValueError:
                continue  # Skip if conversion to int fails (e.g., NaN)

            gaze_end_point_2d = (gaze_end_x, gaze_end_y)
            cv2.arrowedLine(frame, nose_tip_2d, gaze_end_point_2d, (255, 0, 0), 3, tipLength=0.2)

            if is_calibrated:
                orig_gaze_end_point_2d = (
                    int(nose_tip_2d[0] + original_gaze[0] * gaze_line_length),
                    int(nose_tip_2d[1] + original_gaze[1] * gaze_line_length),
                )  # Fixed Y
                cv2.arrowedLine(frame, nose_tip_2d, orig_gaze_end_point_2d, (0, 0, 255), 1, tipLength=0.2)

            cv2.putText(frame, f"Yaw: {yaw:.1f}", (x_min, y_min - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Gaze:({gaze_3d_pred[0]:.2f},{gaze_3d_pred[1]:.2f},{gaze_3d_pred[2]:.2f})",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 100, 0),
                1,
            )

            if is_calibrated:
                cv2.putText(
                    frame,
                    f"Offset:({calibration_gaze_offset[0]:.2f},{calibration_gaze_offset[1]:.2f})",
                    (x_min, y_min + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 100, 255),
                    1,
                )
    else:
        distracted_reason = "NO FACE"  # Handle no face case

    if distracted_reason:
        cv2.putText(frame, f"DISTRACTED: {distracted_reason}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "ATTENTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Gaze Inference", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        calibration_active = True
        calibration_samples = []
    elif key == ord("r"):
        is_calibrated = False
        calibration_active = False
        calibration_gaze_offset = np.array([0.0, 0.0, 0.0])

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

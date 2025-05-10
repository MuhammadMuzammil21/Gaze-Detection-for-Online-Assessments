# backend/utils/gaze_detector.py

import cv2, torch, pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from models.model import GazeEstimationModel
import mediapipe as mp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/gaze_model_final.pth"
SCALERS_PATH = "models/scalers.pkl"

with open(SCALERS_PATH, "rb") as f:
    scalers = pickle.load(f)
    landmark_scaler = scalers["landmark_scaler"]
    pose_scaler = scalers["pose_scaler"]

model = GazeEstimationModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Constants
LANDMARKS = [33, 133, 263, 362, 61, 291]
PNP_POINTS = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

GAZE_THRESHOLD = 0.6
YAW_THRESH = 25
PITCH_THRESH = 20

def process_frame(frame: np.ndarray):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    distracted = False
    gaze_vector = [0, 0, 1]  # Default forward

    if not result.multi_face_landmarks:
        _, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes(), gaze_vector, distracted

    face = result.multi_face_landmarks[0]
    
    image_pts = np.array([
        (face.landmark[i].x * w, face.landmark[i].y * h)
        for i in PNP_POINTS
    ], dtype=np.float64)

    cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_pts, cam_matrix, np.zeros((4, 1)))
    if not success:
        _, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes(), gaze_vector, distracted

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
    yaw, pitch, _ = angles.flatten()

    if abs(yaw) > YAW_THRESH or abs(pitch) > PITCH_THRESH:
        distracted = True

    xs = [l.x for l in face.landmark]
    ys = [l.y for l in face.landmark]
    xmin, xmax = int(min(xs)*w), int(max(xs)*w)
    ymin, ymax = int(min(ys)*h), int(max(ys)*h)
    crop = frame[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        _, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes(), gaze_vector, distracted

    face_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(face_img).unsqueeze(0).to(DEVICE)

    lm_vec = []
    for i in LANDMARKS:
        lx = face.landmark[i].x * w - xmin
        ly = face.landmark[i].y * h - ymin
        lm_vec.extend([lx / 450.0, ly / 450.0])
    lm_vec = torch.tensor(
        landmark_scaler.transform([lm_vec])[0],
        dtype=torch.float32).unsqueeze(0).to(DEVICE)

    pose_vec = np.hstack([rvec.flatten(), tvec.flatten()])
    pose_vec = torch.tensor(
        pose_scaler.transform([pose_vec])[0],
        dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        gaze = model(input_tensor, lm_vec, pose_vec).cpu().numpy().flatten()
    gaze_vector = gaze.tolist()

    if abs(gaze_vector[0]) > GAZE_THRESHOLD or abs(gaze_vector[1]) > GAZE_THRESHOLD:
        distracted = True

    center = (int(face.landmark[1].x * w), int(face.landmark[1].y * h))
    end = (int(center[0] + gaze_vector[0]*150), int(center[1] - gaze_vector[1]*150))
    cv2.arrowedLine(frame, center, end, (0, 0, 255), 2)

    _, jpeg = cv2.imencode(".jpg", frame)
    return jpeg.tobytes(), gaze_vector, distracted

import os
import math
import scipy.io as sio
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import LeaveOneGroupOut

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_root = 'data/MPIIFaceGaze'  # Update this path
num_epochs = 30
batch_size = 64

# MediaPipe Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 1. Data Loading with Full Annotations
def load_all_data():
    participants = [f'p{i:02d}' for i in range(15)]
    all_data = []
    
    for p in participants:
        participant_path = os.path.join(dataset_root, p)
        annotation_file = os.path.join(participant_path, f'{p}.txt')
        
        with open(annotation_file, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]
            
        for parts in lines:
            item = {
                'image_path': os.path.join(participant_path, parts[0]),
                'gaze_screen': [float(parts[1]), float(parts[2])],
                'landmarks': [float(x) for x in parts[3:15]],
                'rvec': [float(x) for x in parts[16:19]],
                'tvec': [float(x) for x in parts[19:22]],
                'fc': [float(x) for x in parts[22:25]],
                'gt': [float(x) for x in parts[25:28]],
                'participant': int(p[1:])
            }
            all_data.append(item)
            
    return all_data

all_data = load_all_data()

# 2. Dataset Class with 3D Gaze Vectors
class MPIIFaceGazeDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        
        # Calculate 3D gaze vector
        fc = np.array(item['fc'])
        gt = np.array(item['gt'])
        gaze_vector = (gt - fc) / np.linalg.norm(gt - fc)
        
        # Head pose features
        head_pose = np.concatenate([item['rvec'], item['tvec']])
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'head_pose': torch.tensor(head_pose, dtype=torch.float32),
            'gaze': torch.tensor(gaze_vector, dtype=torch.float32),
            'participant': item['participant']
        }

# 3. Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Leave-One-Subject-Out Validation
logo = LeaveOneGroupOut()
groups = [d['participant'] for d in all_data]

for train_idx, test_idx in logo.split(all_data, groups=groups):
    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in test_idx]
    
    train_dataset = MPIIFaceGazeDataset(train_data, transform)
    val_dataset = MPIIFaceGazeDataset(val_data, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train model here (see next section)

# 5. Enhanced Model Architecture
class GazeEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)
        
        self.head_pose_net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.combined_net = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3)
        )
        
    def forward(self, image, head_pose):
        img_features = self.cnn(image)
        pose_features = self.head_pose_net(head_pose)
        return self.combined_net(torch.cat([img_features, pose_features], dim=1))

# 6. Head Pose Estimation from MediaPipe
def estimate_head_pose(landmarks, frame_shape):
    # 3D face model points (from MPIIFaceGaze documentation)
    face_3d = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -330.0, -65.0],      # Chin
        [-225.0, 170.0, -135.0],   # Left eye left corner
        [225.0, 170.0, -135.0],    # Right eye right corner
        [-150.0, -150.0, -125.0],  # Left Mouth corner
        [150.0, -150.0, -125.0]    # Right mouth corner
    ], dtype=np.float64) / 4.5  # Scaling factor
    
    # 2D image points
    image_points = np.array([
        [landmarks[4].x * frame_shape[1], landmarks[4].y * frame_shape[0]],
        [landmarks[152].x * frame_shape[1], landmarks[152].y * frame_shape[0]],
        [landmarks[263].x * frame_shape[1], landmarks[263].y * frame_shape[0]],
        [landmarks[33].x * frame_shape[1], landmarks[33].y * frame_shape[0]],
        [landmarks[287].x * frame_shape[1], landmarks[287].y * frame_shape[0]],
        [landmarks[57].x * frame_shape[1], landmarks[57].y * frame_shape[0]]
    ], dtype=np.float64)
    
    # Camera matrix (from dataset calibration)
    camera_matrix = np.array([
        [1000.0, 0.0, 320.0],
        [0.0, 1000.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Solve PnP
    _, rvec, tvec = cv2.solvePnP(
        face_3d,
        image_points,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    return torch.tensor(np.concatenate([rvec.flatten(), tvec.flatten()]), dtype=torch.float32)

# 7. Complete Training Loop
def train_model(train_loader, val_loader):
    model = GazeEstimationModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CosineEmbeddingLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            head_poses = batch['head_pose'].to(device)
            gazes = batch['gaze'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, head_poses)
            loss = criterion(outputs, gazes, torch.ones(gazes.size(0)).to(device))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                head_poses = batch['head_pose'].to(device)
                gazes = batch['gaze'].to(device)
                
                outputs = model(images, head_poses)
                loss = criterion(outputs, gazes, torch.ones(gazes.size(0)).to(device))
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pth')
            best_val_loss = val_loss

# 8. Real-Time Inference with All Components
class GazeTrackingSystem:
    def __init__(self):
        self.model = GazeEstimationModel().load_state_dict(
            torch.load('best_model.pth', map_location=device))
        self.model.eval()
        
        self.calibrator = GazeCalibrator()
        self.attention_monitor = AttentionMonitor()
        self.cap = cv2.VideoCapture(0)
        
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Face detection and cropping
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Crop face region
                face_crop = self.crop_face(frame, landmarks)
                
                # Estimate head pose
                head_pose = estimate_head_pose(landmarks, frame.shape)
                
                # Predict gaze vector
                with torch.no_grad():
                    image_tensor = transform(Image.fromarray(face_crop)).unsqueeze(0).to(device)
                    gaze_vector = self.model(image_tensor, head_pose.unsqueeze(0).to(device))
                    gaze_vector = gaze_vector.cpu().numpy()[0]
                
                # Apply calibration
                if self.calibrator.cal_matrix is not None:
                    screen_point = self.calibrator.apply_calibration(gaze_vector)
                
                # Check attention
                if self.attention_monitor.check_distraction(head_pose, gaze_vector):
                    cv2.putText(frame, "DISTRACTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow('Gaze Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
    
    def crop_face(self, frame, landmarks):
        # Implementation matching dataset's cropping style
        x_coords = [l.x * frame.shape[1] for l in landmarks]
        y_coords = [l.y * frame.shape[0] for l in landmarks]
        
        x_min = int(max(0, min(x_coords) - 0.1 * (max(x_coords) - min(x_coords))))
        y_min = int(max(0, min(y_coords) - 0.1 * (max(y_coords) - min(y_coords))))
        x_max = int(min(frame.shape[1], max(x_coords) + 0.1 * (max(x_coords) - min(x_coords))))
        y_max = int(min(frame.shape[0], max(y_coords) + 0.1 * (max(y_coords) - min(y_coords))))
        
        # Create black background canvas
        cropped = frame[y_min:y_max, x_min:x_max]
        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        h, w = cropped.shape[:2]
        canvas[:h, :w] = cv2.resize(cropped, (224, 224))
        
        return canvas

# 9. Auxiliary Classes
class GazeCalibrator:
    def __init__(self, screen_size=(1920, 1080)):
        self.cal_points = []  # Store calibration points
        self.screen_size = screen_size
        self.cal_matrix = None
        
    def add_calibration_point(self, gaze_pred, target_pos):
        """Add a calibration point where target_pos is known screen position"""
        self.cal_points.append((gaze_pred, target_pos))
        
    def compute_calibration(self):
        """Compute affine transformation matrix"""
        src = np.array([p[0] for p in self.cal_points])
        dst = np.array([p[1] for p in self.cal_points])
        self.cal_matrix, _ = cv2.estimateAffine2D(src, dst)
        
    def apply_calibration(self, gaze_vector):
        """Apply calibration to raw gaze vector"""
        if self.cal_matrix is None:
            return gaze_vector
        return cv2.transform(np.array([gaze_vector]), self.cal_matrix)[0]

class AttentionMonitor:
    def __init__(self, max_head_angle=30, max_gaze_deviation=0.4):
        self.max_head_angle = max_head_angle
        self.max_gaze_deviation = max_gaze_deviation
        
    def check_distraction(self, head_pose, gaze_vector):
        # Convert rotation vector to Euler angles
        rvec = head_pose[:3]
        R, _ = cv2.Rodrigues(rvec.numpy())
        angles = np.degrees(cv2.RQDecomp3x3(R)[0])
        
        # Check head orientation
        head_distracted = any(abs(a) > self.max_head_angle for a in angles)
        
        # Check gaze deviation from center (assuming screen center is [0.5, 0.5])
        gaze_deviation = np.linalg.norm(gaze_vector[:2])
        gaze_distracted = gaze_deviation > self.max_gaze_deviation
        
        return head_distracted or gaze_distracted

# Main Execution
if __name__ == "__main__":
    # Train the model
    train_model(train_loader, val_loader)
    
    # Run real-time system
    system = GazeTrackingSystem()
    system.run()
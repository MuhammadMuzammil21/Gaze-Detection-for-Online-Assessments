import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import scipy as sio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import pickle

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_ROOT = 'data/MPIIFaceGaze'
NUM_PARTICIPANTS = 15
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # Adjusted learning rate
NUM_EPOCHS_LOSO = 5 # Reduced for faster LOSO demo, increase for better results (e.g., 10-20)
NUM_EPOCHS_FINAL = 15 # Epochs for final model training, increase for better results (e.g., 20-30)
MODEL_SAVE_PATH = 'gaze_model_final.pth'
SCALERS_SAVE_PATH = 'scalers.pkl'

# --- Data Loading and Preprocessing ---
def load_annotations_and_calib(participant_id_str):
    participant_folder = os.path.join(DATASET_ROOT, participant_id_str)
    annotation_file = os.path.join(participant_folder, f'{participant_id_str}.txt')
    
    # Screen size (for normalizing original 2D gaze if needed, not primary target)
    calib_folder = os.path.join(participant_folder, 'Calibration')
    screen_size_file = os.path.join(calib_folder, 'screenSize.mat')
    mat = sio.loadmat(screen_size_file)
    width_pixel = mat['width_pixel'][0,0]
    height_pixel = mat['height_pixel'][0,0]

    data = []
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        image_path = os.path.join(participant_folder, parts[0])
        
        # Dimension 4~15: (x,y) for 6 facial landmarks (12 values)
        landmarks = np.array([float(p) for p in parts[3:15]], dtype=np.float32)
        
        # Dimension 16~21: 3D head pose (rotation, translation - 6 values)
        head_pose_rot = np.array([float(p) for p in parts[15:18]], dtype=np.float32) # rvec
        head_pose_trans = np.array([float(p) for p in parts[18:21]], dtype=np.float32) # tvec
        head_pose = np.concatenate((head_pose_rot, head_pose_trans))

        # Dimension 22~24 (fc): Face center
        face_center_3d = np.array([float(p) for p in parts[21:24]], dtype=np.float32)
        
        # Dimension 25~27 (gt): 3D gaze target
        gaze_target_3d = np.array([float(p) for p in parts[24:27]], dtype=np.float32)
        
        # 3D Gaze Vector = gt - fc
        gaze_vector_3d = gaze_target_3d - face_center_3d
        # Normalize gaze vector to unit length
        norm = np.linalg.norm(gaze_vector_3d)
        if norm > 0:
            gaze_vector_3d = gaze_vector_3d / norm
            
        data.append({
            'image_path': image_path,
            'landmarks': landmarks, # 12D
            'head_pose': head_pose, # 6D
            'gaze_3d': gaze_vector_3d # 3D
        })
    return data

class MPIIFaceGazeDataset(Dataset):
    def __init__(self, data_list, transform, landmark_scaler=None, pose_scaler=None):
        self.data_list = data_list
        self.transform = transform
        self.landmark_scaler = landmark_scaler
        self.pose_scaler = pose_scaler

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)
        
        landmarks = item['landmarks'].copy()
        # Normalize landmarks by image dimensions (assuming dataset images are roughly square and landmarks are within)
        # The images in MPIIFaceGaze are not uniformly sized before cropping for the paper.
        # However, they are often presented as ~448x448 or similar before final cropping.
        # For simplicity, we'll scale them assuming they are in a ~[0, IMAGE_SIZE[0]] range.
        # A more robust way would be to use the original image dimensions if available or bounding box.
        # Here, we use a fixed scaling factor assuming landmarks are within a ~450px box.
        landmarks = landmarks / 450.0 # Approximate normalization
        if self.landmark_scaler:
            landmarks = self.landmark_scaler.transform(landmarks.reshape(1, -1)).flatten()
        
        head_pose = item['head_pose'].copy()
        if self.pose_scaler:
            head_pose = self.pose_scaler.transform(head_pose.reshape(1, -1)).flatten()

        gaze_3d = torch.tensor(item['gaze_3d'], dtype=torch.float32)
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        head_pose = torch.tensor(head_pose, dtype=torch.float32)
        
        return image, landmarks, head_pose, gaze_3d

# Image transforms
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Model Definition ---
class GazeEstimationModel(nn.Module):
    def __init__(self, num_landmarks=12, num_head_pose=6, num_gaze_outputs=3):
        super(GazeEstimationModel, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs_image = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # Remove original FC layer

        self.landmark_fc = nn.Sequential(
            nn.Linear(num_landmarks, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.head_pose_fc = nn.Sequential(
            nn.Linear(num_head_pose, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combined features: image_features + landmark_features + head_pose_features
        combined_feature_size = num_ftrs_image + 32 + 32
        
        self.gaze_fc = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_gaze_outputs)
        )

    def forward(self, image, landmarks, head_pose):
        image_features = self.resnet(image)
        landmark_features = self.landmark_fc(landmarks)
        head_pose_features = self.head_pose_fc(head_pose)
        
        combined_features = torch.cat((image_features, landmark_features, head_pose_features), dim=1)
        gaze = self.gaze_fc(combined_features)
        return gaze

# --- Training Loop ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, landmarks, head_poses, gazes_3d in dataloader:
        images, landmarks, head_poses, gazes_3d = \
            images.to(device), landmarks.to(device), head_poses.to(device), gazes_3d.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, landmarks, head_poses)
        loss = criterion(outputs, gazes_3d)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, landmarks, head_poses, gazes_3d in dataloader:
            images, landmarks, head_poses, gazes_3d = \
                images.to(device), landmarks.to(device), head_poses.to(device), gazes_3d.to(device)
            outputs = model(images, landmarks, head_poses)
            loss = criterion(outputs, gazes_3d)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

# --- Main Script ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    all_data_list = []
    participant_groups = [] # For LOSO splitting
    
    print("Loading dataset...")
    for i in range(NUM_PARTICIPANTS):
        p_id_str = f'p{i:02d}'
        p_data = load_annotations_and_calib(p_id_str)
        all_data_list.extend(p_data)
        participant_groups.extend([i] * len(p_data)) # Group ID for each sample
        print(f"Loaded data for {p_id_str}, {len(p_data)} samples.")

    print(f"Total samples loaded: {len(all_data_list)}")

    # Prepare scalers for landmarks and head pose based on the entire dataset
    # This is a simplification; ideally, scalers are fit only on training data within each LOSO fold.
    # For the final model, fitting on all data is common.
    all_landmarks_np = np.array([d['landmarks']/450.0 for d in all_data_list])
    all_head_poses_np = np.array([d['head_pose'] for d in all_data_list])

    landmark_scaler = StandardScaler().fit(all_landmarks_np)
    pose_scaler = StandardScaler().fit(all_head_poses_np)
    
    with open(SCALERS_SAVE_PATH, 'wb') as f:
        pickle.dump({'landmark_scaler': landmark_scaler, 'pose_scaler': pose_scaler}, f)
    print(f"Scalers saved to {SCALERS_SAVE_PATH}")

    # --- Leave-One-Subject-Out (LOSO) Cross-Validation (Optional Demonstration) ---
    print("\n--- Starting Leave-One-Subject-Out Cross-Validation (Demonstration) ---")
    logo = LeaveOneGroupOut()
    fold_val_losses = []

    for fold_idx, (train_indices, val_indices) in enumerate(logo.split(all_data_list, groups=participant_groups)):
        print(f"\n--- Fold {fold_idx+1}/{NUM_PARTICIPANTS} ---")
        left_out_participant_id = participant_groups[val_indices[0]]
        print(f"Validating on participant: p{left_out_participant_id:02d}")

        train_data_fold = [all_data_list[i] for i in train_indices]
        val_data_fold = [all_data_list[i] for i in val_indices]

        # Re-fit scalers on this specific training fold for strict LOSO (more robust)
        # For this script, we are using global scalers for simplicity to match final model.
        # If you need strict LOSO results, fit scalers here:
        current_train_landmarks_np = np.array([d['landmarks']/450.0 for d in train_data_fold])
        current_train_head_poses_np = np.array([d['head_pose'] for d in train_data_fold])
        current_landmark_scaler = StandardScaler().fit(current_train_landmarks_np)
        current_pose_scaler = StandardScaler().fit(current_train_head_poses_np)
        train_dataset_fold = MPIIFaceGazeDataset(train_data_fold, transform, current_landmark_scaler, current_pose_scaler)
        val_dataset_fold = MPIIFaceGazeDataset(val_data_fold, transform, current_landmark_scaler, current_pose_scaler) # Use train scaler on val

        train_dataset_fold = MPIIFaceGazeDataset(train_data_fold, transform, landmark_scaler, pose_scaler)
        val_dataset_fold = MPIIFaceGazeDataset(val_data_fold, transform, landmark_scaler, pose_scaler)

        train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = GazeEstimationModel().to(DEVICE)
        criterion = nn.MSELoss() # Suitable for regressing 3D vectors
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS_LOSO):
            train_loss = train_epoch(model, train_loader_fold, criterion, optimizer, DEVICE)
            val_loss = validate_epoch(model, val_loader_fold, criterion, DEVICE)
            print(f"Fold {fold_idx+1} Epoch {epoch+1}/{NUM_EPOCHS_LOSO}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        fold_val_losses.append(val_loss) # Store final validation loss for this fold
        print(f"Fold {fold_idx+1} final validation loss: {val_loss:.6f}")

    if fold_val_losses:
      print(f"\nAverage LOSO Validation Loss: {np.mean(fold_val_losses):.6f}")
    else:
      print("LOSO did not run (possibly due to small dataset size or groups issue)")


    # --- Train Final Model on All Data ---
    print("\n--- Training Final Model on All Data ---")
    full_train_dataset = MPIIFaceGazeDataset(all_data_list, transform, landmark_scaler, pose_scaler)
    # For final model, usually we don't have a separate val set, or use a small holdout if desired
    # Here, we train on all data.
    full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    final_model = GazeEstimationModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS_FINAL):
        train_loss = train_epoch(final_model, full_train_loader, criterion, optimizer, DEVICE)
        # Optionally, could validate on a small portion of data if held out
        print(f"Final Model Epoch {epoch+1}/{NUM_EPOCHS_FINAL}, Train Loss: {train_loss:.6f}")

    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")
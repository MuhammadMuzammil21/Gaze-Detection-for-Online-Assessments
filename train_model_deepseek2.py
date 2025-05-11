"""
train_model.py
Train a 3D gaze estimation model with MPIIFaceGaze dataset.
Implements leave-subject-out validation, checkpoints, and multi-modal inputs.
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.preprocessing import StandardScaler

# Constants
DATA_ROOT = 'data/MPIIFaceGaze'  # Update to your path
PARTICIPANTS = [f'p{i:02d}' for i in range(15)]
QUICK_VAL_PARTICIPANTS = ['p00', 'p07', 'p14']  # First, middle, last
BATCH_SIZE = 64
INIT_LR = 3e-4
QUICK_EPOCHS = 3
FINAL_EPOCHS = 10
CHECKPOINT_DIR = 'checkpoints'
IMAGE_SIZE = (224, 224)

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Custom Dataset Class
class MPIIGazeDataset(Dataset):
    def __init__(self, data_samples, transform=None):
        self.data = data_samples
        self.transform = transform
        self.scalers = {
            'head_pose': StandardScaler(),
            'eye_landmarks': StandardScaler()
        }
        self._fit_scalers()

    def _fit_scalers(self):
        all_head_pose = [s['head_pose'] for s in self.data]
        all_eyes = [s['eye_landmarks'] for s in self.data]
        self.scalers['head_pose'].fit(all_head_pose)
        self.scalers['eye_landmarks'].fit(all_eyes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Image processing
        img = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Numerical features
        head_pose = self.scalers['head_pose'].transform([sample['head_pose']])[0]
        eye_landmarks = self.scalers['eye_landmarks'].transform([sample['eye_landmarks']])[0]
        
        # 3D gaze vector (normalized)
        gaze_vector = sample['gaze_vector']
        
        return {
            'image': img,
            'head_pose': torch.FloatTensor(head_pose),
            'eye_landmarks': torch.FloatTensor(eye_landmarks),
            'gaze_vector': torch.FloatTensor(gaze_vector)
        }

# Multi-Input Model Architecture
class GazeEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Image branch
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove final FC layer
        
        # Head pose branch (6 dimensions)
        self.head_pose_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU()
        )
        
        # Eye landmarks branch (12 dimensions)
        self.eye_fc = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU()
        )
        
        # Combined fusion
        self.fc = nn.Sequential(
            nn.Linear(512 + 32 + 64, 256),  # CNN features + head pose + eyes
            nn.ReLU(),
            nn.Linear(256, 3)  # 3D gaze vector
        )
        
    def forward(self, x_img, x_head_pose, x_eyes):
        img_features = self.cnn(x_img)
        head_features = self.head_pose_fc(x_head_pose)
        eye_features = self.eye_fc(x_eyes)
        combined = torch.cat([img_features, head_features, eye_features], dim=1)
        return self.fc(combined)

# Data Loading Utilities
def load_participant_data(participant):
    data = []
    ann_file = os.path.join(DATA_ROOT, participant, f"{participant}.txt")
    
    with open(ann_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # Extract 3D gaze vector (gt - fc)
            fc = np.array([float(x) for x in parts[22:25]])  # Face center
            gt = np.array([float(x) for x in parts[25:28]])  # Gaze target
            gaze_vector = gt - fc
            
            # Head pose (rotation + translation)
            head_pose = [float(x) for x in parts[16:22]]
            
            # Eye landmarks (6 points x,y)
            eye_landmarks = [float(x) for x in parts[4:16]]
            
            # Image path
            img_path = os.path.join(DATA_ROOT, participant, parts[0])
            
            data.append({
                'image_path': img_path,
                'gaze_vector': gaze_vector,
                'head_pose': head_pose,
                'eye_landmarks': eye_landmarks
            })
    return data

# Main Training Function
def train_model():
    # Load all data
    all_data = []
    for p in PARTICIPANTS:
        if p in QUICK_VAL_PARTICIPANTS:
            continue  # Exclude validation participants initially
        all_data.extend(load_participant_data(p))
    
    # Quick validation set
    val_data = []
    for vp in QUICK_VAL_PARTICIPANTS:
        val_data.extend(load_participant_data(vp))
    
    # Create datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = MPIIGazeDataset(all_data, transform=transform)
    val_dataset = MPIIGazeDataset(val_data, transform=transform)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = GazeEstimationModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    
    # Training loop
    for epoch in range(QUICK_EPOCHS):
        # Train epoch
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move data to device
            img = batch['image'].to(device)
            head_pose = batch['head_pose'].to(device)
            eyes = batch['eye_landmarks'].to(device)
            gaze = batch['gaze_vector'].to(device)
            
            # Forward pass
            outputs = model(img, head_pose, eyes)
            loss = criterion(outputs, gaze)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * img.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device)
                head_pose = batch['head_pose'].to(device)
                eyes = batch['eye_landmarks'].to(device)
                gaze = batch['gaze_vector'].to(device)
                
                outputs = model(img, head_pose, eyes)
                val_loss += criterion(outputs, gaze).item() * img.size(0)
        
        # Save checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'quick_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss.item()
        }, ckpt_path)
        
        # Print stats
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        print(f"Quick Epoch {epoch+1}/{QUICK_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Final training on full data (including validation participants)
    print("\nStarting final training on all data...")
    all_data += val_data  # Combine all data
    full_dataset = MPIIGazeDataset(all_data, transform=transform)
    full_loader = DataLoader(full_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    
    for epoch in range(FINAL_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch in full_loader:
            optimizer.zero_grad()
            
            img = batch['image'].to(device)
            head_pose = batch['head_pose'].to(device)
            eyes = batch['eye_landmarks'].to(device)
            gaze = batch['gaze_vector'].to(device)
            
            outputs = model(img, head_pose, eyes)
            loss = criterion(outputs, gaze)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * img.size(0)
        
        # Save final checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'final_{epoch+1}.pth')
        torch.save({
            'epoch': QUICK_EPOCHS + epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss.item()
        }, ckpt_path)
        
        train_loss /= len(full_dataset)
        print(f"Final Epoch {epoch+1}/{FINAL_EPOCHS} | Train Loss: {train_loss:.4f}")

if __name__ == "__main__":
    train_model()
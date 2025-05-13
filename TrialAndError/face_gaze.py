import os
import scipy.io as sio
from PIL import Image
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loading Functions
def load_screen_size(participant_folder):
    calib_folder = os.path.join(participant_folder, 'Calibration')
    screen_size_file = os.path.join(calib_folder, 'screenSize.mat')
    mat = sio.loadmat(screen_size_file)
    width_pixel = mat['width_pixel'][0,0]
    height_pixel = mat['height_pixel'][0,0]
    return width_pixel, height_pixel

def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        image_path = parts[0]
        gaze_x = float(parts[1])
        gaze_y = float(parts[2])
        data.append({'image_path': image_path, 'gaze_x': gaze_x, 'gaze_y': gaze_y})
    return data

# Load all data
dataset_root = 'data/MPIIFaceGaze'
participants = [f'p{i:02d}' for i in range(15)]
all_data = []
for p in participants:
    participant_folder = os.path.join(dataset_root, p)
    annotation_file = os.path.join(participant_folder, f'{p}.txt')
    width_pixel, height_pixel = load_screen_size(participant_folder)
    annotations = load_annotations(annotation_file)
    for ann in annotations:
        image_path = os.path.join(participant_folder, ann['image_path'])
        gaze_x_norm = ann['gaze_x'] / width_pixel
        gaze_y_norm = ann['gaze_y'] / height_pixel
        all_data.append({'image_path': image_path, 'gaze': [gaze_x_norm, gaze_y_norm]})

# Dataset Class
class MPIIFaceGazeDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        gaze = item['gaze']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(gaze, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Split data and create DataLoaders
train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
train_dataset = MPIIFaceGazeDataset(train_data, transform=transform)
val_dataset = MPIIFaceGazeDataset(val_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model Definition
class GazeEstimationModel(nn.Module):
    def __init__(self):
        super(GazeEstimationModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)

model = GazeEstimationModel().to(device)

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, gazes in train_loader:
        images = images.to(device)
        gazes = gazes.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, gazes)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, gazes in val_loader:
            images = images.to(device)
            gazes = gazes.to(device)
            outputs = model(images)
            loss = criterion(outputs, gazes)
            val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

# Real-Time Inference
model.load_state_dict(torch.load('checkpoint_epoch_10.pth'))
model.eval()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
screen_width = 1920  # Adjust to actual screen size
screen_height = 1080  # Adjust to actual screen size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_coords = [landmark.x * frame.shape[1] for landmark in face_landmarks.landmark]
            y_coords = [landmark.y * frame.shape[0] for landmark in face_landmarks.landmark]
            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))
            face_box = [x_min, y_min, x_max - x_min, y_max - y_min]

            face_img = frame[y_min:y_max, x_min:x_max]
            if face_img.size == 0:
                continue

            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = Image.fromarray(face_img)
            face_img = transform(face_img).unsqueeze(0).to(device)

            with torch.no_grad():
                gaze_pred = model(face_img).cpu().numpy()[0]
            gaze_x_screen = gaze_pred[0] * screen_width
            gaze_y_screen = gaze_pred[1] * screen_height

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133]
            left_eye_x = [face_landmarks.landmark[i].x * frame.shape[1] for i in left_eye_indices]
            left_eye_y = [face_landmarks.landmark[i].y * frame.shape[0] for i in left_eye_indices]
            left_eye_box = [int(min(left_eye_x)), int(min(left_eye_y)), int(max(left_eye_x)) - int(min(left_eye_x)), int(max(left_eye_y)) - int(min(left_eye_y))]

            right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362]
            right_eye_x = [face_landmarks.landmark[i].x * frame.shape[1] for i in right_eye_indices]
            right_eye_y = [face_landmarks.landmark[i].y * frame.shape[0] for i in right_eye_indices]
            right_eye_box = [int(min(right_eye_x)), int(min(right_eye_y)), int(max(right_eye_x)) - int(min(right_eye_x)), int(max(right_eye_y)) - int(min(right_eye_y))]

            cv2.rectangle(frame, (left_eye_box[0], left_eye_box[1]), (left_eye_box[0] + left_eye_box[2], left_eye_box[1] + left_eye_box[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (right_eye_box[0], right_eye_box[1]), (right_eye_box[0] + right_eye_box[2], right_eye_box[1] + right_eye_box[3]), (255, 0, 0), 2)

            cv2.putText(frame, f'Gaze: ({gaze_x_screen:.0f}, {gaze_y_screen:.0f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Gaze Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
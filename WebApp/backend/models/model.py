# backend/models/model.py

import torch.nn as nn
from torchvision.models import resnet18

class GazeEstimationModel(nn.Module):
    def __init__(self, num_landmarks=12, num_head_pose=6, num_gaze_outputs=3):
        super(GazeEstimationModel, self).__init__()
        self.resnet = resnet18(weights=None)
        num_ftrs_image = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

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

import os
import cv2
import numpy as np
from shutil import copyfile

# Step 1: Install YOLOv5 (uncomment to run)
# !git clone https://github.com/ultralytics/yolov5
# %cd yolov5
# !pip install -r requirements.txt

# Step 2: Preprocess MPIIGaze Dataset
data_dir = 'path/to/MPIIGaze/Data/Original'
output_dir = 'path/to/output'
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

def parse_annotation(annotation_path, image_path, output_label_path, image_width, image_height):
    with open(annotation_path, 'r') as f:
        values = list(map(float, f.read().split()))
    
    # Extract landmarks (assuming dimensions 1-24 are eye landmarks)
    left_eye_landmarks = values[0:12]  # x1,y1,x2,y2,...,x6,y6
    right_eye_landmarks = values[12:24]
    
    # Compute bounding boxes
    left_x = [left_eye_landmarks[i] for i in range(0, 12, 2)]
    left_y = [left_eye_landmarks[i] for i in range(1, 12, 2)]
    left_bbox = [min(left_x), min(left_y), max(left_x), max(left_y)]
    
    right_x = [right_eye_landmarks[i] for i in range(0, 12, 2)]
    right_y = [right_eye_landmarks[i] for i in range(1, 12, 2)]
    right_bbox = [min(right_x), min(right_y), max(right_x), max(right_y)]
    
    # Face bounding box: Entire image
    face_bbox = [0, 0, image_width, image_height]
    
    # Convert to YOLO format
    def to_yolo(bbox, class_id):
        x_center = (bbox[0] + bbox[2]) / 2 / image_width
        y_center = (bbox[1] + bbox[3]) / 2 / image_height
        width = (bbox[2] - bbox[0]) / image_width
        height = (bbox[3] - bbox[1]) / image_height
        return f"{class_id} {x_center} {y_center} {width} {height}\n"
    
    with open(output_label_path, 'w') as f:
        f.write(to_yolo(face_bbox, 0))  # Face
        f.write(to_yolo(left_bbox, 1))  # Left eye
        f.write(to_yolo(right_bbox, 2))  # Right eye

# Process images
for participant in os.listdir(data_dir):
    participant_dir = os.path.join(data_dir, participant)
    for day in os.listdir(participant_dir):
        day_dir = os.path.join(participant_dir, day)
        for file in os.listdir(day_dir):
            if file.endswith('.jpg'):
                image_path = os.path.join(day_dir, file)
                base_name = os.path.splitext(file)[0]
                annotation_file = base_name + '.txt'
                annotation_path = os.path.join(day_dir, annotation_file)
                
                if os.path.exists(annotation_path):
                    image = cv2.imread(image_path)
                    h, w = image.shape[:2]
                    copyfile(image_path, os.path.join(output_dir, 'images', file))
                    label_path = os.path.join(output_dir, 'labels', base_name + '.txt')
                    parse_annotation(annotation_path, image_path, label_path, w, h)

# Create train.txt
with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    for file in os.listdir(os.path.join(output_dir, 'images')):
        f.write(os.path.join('images', file) + '\n')

# Create data.yaml
data_yaml = """
train: /path/to/output/train.txt
val: /path/to/output/train.txt
nc: 3
names: ['face', 'left_eye', 'right_eye']
"""
with open('data.yaml', 'w') as f:
    f.write(data_yaml)

# Step 3: Train YOLOv5 (uncomment to run)
# !python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt --cache

# Step 4: Deploy on Webcam
# !pip install opencv-python
import cv2
from yolov5 import YOLO

model = YOLO('runs/train/exp/weights/best.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results.render()[0]
    cv2.imshow('Gaze Detector', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
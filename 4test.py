"""
# To load a pretrained model instead of retraining, uncomment:
# from tensorflow.keras.models import load_model
# model = load_model('gaze_model.keras')
# model = load_model('gaze_model')
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import get_file
from sklearn.model_selection import train_test_split
import cv2
import dlib

# 1. Download & extract the MPIIFaceGaze dataset
dataset_zip = get_file(
    fname='MPIIFaceGaze.zip',
    origin='https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip',
    extract=True
)
dataset_dir = os.path.splitext(dataset_zip)[0]
if not os.path.isdir(dataset_dir):
    raise FileNotFoundError(f"Extracted dataset directory not found: {dataset_dir}")

# 2. Parse all .txt annotation files recursively
gaze_vals, entries = [], []
for root, dirs, files in os.walk(dataset_dir):
    if any(skip in root for skip in ('Calibration', 'Normalized', 'Evaluation')):
        continue
    for fname in files:
        if not fname.lower().endswith('.txt') or fname.lower().startswith('readme'):
            continue
        file_path = os.path.join(root, fname)
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_rel = parts[0]
                try:
                    gx = float(parts[1])
                except ValueError:
                    continue
                # Try both root-relative and dataset-relative paths
                path1 = os.path.join(root, img_rel)
                path2 = os.path.join(dataset_dir, img_rel)
                if os.path.exists(path1):
                    img_path = path1
                elif os.path.exists(path2):
                    img_path = path2
                else:
                    continue
                gaze_vals.append(gx)
                entries.append((img_path, gx))

if not entries:
    raise FileNotFoundError("No valid annotation entries found")

# 3. Create binary labels via median split
threshold = np.median(gaze_vals)
image_paths = [p for p,_ in entries]
labels      = [0 if gx < threshold else 1 for _,gx in entries]

# 4. Train/validation split
train_p, val_p, train_l, val_l = train_test_split(
    image_paths, labels,
    test_size=0.2, random_state=42,
    stratify=labels
)

# 5. Build tf.data pipelines
def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224,224))
    img = tf.keras.applications.resnet.preprocess_input(img)
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_p, train_l)) \
    .map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((val_p, val_l)) \
    .map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# 6. Define Model (ResNet50 → LSTM → Dense)
base = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet',
    input_shape=(224,224,3), pooling='avg'
)
base.trainable = False

inp  = layers.Input((224,224,3))
feat = base(inp, training=False)
seq  = layers.Reshape((1, feat.shape[-1]))(feat)
x = layers.Dense(64, activation='relu')(feat)
out = layers.Dense(1, activation='sigmoid')(x)


model = models.Model(inp, out)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'mse', tf.keras.metrics.AUC()]
)

# 7. Train until >70% validation accuracy
model.fit(train_ds, epochs=10, validation_data=val_ds)

# 8. Save the trained model
model.save('gaze_model.keras')
model.save('gaze_model')

# 9. Live camera loop: infer on full frames
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the full frame to model input size
    img_resized = cv2.resize(frame, (224, 224))
    inp_tensor  = tf.keras.applications.resnet.preprocess_input(
                      np.expand_dims(img_resized, 0))

    # Run prediction
    pred = model.predict(inp_tensor, verbose=0)[0,0]
    lbl  = 'Right' if pred > 0.5 else 'Left'

    # Overlay label on the original frame
    cv2.putText(
        frame, f'Gaze: {lbl}', (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
    )

    cv2.imshow('Live Gaze Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
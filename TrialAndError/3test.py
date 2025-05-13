# ---------------------------------------
# Uncomment these lines to load a saved model for inference:
# from tensorflow.keras.models import load_model
# model = load_model('gaze_model.h5')
# ---------------------------------------

# 1. Download dataset via Kaggle API
# Requires: pip install kaggle
# & setting KAGGLE_USERNAME and KAGGLE_KEY in your environment
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    'bauthantekmen/gaze-locking-interpreted-from-columbia-gaze',
    path='data',
    unzip=True
)

# 2. Imports and parameters
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = 'data/images'      # adjust if needed
LABELS_CSV = 'data/labels.csv'  # adjust if provided
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# 3. Prepare labels and split
labels = pd.read_csv(LABELS_CSV)
# Define binary classes: on_screen if |yaw|<5° and |pitch|<5°, else off_screen
labels['class'] = labels.apply(
    lambda r: 'on_screen' if abs(r['yaw']) < 5 and abs(r['pitch']) < 5 else 'off_screen',
    axis=1
)
# Split into train/val
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(labels, test_size=0.2, random_state=42, stratify=labels['class'])

# 4. Data pipelines
def preprocess(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

def make_dataset(df):
    paths = df['file_name'].apply(lambda x: os.path.join(DATA_DIR, x)).values
    lbls = df['class'].map({'on_screen':0, 'off_screen':1}).values
    ds = tf.data.Dataset.from_tensor_slices((paths, lbls))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df)
val_ds   = make_dataset(val_df)

# 5. Model definition with Transfer Learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Training with EarlyStopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 7. Evaluate and save
val_loss, val_acc = model.evaluate(val_ds)
print(f'Validation accuracy: {val_acc:.2%}')
assert val_acc >= 0.70, "Desired accuracy not reached"

model.save('gaze_model.h5')
print("Model saved to gaze_model.h5")

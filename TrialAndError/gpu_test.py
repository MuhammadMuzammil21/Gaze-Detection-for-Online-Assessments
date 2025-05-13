import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Dummy data
x = np.random.rand(10000, 32)
y = np.random.randint(0, 10, size=(10000,))

model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x, y, epochs=5, batch_size=64)

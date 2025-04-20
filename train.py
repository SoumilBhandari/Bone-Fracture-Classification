import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications

# Paths
base_dir = 'data'
batch_size = 32
img_size = (224, 224)

# Data loading
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Build model
base_model = applications.ResNet50(
    input_shape=(*img_size, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    layers.InputLayer(input_shape=(*img_size, 3)),
    data_augmentation,
    applications.resnet.preprocess_input,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save
model.save('bone_fracture_detector.h5')

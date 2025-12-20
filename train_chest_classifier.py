#!/usr/bin/env python3
"""
Train a binary classifier to detect CHEST X-rays vs OTHER X-ray types.
Uses MobileNetV2 transfer learning with heavy data augmentation.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
TRAIN_DIR = 'chest_classifier_data/train'
VAL_DIR = 'chest_classifier_data/val'
MODEL_PATH = 'results/models/chest_classifier.keras'

print("=" * 60)
print("CHEST X-RAY CLASSIFIER TRAINING")
print("Distinguishes chest X-rays from other X-ray types")
print("=" * 60)

# Heavy Data Augmentation for not_chest class (to compensate for fewer samples)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    shear_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
print("\nLoading datasets...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['not_chest', 'chest']  # 0 = not_chest, 1 = chest
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['not_chest', 'chest']
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Classes: {train_generator.class_indices}")

# Calculate class weights to handle imbalanced data
n_not_chest = len(os.listdir(os.path.join(TRAIN_DIR, 'not_chest')))
n_chest = len(os.listdir(os.path.join(TRAIN_DIR, 'chest')))
total = n_not_chest + n_chest

# Give more weight to the minority class (not_chest)
weight_for_not_chest = (1 / n_not_chest) * (total / 2.0)
weight_for_chest = (1 / n_chest) * (total / 2.0)
class_weight = {0: weight_for_not_chest, 1: weight_for_chest}

print(f"\nClass weights: not_chest={weight_for_not_chest:.2f}, chest={weight_for_chest:.2f}")

# Build Model
print("\nBuilding model with MobileNetV2 backbone...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Model parameters: {model.count_params():,}")

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH.replace('.keras', '_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Train
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# Fine-tuning: Unfreeze last layers
print("\n" + "=" * 60)
print("FINE-TUNING")
print("=" * 60)

base_model.trainable = True
# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# Evaluate
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")

# Save final model
print(f"\nSaving model to {MODEL_PATH}...")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print("Model saved successfully!")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"Model saved to: {MODEL_PATH}")
print("=" * 60)

#!/usr/bin/env python3
"""
PHASE 4: QUICK MODEL TRAINING - 5 EPOCHS
========================================
Fast training implementation for Lightweight U-Net
5 epochs for quick validation of model performance
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split

print("🚀 PHASE 4: QUICK MODEL TRAINING (5 EPOCHS)")
print("=" * 60)
print("Objective: Quick training validation with 5 epochs")
print("Fast iteration for performance testing")
print()

def create_lightweight_unet(input_shape=(80, 160, 3), num_classes=1):
    """Create and compile Lightweight U-Net model"""
    print("🏗️ Creating Lightweight U-Net for quick training...")
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name="input_image")
    
    # Simplified encoder (fewer layers for faster training)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    
    # Simplified decoder
    up5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
    concat5 = layers.concatenate([up5, conv3])
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat5)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat6 = layers.concatenate([up6, conv2])
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat7 = layers.concatenate([up7, conv1])
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat7)
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    
    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(conv7)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs, name='Quick_UNet')
    
    # Compile with optimized settings
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.BinaryIoU(name='iou')
        ]
    )
    
    print(f"✅ Quick U-Net created and compiled!")
    print(f"   Parameters: {model.count_params():,}")
    
    return model

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("📂 Loading and preprocessing dataset...")
    
    dataset_dir = Path("../../Dataset")
    
    # Load training images
    with open(dataset_dir / "train_dataset.p", 'rb') as f:
        train_data = pickle.load(f)
    
    # Load labels  
    with open(dataset_dir / "labels_dataset.p", 'rb') as f:
        labels_data = pickle.load(f)
    
    # Convert to numpy arrays and preprocess
    train_data = np.array(train_data, dtype=np.float32) / 255.0  # Normalize
    labels_data = np.array(labels_data, dtype=np.float32)
    labels_data = (labels_data > 240).astype(np.float32)  # Binary threshold
    
    print(f"✅ Data loaded and preprocessed:")
    print(f"   Training images: {train_data.shape}")
    print(f"   Labels: {labels_data.shape}")
    
    return train_data, labels_data

def create_training_datasets(images, labels, batch_size=16):
    """Create training datasets with splits (larger batch for speed)"""
    print(f"\n📂 Creating training datasets (batch_size={batch_size})...")
    
    # Create splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42
    )
    
    print(f"✅ Data splits created:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # Simple augmentation for training
    def augment_data(image, mask):
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        return image, mask
    
    # Optimize datasets
    train_dataset = (train_dataset
                    .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))
    
    val_dataset = (val_dataset
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE))
    
    test_dataset = (test_dataset
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))
    
    print(f"✅ TensorFlow datasets created")
    
    return train_dataset, val_dataset, test_dataset, (X_test, y_test)

def train_quick_model(model, train_dataset, val_dataset, epochs=5):
    """Quick training with 5 epochs"""
    print(f"\n🚀 Starting quick training for {epochs} epochs...")
    
    # Simple callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_and_save(model, test_dataset, test_data, history):
    """Quick evaluation and save results"""
    print("\n📊 Quick evaluation...")
    
    # Evaluate on test set
    test_results = model.evaluate(test_dataset, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))
    
    print(f"\nModel metrics names: {model.metrics_names}")
    print(f"Test results: {test_results}")
    
    print(f"\n✅ Quick training completed!")
    print(f"   Test Accuracy: {test_results[1]:.4f}")
    print(f"   Test IoU: {test_results[2]:.4f}")
    print(f"   Test Loss: {test_results[0]:.4f}")
    
    # Check if we're approaching target
    accuracy = test_results[1]
    if accuracy >= 0.70:
        print(f"   🎉 Great progress! {accuracy:.1%} accuracy in just {len(history.history['loss'])} epochs!")
    
    if accuracy >= 0.87:
        print(f"   🚀 AMAZING! Already exceeded 87% target!")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model.save(models_dir / "quick_trained_model.keras")
    print("✅ Model saved: quick_trained_model.keras")
    
    # Create quick visualization
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.axhline(y=0.87, color='red', linestyle='--', label='Target (87%)')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['iou'], label='Training IoU')
    plt.plot(history.history['val_iou'], label='Validation IoU')
    plt.title('Model IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_dir / "quick_training_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Quick results visualization saved")
    
    return test_results

def main():
    """Main quick training pipeline"""
    
    # Create model
    model = create_lightweight_unet()
    
    # Load data
    images, labels = load_and_preprocess_data()
    
    # Create datasets (larger batch size for speed)
    train_dataset, val_dataset, test_dataset, test_data = create_training_datasets(images, labels, batch_size=16)
    
    # Quick training
    history = train_quick_model(model, train_dataset, val_dataset, epochs=5)
    
    # Evaluate and save
    test_metrics = evaluate_and_save(model, test_dataset, test_data, history)
    
    print("\n" + "=" * 60)
    print("✅ QUICK TRAINING (5 EPOCHS) COMPLETE!")
    print("=" * 60)
    
    print(f"\n🏆 Quick Results:")
    print(f"   • Epochs: {len(history.history['loss'])}")
    print(f"   • Test Accuracy: {test_metrics[1]:.4f}")
    print(f"   • Test IoU: {test_metrics[2]:.4f}")
    print(f"   • Progress: {'🎉 Excellent!' if test_metrics[1] > 0.7 else '📈 Good start!'}")
    
    if test_metrics[1] >= 0.87:
        print(f"   • PPT Target: ✅ ACHIEVED!")
    else:
        print(f"   • PPT Target: 📊 {test_metrics[1]:.1%} / 87% (great progress!)")
    
    print(f"\n📁 Quick Outputs:")
    print(f"   🏗️ Model: models/quick_trained_model.keras")
    print(f"   📊 Results: visualizations/quick_training_results.png")

if __name__ == "__main__":
    main()
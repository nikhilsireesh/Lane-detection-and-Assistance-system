#!/usr/bin/env python3
"""
PHASE 5: MODEL VALIDATION
========================
Comprehensive model validation and performance analysis
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import time

print("🚀 PHASE 5: MODEL VALIDATION")
print("=" * 50)
print("Comprehensive model validation and performance analysis")
print()

def load_trained_model():
    """Load the trained model"""
    print("📂 Loading trained model...")
    
    possible_paths = [
        Path("../phase4_model_training/scripts/models/quick_trained_model.keras"),
        Path("../../phase4_model_training/scripts/models/quick_trained_model.keras")
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        print("❌ Error: Trained model not found.")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_test_data():
    """Load and prepare test dataset"""
    print("\n📂 Loading test dataset...")
    
    import pickle
    dataset_dir = Path("../../Dataset")
    
    with open(dataset_dir / "train_dataset.p", 'rb') as f:
        train_data = pickle.load(f)
    
    with open(dataset_dir / "labels_dataset.p", 'rb') as f:
        labels_data = pickle.load(f)
    
    train_data = np.array(train_data, dtype=np.float32) / 255.0
    labels_data = np.array(labels_data, dtype=np.float32)
    labels_data = (labels_data > 240).astype(np.float32)
    
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        train_data, labels_data, test_size=0.2, random_state=42
    )
    
    print(f"✅ Test data prepared:")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Image shape: {X_test.shape[1:]}")
    print(f"   Label shape: {y_test.shape[1:]}")
    
    return X_test, y_test

def validate_model_performance(model, X_test, y_test):
    """Validate model performance"""
    print("\n📊 Validating model performance...")
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(8).prefetch(tf.data.AUTOTUNE)
    
    print("Evaluating model...")
    test_results = model.evaluate(test_dataset, verbose=1)
    
    loss = test_results[0]
    accuracy = test_results[1] if len(test_results) > 1 else 0.0
    iou = test_results[2] if len(test_results) > 2 else 0.0
    
    print("Generating predictions...")
    predictions = model.predict(test_dataset, verbose=1)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate additional metrics
    intersection = np.logical_and(y_test, binary_predictions)
    union = np.logical_or(y_test, binary_predictions)
    iou_manual = np.sum(intersection) / np.sum(union)
    
    pixel_accuracy = np.mean(binary_predictions == y_test)
    
    true_positives = np.sum(np.logical_and(y_test == 1, binary_predictions == 1))
    false_positives = np.sum(np.logical_and(y_test == 0, binary_predictions == 1))
    false_negatives = np.sum(np.logical_and(y_test == 1, binary_predictions == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        "accuracy": float(accuracy),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "loss": float(loss),
        "manual_iou": float(iou_manual),
        "pixel_accuracy": float(pixel_accuracy),
        "f1_score": float(f1_score)
    }
    
    print(f"✅ Validation completed:")
    print(f"   🎯 Accuracy: {results['accuracy']:.4f} ({results['accuracy']:.1%})")
    print(f"   📊 IoU: {results['iou']:.4f}")
    print(f"   ⚖️ Precision: {results['precision']:.4f}")
    print(f"   🔄 Recall: {results['recall']:.4f}")
    print(f"   📈 F1 Score: {results['f1_score']:.4f}")
    print(f"   🎯 Pixel Accuracy: {results['pixel_accuracy']:.4f}")
    
    return results, predictions, binary_predictions

def test_inference_speed(model, X_test):
    """Test inference speed"""
    print("\n⏱️ Testing inference speed...")
    
    batch_sizes = [1, 4, 8, 16]
    timing_results = {}
    
    print("Warming up model...")
    _ = model.predict(X_test[:8], verbose=0)
    
    for batch_size in batch_sizes:
        if batch_size > len(X_test):
            continue
            
        print(f"Testing batch size {batch_size}...")
        
        test_batch = X_test[:batch_size]
        
        times = []
        for i in range(5):
            start_time = time.time()
            _ = model.predict(test_batch, verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        per_image_time = avg_time / batch_size * 1000  # milliseconds
        
        timing_results[batch_size] = {
            "avg_time": avg_time,
            "per_image_ms": per_image_time
        }
        
        print(f"   Batch {batch_size}: {per_image_time:.1f}ms per image")
    
    return timing_results

def create_validation_visualizations(results, X_test, y_test, predictions):
    """Create validation visualizations"""
    print("\n🎨 Creating validation visualizations...")
    
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    # 1. Performance metrics visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Validation Results', fontsize=16, fontweight='bold')
    
    # Metrics bar chart
    metrics_names = ['Accuracy', 'IoU', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [results['accuracy'], results['iou'], results['precision'], 
                     results['recall'], results['f1_score']]
    
    bars = axes[0, 0].bar(metrics_names, metrics_values, 
                         color=['skyblue', 'lightgreen', 'orange', 'pink', 'gold'])
    axes[0, 0].set_title('Model Performance Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, metrics_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Sample prediction
    sample_idx = np.random.choice(len(X_test))
    
    axes[0, 1].imshow(X_test[sample_idx])
    axes[0, 1].set_title('Test Image')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(y_test[sample_idx, :, :, 0], cmap='gray')
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(predictions[sample_idx, :, :, 0], cmap='gray')
    axes[1, 1].set_title('Prediction')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / "validation_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed prediction comparison
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Prediction Results Comparison', fontsize=16, fontweight='bold')
    
    sample_indices = np.random.choice(len(X_test), 4, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Original image
        axes[i, 0].imshow(X_test[idx])
        axes[i, 0].set_title(f'Original {i+1}')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(y_test[idx, :, :, 0], cmap='gray')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[idx, :, :, 0], cmap='gray')
        axes[i, 2].set_title(f'Prediction {i+1}')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = X_test[idx].copy()
        mask = predictions[idx, :, :, 0] > 0.5
        overlay[mask] = [1, 0, 0]  # Red overlay for detected lanes
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'Overlay {i+1}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(vis_dir / "prediction_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
    
    # Performance pie chart
    performance_data = [results['accuracy'], 1 - results['accuracy']]
    performance_labels = [f"Correct ({results['accuracy']:.1%})", 
                         f"Incorrect ({1-results['accuracy']:.1%})"]
    colors = ['lightgreen', 'lightcoral']
    
    axes[0].pie(performance_data, labels=performance_labels, colors=colors, 
               autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Overall Accuracy')
    
    # Metrics comparison
    metric_labels = ['Precision', 'Recall', 'F1-Score', 'IoU']
    metric_values = [results['precision'], results['recall'], 
                    results['f1_score'], results['iou']]
    
    axes[1].barh(metric_labels, metric_values, color='lightblue')
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Detailed Metrics')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(metric_values):
        axes[1].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(vis_dir / "performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Validation visualizations saved:")
    print(f"   📊 validation_overview.png")
    print(f"   🔍 prediction_comparison.png") 
    print(f"   📈 performance_summary.png")

def save_results(results, timing_results):
    """Save validation results"""
    print("\n💾 Saving validation results...")
    
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    complete_results = {
        "timestamp": datetime.now().isoformat(),
        "phase": "5_model_validation",
        "validation_results": results,
        "timing_results": timing_results,
        "summary": {
            "accuracy_percentage": f"{results['accuracy']:.1%}",
            "iou_score": f"{results['iou']:.3f}",
            "f1_score": f"{results['f1_score']:.3f}",
            "fastest_inference": f"{min([t['per_image_ms'] for t in timing_results.values()]):.1f}ms"
        }
    }
    
    with open(outputs_dir / "validation_results.json", 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    # Create simple report
    report_content = f"""# Model Validation Report

## Performance Results
- **Accuracy:** {results['accuracy']:.1%}
- **IoU:** {results['iou']:.3f}
- **Precision:** {results['precision']:.3f}
- **Recall:** {results['recall']:.3f}
- **F1 Score:** {results['f1_score']:.3f}

## Inference Speed
- **Single Image:** {timing_results[1]['per_image_ms']:.1f}ms
- **Batch (8 images):** {timing_results[8]['per_image_ms']:.1f}ms per image

## Files Generated
- validation_overview.png - Overall performance metrics
- prediction_comparison.png - Detailed prediction results
- performance_summary.png - Summary charts
- validation_results.json - Complete data

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    with open(reports_dir / "validation_report.md", 'w') as f:
        f.write(report_content)
    
    print("✅ Results saved:")
    print("   📊 validation_results.json")
    print("   📄 validation_report.md")

def main():
    """Main validation pipeline"""
    
    model = load_trained_model()
    if model is None:
        return
    
    X_test, y_test = load_test_data()
    
    results, predictions, binary_predictions = validate_model_performance(model, X_test, y_test)
    
    timing_results = test_inference_speed(model, X_test)
    
    create_validation_visualizations(results, X_test, y_test, predictions)
    
    save_results(results, timing_results)
    
    print("\n" + "=" * 50)
    print("✅ PHASE 5: MODEL VALIDATION COMPLETE!")
    print("=" * 50)
    
    print(f"\n🏆 Final Results:")
    print(f"   • Accuracy: {results['accuracy']:.1%}")
    print(f"   • IoU: {results['iou']:.3f}")
    print(f"   • F1 Score: {results['f1_score']:.3f}")
    print(f"   • Inference: {timing_results[1]['per_image_ms']:.1f}ms per image")
    
    print(f"\n📁 Generated Files:")
    print(f"   📊 visualizations/ - Performance charts and predictions")
    print(f"   📄 reports/ - Validation report")
    print(f"   💾 outputs/ - Raw data and metrics")

if __name__ == "__main__":
    main()
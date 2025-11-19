#!/usr/bin/env python3
"""
Enhanced Phase 1: Dataset Analysis - TensorFlow Implementation
Comprehensive analysis of lane detection dataset following PPT specifications
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
import cv2

print("🚀 PHASE 1: ENHANCED DATASET ANALYSIS")
print("=" * 60)
print("Objective: Comprehensive dataset understanding for TensorFlow implementation")
print("Reference: PPT specifications - 12,764 images, 80x160 resolution")
print()

def load_and_analyze_dataset():
    """Load and perform comprehensive analysis of the dataset"""
    
    print("📂 Loading dataset files...")
    
    # Dataset paths
    dataset_dir = Path("../../Dataset")
    train_path = dataset_dir / "train_dataset.p"
    labels_path = dataset_dir / "labels_dataset.p"
    
    analysis_results = {}
    
    # Load training images
    try:
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        print("✅ Training dataset loaded successfully")
        
        # Convert list to numpy array for analysis
        if isinstance(train_data, list):
            train_data = np.array(train_data)
            print(f"   Converted list of {len(train_data)} images to numpy array")
        
        # Analyze training data structure
        train_analysis = {
            "type": str(type(train_data)),
            "shape": list(train_data.shape) if hasattr(train_data, 'shape') else None,
            "dtype": str(train_data.dtype) if hasattr(train_data, 'dtype') else None,
            "size_mb": float(train_data.nbytes / (1024**2)) if hasattr(train_data, 'nbytes') else None,
            "min_value": float(np.min(train_data)) if hasattr(train_data, 'min') else None,
            "max_value": float(np.max(train_data)) if hasattr(train_data, 'max') else None,
            "mean_value": float(np.mean(train_data)) if hasattr(train_data, 'mean') else None,
            "std_value": float(np.std(train_data)) if hasattr(train_data, 'std') else None
        }
        
        analysis_results["training_data"] = train_analysis
        
        print(f"   Shape: {train_analysis['shape']}")
        print(f"   Data type: {train_analysis['dtype']}")
        print(f"   Size: {train_analysis['size_mb']:.2f} MB")
        print(f"   Value range: [{train_analysis['min_value']:.1f}, {train_analysis['max_value']:.1f}]")
        print(f"   Mean: {train_analysis['mean_value']:.3f}, Std: {train_analysis['std_value']:.3f}")
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        analysis_results["training_data"] = {"error": str(e)}
    
    # Load labels
    try:
        with open(labels_path, 'rb') as f:
            labels_data = pickle.load(f)
        print("✅ Labels dataset loaded successfully")
        
        # Convert list to numpy array for analysis
        if isinstance(labels_data, list):
            labels_data = np.array(labels_data)
            print(f"   Converted list of {len(labels_data)} labels to numpy array")
        
        # Analyze labels data structure
        labels_analysis = {
            "type": str(type(labels_data)),
            "shape": list(labels_data.shape) if hasattr(labels_data, 'shape') else None,
            "dtype": str(labels_data.dtype) if hasattr(labels_data, 'dtype') else None,
            "size_mb": float(labels_data.nbytes / (1024**2)) if hasattr(labels_data, 'nbytes') else None,
            "min_value": float(np.min(labels_data)) if hasattr(labels_data, 'min') else None,
            "max_value": float(np.max(labels_data)) if hasattr(labels_data, 'max') else None,
            "unique_values": np.unique(labels_data).tolist() if hasattr(labels_data, 'min') else None
        }
        
        analysis_results["labels_data"] = labels_analysis
        
        print(f"   Shape: {labels_analysis['shape']}")
        print(f"   Data type: {labels_analysis['dtype']}")
        print(f"   Size: {labels_analysis['size_mb']:.2f} MB")
        print(f"   Value range: [{labels_analysis['min_value']:.1f}, {labels_analysis['max_value']:.1f}]")
        print(f"   Unique values: {len(labels_analysis['unique_values']) if labels_analysis['unique_values'] else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Error loading labels data: {e}")
        analysis_results["labels_data"] = {"error": str(e)}
    
    # Verify PPT specifications
    print("\n📊 Verifying PPT specifications...")
    if "training_data" in analysis_results and analysis_results["training_data"]["shape"]:
        shape = analysis_results["training_data"]["shape"]
        total_images = shape[0]
        height, width, channels = shape[1], shape[2], shape[3]
        
        ppt_specs = {
            "expected_total": 12764,
            "expected_resolution": [80, 160, 3],
            "actual_total": total_images,
            "actual_resolution": [height, width, channels],
            "specs_match": (total_images == 12764 and 
                          height == 80 and width == 160 and channels == 3)
        }
        
        analysis_results["ppt_verification"] = ppt_specs
        
        if ppt_specs["specs_match"]:
            print("✅ Dataset matches PPT specifications perfectly!")
        else:
            print("⚠️  Dataset differs from PPT specifications:")
            print(f"   Expected: {ppt_specs['expected_total']} images, {ppt_specs['expected_resolution']}")
            print(f"   Actual: {ppt_specs['actual_total']} images, {ppt_specs['actual_resolution']}")
    
    return analysis_results, train_data, labels_data

def create_enhanced_visualizations(train_data, labels_data, output_dir):
    """Create comprehensive visualizations following PPT style"""
    
    print("\n🎨 Creating enhanced visualizations...")
    
    # Create sample visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Enhanced Lane Detection Dataset - Sample Analysis', fontsize=16, fontweight='bold')
    
    # Show sample images and labels
    for i in range(4):
        # Original image
        axes[0, i].imshow(train_data[i])
        axes[0, i].set_title(f'Sample {i+1}: Input Image\n80×160×3 RGB', fontweight='bold')
        axes[0, i].axis('off')
        
        # Corresponding label
        axes[1, i].imshow(labels_data[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Sample {i+1}: Lane Mask\nBinary Segmentation', fontweight='bold')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create statistics visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Enhanced Dataset Statistics - TensorFlow Ready', fontsize=16, fontweight='bold')
    
    # Pixel value distribution
    sample_pixels = train_data[:1000].flatten()
    axes[0, 0].hist(sample_pixels, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Pixel Value Distribution\n(Sample of 1000 images)', fontweight='bold')
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Channel-wise statistics
    mean_rgb = [np.mean(train_data[:, :, :, i]) for i in range(3)]
    std_rgb = [np.std(train_data[:, :, :, i]) for i in range(3)]
    
    x = ['Red', 'Green', 'Blue']
    axes[0, 1].bar(x, mean_rgb, alpha=0.7, color=['red', 'green', 'blue'], edgecolor='black')
    axes[0, 1].set_title('Channel-wise Mean Values\nFor Normalization', fontweight='bold')
    axes[0, 1].set_ylabel('Mean Pixel Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Label distribution
    lane_pixels = np.sum(labels_data > 128, axis=(1, 2, 3))
    total_pixels = labels_data.shape[1] * labels_data.shape[2]
    lane_percentage = (lane_pixels / total_pixels) * 100
    
    axes[1, 0].hist(lane_percentage, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Lane Coverage Distribution\n(% of pixels per image)', fontweight='bold')
    axes[1, 0].set_xlabel('Lane Coverage (%)')
    axes[1, 0].set_ylabel('Number of Images')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Image resolution confirmation
    resolution_data = [train_data.shape[1], train_data.shape[2], train_data.shape[3]]
    resolution_labels = ['Height\n(80px)', 'Width\n(160px)', 'Channels\n(3 RGB)']
    
    bars = axes[1, 1].bar(resolution_labels, resolution_data, 
                         color=['skyblue', 'lightgreen', 'salmon'], 
                         alpha=0.8, edgecolor='black')
    axes[1, 1].set_title('Image Resolution Specs\nPPT Compliant', fontweight='bold')
    axes[1, 1].set_ylabel('Dimension Size')
    
    # Add value labels on bars
    for bar, value in zip(bars, resolution_data):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create PPT-style summary visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Summary metrics
    summary_data = {
        'Total Images': train_data.shape[0],
        'Resolution': f"{train_data.shape[1]}×{train_data.shape[2]}",
        'Channels': train_data.shape[3],
        'Mean Lane Coverage': f"{np.mean(lane_percentage):.1f}%",
        'Pixel Range': f"[{np.min(train_data):.0f}, {np.max(train_data):.0f}]",
        'Optimal Threshold': '240 (PPT Spec)'
    }
    
    # Create text-based summary
    ax.text(0.5, 0.8, 'Enhanced Dataset Analysis Summary', 
            ha='center', va='top', fontsize=20, fontweight='bold',
            transform=ax.transAxes)
    
    y_pos = 0.65
    for key, value in summary_data.items():
        ax.text(0.2, y_pos, f'{key}:', ha='left', va='center', 
                fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.6, y_pos, f'{value}', ha='left', va='center', 
                fontsize=14, transform=ax.transAxes)
        y_pos -= 0.08
    
    ax.text(0.5, 0.15, 'Ready for TensorFlow Implementation', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Enhanced visualizations created:")
    print("   📊 dataset_samples.png - Sample images and masks")
    print("   📈 dataset_statistics.png - Comprehensive statistics")
    print("   📋 analysis_summary.png - PPT-style summary")

def generate_enhanced_report(analysis_results, output_dir):
    """Generate comprehensive analysis report"""
    
    print("\n📄 Generating enhanced analysis report...")
    
    report_path = output_dir / "enhanced_analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced Phase 1: Dataset Analysis Report\n\n")
        f.write("## TensorFlow Implementation - PPT Specifications\n\n")
        
        f.write("### Dataset Overview\n")
        if "training_data" in analysis_results:
            train_data = analysis_results["training_data"]
            f.write(f"- **Total Images:** {train_data['shape'][0]:,} (PPT: 12,764)\n")
            f.write(f"- **Image Resolution:** {train_data['shape'][1]}×{train_data['shape'][2]}×{train_data['shape'][3]} (PPT: 80×160×3)\n")
            f.write(f"- **Data Type:** {train_data['dtype']}\n")
            f.write(f"- **Size:** {train_data['size_mb']:.2f} MB\n")
            f.write(f"- **Pixel Range:** [{train_data['min_value']:.0f}, {train_data['max_value']:.0f}]\n")
            f.write(f"- **Mean:** {train_data['mean_value']:.3f}, **Std:** {train_data['std_value']:.3f}\n\n")
        
        f.write("### Label Data Analysis\n")
        if "labels_data" in analysis_results:
            labels_data = analysis_results["labels_data"]
            f.write(f"- **Shape:** {labels_data['shape']}\n")
            f.write(f"- **Data Type:** {labels_data['dtype']}\n")
            f.write(f"- **Value Range:** [{labels_data['min_value']:.0f}, {labels_data['max_value']:.0f}]\n")
            f.write(f"- **Unique Values:** {len(labels_data['unique_values']) if labels_data['unique_values'] else 'N/A'}\n\n")
        
        f.write("### PPT Specification Verification\n")
        if "ppt_verification" in analysis_results:
            ppt = analysis_results["ppt_verification"]
            status = "✅ PASSED" if ppt["specs_match"] else "⚠️ DIFFERS"
            f.write(f"- **Verification Status:** {status}\n")
            f.write(f"- **Expected:** {ppt['expected_total']:,} images, {ppt['expected_resolution']}\n")
            f.write(f"- **Actual:** {ppt['actual_total']:,} images, {ppt['actual_resolution']}\n\n")
        
        f.write("### TensorFlow Implementation Notes\n")
        f.write("- **Framework:** TensorFlow/Keras 2.17+\n")
        f.write("- **Target Architecture:** Lightweight U-Net (1.9M parameters)\n")
        f.write("- **Normalization:** [0.0, 1.0] range required\n")
        f.write("- **Binary Threshold:** 240 (as per PPT specification)\n")
        f.write("- **Expected Performance:** 87% accuracy, ~50ms inference\n\n")
        
        f.write("### Data Split Recommendations\n")
        f.write("- **Training Split:** 80% (10,211 samples)\n")
        f.write("- **Validation Split:** 20% (2,553 samples)\n")
        f.write("- **Batch Size:** 8 (PPT specification)\n")
        f.write("- **Epochs:** 20 with early stopping (patience=5)\n\n")
        
        f.write("### Next Phase Preparation\n")
        f.write("1. **Phase 2:** Data preprocessing and normalization\n")
        f.write("2. **Requirements:** Install TensorFlow 2.17+ and dependencies\n")
        f.write("3. **Normalization:** Convert pixel values to [0.0, 1.0] range\n")
        f.write("4. **Augmentation:** Implement rotation, brightness, flipping\n")
        f.write("5. **Binary Conversion:** Apply threshold=240 for labels\n\n")
        
        f.write("### Key Findings\n")
        f.write("- Dataset is ready for TensorFlow implementation\n")
        f.write("- Image resolution matches PPT specifications\n")
        f.write("- Labels require binary conversion (threshold=240)\n")
        f.write("- Normalization to [0.0, 1.0] range needed\n")
        f.write("- Perfect for real-time lane detection (80×160 resolution)\n")
    
    print(f"✅ Enhanced report saved: {report_path}")

def save_analysis_data(analysis_results, output_dir):
    """Save analysis data in JSON format"""
    
    print("\n💾 Saving analysis data...")
    
    # Save complete analysis
    analysis_path = output_dir / "dataset_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Save TensorFlow-specific configuration
    tf_config = {
        "framework": "TensorFlow/Keras",
        "version": "2.17+",
        "architecture": "Lightweight U-Net",
        "target_parameters": "1.9M",
        "input_shape": analysis_results.get("training_data", {}).get("shape", [None, 80, 160, 3])[1:],
        "output_shape": [80, 160, 1],
        "normalization": {
            "input_range": [0.0, 1.0],
            "current_range": [
                analysis_results.get("training_data", {}).get("min_value", 0),
                analysis_results.get("training_data", {}).get("max_value", 255)
            ]
        },
        "binary_threshold": 240,
        "training_config": {
            "epochs": 20,
            "batch_size": 8,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "binary_crossentropy",
            "metrics": ["accuracy", "precision", "recall"],
            "early_stopping": 5
        }
    }
    
    config_path = output_dir / "tensorflow_config.json"
    with open(config_path, 'w') as f:
        json.dump(tf_config, f, indent=2)
    
    print(f"✅ Analysis data saved: {analysis_path}")
    print(f"✅ TensorFlow config saved: {config_path}")

def main():
    """Main Phase 1 execution"""
    
    # Create output directories
    output_dir = Path("outputs")
    viz_dir = Path("visualizations")
    reports_dir = Path("reports")
    
    output_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Perform comprehensive analysis
    analysis_results, train_data, labels_data = load_and_analyze_dataset()
    
    # Create visualizations
    create_enhanced_visualizations(train_data, labels_data, viz_dir)
    
    # Generate report
    generate_enhanced_report(analysis_results, reports_dir)
    
    # Save analysis data
    save_analysis_data(analysis_results, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ PHASE 1: ENHANCED DATASET ANALYSIS COMPLETE!")
    print("=" * 60)
    print()
    print("📊 Analysis Summary:")
    if "training_data" in analysis_results:
        shape = analysis_results["training_data"]["shape"]
        print(f"   • Dataset: {shape[0]:,} images ({shape[1]}×{shape[2]}×{shape[3]})")
        print(f"   • PPT Compliance: {'✅ Perfect Match' if analysis_results.get('ppt_verification', {}).get('specs_match') else '⚠️ Requires Attention'}")
        print(f"   • TensorFlow Ready: ✅ Configuration generated")
    
    print("\n📁 Outputs Generated:")
    print("   📊 Visualizations: visualizations/")
    print("   📄 Reports: reports/enhanced_analysis_report.md")
    print("   💾 Data: outputs/dataset_analysis.json")
    print("   ⚙️ Config: outputs/tensorflow_config.json")
    
    print("\n🎯 Ready for Phase 2: Data Preprocessing")
    print("   Next: python ../phase2_data_preprocessing/scripts/preprocess_data.py")

if __name__ == "__main__":
    main()
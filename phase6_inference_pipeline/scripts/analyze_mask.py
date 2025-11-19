#!/usr/bin/env python3
"""
Quick script to analyze and display the mask image
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_mask_image(mask_path):
    """Analyze and display information about the mask image"""
    
    # Load the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"❌ Could not load image: {mask_path}")
        return
    
    print(f"🔍 Analyzing mask image: {Path(mask_path).name}")
    print(f"📏 Image dimensions: {mask.shape[1]} x {mask.shape[0]} pixels")
    print(f"📊 Data type: {mask.dtype}")
    print(f"📈 Value range: {mask.min()} to {mask.max()}")
    
    # Count lane pixels (white pixels)
    lane_pixels = np.sum(mask > 127)  # Assuming lanes are white (>127)
    total_pixels = mask.shape[0] * mask.shape[1]
    lane_percentage = (lane_pixels / total_pixels) * 100
    
    print(f"🛣️  Lane pixels: {lane_pixels:,} ({lane_percentage:.2f}% of image)")
    print(f"⚫ Background pixels: {total_pixels - lane_pixels:,}")
    
    # Unique values in the mask
    unique_values = np.unique(mask)
    print(f"🎨 Unique pixel values: {unique_values}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Original mask
    plt.subplot(2, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Lane Detection Mask')
    plt.axis('off')
    
    # Histogram of pixel values
    plt.subplot(2, 2, 2)
    plt.hist(mask.flatten(), bins=50, alpha=0.7, color='blue')
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Binary visualization (lanes highlighted)
    plt.subplot(2, 2, 3)
    binary_mask = (mask > 127).astype(np.uint8) * 255
    plt.imshow(binary_mask, cmap='RdYlBu_r')
    plt.title('Binary Lane Mask (Lanes in Red)')
    plt.axis('off')
    
    # Lane density heatmap
    plt.subplot(2, 2, 4)
    plt.imshow(mask, cmap='hot')
    plt.title('Lane Confidence Heatmap')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save analysis
    output_path = Path(mask_path).parent / f"analysis_{Path(mask_path).name}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Analysis saved to: {output_path}")
    
    plt.show()
    
    return {
        'dimensions': mask.shape,
        'lane_pixels': int(lane_pixels),
        'total_pixels': int(total_pixels),
        'lane_percentage': float(lane_percentage),
        'value_range': (int(mask.min()), int(mask.max())),
        'unique_values': unique_values.tolist()
    }

if __name__ == "__main__":
    mask_path = "/Users/nikhilsireesh/Lane detection and Assistance system using CNN/phase6_inference_pipeline/inference_results/masks/demo_inference_test_image_001_mask.png"
    
    print("🔍 MASK IMAGE ANALYSIS")
    print("=" * 50)
    
    result = analyze_mask_image(mask_path)
    
    if result:
        print("\n📋 SUMMARY:")
        print(f"   🖼️  Image: {result['dimensions'][1]}x{result['dimensions'][0]} pixels")
        print(f"   🛣️  Lane coverage: {result['lane_percentage']:.1f}%")
        print(f"   📊 Value range: {result['value_range'][0]}-{result['value_range'][1]}")
        print(f"   🎨 Unique values: {len(result['unique_values'])} different intensities")
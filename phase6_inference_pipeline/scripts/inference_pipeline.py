#!/usr/bin/env python3
"""
🚀 PHASE 6: INFERENCE PIPELINE
=================================
Comprehensive inference pipeline for lane detection model.
Supports single images, batch processing, and real-time inference.
"""

import os
import sys
import time
import pickle
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LaneDetectionInference:
    """
    Advanced inference pipeline for lane detection with multiple processing modes.
    """
    
    def __init__(self, model_path: str, config: dict = None):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to the trained model
            config: Configuration dictionary for inference settings
        """
        self.model_path = Path(model_path)
        self.config = config or self._default_config()
        self.model = None
        self.input_shape = None
        self.output_shape = None
        
        # Performance tracking
        self.inference_times = []
        self.processing_stats = {
            'total_images': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'avg_inference_time': 0.0
        }
        
        # Load model
        self._load_model()
        
        # Setup output directories
        self._setup_directories()
        
    def _default_config(self) -> dict:
        """Default configuration for inference pipeline."""
        return {
            'input_size': (80, 160),
            'batch_size': 16,
            'confidence_threshold': 0.5,
            'post_processing': True,
            'save_visualizations': True,
            'output_format': 'both',  # 'mask', 'overlay', 'both'
            'timing_analysis': True
        }
    
    def _setup_directories(self):
        """Setup output directories for inference results."""
        base_dir = Path(__file__).parent.parent
        
        self.output_dirs = {
            'results': base_dir / 'inference_results',
            'visualizations': base_dir / 'inference_results' / 'visualizations',
            'masks': base_dir / 'inference_results' / 'masks',
            'overlays': base_dir / 'inference_results' / 'overlays',
            'reports': base_dir / 'inference_results' / 'reports'
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Load the trained model."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Custom objects for model loading
            custom_objects = {
                'iou': self._iou_metric
            }
            
            self.model = tf.keras.models.load_model(
                self.model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            
            # Get model information
            self.input_shape = self.model.input_shape
            self.output_shape = self.model.output_shape
            
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   Input shape: {self.input_shape}")
            logger.info(f"   Output shape: {self.output_shape}")
            logger.info(f"   Parameters: {self.model.count_params():,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _iou_metric(self, y_true, y_pred):
        """IoU metric for model loading."""
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(tf.cast(tf.logical_or(y_true > 0, y_pred > 0), tf.float32), axis=[1, 2, 3])
        iou = tf.reduce_mean(intersection / (union + 1e-7))
        return iou
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess single image for inference.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image ready for model input
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            target_size = self.config['input_size']
            image_resized = cv2.resize(image, (target_size[1], target_size[0]))
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            return image_normalized
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def postprocess_prediction(self, prediction: np.ndarray, 
                             original_shape: Tuple[int, int] = None) -> np.ndarray:
        """
        Postprocess model prediction.
        
        Args:
            prediction: Raw model output
            original_shape: Original image shape for resizing back
            
        Returns:
            Postprocessed prediction mask
        """
        try:
            # Apply confidence threshold
            mask = (prediction > self.config['confidence_threshold']).astype(np.uint8)
            
            # Resize back to original size if provided
            if original_shape is not None:
                mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
            
            # Apply morphological operations if enabled
            if self.config['post_processing']:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            return prediction
    
    def infer_single_image(self, image_path: str) -> dict:
        """
        Perform inference on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing inference results
        """
        try:
            # Load and preprocess image
            start_time = time.time()
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            original_shape = image.shape[:2]
            preprocessed = self.preprocess_image(image)
            
            # Add batch dimension
            input_batch = np.expand_dims(preprocessed, axis=0)
            
            # Perform inference
            inference_start = time.time()
            prediction = self.model.predict(input_batch, verbose=0)
            inference_time = time.time() - inference_start
            
            # Postprocess
            mask = self.postprocess_prediction(prediction[0, :, :, 0], original_shape)
            
            total_time = time.time() - start_time
            
            # Update statistics
            self.inference_times.append(inference_time)
            self.processing_stats['total_images'] += 1
            self.processing_stats['successful_inferences'] += 1
            
            # Create results dictionary
            results = {
                'image_path': image_path,
                'original_shape': original_shape,
                'inference_time': inference_time,
                'total_processing_time': total_time,
                'mask': mask,
                'raw_prediction': prediction[0, :, :, 0],
                'confidence_scores': {
                    'mean': float(np.mean(prediction)),
                    'max': float(np.max(prediction)),
                    'min': float(np.min(prediction))
                }
            }
            
            logger.info(f"✅ Inference completed: {Path(image_path).name} ({inference_time*1000:.1f}ms)")
            return results
            
        except Exception as e:
            logger.error(f"❌ Inference failed for {image_path}: {e}")
            self.processing_stats['failed_inferences'] += 1
            return None
    
    def infer_batch(self, image_paths: List[str]) -> List[dict]:
        """
        Perform batch inference on multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of inference results
        """
        logger.info(f"🔄 Starting batch inference on {len(image_paths)} images...")
        
        results = []
        batch_size = self.config['batch_size']
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            original_shapes = []
            
            # Load and preprocess batch
            for path in batch_paths:
                try:
                    image = cv2.imread(path)
                    if image is not None:
                        original_shapes.append(image.shape[:2])
                        preprocessed = self.preprocess_image(image)
                        batch_images.append(preprocessed)
                        valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"⚠️ Skipping {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Perform batch inference
            try:
                batch_array = np.array(batch_images)
                start_time = time.time()
                predictions = self.model.predict(batch_array, verbose=0)
                inference_time = time.time() - start_time
                
                # Process each prediction
                for j, (path, pred, orig_shape) in enumerate(zip(valid_paths, predictions, original_shapes)):
                    mask = self.postprocess_prediction(pred[:, :, 0], orig_shape)
                    
                    result = {
                        'image_path': path,
                        'original_shape': orig_shape,
                        'inference_time': inference_time / len(batch_images),
                        'mask': mask,
                        'raw_prediction': pred[:, :, 0],
                        'batch_index': j,
                        'confidence_scores': {
                            'mean': float(np.mean(pred)),
                            'max': float(np.max(pred)),
                            'min': float(np.min(pred))
                        }
                    }
                    results.append(result)
                    self.processing_stats['successful_inferences'] += 1
                
                self.processing_stats['total_images'] += len(valid_paths)
                avg_time = inference_time / len(batch_images) * 1000
                logger.info(f"✅ Batch {i//batch_size + 1} completed: {len(valid_paths)} images ({avg_time:.1f}ms/image)")
                
            except Exception as e:
                logger.error(f"❌ Batch inference failed: {e}")
                self.processing_stats['failed_inferences'] += len(batch_images)
        
        logger.info(f"🎯 Batch inference completed: {len(results)} successful inferences")
        return results
    
    def create_visualization(self, image_path: str, mask: np.ndarray, 
                           save_path: str = None) -> np.ndarray:
        """
        Create visualization overlay of lane detection results.
        
        Args:
            image_path: Path to original image
            mask: Predicted lane mask
            save_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        try:
            # Load original image
            original = cv2.imread(image_path)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Create colored overlay
            overlay = original_rgb.copy()
            
            # Apply lane mask (green color for lanes)
            lane_pixels = mask > 0
            overlay[lane_pixels] = [0, 255, 0]  # Green for lanes
            
            # Blend with original image
            alpha = 0.3
            visualization = cv2.addWeighted(original_rgb, 1-alpha, overlay, alpha, 0)
            
            # Save if path provided
            if save_path:
                visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, visualization_bgr)
            
            return visualization
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
    
    def save_results(self, results: List[dict], output_prefix: str = "inference"):
        """
        Save inference results to files.
        
        Args:
            results: List of inference results
            output_prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save visualizations and masks
            for i, result in enumerate(results):
                if result is None:
                    continue
                
                image_name = Path(result['image_path']).stem
                
                # Save mask
                if self.config['output_format'] in ['mask', 'both']:
                    mask_path = self.output_dirs['masks'] / f"{output_prefix}_{image_name}_mask.png"
                    cv2.imwrite(str(mask_path), result['mask'] * 255)
                
                # Save overlay visualization
                if self.config['output_format'] in ['overlay', 'both']:
                    overlay_path = self.output_dirs['overlays'] / f"{output_prefix}_{image_name}_overlay.png"
                    self.create_visualization(result['image_path'], result['mask'], str(overlay_path))
            
            # Save performance report
            self._save_performance_report(results, output_prefix, timestamp)
            
            # Save detailed results as JSON
            json_results = []
            for result in results:
                if result is not None:
                    json_result = {
                        'image_path': result['image_path'],
                        'original_shape': result['original_shape'],
                        'inference_time': result['inference_time'],
                        'confidence_scores': result['confidence_scores']
                    }
                    json_results.append(json_result)
            
            json_path = self.output_dirs['reports'] / f"{output_prefix}_results_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"✅ Results saved to: {self.output_dirs['results']}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
    
    def _save_performance_report(self, results: List[dict], prefix: str, timestamp: str):
        """Save detailed performance report."""
        try:
            # Calculate statistics
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return
            
            inference_times = [r['inference_time'] for r in valid_results]
            
            stats = {
                'total_images': len(results),
                'successful_inferences': len(valid_results),
                'failed_inferences': len(results) - len(valid_results),
                'success_rate': len(valid_results) / len(results) * 100,
                'inference_times': {
                    'mean': np.mean(inference_times),
                    'median': np.median(inference_times),
                    'min': np.min(inference_times),
                    'max': np.max(inference_times),
                    'std': np.std(inference_times)
                },
                'throughput': {
                    'images_per_second': 1.0 / np.mean(inference_times),
                    'ms_per_image': np.mean(inference_times) * 1000
                }
            }
            
            # Create performance report
            report_content = f"""# Lane Detection Inference Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- **Total Images:** {stats['total_images']}
- **Successful Inferences:** {stats['successful_inferences']}
- **Success Rate:** {stats['success_rate']:.1f}%

## Performance Metrics
- **Average Inference Time:** {stats['inference_times']['mean']*1000:.1f}ms
- **Median Inference Time:** {stats['inference_times']['median']*1000:.1f}ms
- **Throughput:** {stats['throughput']['images_per_second']:.1f} images/second

## Detailed Statistics
- **Min Time:** {stats['inference_times']['min']*1000:.1f}ms
- **Max Time:** {stats['inference_times']['max']*1000:.1f}ms
- **Standard Deviation:** {stats['inference_times']['std']*1000:.1f}ms

## Model Information
- **Input Shape:** {self.input_shape}
- **Output Shape:** {self.output_shape}
- **Parameters:** {self.model.count_params():,}
"""
            
            report_path = self.output_dirs['reports'] / f"{prefix}_performance_{timestamp}.md"
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            # Save JSON stats
            json_path = self.output_dirs['reports'] / f"{prefix}_stats_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")


def main():
    """Main inference pipeline execution."""
    print("\n" + "="*50)
    print("🚀 PHASE 6: INFERENCE PIPELINE")
    print("="*50)
    print("Comprehensive inference system for lane detection")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "phase4_model_training" / "scripts" / "models" / "quick_trained_model.keras"
    dataset_path = base_dir / "Dataset"
    
    try:
        # Initialize inference pipeline
        print(f"\n📂 Initializing inference pipeline...")
        pipeline = LaneDetectionInference(str(model_path))
        
        # Load test dataset for demonstration
        print(f"\n📂 Loading test dataset...")
        with open(dataset_path / "train_dataset.p", "rb") as f:
            data = pickle.load(f)
        
        # Use a subset of test images for demonstration
        # Handle different dataset formats
        if isinstance(data, dict):
            X_test = data['features'][:50] if 'features' in data else data['images'][:50]
            y_test = data['labels'][:50] if 'labels' in data else data['masks'][:50]
        elif isinstance(data, list) and len(data) >= 2:
            X_test = data[0][:50]  # Images
            y_test = data[1][:50]  # Labels
        else:
            # Assume it's a single array of images
            X_test = data[:50]
            y_test = None
        
        print(f"✅ Test data loaded: {len(X_test)} images")
        
        # Save test images temporarily for inference
        temp_dir = Path(__file__).parent / "temp_images"
        temp_dir.mkdir(exist_ok=True)
        
        image_paths = []
        for i in range(min(10, len(X_test))):  # Process first 10 images
            image_path = temp_dir / f"test_image_{i:03d}.png"
            
            # Convert from normalized to uint8
            image_uint8 = (X_test[i] * 255).astype(np.uint8)
            cv2.imwrite(str(image_path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
            image_paths.append(str(image_path))
        
        print(f"✅ Prepared {len(image_paths)} test images")
        
        # Demonstrate single image inference
        print(f"\n🔍 Testing single image inference...")
        single_result = pipeline.infer_single_image(image_paths[0])
        if single_result:
            print(f"   ✅ Single inference: {single_result['inference_time']*1000:.1f}ms")
        
        # Demonstrate batch inference
        print(f"\n🔄 Testing batch inference...")
        batch_results = pipeline.infer_batch(image_paths)
        
        # Save results
        print(f"\n💾 Saving inference results...")
        pipeline.save_results(batch_results, "demo_inference")
        
        # Performance summary
        if batch_results:
            avg_time = np.mean([r['inference_time'] for r in batch_results if r]) * 1000
            print(f"\n📊 Performance Summary:")
            print(f"   • Processed: {len(batch_results)} images")
            print(f"   • Average time: {avg_time:.1f}ms per image")
            print(f"   • Throughput: {1000/avg_time:.1f} images/second")
        
        # Cleanup temp files
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        print(f"\n" + "="*50)
        print("✅ PHASE 6: INFERENCE PIPELINE COMPLETE!")
        print("="*50)
        print(f"\n🎯 Results saved to:")
        print(f"   📊 Visualizations: phase6_inference_pipeline/inference_results/")
        print(f"   📄 Reports: phase6_inference_pipeline/inference_results/reports/")
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
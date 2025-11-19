#!/usr/bin/env python3
"""
🚀 PHASE 7: REAL-TIME PROCESSING WEB APPLICATION - DEMO VERSION
===============================================================
Simplified version for testing web interface without model loading
"""

from flask import Flask, render_template, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64
import io
import os

# Set environment variable to handle macOS camera authorization
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
import time
from pathlib import Path
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the correct template directory
template_dir = Path(__file__).parent.parent / 'templates'
app = Flask(__name__, template_folder=str(template_dir))
app.config['SECRET_KEY'] = 'lane_detection_secret_key'

# Set absolute path for uploads folder (in the scripts directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(script_dir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def validate_video_file(video_path):
    """Validate if a video file is properly created and readable"""
    try:
        if not os.path.exists(video_path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(video_path)
        if file_size < 1000:  # Less than 1KB is likely corrupted
            return False, f"File too small ({file_size} bytes) - likely corrupted"
        
        # Try to open with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return False, "Cannot open video with OpenCV"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if frame_count == 0:
            return False, "Video has no frames"
        
        if fps == 0 or width == 0 or height == 0:
            return False, "Invalid video properties"
        
        return True, f"Valid video: {frame_count} frames, {fps} FPS, {width}x{height}"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"

class LaneDetectionWebDemo:
    """Demo version of lane detection web application"""
    
    def __init__(self):
        self.model_loaded = False  # Demo mode - no actual model
        self.camera = None
        self.camera_active = False
        self.processing_stats = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'last_processing_time': 0.0
        }
        
        # Simulate model loading after a delay
        self.simulate_model_loading()
    
    def simulate_model_loading(self):
        """Load the actual model"""
        def load_later():
            time.sleep(2)  # Brief delay for UI responsiveness
            try:
                self.actual_model = self.load_actual_model()
                self.model_loaded = True
                if self.actual_model is not None:
                    logger.info("✅ Trained model loaded successfully!")
                else:
                    logger.info("✅ Demo mode: Using fallback simulation")
            except Exception as e:
                logger.error(f"Model loading error: {e}")
                self.model_loaded = True  # Still mark as loaded for demo fallback
        
        import threading
        threading.Thread(target=load_later, daemon=True).start()
    
    def preprocess_for_model(self, image):
        """Preprocess image for the trained model"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Resize to model input size (80x160)
            image_resized = cv2.resize(image_rgb, (160, 80))
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            return image_normalized
        except Exception as e:
            logger.error(f"Model preprocessing error: {e}")
            return None
    
    def load_actual_model(self):
        """Load the actual trained model"""
        try:
            # Import TensorFlow
            import tensorflow as tf
            
            # Path to the trained model
            base_dir = Path(__file__).parent.parent.parent
            model_path = base_dir / "phase4_model_training" / "scripts" / "models" / "quick_trained_model.keras"
            
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}, using demo mode")
                return None
            
            # Custom objects for model loading
            def iou_metric(y_true, y_pred):
                y_pred = tf.cast(y_pred > 0.5, tf.float32)
                intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
                union = tf.reduce_sum(tf.cast(tf.logical_or(y_true > 0, y_pred > 0), tf.float32), axis=[1, 2, 3])
                iou = tf.reduce_mean(intersection / (union + 1e-7))
                return iou
            
            custom_objects = {'iou': iou_metric}
            
            model = tf.keras.models.load_model(
                str(model_path), 
                custom_objects=custom_objects,
                compile=False
            )
            
            logger.info("✅ Actual model loaded successfully!")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load actual model: {e}")
            return None
    
    def predict_with_model(self, image):
        """Use the actual trained model for prediction"""
        try:
            # Try to load model if not already loaded
            if not hasattr(self, 'actual_model'):
                self.actual_model = self.load_actual_model()
            
            if self.actual_model is None:
                return self.create_demo_mask(image)
            
            # Preprocess image
            preprocessed = self.preprocess_for_model(image)
            if preprocessed is None:
                return self.create_demo_mask(image)
            
            # Add batch dimension
            input_batch = np.expand_dims(preprocessed, axis=0)
            
            # Predict
            prediction = self.actual_model.predict(input_batch, verbose=0)
            
            # Get mask from prediction
            mask_small = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
            
            # Resize mask back to original image size
            original_height, original_width = image.shape[:2]
            mask = cv2.resize(mask_small, (original_width, original_height))
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return self.create_demo_mask(image)
    
    def create_demo_mask(self, image):
        """Create a demo lane mask for testing (fallback)"""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create fake lane lines as fallback
        # Left lane
        cv2.line(mask, (int(width*0.3), height), (int(width*0.4), int(height*0.6)), 255, 8)
        # Right lane
        cv2.line(mask, (int(width*0.6), int(height*0.6)), (int(width*0.7), height), 255, 8)
        
        # Add some curve
        points = np.array([
            [int(width*0.35), int(height*0.8)],
            [int(width*0.45), int(height*0.7)],
            [int(width*0.55), int(height*0.7)],
            [int(width*0.65), int(height*0.8)]
        ], np.int32)
        
        cv2.polylines(mask, [points], False, 255, 6)
        
        return mask
    
    def process_image(self, image):
        """Process single image and return demo results"""
        if not self.model_loaded:
            return None, "Model still loading, please wait..."
        
        start_time = time.time()
        
        try:
            # Use actual model for prediction
            mask = self.predict_with_model(image)
            
            # Create overlay
            overlay = self.create_overlay(image, mask)
            
            # Simulate processing time
            time.sleep(0.05)  # 50ms simulation
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.processing_stats['total_processed'] += 1
            self.processing_stats['last_processing_time'] = processing_time
            
            # Calculate rolling average
            total = self.processing_stats['total_processed']
            current_avg = self.processing_stats['avg_processing_time']
            self.processing_stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total
            
            lane_coverage = float(np.sum(mask > 127) / (mask.shape[0] * mask.shape[1]) * 100)
            
            # Add lane assistance analysis
            position_status = assistance_system.analyze_lane_position(lane_coverage)
            alert = assistance_system.generate_alert(position_status, lane_coverage)
            safety_metrics = assistance_system.get_safety_metrics()
            
            # Add video alerts overlay to the image
            overlay_with_alerts = self.add_video_alerts_overlay(overlay, alert, lane_coverage, safety_metrics)
            
            results = {
                'mask': mask,
                'overlay': overlay_with_alerts,
                'processing_time': processing_time,
                'confidence_mean': 0.85,  # Demo confidence
                'lane_coverage': lane_coverage,
                'assistance': {
                    'position_status': position_status,
                    'alert': alert,
                    'safety_metrics': safety_metrics
                }
            }
            
            return results, None
            
        except Exception as e:
            error_msg = f"Processing failed: {e}"
            logger.error(error_msg)
            return None, error_msg
    
    def add_video_alerts_overlay(self, image, alert, lane_coverage, safety_metrics):
        """Add professional ADAS-style alert overlays to video frame"""
        try:
            # Convert to BGR for OpenCV operations
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume input is RGB, convert to BGR for OpenCV
                overlay_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                overlay_image = image.copy()
            
            height, width = overlay_image.shape[:2]
            
            # Alert colors mapping
            color_map = {
                'critical': (0, 0, 255),      # Red
                'warning': (0, 140, 255),     # Orange
                'safe': (0, 170, 0),          # Green
                'excellent': (255, 102, 0)    # Blue
            }
            
            alert_color = color_map.get(alert['level'], (0, 170, 0))
            
            # 1. Top Alert Banner
            banner_height = 60
            banner_bg = alert_color + (180,)  # Semi-transparent
            cv2.rectangle(overlay_image, (0, 0), (width, banner_height), alert_color, -1)
            cv2.rectangle(overlay_image, (0, 0), (width, banner_height), (255, 255, 255), 2)
            
            # Alert message
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            thickness = 2
            
            # Center the alert message
            text_size = cv2.getTextSize(alert['message'], font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = 35
            
            cv2.putText(overlay_image, alert['message'], (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # 2. Bottom HUD Panel
            hud_height = 80
            hud_y = height - hud_height
            
            # Semi-transparent background
            hud_overlay = overlay_image.copy()
            cv2.rectangle(hud_overlay, (0, hud_y), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay_image, 0.7, hud_overlay, 0.3, 0, overlay_image)
            
            # HUD border
            cv2.rectangle(overlay_image, (0, hud_y), (width, height), (100, 100, 100), 2)
            
            # HUD Information
            hud_font = cv2.FONT_HERSHEY_SIMPLEX
            hud_font_scale = 0.6
            hud_thickness = 2
            hud_color = (255, 255, 255)
            
            # Left side - Safety metrics
            y_offset = hud_y + 25
            cv2.putText(overlay_image, f"Safety Score: {safety_metrics['safety_score']}", 
                       (15, y_offset), hud_font, hud_font_scale, hud_color, hud_thickness)
            
            cv2.putText(overlay_image, f"Lane Coverage: {lane_coverage:.1f}%", 
                       (15, y_offset + 25), hud_font, hud_font_scale, hud_color, hud_thickness)
            
            # Right side - Alert info
            cv2.putText(overlay_image, f"Alerts: {safety_metrics['total_alerts']}", 
                       (width - 200, y_offset), hud_font, hud_font_scale, hud_color, hud_thickness)
            
            cv2.putText(overlay_image, f"Critical: {safety_metrics['critical_alerts']}", 
                       (width - 200, y_offset + 25), hud_font, hud_font_scale, hud_color, hud_thickness)
            
            # Center - Current time
            current_time = datetime.now().strftime("%H:%M:%S")
            time_size = cv2.getTextSize(current_time, hud_font, hud_font_scale, hud_thickness)[0]
            time_x = (width - time_size[0]) // 2
            cv2.putText(overlay_image, current_time, (time_x, y_offset + 12), 
                       hud_font, hud_font_scale, hud_color, hud_thickness)
            
            # 3. Side Warning Indicators for Critical Alerts
            if alert['level'] == 'critical':
                # Flashing border effect
                border_color = (0, 0, 255)  # Red
                border_thickness = 8
                
                # Top and bottom borders
                cv2.rectangle(overlay_image, (0, banner_height), (width, banner_height + border_thickness), 
                             border_color, -1)
                cv2.rectangle(overlay_image, (0, hud_y - border_thickness), (width, hud_y), 
                             border_color, -1)
                
                # Side borders
                cv2.rectangle(overlay_image, (0, banner_height), (border_thickness, hud_y), 
                             border_color, -1)
                cv2.rectangle(overlay_image, (width - border_thickness, banner_height), (width, hud_y), 
                             border_color, -1)
            
            # 4. Lane Position Indicator
            indicator_width = 200
            indicator_height = 20
            indicator_x = (width - indicator_width) // 2
            indicator_y = banner_height + 20
            
            # Background bar
            cv2.rectangle(overlay_image, (indicator_x, indicator_y), 
                         (indicator_x + indicator_width, indicator_y + indicator_height), 
                         (50, 50, 50), -1)
            
            # Coverage bar
            coverage_width = int((lane_coverage / 100.0) * indicator_width)
            coverage_color = alert_color
            cv2.rectangle(overlay_image, (indicator_x, indicator_y), 
                         (indicator_x + coverage_width, indicator_y + indicator_height), 
                         coverage_color, -1)
            
            # Border
            cv2.rectangle(overlay_image, (indicator_x, indicator_y), 
                         (indicator_x + indicator_width, indicator_y + indicator_height), 
                         (255, 255, 255), 1)
            
            # Convert back to RGB for return
            return cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            logger.error(f"Error adding video alerts overlay: {e}")
            return image  # Return original image if overlay fails
    
    def create_overlay(self, image, mask):
        """Create lane overlay on original image"""
        try:
            # Convert image to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image.copy()
            
            # Create overlay
            overlay = image_rgb.copy()
            lane_pixels = mask > 127
            overlay[lane_pixels] = [0, 255, 0]  # Green for lanes
            
            # Blend with original
            result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
            
            return result
        except Exception as e:
            logger.error(f"Overlay creation error: {e}")
            return image
    
    def get_camera_frame(self):
        """Get frame from camera"""
        if self.camera is None or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def start_camera(self, camera_id=0):
        """Start camera capture"""
        try:
            # Try to open camera with proper error handling
            self.camera = cv2.VideoCapture(camera_id)
            
            # Give camera time to initialize
            time.sleep(0.5)
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test if camera is actually working by trying to read a frame
            if self.camera.isOpened():
                ret, test_frame = self.camera.read()
                if ret and test_frame is not None:
                    self.camera_active = True
                    logger.info("✅ Camera started successfully")
                    return True
                else:
                    self.camera.release()
                    self.camera = None
                    if not hasattr(self, '_camera_error_logged'):
                        logger.info("📷 Camera disabled - using file upload mode only")
                        self._camera_error_logged = True
                    return False
            else:
                self.camera.release()
                self.camera = None
                if not hasattr(self, '_camera_error_logged'):
                    logger.info("📷 Camera disabled - using file upload mode only")
                    self._camera_error_logged = True
                return False
        except Exception as e:
            if self.camera:
                self.camera.release()
                self.camera = None
            if not hasattr(self, '_camera_error_logged'):
                logger.info("📷 Camera disabled - file upload and video processing available")
                self._camera_error_logged = True
            return False
    
    def stop_camera(self):
        """Stop camera processing"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.camera_active = False
        logger.info("📷 Camera stopped")
    
    def _create_web_optimized_video(self, input_path, output_path):
        """Create a web-optimized version of the video for better browser compatibility"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Cannot open input video for web optimization")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use H.264 codec for best browser compatibility
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'H264'),  # H.264 - best browser support
            cv2.VideoWriter_fourcc(*'avc1'),  # Alternative H.264
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 fallback
        ]
        
        out = None
        for fourcc in fourcc_options:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                logger.info(f"Web optimization using fourcc: {fourcc}")
                break
            else:
                out.release()
        
        if not out.isOpened():
            cap.release()
            raise Exception("Cannot create web-optimized video writer")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ensure frame is in correct format
            if frame.dtype != 'uint8':
                frame = frame.astype('uint8')
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                out.write(frame)
                frame_count += 1
        
        cap.release()
        out.release()
        
        web_size = os.path.getsize(output_path)
        logger.info(f"Web-optimized video created: {frame_count} frames, {web_size} bytes")
        return True
    
    def process_video(self, video_path):
        """Process video file with lane detection"""
        try:
            start_time = time.time()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, "Could not open video file"
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video with better error handling
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"processed_{timestamp}.mp4"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Use a simple, widely supported codec approach
            # Try codecs in order of reliability on macOS
            fourcc_options = [
                ('mp4v', '.mp4'),  # MPEG-4 - most reliable
                ('MJPG', '.avi'),  # Motion JPEG - fallback
                ('XVID', '.avi'),  # Xvid - alternative
                (-1, '.mp4')       # Default codec
            ]
            
            out = None
            final_output_path = output_path
            
            for codec, ext in fourcc_options:
                try:
                    test_output_path = output_path.replace('.mp4', ext) if ext != '.mp4' else output_path
                    
                    if codec == -1:
                        # Try default codec
                        out = cv2.VideoWriter(test_output_path, -1, fps, (width, height))
                        codec_name = "default"
                    else:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        out = cv2.VideoWriter(test_output_path, fourcc, fps, (width, height))
                        codec_name = codec
                    
                    # Simple test - just check if VideoWriter opened successfully
                    if out.isOpened():
                        logger.info(f"Successfully created VideoWriter with codec: {codec_name}")
                        final_output_path = test_output_path
                        break
                    else:
                        logger.warning(f"Failed to open VideoWriter with codec: {codec_name}")
                        if out:
                            out.release()
                            out = None
                        
                except Exception as e:
                    logger.warning(f"Exception with codec {codec}: {e}")
                    if out:
                        out.release()
                        out = None
                    continue
            
            # Last resort: try uncompressed video
            if out is None or not out.isOpened():
                try:
                    logger.warning("Attempting uncompressed video as last resort...")
                    uncompressed_path = output_path.replace('.mp4', '_uncompressed.avi')
                    # Use uncompressed format
                    out = cv2.VideoWriter(uncompressed_path, 0, fps, (width, height))
                    if out.isOpened():
                        logger.info("Using uncompressed video format")
                        final_output_path = uncompressed_path
                        output_filename = os.path.basename(final_output_path)
                        output_path = final_output_path
                    else:
                        cap.release()
                        return None, f"Could not create output video with any codec including uncompressed. OpenCV version: {cv2.__version__}"
                except Exception as e:
                    cap.release()
                    return None, f"All video creation methods failed. Error: {e}"
            
            # Update the output filename to match the actual file created
            output_filename = os.path.basename(final_output_path)
            output_path = final_output_path
            
            frames_processed = 0
            lane_coverages = []
            
            logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
            logger.info(f"Output file: {output_path}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with lane detection
                results, error = self.process_image(frame)
                
                if results is not None:
                    # Use overlay frame with video alerts and ensure proper format
                    processed_frame = cv2.cvtColor(results['overlay'], cv2.COLOR_RGB2BGR)
                    lane_coverages.append(results['lane_coverage'])
                else:
                    # Use original frame if processing fails
                    processed_frame = frame.copy()
                    lane_coverages.append(0.0)
                
                # Ensure frame has correct dimensions and type
                if processed_frame.shape[:2] != (height, width):
                    processed_frame = cv2.resize(processed_frame, (width, height))
                
                # Ensure frame is in correct format (uint8) and has 3 channels
                if processed_frame.dtype != np.uint8:
                    processed_frame = processed_frame.astype(np.uint8)
                
                if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
                    # Frame is already BGR, good to go
                    pass
                elif len(processed_frame.shape) == 3 and processed_frame.shape[2] == 4:
                    # Convert BGRA to BGR
                    processed_frame = processed_frame[:, :, :3]
                elif len(processed_frame.shape) == 2:
                    # Convert grayscale to BGR
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                
                # Write frame to output video
                try:
                    out.write(processed_frame)
                except Exception as e:
                    logger.warning(f"Exception writing frame {frames_processed}: {e}")
                
                frames_processed += 1
                
                # Log progress every 30 frames
                if frames_processed % 30 == 0:
                    progress = (frames_processed / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frames_processed}/{total_frames})")
            
            # Cleanup
            cap.release()
            out.release()
            
            # Verify file was created
            if not os.path.exists(output_path):
                return None, f"Output video file was not created: {output_path}"
            
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                return None, f"Output video file is empty: {output_path}"
            
            total_time = time.time() - start_time
            avg_fps = frames_processed / total_time if total_time > 0 else 0
            avg_lane_coverage = sum(lane_coverages) / len(lane_coverages) if lane_coverages else 0
            
            results = {
                'output_filename': output_filename,
                'total_time': total_time,
                'frames_processed': frames_processed,
                'avg_fps': avg_fps,
                'avg_lane_coverage': avg_lane_coverage
            }
            
            logger.info(f"Video processing complete: {frames_processed} frames in {total_time:.2f}s")
            logger.info(f"Output file size: {file_size} bytes")
            return results, None
            
        except Exception as e:
            error_msg = f"Video processing failed: {e}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

class LaneAssistanceSystem:
    """Basic Lane Departure Warning and Assistance System"""
    
    def __init__(self):
        self.alert_history = []
        self.safety_score = 100.0
        self.consecutive_warnings = 0
        self.total_frames_processed = 0
        self.safe_frames = 0
        
        # Configurable thresholds
        self.thresholds = {
            'critical_departure': 15,  # < 15% lane coverage
            'warning_drift': 35,       # < 35% lane coverage
            'safe_position': 50,       # >= 50% lane coverage
            'excellent_position': 70   # >= 70% lane coverage
        }
    
    def analyze_lane_position(self, lane_coverage):
        """Analyze current lane position and return status"""
        self.total_frames_processed += 1
        
        if lane_coverage < self.thresholds['critical_departure']:
            status = "CRITICAL_DEPARTURE"
            self.consecutive_warnings += 1
            self.safety_score = max(0, self.safety_score - 2.0)
        elif lane_coverage < self.thresholds['warning_drift']:
            status = "WARNING_DRIFT"
            self.consecutive_warnings += 1
            self.safety_score = max(0, self.safety_score - 0.5)
        elif lane_coverage >= self.thresholds['excellent_position']:
            status = "EXCELLENT_POSITION"
            self.consecutive_warnings = 0
            self.safe_frames += 1
            self.safety_score = min(100, self.safety_score + 0.2)
        else:
            status = "SAFE_POSITION"
            self.consecutive_warnings = 0
            self.safe_frames += 1
            self.safety_score = min(100, self.safety_score + 0.1)
        
        return status
    
    def generate_alert(self, position_status, lane_coverage):
        """Generate appropriate alert based on lane position"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        alerts = {
            "CRITICAL_DEPARTURE": {
                "level": "critical",
                "message": "🚨 LANE DEPARTURE CRITICAL",
                "action": "IMMEDIATE CORRECTION NEEDED",
                "color": "#ff0000",
                "sound": "critical"
            },
            "WARNING_DRIFT": {
                "level": "warning",
                "message": "⚠️ LANE DRIFT DETECTED",
                "action": "ADJUST STEERING GENTLY",
                "color": "#ff8800",
                "sound": "warning"
            },
            "SAFE_POSITION": {
                "level": "safe",
                "message": "✅ LANE POSITION GOOD",
                "action": "CONTINUE CURRENT PATH",
                "color": "#00aa00",
                "sound": "none"
            },
            "EXCELLENT_POSITION": {
                "level": "excellent",
                "message": "🌟 EXCELLENT LANE KEEPING",
                "action": "PERFECT DRIVING",
                "color": "#0066cc",
                "sound": "none"
            }
        }
        
        alert = alerts.get(position_status, alerts["SAFE_POSITION"])
        alert.update({
            "timestamp": timestamp,
            "lane_coverage": lane_coverage,
            "consecutive_warnings": self.consecutive_warnings,
            "safety_score": round(self.safety_score, 1)
        })
        
        # Store in history (keep last 50 alerts)
        self.alert_history.append(alert)
        if len(self.alert_history) > 50:
            self.alert_history.pop(0)
        
        return alert
    
    def get_safety_metrics(self):
        """Get current safety metrics and statistics"""
        if self.total_frames_processed == 0:
            return {
                "safety_score": 100.0,
                "safe_percentage": 100.0,
                "total_alerts": 0,
                "critical_alerts": 0,
                "warning_alerts": 0
            }
        
        safe_percentage = (self.safe_frames / self.total_frames_processed) * 100
        
        # Count alert types
        critical_alerts = sum(1 for alert in self.alert_history if alert["level"] == "critical")
        warning_alerts = sum(1 for alert in self.alert_history if alert["level"] == "warning")
        
        return {
            "safety_score": round(self.safety_score, 1),
            "safe_percentage": round(safe_percentage, 1),
            "total_alerts": len(self.alert_history),
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "consecutive_warnings": self.consecutive_warnings,
            "frames_processed": self.total_frames_processed
        }
    
    def reset_session(self):
        """Reset assistance session data"""
        self.alert_history = []
        self.safety_score = 100.0
        self.consecutive_warnings = 0
        self.total_frames_processed = 0
        self.safe_frames = 0

# Initialize the demo app and assistance system
lane_app = LaneDetectionWebDemo()
assistance_system = LaneAssistanceSystem()

@app.route('/')
def index():
    """Main page - Stop camera on page load/refresh"""
    # Stop camera when page is loaded/refreshed
    lane_app.stop_camera()
    logger.info("📱 Page loaded - Camera stopped until user requests it")
    return render_template('index.html')

@app.route('/test-videos')
def test_videos():
    """Video test and download page"""
    return send_file('video_test.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process image
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Process with lane detection
            results, error = lane_app.process_image(image)
            if error:
                return jsonify({'error': error}), 500
            
            # Convert results to base64 for web display
            _, mask_buffer = cv2.imencode('.png', results['mask'])
            mask_b64 = base64.b64encode(mask_buffer).decode('utf-8')
            
            _, overlay_buffer = cv2.imencode('.png', cv2.cvtColor(results['overlay'], cv2.COLOR_RGB2BGR))
            overlay_b64 = base64.b64encode(overlay_buffer).decode('utf-8')
            
            # Original image
            _, orig_buffer = cv2.imencode('.png', image)
            orig_b64 = base64.b64encode(orig_buffer).decode('utf-8')
            
            response = {
                'success': True,
                'original_image': f"data:image/png;base64,{orig_b64}",
                'mask_image': f"data:image/png;base64,{mask_b64}",
                'overlay_image': f"data:image/png;base64,{overlay_b64}",
                'processing_time': round(results['processing_time'] * 1000, 2),
                'lane_coverage': round(results['lane_coverage'], 2),
                'confidence': round(results['confidence_mean'] * 100, 2)
            }
            
            # Add assistance data if available
            if 'assistance' in results:
                response['assistance'] = results['assistance']
            
            return jsonify(response)
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_video_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_filename = f"{timestamp}_{filename}"
            input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            
            logger.info(f"Saving uploaded video: {input_filepath}")
            file.save(input_filepath)
            
            # Verify input file was saved
            if not os.path.exists(input_filepath):
                return jsonify({'error': 'Failed to save uploaded file'}), 500
            
            # Process video
            logger.info(f"Starting video processing for: {input_filename}")
            results, error = lane_app.process_video(input_filepath)
            if error:
                logger.error(f"Video processing failed: {error}")
                return jsonify({'error': error}), 500
            
            # Verify output file exists
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], results['output_filename'])
            if not os.path.exists(output_path):
                logger.error(f"Processed video file not found: {output_path}")
                return jsonify({'error': 'Processed video file was not created'}), 500
            
            logger.info(f"Video processing successful: {results['output_filename']}")
            
            # Try to create web-optimized version for better browser compatibility
            web_optimized_filename = None
            try:
                web_optimized_path = output_path.replace('.mp4', '_web.mp4').replace('.avi', '_web.mp4')
                web_optimized_filename = os.path.basename(web_optimized_path)
                
                logger.info(f"Creating web-optimized version: {web_optimized_path}")
                
                # Open processed video for web optimization
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Use H.264 codec specifically for web compatibility
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                    web_out = cv2.VideoWriter(web_optimized_path, fourcc, fps, (width, height))
                    
                    if web_out.isOpened():
                        # Copy all frames to web-optimized video
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            web_out.write(frame)
                        
                        web_out.release()
                        logger.info(f"Web-optimized version created: {web_optimized_filename}")
                    else:
                        logger.warning("Failed to create web-optimized version with H.264")
                        web_optimized_filename = None
                    
                    cap.release()
                else:
                    logger.warning("Could not open processed video for web optimization")
                    web_optimized_filename = None
                    
            except Exception as e:
                logger.warning(f"Web optimization failed: {e}")
                web_optimized_filename = None
            
            response = {
                'success': True,
                'original_video': f"/video/{input_filename}",
                'processed_video': f"/video/{results['output_filename']}",
                'web_optimized_video': f"/video/{web_optimized_filename}" if web_optimized_filename else None,
                'total_time': round(results['total_time'], 2),
                'frames_processed': results['frames_processed'],
                'avg_fps': round(results['avg_fps'], 1),
                'avg_lane_coverage': round(results['avg_lane_coverage'], 2)
            }
            
            logger.info(f"Returning response: {response}")
            return jsonify(response)
        
        return jsonify({'error': 'Invalid video file type'}), 400
        
    except Exception as e:
        logger.error(f"Video upload error: {e}", exc_info=True)
        return jsonify({'error': 'Video processing failed'}), 500

@app.route('/camera/start')
def start_camera():
    """Start camera feed"""
    success = lane_app.start_camera()
    message = 'Camera started successfully - Live detection active' if success else 'Camera access denied - Check System Preferences → Security & Privacy → Camera'
    return jsonify({'success': success, 'message': message})

@app.route('/camera/stop')
def stop_camera():
    """Stop camera feed"""
    lane_app.stop_camera()
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/page/loaded')
def page_loaded():
    """Handle page load - automatically stop camera"""
    lane_app.stop_camera()
    return jsonify({'success': True, 'message': 'Camera stopped on page load'})

@app.route('/camera/feed')
def camera_feed():
    """Video streaming route"""
    def generate_frames():
        while lane_app.camera_active:
            frame = lane_app.get_camera_frame()
            if frame is not None:
                # Process frame with demo lane detection
                results, error = lane_app.process_image(frame)
                
                if results is not None:
                    # Use overlay image for display
                    display_frame = cv2.cvtColor(results['overlay'], cv2.COLOR_RGB2BGR)
                    
                    # Add processing info overlay
                    processing_time = results['processing_time'] * 1000
                    lane_coverage = results['lane_coverage']
                    
                    # Check if using actual model
                    model_status = "AI MODEL" if hasattr(lane_app, 'actual_model') and lane_app.actual_model is not None else "DEMO MODE"
                    
                    cv2.putText(display_frame, f"{model_status} - Processing: {processing_time:.1f}ms", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Lane Coverage: {lane_coverage:.1f}%", 
                              (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"FPS: {1/results['processing_time']:.1f}", 
                              (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    display_frame = frame
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', display_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # ~30 FPS
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/<filename>')
def serve_video(filename):
    """Serve video files"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Attempting to serve video: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Video file not found: {file_path}")
            # List available files for debugging
            try:
                available_files = os.listdir(app.config['UPLOAD_FOLDER'])
                logger.info(f"Available files in upload folder: {available_files}")
            except:
                logger.error("Could not list upload folder contents")
            return jsonify({'error': f'Video file not found: {filename}'}), 404
        
        # Determine MIME type based on file extension
        if filename.lower().endswith('.mp4'):
            mimetype = 'video/mp4'
        elif filename.lower().endswith('.avi'):
            mimetype = 'video/avi'
        elif filename.lower().endswith('.mov'):
            mimetype = 'video/quicktime'
        else:
            mimetype = 'video/mp4'  # Default fallback
            
        return send_file(file_path, mimetype=mimetype)
    except Exception as e:
        logger.error(f"Video serving error: {e}", exc_info=True)
        return jsonify({'error': 'Video serving failed'}), 500

@app.route('/download/<filename>')
def download_video(filename):
    """Download video files directly as attachment"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Attempting to download video: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Video file not found for download: {file_path}")
            return jsonify({'error': f'Video file not found: {filename}'}), 404
        
        # Get file size for logging
        file_size = os.path.getsize(file_path)
        logger.info(f"Serving download: {filename} ({file_size} bytes)")
        
        # Determine MIME type based on file extension
        if filename.lower().endswith('.mp4'):
            mimetype = 'video/mp4'
        elif filename.lower().endswith('.avi'):
            mimetype = 'video/avi'
        elif filename.lower().endswith('.mov'):
            mimetype = 'video/quicktime'
        else:
            mimetype = 'video/mp4'  # Default fallback
            
        return send_file(
            file_path, 
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Video download error: {e}", exc_info=True)
        return jsonify({'error': 'Video download failed'}), 500

@app.route('/assistance/metrics')
def get_assistance_metrics():
    """Get current assistance metrics"""
    try:
        metrics = assistance_system.get_safety_metrics()
        return jsonify({
            'success': True,
            'metrics': metrics,
            'recent_alerts': assistance_system.alert_history[-10:] if assistance_system.alert_history else []
        })
    except Exception as e:
        logger.error(f"Assistance metrics error: {e}")
        return jsonify({'error': 'Failed to get assistance metrics'}), 500

@app.route('/assistance/reset', methods=['POST'])
def reset_assistance():
    """Reset assistance session"""
    try:
        assistance_system.reset_session()
        return jsonify({
            'success': True,
            'message': 'Assistance session reset successfully'
        })
    except Exception as e:
        logger.error(f"Assistance reset error: {e}")
        return jsonify({'error': 'Failed to reset assistance'}), 500

@app.route('/assistance/settings', methods=['GET', 'POST'])
def assistance_settings():
    """Get or update assistance settings"""
    try:
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'settings': assistance_system.thresholds
            })
        else:
            # Update settings
            data = request.get_json()
            if 'thresholds' in data:
                assistance_system.thresholds.update(data['thresholds'])
                return jsonify({
                    'success': True,
                    'message': 'Settings updated successfully',
                    'settings': assistance_system.thresholds
                })
            return jsonify({'error': 'Invalid settings data'}), 400
    except Exception as e:
        logger.error(f"Assistance settings error: {e}")
        return jsonify({'error': 'Failed to handle settings'}), 500

@app.route('/stats')
def get_stats():
    """Get processing statistics"""
    # Get assistance metrics for stats
    assistance_metrics = assistance_system.get_safety_metrics()
    
    return jsonify({
        'model_loaded': lane_app.model_loaded,
        'camera_active': lane_app.camera_active,
        'assistance_metrics': assistance_metrics,
        'total_processed': lane_app.processing_stats['total_processed'],
        'avg_processing_time': round(lane_app.processing_stats['avg_processing_time'] * 1000, 2),
        'last_processing_time': round(lane_app.processing_stats['last_processing_time'] * 1000, 2)
    })

def allowed_file(filename):
    """Check if image file extension is allowed"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def allowed_video_file(filename):
    """Check if video file extension is allowed"""
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 PHASE 7: REAL-TIME PROCESSING WEB APP - DEMO MODE")
    print("="*60)
    print("🌐 Starting Flask web server...")
    print("📱 Access the web interface at: http://localhost:5001")
    print("🎥 Camera support: Available")
    print("📤 Upload support: Available")
    print("⚡ Real-time processing: Demo Mode (Simulated Lanes)")
    print("💡 Note: This is a demo version with simulated lane detection")
    print("="*60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
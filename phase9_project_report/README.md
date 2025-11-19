# 🚗 Lane Detection & Assistance System using CNN

## Overview

A comprehensive AI-powered lane detection and driver assistance system built with Lightweight U-Net architecture using TensorFlow and OpenCV. This project implements a complete 9-phase development cycle, achieving >90% accuracy in lane detection with real-time processing capabilities and professional ADAS-style visual alerts.

## 🎯 Key Features

- **Real-time Lane Detection**: >90% accuracy using Lightweight U-Net with TensorFlow
- **Web Interface**: Professional Flask-based application with modern UI
- **Driver Assistance**: 4-level visual alert system (Critical, Warning, Safe, Excellent)
- **Multi-input Support**: Images, videos, and live camera processing
- **ADAS-style Overlays**: Professional video overlays with HUD displays
- **Browser Compatibility**: H.264 optimized video processing
- **Safety Dashboard**: Real-time metrics and alert history tracking

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Accuracy | >90% |
| Processing Speed | 9.4 FPS |
| Architecture | Lightweight U-Net |
| Model Size | 22MB |
| Development Phases | 9 |

## 🏗️ System Architecture

```
Input → Preprocessing → CNN Model → Lane Detection → Assistance Analysis → Alert Generation → Video Overlay → Web Display
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 4GB+ RAM
- Webcam (optional, for live processing)
- Modern web browser

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Lane detection and Assistance system using CNN"
```

2. **Install dependencies**
```bash
pip install tensorflow==2.17.0 opencv-python flask numpy pillow
```

3. **Run the web application**
```bash
cd phase7_real_time_processing
python web_demo.py
```

4. **Access the application**
Open your browser and navigate to: `http://localhost:5001`

## 📁 Project Structure

```
Lane detection and Assistance system using CNN/
├── Dataset/                          # Training data (625MB)
│   ├── train_dataset.p
│   └── labels_dataset.p
├── phase1_dataset_analysis/          # Data exploration and visualization
├── phase2_data_preprocessing/        # Data preparation pipeline
├── phase3_model_architecture/        # CNN architecture design
├── phase4_model_training/            # Training implementation
├── phase5_model_validation/          # Performance validation
├── phase6_inference_pipeline/        # Prediction system
├── phase7_real_time_processing/      # Web application
│   ├── web_demo.py                   # Main Flask application
│   ├── templates/
│   │   └── index.html               # Web interface
│   ├── static/                      # CSS, JS, and assets
│   ├── uploads/                     # Uploaded files
│   ├── outputs/                     # Processed results
│   └── models/                      # Trained model files
├── phase8_driver_assistance/         # ADAS features
└── phase9_project_report/           # Comprehensive documentation
    └── project_report.html          # Full project report
```

## 🔧 Development Phases

### Phase 1: Dataset Analysis
- Comprehensive data exploration
- Statistical analysis and visualization
- Quality assessment of training data

### Phase 2: Data Preprocessing
- Data normalization and augmentation
- Pipeline optimization
- Format standardization

### Phase 3: Model Architecture
- CNN design for lane detection
- Layer optimization
- Architecture documentation

### Phase 4: Model Training
- TensorFlow implementation
- Training process optimization
- Checkpoint management

### Phase 5: Model Validation
- Performance evaluation
- Cross-validation
- Accuracy metrics analysis

### Phase 6: Inference Pipeline
- Real-time prediction system
- Optimization for production
- Performance benchmarking

### Phase 7: Real-Time Processing
- Flask web application
- Multi-input processing (image/video/camera)
- Professional web interface

### Phase 8: Driver Assistance
- ADAS implementation
- 4-level alert system
- Safety metrics dashboard
- Professional video overlays

### Phase 9: Project Report
- Comprehensive documentation
- Performance analysis
- Technical specifications

## 🚨 Driver Assistance Features

### Alert Levels

| Alert Type | Lane Coverage | Description | Visual Indicator |
|------------|---------------|-------------|------------------|
| 🚨 Critical | < 15% | Immediate correction needed | Red border + Visual |
| ⚠️ Warning | 15-35% | Gentle adjustment recommended | Orange indicators |
| ✅ Safe | 35-70% | Good lane keeping | Green status |
| 🌟 Excellent | > 70% | Perfect driving | Blue excellence |

### Safety Dashboard
- Real-time lane position monitoring
- Safety score calculation
- Alert history tracking
- Performance metrics display

## 🛠️ Technical Specifications

### Model Details
- **Framework**: TensorFlow 2.17
- **Architecture**: Lightweight U-Net (7.76M parameters)
- **Layers**: 32 layers with encoder-decoder structure
- **Input**: 80×160×3 (auto-scaled preprocessing)
- **Output**: Binary lane segmentation masks (80×160×1)
- **Accuracy**: >90% with robust performance
- **File Size**: 22MB (.keras format)

### Web Application
- **Backend**: Flask 2.3
- **Frontend**: HTML5, CSS3, JavaScript
- **Video Processing**: OpenCV with H.264 optimization
- **Real-time Features**: WebRTC camera integration
- **Browser Support**: Chrome, Firefox, Safari, Edge

### System Requirements
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **GPU**: Optional (CUDA support for faster processing)

## 📈 Usage Examples

### Image Processing
```python
# Upload image through web interface
# Automatic lane detection and overlay generation
# Download processed result with lane annotations
```

### Video Processing
```python
# Upload video file (MP4, AVI, MOV supported)
# Batch processing with progress tracking
# Browser-compatible output with H.264 encoding
```

### Live Camera
```python
# Access webcam through browser
# Real-time lane detection at 9.4 FPS
# Live assistance alerts and safety scoring
```

## 🔍 API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Image/video upload and processing
- `GET /camera` - Live camera processing
- `GET /assistance/metrics` - Real-time safety metrics
- `POST /assistance/reset` - Reset assistance history
- `GET /assistance/settings` - Configuration management

## 🎨 Web Interface Features

- **Responsive Design**: Mobile and desktop compatibility
- **Real-time Dashboard**: Live metrics and status updates
- **File Management**: Upload, process, and download files
- **Camera Integration**: Live video processing
- **Alert System**: Visual and audio notifications
- **Performance Monitoring**: Processing speed and accuracy display

## 📊 Performance Benchmarks

### Processing Speed
- **Images**: ~0.5 seconds per image
- **Videos**: 9.4 FPS processing rate
- **Live Camera**: Real-time at 720p resolution

### Accuracy Metrics
- **Lane Detection**: >90% accuracy
- **False Positives**: < 1%
- **Lane Coverage**: 85% average detection

## 🔮 Future Enhancements

- **Hardware Integration**: CAN bus connectivity for real vehicles
- **Advanced ML**: Transfer learning and model fine-tuning
- **Multi-lane Detection**: Support for complex road scenarios
- **Mobile Apps**: iOS and Android applications
- **Cloud Deployment**: Scalable processing infrastructure
- **Advanced Analytics**: Driving behavior analysis
- **Additional ADAS**: Collision detection, traffic sign recognition

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or support:
- Open an issue on GitHub
- Review the comprehensive project report in `phase9_project_report/`
- Check the technical documentation in each phase folder

## 🏆 Achievements

- ✅ >90% model accuracy achieved with Lightweight U-Net
- ✅ Complete 9-phase development cycle
- ✅ Professional web interface implementation
- ✅ Real-time processing capabilities
- ✅ ADAS-standard driver assistance features
- ✅ Browser-compatible video processing
- ✅ Comprehensive documentation and reporting

---

**Developed with ❤️ using TensorFlow, OpenCV, and Flask**

*Lane Detection & Assistance System - Your AI-Powered Driving Companion*
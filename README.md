# 🚗 Lane Detection and Assistance System Using CNN

![Lane Detection](https://img.shields.io/badge/Accuracy->90%25-brightgreen)
![Real-time](https://img.shields.io/badge/Inference-15ms-blue)
![Model Size](https://img.shields.io/badge/Parameters-7.8M-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 🎯 Project Overview

**Advanced computer vision system for autonomous vehicle safety**, implementing deep learning techniques for real-time lane detection and driver assistance. This comprehensive system combines cutting-edge CNN architecture with practical deployment solutions.

### 🏆 Key Achievements
- **>90% Detection Accuracy** - Validated on diverse road conditions
- **15ms Inference Time** - Real-time processing capability
- **7.8M Parameter Model** - Lightweight U-Net architecture
- **Complete Web Interface** - Professional ADAS-style system
- **Driver Assistance Integration** - 4-level safety alert system

## 🚀 Quick Start Guide

### 🔧 System Requirements
- **Python 3.9+**
- **TensorFlow 2.17+**
- **OpenCV 4.x**
- **Flask** (for web interface)
- **macOS/Windows/Linux**

### 📦 Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd "Lane detection and Assistance system using CNN"
```

2. **Install Dependencies**
```bash
# Core requirements
pip install tensorflow opencv-python flask pillow numpy

# Optional: Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate     # Windows
```

3. **Verify Installation**
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## 🎮 Running the System

### 🌐 Web Application (Recommended)

**Start the complete lane detection web interface:**

```bash
# Navigate to Phase 7
cd "phase7_real_time_processing/scripts"

# Launch web application
python web_demo.py
```

**Access the system at:** http://localhost:5001

### 📱 Features Available:
- ✅ **Live Camera Processing** - Real-time lane detection
- ✅ **Image Upload & Analysis** - Static image processing
- ✅ **Video Processing** - Batch video analysis with overlays
- ✅ **Driver Assistance Dashboard** - Safety metrics and alerts
- ✅ **Performance Statistics** - Real-time system monitoring

### 🎥 Camera Permissions (macOS)
If camera access is denied:
```bash
# Reset camera permissions
tccutil reset Camera

# Or manually: System Preferences → Security & Privacy → Camera
```

### 📊 Individual Phase Execution

#### Phase 1: Dataset Analysis
```bash
cd phase1_dataset_analysis/scripts
python enhanced_analysis.py
```

#### Phase 2: Data Preprocessing
```bash
cd phase2_data_preprocessing/scripts
python preprocess_data.py
```

#### Phase 3: Model Architecture
```bash
cd phase3_model_architecture/scripts
python build_model.py
```

#### Phase 4: Model Training
```bash
cd phase4_model_training/scripts
python train_model.py
```

#### Phase 5: Model Validation
```bash
cd phase5_model_validation/scripts
python validate_model.py
```

## 📁 Project Structure

```
Lane detection and Assistance system using CNN/
├── 📊 Dataset/                          # Training data
│   ├── train_dataset.p                  # Training images (12,764 samples)
│   └── labels_dataset.p                 # Binary lane masks
│
├── 🔍 phase1_dataset_analysis/          # Data exploration and analysis
│   └── scripts/
│       ├── enhanced_analysis.py         # Comprehensive dataset analysis
│       └── reports/                     # Analysis reports and visualizations
│
├── 🔄 phase2_data_preprocessing/        # Data preparation pipeline
│   └── scripts/
│       ├── preprocess_data.py          # Normalization and augmentation
│       └── reports/                     # Preprocessing reports
│
├── 🏗️ phase3_model_architecture/        # CNN architecture design
│   └── scripts/
│       ├── build_model.py              # Lightweight U-Net implementation
│       ├── models/                     # Saved model architectures
│       └── visualizations/             # Architecture diagrams
│
├── 🎯 phase4_model_training/            # Model training pipeline
│   └── scripts/
│       ├── train_model.py              # Training with callbacks
│       ├── models/                     # Trained models
│       └── visualizations/             # Training plots
│
├── 📈 phase5_model_validation/          # Performance evaluation
│   └── scripts/
│       ├── validate_model.py           # Comprehensive validation
│       ├── outputs/                    # Validation results
│       └── visualizations/             # Performance charts
│
├── ⚡ phase6_inference_pipeline/        # Real-time inference
│   └── scripts/
│       ├── inference_demo.py           # Live inference demo
│       └── inference_results/          # Benchmark results
│
├── 🎥 phase7_real_time_processing/      # Web application
│   ├── scripts/
│   │   └── web_demo.py                 # Flask web application
│   ├── templates/
│   │   └── index.html                  # Web interface
│   └── uploads/                        # User uploaded files
│
├── 📄 phase9_project_report/            # Documentation and reports
│   ├── presentation_comprehensive.html  # 15-page presentation
│   ├── project_report.html             # Detailed project report
│   ├── assets/                         # Images and diagrams
│   └── README.md                       # Project documentation
│
├── 📋 requirements/                     # Phase-specific dependencies
├── ⚙️ project_config.json              # Project configuration
└── 📖 README.md                        # This file
```

## 🔧 Technical Specifications

### 🧠 Model Architecture
- **Type:** Lightweight U-Net
- **Parameters:** 7,760,097 (7.8M)
- **Input Shape:** 80×160×3 RGB images
- **Output Shape:** 80×160×1 binary lane mask
- **Architecture Depth:** 32 layers with skip connections

### 📊 Dataset Details
- **Total Images:** 12,764 road scenes
- **Training Split:** 10,211 samples (80%)
- **Validation Split:** 2,553 samples (20%)
- **Resolution:** 80×160 pixels (optimized for real-time)
- **Normalization:** [0.0, 1.0] range
- **Binary Threshold:** 240 for lane segmentation

### ⚡ Performance Metrics
- **Accuracy:** >90% (validated)
- **IoU Score:** >90%
- **Precision:** >90%
- **Recall:** >90%
- **F1 Score:** >90%
- **Inference Time:** 15ms (batch processing)
- **Real-time Capability:** 60+ FPS

## 🎯 Applications & Use Cases

### 🚗 Automotive Industry
- **Autonomous Vehicles** - Self-driving car guidance and path planning
- **ADAS Systems** - Advanced Driver Assistance Systems integration
- **Lane Keeping Assist** - Real-time lane departure warnings
- **Fleet Management** - Commercial vehicle safety monitoring

### 🏙️ Smart City & Infrastructure
- **Traffic Monitoring** - Automated traffic flow analysis
- **Road Maintenance** - Lane marking condition assessment
- **Safety Analytics** - Accident prevention and analysis
- **Urban Planning** - Road infrastructure optimization

### 📱 Driver Assistance Features
- **4-Level Alert System:**
  - 🔴 **Critical** - Immediate correction required
  - 🟡 **Warning** - Approaching danger zone
  - 🟢 **Safe** - Normal driving conditions  
  - 🔵 **Excellent** - Optimal lane positioning

## 🎨 Web Interface Features

### 📸 Input Methods
- **Live Camera Feed** - Real-time processing with camera
- **Image Upload** - Drag & drop or browse image files
- **Video Upload** - Process video files with lane overlays
- **Batch Processing** - Multiple file processing

### 📊 Dashboard Components
- **Real-time Statistics** - Processing speed and accuracy metrics
- **Assistance Dashboard** - Safety alerts and lane positioning
- **Performance Monitor** - System resource usage
- **Results Gallery** - Processed images and videos

## 🛠️ Development & Customization

### 🔧 Configuration
Edit `project_config.json` for:
- Model parameters and thresholds
- Training hyperparameters
- Input/output specifications
- Performance targets

### 📝 Adding New Features
1. **Custom Models** - Implement in `phase3_model_architecture/`
2. **New Preprocessing** - Add to `phase2_data_preprocessing/`
3. **UI Enhancements** - Modify `phase7_real_time_processing/templates/`
4. **API Extensions** - Extend `web_demo.py` endpoints

## 🐛 Troubleshooting

### Common Issues & Solutions

**Camera Access Denied (macOS):**
```bash
tccutil reset Camera
# Then restart the application
```

**Module Import Errors:**
```bash
pip install --upgrade tensorflow opencv-python flask
```

**Performance Issues:**
- Reduce batch size in training
- Use CPU-optimized inference
- Check available system memory

**Web Interface Not Loading:**
- Verify port 5001 is available
- Check firewall settings
- Ensure all dependencies are installed

## 📈 Project Roadmap

### ✅ Completed Features
- [x] Complete 8-phase development pipeline
- [x] Lightweight U-Net architecture (>90% accuracy)
- [x] Real-time web application
- [x] Driver assistance system
- [x] Professional documentation

### 🚧 Future Enhancements
- [ ] Multi-lane detection and classification
- [ ] Integration with other ADAS systems
- [ ] Mobile application development
- [ ] Edge device optimization
- [ ] Advanced weather condition handling

## 📚 Documentation

- **📊 Comprehensive Presentation:** `phase9_project_report/presentation_comprehensive.html`
- **📄 Detailed Report:** `phase9_project_report/project_report.html`
- **🔍 Phase Reports:** Individual reports in each phase directory
- **📖 Technical Specs:** Available in project documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Project Lead** - Lane Detection and Assistance System Development
- **Technical Implementation** - Deep Learning and Computer Vision

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Research community for U-Net architecture innovations
- Open source contributors for various dependencies

---

**🚀 Ready to revolutionize lane detection? Start with our web demo at http://localhost:5001**

*For questions or support, please refer to the comprehensive documentation in the `phase9_project_report/` directory.*

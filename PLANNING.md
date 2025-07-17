# PLANNING.md - PCB Quality Inspection System

## 1. Project Vision

### 1.1 Executive Summary
Develop an automated PCB (Printed Circuit Board) quality inspection system using Deep Learning to detect manufacturing defects in bare PCBs (without components). The system leverages pre-trained YOLOv11 weights to identify 6 common defect types, providing real-time inspection capabilities for manufacturing quality control.

### 1.2 Problem Statement
- Manual PCB inspection is time-consuming, error-prone, and inconsistent
- Manufacturing defects in PCBs lead to product failures and increased costs
- Need for automated, accurate, and fast quality control system
- Requirement for real-time defect detection and tracking

### 1.3 Solution Overview
An automated inspection system that:
- Continuously monitors PCB production line using industrial camera
- Automatically detects when PCB is present and stable
- Performs AI-based defect detection using YOLOv11
- Provides real-time feedback and defect visualization
- Tracks inspection history and generates analytics

### 1.4 Key Benefits
- **Accuracy**: AI-powered detection with proven YOLOv11 model
- **Speed**: Real-time inspection at 30 FPS preview, instant defect detection
- **Automation**: Auto-trigger eliminates manual intervention
- **Traceability**: Complete inspection history and analytics
- **Cost-effective**: Uses existing trained model, minimal training required

## 2. System Architecture

### 2.1 High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                         │
│         (GUI with Live Preview & Inspection Results)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                    Analytics Layer                           │
│           (Statistics, Reports, Trend Analysis)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                      Data Layer                              │
│        (SQLite DB, Defect Image Storage, Metadata)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                   Processing Layer                           │
│    (Image Enhancement, PCB Detection, Post-processing)      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                      AI Layer                                │
│          (YOLOv11 Inference Engine, GPU Optimized)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                   Hardware Layer                             │
│        (Basler Camera Controller, Raw Image Stream)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                     Core Layer                               │
│        (Configuration, Interfaces, Common Utilities)         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Architecture
```
Camera (Raw Bayer) → Preview Stream → PCB Detection → Auto-Trigger
                                          ↓
                     High-Quality Capture → Pre-processing → AI Detection
                                                                 ↓
                     Database ← Results ← Post-processing ← Defect Analysis
                        ↓
                    Analytics → GUI Display → User
```

### 2.3 Component Interactions
- **One-way dependency**: Upper layers depend on lower layers only
- **Interface-based communication**: All interactions through defined interfaces
- **Event-driven triggers**: Auto-detection triggers inspection pipeline
- **Async processing**: Preview and inspection run in separate threads

## 3. Technology Stack

### 3.1 Programming Language
- **Python 3.10+**: Primary language for entire system
  - Rich ecosystem for computer vision and AI
  - Native support for all required libraries
  - Good performance with proper optimization

### 3.2 Core Technologies

#### Computer Vision & AI
- **YOLOv11 (Ultralytics)**: Pre-trained model for defect detection
- **OpenCV 4.8.0**: Image processing and computer vision
- **PyTorch 2.0.0**: Deep learning framework (YOLOv11 backend)
- **CUDA**: GPU acceleration on NVIDIA Tesla P4

#### Camera & Hardware
- **pypylon 3.0.0**: Official Basler camera SDK Python wrapper
- **Basler Pylon SDK**: Low-level camera control

#### Data Management
- **SQLite3**: Lightweight database for inspection records
- **JSON**: Metadata serialization
- **Pandas 2.0.0**: Data analysis and statistics

#### User Interface
- **Tkinter**: Built-in Python GUI framework
- **Pillow 10.0.0**: Image display and manipulation

#### System Integration
- **Threading**: Concurrent preview and inspection
- **Queue**: Thread-safe communication
- **Logging**: System monitoring and debugging

### 3.3 Hardware Requirements
- **Camera**: Basler acA3800-10gm Mono
- **GPU**: NVIDIA Tesla P4 8GB
- **OS**: Ubuntu 22.04 LTS
- **RAM**: Minimum 16GB recommended
- **Storage**: 100GB+ for defect image archive

## 4. Required Tools & Setup

### 4.1 Development Tools
```bash
# Version Control
- Git (for code management)

# Python Environment
- Python 3.10+
- pip (package manager)
- venv (virtual environment)

# IDE/Editor (recommended)
- VS Code with Python extension
- PyCharm Professional
- Jupyter Lab (for testing/debugging)

# System Monitoring
- nvidia-smi (GPU monitoring)
- htop (system resources)
```

### 4.2 Required Software Installation

#### 4.2.1 System Dependencies
```bash
# NVIDIA Driver & CUDA
sudo apt update
sudo apt install nvidia-driver-525  # or latest
sudo apt install nvidia-cuda-toolkit

# Python and development tools
sudo apt install python3.10 python3.10-venv python3-pip
sudo apt install build-essential cmake
sudo apt install libopencv-dev python3-opencv
```

#### 4.2.2 Basler Pylon SDK
```bash
# Download from Basler website
wget https://www.baslerweb.com/[pylon-download-link]
tar -xzf pylon_*.tar.gz
sudo tar -C /opt -xzf pylon_*.tar.gz
# Follow Basler installation guide
```

#### 4.2.3 Python Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Project Structure Setup
```bash
# Create directory structure
mkdir -p pcb-inspection/{core,hardware,ai,processing,data,analytics,presentation}
mkdir -p pcb-inspection/data/{images,defects}
mkdir -p pcb-inspection/weights

# Initialize Python packages
touch pcb-inspection/{core,hardware,ai,processing,data,analytics,presentation}/__init__.py
```

### 4.4 Pre-trained Model Setup
```bash
# Download or copy YOLOv11 weights trained on PCB dataset
# Place in weights/ directory
cp /path/to/yolov11_pcb_defects.pt pcb-inspection/weights/
```

### 4.5 Configuration Files

#### requirements.txt
```txt
ultralytics==8.0.0
pypylon==3.0.0
opencv-python==4.8.0
pillow==10.0.0
pandas==2.0.0
numpy==1.24.0
torch==2.0.0
torchvision==0.15.0
```

#### .gitignore
```
__pycache__/
*.py[cod]
*$py.class
venv/
.env
data/images/
data/defects/
*.db
*.log
.DS_Store
.idea/
.vscode/
```

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- Set up development environment
- Implement Core Layer (config, interfaces, utils)
- Basic Hardware Layer (camera connection)
- Unit tests for core components

### Phase 2: Detection Pipeline (Week 2)
- Processing Layer (preprocessor, PCB detector)
- AI Layer integration with YOLOv11
- Basic defect detection functionality
- Integration testing

### Phase 3: Auto-Trigger System (Week 3)
- Implement preview streaming
- PCB detection and stability checking
- Auto-trigger logic
- Performance optimization

### Phase 4: Data & Analytics (Week 4)
- Database schema and operations
- Analytics calculations
- Historical data tracking
- Report generation

### Phase 5: User Interface (Week 5)
- GUI implementation with dual display
- Real-time updates
- User controls and settings
- Error handling and messages

### Phase 6: Testing & Deployment (Week 6)
- System integration testing
- Performance benchmarking
- Documentation completion
- Production deployment

## 6. Success Metrics

### 6.1 Technical Metrics
- **Detection Accuracy**: >95% for trained defect types
- **Processing Speed**: <1 second per inspection
- **System Uptime**: >99% availability
- **False Positive Rate**: <5%

### 6.2 Operational Metrics
- **Throughput**: 1800+ PCBs/hour capability
- **Storage Efficiency**: <1MB per inspection (with defects)
- **Response Time**: <100ms for auto-trigger
- **Resource Usage**: <50% GPU utilization during operation

### 6.3 Business Metrics
- **ROI**: Cost recovery within 6 months
- **Quality Improvement**: 90% reduction in defective products shipped
- **Labor Savings**: 2-3 FTE equivalent
- **Inspection Coverage**: 100% of production

## 7. Risk Management

### 7.1 Technical Risks
- **Camera compatibility**: Test with exact Basler model early
- **GPU driver issues**: Maintain stable CUDA environment
- **Model accuracy**: Validate on actual production PCBs
- **Performance bottlenecks**: Profile and optimize critical paths

### 7.2 Mitigation Strategies
- Maintain fallback manual inspection option
- Regular model validation and retraining if needed
- Comprehensive error handling and logging
- Redundant data backup procedures

## 8. Future Enhancements

### 8.1 Short-term (3-6 months)
- Multi-camera support for larger PCBs
- Web-based dashboard for remote monitoring
- Integration with MES/ERP systems
- Custom training for new defect types

### 8.2 Long-term (6-12 months)
- Edge AI deployment for distributed inspection
- 3D inspection capabilities
- Predictive maintenance based on defect trends
- Mobile app for notifications and control

## 9. Conclusion

This planning document outlines a comprehensive approach to building an automated PCB inspection system that is:
- **Simple**: Minimal complexity while meeting all requirements
- **Effective**: Leverages proven AI technology
- **Scalable**: Can grow with production needs
- **Maintainable**: Clear architecture and documentation

The system design prioritizes reliability and ease of use while maintaining the flexibility to adapt to changing requirements.
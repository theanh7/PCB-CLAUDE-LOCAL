# 🚀 PCB Inspection System - Deployment Checklist

## 📋 Pre-Deployment Validation

### ✅ **1. System Requirements Verification**

#### Hardware Requirements
- [ ] **Camera**: Basler acA3800-10gm (10 GigE)
- [ ] **GPU**: NVIDIA Tesla P4 (minimum 4GB VRAM)
- [ ] **CPU**: Intel/AMD 4+ cores, 2.5GHz+
- [ ] **RAM**: 16GB minimum, 32GB recommended
- [ ] **Storage**: 100GB available (SSD recommended)
- [ ] **Network**: Stable connection for updates

#### Software Requirements  
- [ ] **OS**: Ubuntu 22.04 LTS / Windows 10+ / macOS 12+
- [ ] **Python**: 3.8-3.11 (tested versions)
- [ ] **CUDA**: 11.8 or 12.x (for GPU acceleration)
- [ ] **Basler Pylon SDK**: Latest version installed

### ✅ **2. Dependencies Installation**

#### Core Dependencies
```bash
# Check if already installed
python -c "import cv2, torch, numpy, pandas; print('✅ Core dependencies OK')"
```

- [ ] **OpenCV**: `opencv-python>=4.8.0`
- [ ] **PyTorch**: `torch>=2.0.0` (with CUDA support)
- [ ] **Ultralytics**: `ultralytics>=8.0.0`
- [ ] **NumPy**: `numpy>=1.24.0`
- [ ] **Pandas**: `pandas>=2.0.0`

#### Hardware-Specific
- [ ] **pypylon**: For Basler camera integration
- [ ] **CUDA drivers**: Compatible with PyTorch version
- [ ] **Camera drivers**: Basler GenTL drivers

### ✅ **3. Model and Data Preparation**

#### AI Model
- [ ] **YOLOv11 weights**: `weights/best.pt` (trained model)
- [ ] **Model validation**: Test inference on sample images
- [ ] **Class mapping**: Verify 6 defect classes mapping
- [ ] **Performance test**: <100ms inference time

#### Database Setup
- [ ] **SQLite database**: Create `data/pcb_inspection.db`
- [ ] **Directory structure**: Create all required folders
- [ ] **Permissions**: Write access to data directory
- [ ] **Backup strategy**: Configure automatic backups

### ✅ **4. Hardware Connection and Testing**

#### Camera Setup
```bash
# Test camera connection
python -c "
from hardware.camera_controller import BaslerCamera
from core.config import CAMERA_CONFIG
try:
    camera = BaslerCamera(CAMERA_CONFIG)
    print('✅ Camera connection successful')
except Exception as e:
    print(f'❌ Camera error: {e}')
"
```

- [ ] **Physical connection**: Camera properly connected via GigE
- [ ] **Network configuration**: IP address configured
- [ ] **Pylon Viewer test**: Camera visible in Pylon software
- [ ] **Frame capture test**: Successful image acquisition
- [ ] **Exposure settings**: Proper lighting configuration

#### GPU Testing
```bash
# Test GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

- [ ] **CUDA detection**: GPU properly detected
- [ ] **Memory test**: Sufficient VRAM available
- [ ] **Model loading**: YOLOv11 loads on GPU
- [ ] **Inference test**: GPU inference working

### ✅ **5. System Integration Testing**

#### Component Tests
```bash
# Run comprehensive tests
python test_system_integration.py
```

- [ ] **Core layer**: Configuration and utilities
- [ ] **Hardware layer**: Camera and preset functionality
- [ ] **Processing layer**: Image preprocessing pipeline
- [ ] **AI layer**: Model inference and result parsing
- [ ] **Data layer**: Database operations and analytics
- [ ] **Presentation layer**: GUI components
- [ ] **System integration**: End-to-end workflow

#### Performance Validation
- [ ] **Preview performance**: 30+ FPS live stream
- [ ] **Inspection speed**: <2 seconds per PCB
- [ ] **Memory usage**: <8GB total system memory
- [ ] **GPU utilization**: Efficient GPU usage
- [ ] **Database performance**: >100 writes/second

### ✅ **6. Configuration Validation**

#### Core Configuration
- [ ] **Camera config**: Exposure, gain, pixel format
- [ ] **AI config**: Model path, confidence threshold
- [ ] **Database config**: Path, backup settings
- [ ] **Trigger config**: Stability, focus thresholds
- [ ] **Logging config**: Level, file rotation

#### Production Settings
```python
# Verify production configuration
PRODUCTION_CONFIG = {
    "debug_mode": False,
    "log_level": "INFO",
    "auto_backup": True,
    "performance_monitoring": True,
    "error_notifications": True
}
```

### ✅ **7. User Interface and Workflow**

#### GUI Testing
- [ ] **Application startup**: Clean GUI launch
- [ ] **Mode switching**: Auto/Manual mode toggle
- [ ] **Live preview**: Real-time camera feed
- [ ] **Inspection results**: Proper result display
- [ ] **Analytics view**: Charts and statistics
- [ ] **History browser**: Inspection history access
- [ ] **Export functions**: Data export capabilities

#### Workflow Validation
- [ ] **Auto-trigger**: PCB detection → inspection
- [ ] **Manual trigger**: Operator-initiated inspection
- [ ] **Result interpretation**: Clear defect indication
- [ ] **Data persistence**: Results properly saved
- [ ] **Error handling**: Graceful error recovery

## 🔧 **Deployment Process**

### **Step 1: Environment Preparation**
```bash
# 1. Clone repository
git clone <repository-url>
cd PCB-CLAUDE

# 2. Run setup script
python setup_dev.py --mode full

# 3. Activate environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### **Step 2: Hardware Configuration**
```bash
# 1. Install Basler Pylon SDK
# Download from: https://www.baslerweb.com/downloads/software-downloads/

# 2. Configure camera network settings
# Use Pylon IP Configuration Tool

# 3. Test camera connection
python -c "from hardware.test_camera import main; main()"
```

### **Step 3: Model Deployment**
```bash
# 1. Copy trained model to weights directory
cp your_trained_model.pt weights/best.pt

# 2. Validate model
python -c "
from ai.validate_ai import main
main()
"
```

### **Step 4: System Validation**
```bash
# 1. Run full system test
python test_system_integration.py

# 2. Performance test
python -c "
from tests.test_performance import run_performance_tests
run_performance_tests()
"
```

### **Step 5: Production Launch**
```bash
# 1. Start application
python main.py

# 2. Verify all components
# - Camera preview working
# - AI detection functional
# - Database storing results
# - GUI responsive
```

## ⚠️ **Common Issues and Solutions**

### **Camera Issues**
- **Problem**: Camera not detected
- **Solution**: Check network configuration, ensure Pylon SDK installed
- **Command**: `python hardware/test_camera.py`

### **GPU Issues**
- **Problem**: CUDA not available
- **Solution**: Install/update NVIDIA drivers and CUDA toolkit
- **Command**: `nvidia-smi` to check GPU status

### **Model Issues**
- **Problem**: Model loading fails
- **Solution**: Verify model file path and format
- **Command**: `python ai/validate_ai.py`

### **Performance Issues**
- **Problem**: Slow inference
- **Solution**: Check GPU utilization, reduce image size
- **Command**: Monitor with `nvidia-smi` during operation

## 📊 **Production Monitoring**

### **Health Checks**
```bash
# System health monitoring script
python -c "
import psutil
import torch
from pathlib import Path

print('=== System Health ===')
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\".\").percent}%')
print(f'GPU: {torch.cuda.is_available()}')

# Check critical files
critical_files = ['weights/best.pt', 'data/pcb_inspection.db']
for file in critical_files:
    exists = Path(file).exists()
    print(f'{file}: {\"✅\" if exists else \"❌\"}')
"
```

### **Performance Metrics**
- [ ] **Inspection rate**: Target 30+ PCBs/hour
- [ ] **False positive rate**: <5%
- [ ] **System uptime**: >99%
- [ ] **Response time**: <100ms preview, <2s inspection

### **Backup and Recovery**
- [ ] **Database backup**: Daily automated backups
- [ ] **Configuration backup**: Settings saved externally
- [ ] **Model versioning**: Track model updates
- [ ] **Recovery procedures**: Documented restoration steps

## ✅ **Sign-off Checklist**

### **Technical Validation**
- [ ] All components tested and functional
- [ ] Performance requirements met
- [ ] Error handling validated
- [ ] Security considerations addressed
- [ ] Documentation complete

### **User Acceptance**
- [ ] Operator training completed
- [ ] Workflow procedures documented
- [ ] User interface approved
- [ ] Acceptance criteria met

### **Production Readiness**
- [ ] System monitoring configured
- [ ] Backup procedures established
- [ ] Support procedures documented
- [ ] Maintenance schedule created

---

**Deployment Date**: _______________  
**Deployed By**: _______________  
**Validated By**: _______________  
**Approved By**: _______________  

**🎉 System Ready for Production Operation**
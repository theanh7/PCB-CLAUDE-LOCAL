# Camera Troubleshooting Guide

This guide provides solutions for common camera-related issues in the PCB inspection system.

## Quick Diagnostics

First, run the camera diagnostics to identify issues:

```bash
python hardware/test_camera.py
```

## Common Issues and Solutions

### 1. Camera Not Detected

**Symptoms:**
- "No camera found" error
- Camera initialization fails
- System cannot connect to camera

**Solutions:**

#### Check Physical Connection
```bash
# For USB cameras
lsusb | grep -i basler

# For GigE cameras
ping [camera_ip_address]
```

#### Verify Pylon Installation
```bash
# Check if Pylon SDK is installed
ls /opt/pylon*/bin/

# Test with Pylon Viewer
/opt/pylon*/bin/pylonviewer
```

#### Check Permissions (Linux)
```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Check udev rules
ls /etc/udev/rules.d/*pylon*

# Restart udev
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 2. pypylon Import Error

**Symptoms:**
- `ImportError: No module named 'pypylon'`
- Camera controller fails to import

**Solutions:**

#### Install pypylon
```bash
# Standard installation
pip install pypylon

# If installation fails, try:
pip install --upgrade pip
pip install pypylon --no-cache-dir

# For specific Python version
python3.10 -m pip install pypylon
```

#### Verify Installation
```python
# Test import
import pypylon
print(pypylon.__version__)

# Test basic functionality
from pypylon import pylon
print(pylon.TlFactory.GetInstance().EnumerateDevices())
```

### 3. Camera Configuration Issues

**Symptoms:**
- Camera connects but settings fail
- Exposure/gain parameters rejected
- Pixel format not supported

**Solutions:**

#### Check Camera Capabilities
```python
from hardware.camera_controller import BaslerCamera

# Create camera instance
camera = BaslerCamera()

# Check supported formats
print("Supported pixel formats:")
for fmt in camera.camera.PixelFormat.GetSymbolics():
    print(f"  {fmt}")

# Check exposure range
print(f"Exposure range: {camera.camera.ExposureTime.GetMin()} - {camera.camera.ExposureTime.GetMax()}")
```

#### Use Compatible Settings
```python
# Use validated presets
from hardware.camera_presets import CameraPresets

# Get working configuration
config = CameraPresets.get_preset("default")
camera = BaslerCamera(config)
```

### 4. Streaming Performance Issues

**Symptoms:**
- Low frame rate
- Dropped frames
- High CPU usage

**Solutions:**

#### Optimize Camera Settings
```python
# Use fast preview preset
from hardware.camera_presets import get_fast_preview_config

config = get_fast_preview_config()
camera = BaslerCamera(config)
```

#### Check System Resources
```bash
# Monitor CPU usage
top -p $(pgrep -f python)

# Check memory usage
free -h

# Monitor network (for GigE cameras)
iftop -i eth0
```

#### Reduce Buffer Size
```python
# Smaller buffer for less latency
config = {
    "buffer_size": 3,
    "preview_exposure": 2000,
    "binning": 2
}
```

### 5. Image Quality Issues

**Symptoms:**
- Dark/bright images
- Noisy images
- Poor focus

**Solutions:**

#### Adjust Exposure and Gain
```python
# For dark images
camera.set_exposure(15000)  # Increase exposure
camera.set_gain(2)          # Add some gain

# For bright images
camera.set_exposure(2000)   # Decrease exposure
camera.set_gain(0)          # Remove gain
```

#### Use Lighting Presets
```python
from hardware.camera_presets import optimize_for_lighting

# Optimize for your lighting conditions
config = optimize_for_lighting("low")     # Low light
config = optimize_for_lighting("bright")  # Bright light
```

#### Check Focus
```python
from core.utils import calculate_focus_score

# Capture image and check focus
image = camera.capture()
focus_score = calculate_focus_score(image)
print(f"Focus score: {focus_score}")

# Focus score should be > 100 for good focus
```

### 6. Threading and Synchronization Issues

**Symptoms:**
- Application hangs
- Frame queue fills up
- Memory leaks

**Solutions:**

#### Monitor Queue Status
```python
# Check queue statistics
stats = camera.get_frame_statistics()
print(f"Queue size: {stats['queue_size']}")
print(f"Frames dropped: {stats['frames_dropped']}")
```

#### Implement Proper Cleanup
```python
# Use context manager
with BaslerCamera() as camera:
    camera.start_streaming()
    # ... use camera
    # Automatic cleanup on exit

# Or manual cleanup
camera.stop_streaming()
camera.close()
```

### 7. High-Quality Capture Issues

**Symptoms:**
- High-quality capture fails
- Mode switching doesn't work
- Exposure changes not applied

**Solutions:**

#### Check Mode Switching
```python
# Verify streaming state
print(f"Streaming: {camera.is_streaming}")

# Test high-quality capture
image = camera.capture_high_quality()
if image is not None:
    print(f"Captured image shape: {image.shape}")
```

#### Adjust Timeout Values
```python
# Increase timeout for high-quality capture
config = {
    "timeout": 10000,  # 10 seconds
    "capture_exposure": 20000
}
```

## Network Configuration (GigE Cameras)

### Check Network Settings
```bash
# Check network interface
ip addr show

# Check MTU size (should be 9000 for jumbo frames)
ip link show eth0

# Set jumbo frames
sudo ip link set eth0 mtu 9000
```

### Configure Camera IP
```python
# Set static IP for GigE camera
# This should be done through Pylon Viewer or camera configuration
```

## Performance Optimization

### System-Level Optimizations
```bash
# Increase network buffer sizes
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.rmem_default = 134217728' >> /etc/sysctl.conf

# Apply settings
sudo sysctl -p
```

### Application-Level Optimizations
```python
# Use appropriate presets
config = CameraPresets.get_speed_preset("fast")

# Reduce image resolution with binning
config["binning"] = 2

# Optimize buffer management
config["buffer_size"] = 5
```

## Debugging Tools

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug camera preset
from hardware.camera_presets import get_debug_config
config = get_debug_config()
```

### Camera Information
```python
# Get detailed camera information
info = camera.get_camera_info()
print(f"Camera info: {info}")

# Check camera capabilities
print(f"Frame rate: {camera.camera.ResultingFrameRate.GetValue()}")
print(f"Bandwidth: {camera.camera.BandwidthReserve.GetValue()}")
```

### Performance Monitoring
```python
# Run performance benchmarks
from hardware.test_camera import benchmark_camera_operations
results = benchmark_camera_operations()
print(f"Performance results: {results}")
```

## Environment Setup Checklist

- [ ] Ubuntu 22.04 LTS or compatible OS
- [ ] Python 3.10 or higher
- [ ] Basler Pylon SDK installed
- [ ] pypylon Python package installed
- [ ] Camera connected and powered
- [ ] Appropriate network configuration (for GigE)
- [ ] User permissions configured
- [ ] System resources adequate (CPU, memory)

## Getting Help

If issues persist:

1. Run full diagnostics: `python hardware/test_camera.py`
2. Check system logs: `dmesg | grep -i usb` or `journalctl -f`
3. Verify camera with Pylon Viewer
4. Check Basler support documentation
5. Review camera datasheet for specifications

## Common Error Messages

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `"No camera found"` | Physical connection | Check USB/network cable |
| `"Access denied"` | Permissions | Add user to dialout group |
| `"Invalid pixel format"` | Unsupported format | Use supported format |
| `"Timeout occurred"` | Network/timing issue | Increase timeout value |
| `"Buffer underrun"` | Performance issue | Reduce frame rate or resolution |

## Contact Support

For additional support:
- Check camera documentation
- Review Basler technical documentation
- Consult system administrator for network issues
- Contact development team for application-specific issues
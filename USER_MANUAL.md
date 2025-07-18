# PCB Auto-Inspection System - User Manual

## Version 1.0 | December 2024

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Operating the System](#operating-the-system)
4. [Understanding Results](#understanding-results)
5. [Maintenance & Troubleshooting](#maintenance--troubleshooting)
6. [Safety Guidelines](#safety-guidelines)
7. [Technical Specifications](#technical-specifications)

---

## System Overview

### What is the PCB Auto-Inspection System?

The PCB Auto-Inspection System is an automated quality control solution designed to detect manufacturing defects in bare printed circuit boards (PCBs). Using advanced AI technology and computer vision, the system can:

- **Automatically detect** when a PCB is placed in the inspection area
- **Identify 6 common defect types** with high accuracy
- **Provide real-time feedback** on inspection results
- **Maintain detailed records** of all inspections
- **Generate analytics** and quality reports

### Key Benefits

- ✅ **Increased Accuracy**: AI-powered detection reduces human error
- ✅ **Higher Throughput**: Automated inspection at up to 1800 PCBs/hour
- ✅ **Consistent Quality**: Standardized inspection criteria
- ✅ **Complete Traceability**: Digital records of all inspections
- ✅ **Cost Reduction**: Reduced labor costs and defect-related rework

### Detected Defect Types

The system can identify these common PCB defects:

1. **Missing Hole** - Drill holes that are absent or incomplete
2. **Mouse Bite** - Small circular cutouts along board edges
3. **Open Circuit** - Broken or incomplete electrical connections
4. **Short Circuit** - Unwanted electrical connections between traces
5. **Spur** - Unwanted copper extensions from traces
6. **Spurious Copper** - Excess copper in unintended areas

---

## Getting Started

### Initial Setup Checklist

Before operating the system, ensure:

- [ ] **Power Supply**: System is connected to stable power
- [ ] **Camera Connection**: Basler camera is properly connected
- [ ] **Lighting**: Inspection area has adequate, even lighting
- [ ] **Software**: System software is running and initialized
- [ ] **Calibration**: Camera and detection parameters are calibrated
- [ ] **Safety**: Work area is clear and safe

### System Startup

1. **Power On**
   - Turn on the main system power
   - Wait for system to complete startup sequence (approximately 30 seconds)

2. **Launch Software**
   - Double-click the PCB Inspection System icon
   - Wait for all system components to initialize
   - Verify camera connection (green status indicator)

3. **System Check**
   - Observe the live preview window
   - Ensure image quality is clear and well-lit
   - Check that all status indicators show "Ready"

### First-Time Calibration

⚠️ **Important**: Calibration should only be performed by trained technicians.

Contact your system administrator if calibration is needed.

---

## Operating the System

### Main Interface Overview

The system interface consists of four main areas:

```
┌─────────────────────────────────────────────────────────┐
│  [AUTO] [Manual Inspect] [Analytics] [History] Status  │
├─────────────────────┬───────────────────────────────────┤
│                     │                                   │
│   LIVE PREVIEW      │     INSPECTION RESULTS           │
│                     │                                   │
│  • PCB Detection    │  • Defect Visualization          │
│  • Focus Status     │  • Defect List                   │
│  • Stability Check  │  • Confidence Scores             │
│                     │                                   │
├─────────────────────┴───────────────────────────────────┤
│              STATISTICS & STATUS                        │
│  Total Inspections: 1,234 | Defects: 56 | Pass Rate: 95.4% │
└─────────────────────────────────────────────────────────┘
```

### Operation Modes

#### AUTO Mode (Recommended)
- **Default operating mode**
- System automatically detects PCB placement
- Inspection triggers when PCB is stable and in focus
- No manual intervention required

**To use AUTO mode:**
1. Ensure AUTO button is highlighted (default)
2. Place PCB in inspection area
3. Wait for automatic detection and inspection
4. Remove PCB after inspection completes

#### MANUAL Mode
- **Operator-controlled inspection**
- Manual trigger required for each inspection
- Useful for testing or difficult PCB orientations

**To use MANUAL mode:**
1. Click "AUTO" button to switch to "MANUAL"
2. Place PCB in inspection area
3. Click "Manual Inspect" button
4. Wait for inspection to complete
5. Remove PCB

### Step-by-Step Operating Procedure

#### Standard Operation (AUTO Mode)

1. **Prepare PCB**
   - Ensure PCB is clean and free of debris
   - Orient PCB with component side facing camera
   - Remove any protective films or covers

2. **Place PCB**
   - Gently place PCB in the inspection area
   - Center the PCB within the camera field of view
   - Avoid touching the PCB surface with fingers

3. **Automatic Detection**
   - System will display "PCB: Detected" when PCB is found
   - Green indicator shows PCB is properly positioned
   - Orange indicator means PCB is detected but not stable
   - Wait for "Stability: OK" status

4. **Automatic Inspection**
   - Inspection automatically triggers when conditions are met:
     - PCB detected and stable for 10 frames
     - Focus score above threshold (typically >100)
     - Minimum 2 seconds since last inspection
   - High-quality image is captured automatically
   - AI analysis begins immediately

5. **Review Results**
   - Results appear in the right panel within 1-2 seconds
   - Green "PASS" indicates no defects found
   - Red "FAIL" with defect list indicates defects detected
   - Defects are highlighted with colored bounding boxes

6. **Remove PCB**
   - Note the inspection result
   - Remove PCB from inspection area
   - Place PCB in appropriate pass/fail bin

#### Quality Control Actions

**For PASS Results:**
- ✅ PCB meets quality standards
- Route to next production step
- No further action required

**For FAIL Results:**
- ❌ Review detected defects carefully
- Verify defects are genuine (not false positives)
- Route PCB to rework or scrap as appropriate
- Document any systemic issues

### Understanding the Display

#### Live Preview Panel
- **PCB Detection**: Shows if PCB is detected in field of view
- **Focus Score**: Numerical value indicating image sharpness
- **Stability Status**: Shows if PCB position is stable
- **Frame Rate**: Current camera preview rate

#### Results Panel
- **Inspection Image**: High-quality captured image with defect markings
- **Defect List**: Text list of all detected defects
- **Confidence Scores**: AI confidence level for each detection
- **Processing Time**: Time taken for inspection (typically <100ms)

#### Status Bar
- **Total Inspections**: Count of all inspections since system start
- **Defects Found**: Total number of defects detected
- **Pass Rate**: Percentage of PCBs passing inspection
- **System Time**: Current date and time

---

## Understanding Results

### Inspection Results

#### PASS Result Example
```
Inspection #1,245
========================================

✓ No defects found
PCB PASSED

Processing Time: 0.08 seconds
Focus Score: 165.2
```

#### FAIL Result Example
```
Inspection #1,246
========================================

Found 2 defects:

1. Missing Hole (Confidence: 92.3%)
2. Spur (Confidence: 87.6%)

Processing Time: 0.12 seconds
Focus Score: 142.8
```

### Defect Visualization

Defects are marked on the inspection image with:
- **Colored bounding boxes** around defect areas
- **Labels** showing defect type and confidence
- **Consistent color coding** for easy identification

#### Color Coding
- 🔴 **Red**: Missing Hole
- 🟠 **Orange**: Mouse Bite  
- 🟡 **Yellow**: Open Circuit
- 🟣 **Magenta**: Short Circuit
- 🔵 **Cyan**: Spur
- 🟣 **Purple**: Spurious Copper

### Confidence Scores

Each defect detection includes a confidence score (0-100%):
- **>90%**: Very high confidence - defect is almost certainly present
- **80-90%**: High confidence - defect is likely present
- **70-80%**: Moderate confidence - review recommended
- **<70%**: Low confidence - manual verification required

### When to Question Results

Consider manual verification if:
- Confidence score is below 80%
- Defect appears very small or unclear
- PCB design has unusual features
- Lighting conditions were poor during inspection

---

## Maintenance & Troubleshooting

### Daily Maintenance

#### Start of Shift
- [ ] **Visual Inspection**: Check camera lens for dust or scratches
- [ ] **Lighting Check**: Verify lighting is even and adequate
- [ ] **Test Inspection**: Run a known-good PCB to verify operation
- [ ] **Clean Work Area**: Remove dust and debris from inspection area

#### End of Shift
- [ ] **System Shutdown**: Properly close software and power down system
- [ ] **Clean Lens**: Gently clean camera lens with appropriate cloth
- [ ] **Data Backup**: Ensure inspection data is backed up (automatic)
- [ ] **Log Review**: Check system logs for any errors or warnings

### Weekly Maintenance

- [ ] **Deep Clean**: Thoroughly clean inspection area and camera housing
- [ ] **Calibration Check**: Verify system accuracy with reference standards
- [ ] **Performance Review**: Analyze inspection statistics and trends
- [ ] **Software Updates**: Check for and install any system updates

### Common Issues and Solutions

#### "Camera Not Connected" Error
**Symptoms**: Red camera status, no preview image
**Solutions**:
1. Check camera USB/Ethernet cable connection
2. Restart system software
3. Power cycle camera
4. Contact technical support if issue persists

#### "Poor Image Quality" Warning  
**Symptoms**: Low focus scores, blurry preview
**Solutions**:
1. Clean camera lens gently with microfiber cloth
2. Check lighting - should be bright and even
3. Adjust PCB position to center in field of view
4. Verify camera mounting is secure

#### "No PCB Detected" Issue
**Symptoms**: PCB in place but not detected by system
**Solutions**:
1. Ensure PCB covers sufficient area (>10% of field of view)
2. Check PCB contrast against background
3. Verify PCB is flat and properly oriented
4. Clean PCB surface if dirty or reflective

#### False Positive Detections
**Symptoms**: System detects defects that aren't present
**Solutions**:
1. Check lighting for shadows or reflections
2. Ensure PCB surface is clean
3. Verify PCB design matches trained specifications
4. Consider recalibration if issue persists

#### Slow Performance
**Symptoms**: Long inspection times, system lag
**Solutions**:
1. Close unnecessary software applications
2. Check available disk space (>10GB recommended)
3. Restart system software
4. Contact IT support if performance doesn't improve

### When to Contact Support

Contact technical support immediately if:
- System displays error messages not covered in this manual
- Inspection accuracy appears significantly degraded
- Hardware damage is suspected
- Software crashes or freezes repeatedly
- Calibration procedure is needed

**Support Contact Information**:
- Technical Support: [Insert contact details]
- Emergency Support: [Insert emergency contact]
- System Administrator: [Insert admin contact]

---

## Safety Guidelines

### Electrical Safety
- ⚠️ **Never** touch electrical connections while system is powered
- ⚠️ **Always** use proper grounding procedures when handling PCBs
- ⚠️ **Keep** liquids away from all electrical equipment
- ⚠️ **Report** any damaged cables or connectors immediately

### Optical Safety
- ⚠️ **Do not** look directly into camera lighting
- ⚠️ **Avoid** pointing lasers or bright lights at camera
- ⚠️ **Use** proper lighting (avoid strobe or flickering lights)

### Mechanical Safety
- ⚠️ **Keep** fingers clear of moving parts
- ⚠️ **Ensure** PCBs are properly supported during inspection
- ⚠️ **Avoid** placing heavy objects on inspection area

### Data Security
- ⚠️ **Do not** share system access credentials
- ⚠️ **Protect** inspection data from unauthorized access
- ⚠️ **Follow** company data handling procedures

---

## Technical Specifications

### System Requirements
- **Operating System**: Windows 10/11 or Ubuntu 22.04 LTS
- **Processor**: Intel i5 8th gen or equivalent
- **Memory**: 16GB RAM minimum
- **Storage**: 100GB available space
- **Graphics**: NVIDIA GPU with CUDA support (recommended)

### Camera Specifications
- **Model**: Basler acA3800-10gm
- **Resolution**: 3840 × 2748 pixels
- **Frame Rate**: Up to 30 FPS (preview mode)
- **Interface**: USB 3.0 or GigE
- **Pixel Size**: 1.67 μm × 1.67 μm

### Performance Specifications
- **Inspection Speed**: <100ms per inspection
- **Throughput**: Up to 1800 PCBs/hour (theoretical)
- **Accuracy**: >95% for trained defect types
- **False Positive Rate**: <5%
- **Field of View**: Configurable (typically 50mm × 40mm)

### Environmental Requirements
- **Operating Temperature**: 15°C to 35°C (59°F to 95°F)
- **Humidity**: 30% to 80% relative humidity (non-condensing)
- **Lighting**: Bright, even illumination (>1000 lux recommended)
- **Vibration**: Minimal vibration environment required

### Network Requirements
- **Bandwidth**: Minimal (primarily for updates and remote support)
- **Connectivity**: Ethernet connection recommended
- **Security**: Standard corporate network security protocols

---

## Appendices

### Appendix A: Keyboard Shortcuts
- **F1**: Help
- **F5**: Refresh display
- **Ctrl+M**: Toggle AUTO/MANUAL mode
- **Ctrl+I**: Manual inspect (MANUAL mode only)
- **Ctrl+A**: View analytics
- **Ctrl+H**: View history

### Appendix B: Error Codes
- **E001**: Camera connection lost
- **E002**: Insufficient lighting
- **E003**: AI model loading failed
- **E004**: Database connection error
- **E005**: Calibration required

### Appendix C: Quality Standards
- **Class A**: Zero defects allowed
- **Class B**: Minor defects acceptable with documentation
- **Class C**: Defects acceptable within specification limits

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Prepared by**: PCB Inspection System Development Team  

For questions about this manual or system operation, contact your system administrator or technical support team.
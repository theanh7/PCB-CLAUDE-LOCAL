# Model Weights Directory

This directory contains the trained YOLOv11 model weights for PCB defect detection.

## Current Files

- `best.pt` - Main YOLOv11 model weights trained on PCB defect dataset
- `last.pt` - Last checkpoint from training (backup)
- `data.yaml` - Model configuration file with class definitions

## Model Specifications

The model is trained to detect the following defect classes (as defined in `data.yaml`):

**Model Classes (by ID):**
- 0: mouse_bite
- 1: spur  
- 2: missing_hole
- 3: short
- 4: open_circuit
- 5: spurious_copper

**Display Names (in system):**
- Mouse Bite
- Spur
- Missing Hole
- Short Circuit
- Open Circuit
- Spurious Copper

## Model Format

- Framework: YOLOv11 (Ultralytics) 
- Base Model: yolo11m.pt
- Format: PyTorch (.pt)
- Input size: 640x640 (configurable)
- Classes: 6 defect types
- Precision: FP16 optimized for Tesla P4

## Class Mapping

The system uses `MODEL_CLASS_MAPPING` in `core/config.py` to map model output IDs to display names:

```python
MODEL_CLASS_MAPPING = {
    0: "Mouse Bite",        # mouse_bite
    1: "Spur",             # spur  
    2: "Missing Hole",     # missing_hole
    3: "Short Circuit",    # short
    4: "Open Circuit",     # open_circuit
    5: "Spurious Copper"   # spurious_copper
}
```

## Configuration

The model is configured in `core/config.py`:

```python
AI_CONFIG = {
    "model_path": "weights/best.pt",
    "confidence": 0.5,
    "device": "cuda:0",  # Tesla P4
    "imgsz": 640,
    "half": True,  # FP16 for faster inference
    ...
}
```

## Usage

1. **Model Loading**: Done automatically by `PCBDefectDetector` class
2. **Inference**: Use `detect()` method for single images or `detect_batch()` for multiple images
3. **Testing**: Run `python ai/test_ai.py --quick` to validate model functionality

## Performance Targets

- **Inference Time**: <100ms per image on Tesla P4
- **Memory Usage**: <2GB GPU memory
- **Throughput**: 10+ FPS for real-time operation
- **Accuracy**: Optimized for PCB defect detection

## Validation

Run validation tests:

```bash
# Quick functionality test
python ai/validate_ai.py

# Full AI test suite  
python ai/test_ai.py --full

# Integration tests
python ai/test_integration.py
```

## Notes

- Model files are excluded from version control due to size
- The system automatically handles class mapping between model output and display names
- GPU optimization with FP16 precision for Tesla P4
- Fallback to CPU if GPU is not available
- Model warmup is performed automatically for consistent performance
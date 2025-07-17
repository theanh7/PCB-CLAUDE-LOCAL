# Model Weights Directory

This directory should contain the trained YOLOv11 model weights for PCB defect detection.

## Required Files

- `yolov11_pcb_defects.pt` - Main YOLOv11 model weights trained on PCB defect dataset

## Model Specifications

The expected model should be trained to detect the following defect classes:

1. Missing Hole
2. Mouse Bite
3. Open Circuit
4. Short Circuit
5. Spur
6. Spurious Copper

## Model Format

- Framework: YOLOv11 (Ultralytics)
- Format: PyTorch (.pt)
- Input size: 640x640 (configurable)
- Classes: 6 defect types

## Installation

1. Obtain the trained model weights from your ML team or training pipeline
2. Place the `.pt` file in this directory
3. Update the `model_path` in `core/config.py` if using a different filename
4. Run configuration validation: `python -c "from core.config import validate_config; print(validate_config())"`

## Notes

- The model file is excluded from version control due to size
- Ensure the model classes match the `DEFECT_CLASSES` in `core/config.py`
- Test the model with sample images before deployment
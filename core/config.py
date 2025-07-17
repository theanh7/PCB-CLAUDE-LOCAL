"""
Core configuration for PCB inspection system.

This module contains all system-wide configuration settings including
camera, AI model, auto-trigger, and database configurations.
"""

import os
from typing import Dict, List, Any

# Camera Configuration
CAMERA_CONFIG: Dict[str, Any] = {
    "model": "Basler_acA3800-10gm",
    "preview_exposure": 5000,    # Exposure for preview mode (μs)
    "capture_exposure": 10000,   # Exposure for high-quality capture (μs)
    "gain": 0,                   # Camera gain setting
    "pixel_format": "BayerRG8",  # Raw Bayer pattern for maximum quality
    "binning": 1,                # No binning for full resolution
    "trigger_mode": "Off",       # Free running for preview mode
    "buffer_size": 10,           # Frame buffer size for streaming
    "timeout": 5000,             # Grab timeout in milliseconds
}

# AI Model Configuration
AI_CONFIG: Dict[str, Any] = {
    "model_path": "weights/best.pt",  # YOLOv11 model trained on PCB defects
    "confidence": 0.5,           # Confidence threshold for detections
    "device": "cuda:0",          # GPU device (Tesla P4)
    "imgsz": 640,               # Input image size for model
    "max_det": 50,              # Maximum detections per image
    "agnostic_nms": False,      # Class-agnostic NMS
    "augment": False,           # Test-time augmentation
    "half": True,               # Use FP16 for faster inference
    "warmup": True,             # Warmup model for consistent performance
    "save_crops": False,        # Don't save detection crops
    "save_txt": False,          # Don't save detection labels
    "save_conf": True,          # Save confidence scores
}

# Auto-Trigger Configuration
TRIGGER_CONFIG: Dict[str, Any] = {
    "stability_frames": 10,      # Frames needed stable before triggering
    "focus_threshold": 100,      # Minimum focus score threshold
    "movement_threshold": 5,     # Pixel tolerance for stability check
    "min_pcb_area": 0.1,        # Minimum PCB area ratio (PCB/frame)
    "inspection_interval": 2.0,  # Minimum seconds between inspections
    "max_inspection_rate": 1800, # Maximum inspections per hour
    "enable_auto_trigger": True, # Enable/disable auto-trigger
}

# Database Configuration
DB_CONFIG: Dict[str, Any] = {
    "path": "data/pcb_inspection.db",
    "save_raw_images": False,    # Don't save raw images to save space
    "save_processed_images": True, # Save processed images only when defects found
    "max_db_size": 1000,        # Maximum number of records before cleanup
    "backup_enabled": True,      # Enable database backup
    "backup_interval": 24,       # Backup interval in hours
}

# Storage Configuration
STORAGE_CONFIG: Dict[str, Any] = {
    "images_dir": "data/images",
    "defects_dir": "data/defects",
    "max_image_size": 1920,     # Maximum image dimension for storage
    "image_quality": 85,        # JPEG quality for saved images
    "cleanup_after_days": 30,   # Auto-cleanup old images after N days
}

# Processing Configuration
PROCESSING_CONFIG: Dict[str, Any] = {
    "preview_resolution": (800, 600),  # Preview display resolution
    "debayer_method": "bilinear",      # Debayer algorithm (bilinear, edgesense)
    "contrast_enhancement": True,       # Enable CLAHE contrast enhancement
    "noise_reduction": True,           # Enable bilateral filtering
    "clahe_clip_limit": 2.0,          # CLAHE clip limit
    "clahe_tile_size": (8, 8),        # CLAHE tile grid size
    "bilateral_d": 9,                  # Bilateral filter diameter
    "bilateral_sigma_color": 75,       # Bilateral filter sigma color
    "bilateral_sigma_space": 75,       # Bilateral filter sigma space
}

# GUI Configuration
GUI_CONFIG: Dict[str, Any] = {
    "window_title": "PCB Auto-Inspection System",
    "window_size": (1400, 800),
    "preview_size": (640, 480),
    "result_size": (600, 600),
    "update_interval": 33,      # GUI update interval in ms (~30 FPS)
    "font_family": "Consolas",
    "font_size": 10,
    "theme": "default",
}

# Logging Configuration
LOG_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/pcb_inspection.log",
    "max_size": 10 * 1024 * 1024,  # 10MB max log file size
    "backup_count": 5,              # Keep 5 backup log files
    "console_output": True,         # Also log to console
}

# Performance Configuration
PERFORMANCE_CONFIG: Dict[str, Any] = {
    "max_threads": 4,           # Maximum number of processing threads
    "gpu_memory_fraction": 0.7, # GPU memory fraction to use
    "cpu_count": None,          # CPU cores to use (None = auto)
    "enable_profiling": False,  # Enable performance profiling
    "benchmark_mode": False,    # Enable benchmarking mode
    "gpu_optimization": True,   # Enable GPU-specific optimizations
    "batch_size": 4,           # Batch size for multi-image processing
    "fp16_inference": True,    # Use FP16 for faster inference
    "memory_cleanup": True,    # Enable automatic memory cleanup
}

# PCB Defect Classes (Display Names)
DEFECT_CLASSES: List[str] = [
    "Missing Hole",
    "Mouse Bite",
    "Open Circuit",
    "Short Circuit", 
    "Spur",
    "Spurious Copper"
]

# Model Class Mapping (from data.yaml to display names)
MODEL_CLASS_MAPPING: Dict[int, str] = {
    0: "Mouse Bite",        # mouse_bite -> Mouse Bite
    1: "Spur",             # spur -> Spur
    2: "Missing Hole",     # missing_hole -> Missing Hole
    3: "Short Circuit",    # short -> Short Circuit
    4: "Open Circuit",     # open_circuit -> Open Circuit
    5: "Spurious Copper"   # spurious_copper -> Spurious Copper
}

# Defect Color Mapping for visualization
DEFECT_COLORS: Dict[str, tuple] = {
    "Missing Hole": (255, 0, 0),      # Red
    "Mouse Bite": (255, 165, 0),      # Orange
    "Open Circuit": (255, 255, 0),    # Yellow
    "Short Circuit": (255, 0, 255),   # Magenta
    "Spur": (0, 255, 255),           # Cyan
    "Spurious Copper": (128, 0, 128) # Purple
}

# System Paths
PATHS: Dict[str, str] = {
    "root": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights": "weights",
    "data": "data",
    "logs": "logs",
    "config": "config",
    "temp": "temp",
}

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist."""
    required_dirs = [
        "data/images",
        "data/defects", 
        "logs",
        "config",
        "temp",
        "weights"
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)

# Configuration validation
def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check AI model path
    if not os.path.exists(AI_CONFIG["model_path"]):
        errors.append(f"AI model not found: {AI_CONFIG['model_path']}")
    
    # Check confidence threshold
    if not 0 < AI_CONFIG["confidence"] < 1:
        errors.append("AI confidence must be between 0 and 1")
    
    # Check focus threshold
    if TRIGGER_CONFIG["focus_threshold"] < 0:
        errors.append("Focus threshold must be positive")
    
    # Check stability frames
    if TRIGGER_CONFIG["stability_frames"] < 1:
        errors.append("Stability frames must be at least 1")
    
    # Check inspection interval
    if TRIGGER_CONFIG["inspection_interval"] < 0.1:
        errors.append("Inspection interval must be at least 0.1 seconds")
    
    return errors

# Get configuration by category
def get_config(category: str) -> Dict[str, Any]:
    """
    Get configuration for a specific category.
    
    Args:
        category: Configuration category name
        
    Returns:
        Configuration dictionary
    """
    config_map = {
        "camera": CAMERA_CONFIG,
        "ai": AI_CONFIG,
        "trigger": TRIGGER_CONFIG,
        "database": DB_CONFIG,
        "storage": STORAGE_CONFIG,
        "processing": PROCESSING_CONFIG,
        "gui": GUI_CONFIG,
        "logging": LOG_CONFIG,
        "performance": PERFORMANCE_CONFIG,
    }
    
    return config_map.get(category, {})

# Initialize configuration
if __name__ == "__main__":
    # Ensure directories exist
    ensure_directories()
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validated successfully")
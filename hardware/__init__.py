"""
Hardware Layer for PCB Inspection System.

This module provides camera control and hardware interface functionality
for the PCB quality inspection system.
"""

# Core camera classes
from .camera_controller import BaslerCamera, CameraImageHandler, create_camera

# Camera presets and configuration
from .camera_presets import (
    CameraPresets,
    get_fast_preview_config,
    get_quality_config,
    get_low_light_config,
    get_debug_config,
    optimize_for_lighting,
    optimize_for_speed,
    validate_config
)

# Version info
__version__ = "1.0.0"

# Public API
__all__ = [
    # Camera classes
    "BaslerCamera",
    "CameraImageHandler",
    "create_camera",
    
    # Presets and configuration
    "CameraPresets", 
    "get_fast_preview_config",
    "get_quality_config",
    "get_low_light_config",
    "get_debug_config",
    "optimize_for_lighting",
    "optimize_for_speed",
    "validate_config",
]

# Module-level documentation
def get_module_info():
    """Get information about the hardware module."""
    return {
        "name": "PCB Inspection Hardware Layer",
        "version": __version__,
        "description": "Camera control and hardware interface",
        "components": [
            "BaslerCamera - Main camera controller",
            "CameraImageHandler - Async frame handling",
            "CameraPresets - Predefined configurations",
        ],
        "dependencies": [
            "pypylon - Basler camera SDK",
            "numpy - Array operations",
            "core - System core components",
        ]
    }


# Convenience functions for quick setup
def create_fast_camera():
    """Create camera optimized for fast preview."""
    config = get_fast_preview_config()
    return BaslerCamera(config)


def create_quality_camera():
    """Create camera optimized for quality."""
    config = get_quality_config()
    return BaslerCamera(config)


def create_camera_for_lighting(lighting_condition):
    """Create camera optimized for specific lighting conditions."""
    config = optimize_for_lighting(lighting_condition)
    return BaslerCamera(config)


# Module initialization
def _check_dependencies():
    """Check if required dependencies are available."""
    try:
        import pypylon
        return True
    except ImportError:
        return False


# Set module-level flags
PYPYLON_AVAILABLE = _check_dependencies()

if not PYPYLON_AVAILABLE:
    import warnings
    warnings.warn(
        "pypylon not available. Camera functionality will be limited. "
        "Install with: pip install pypylon",
        ImportWarning
    )
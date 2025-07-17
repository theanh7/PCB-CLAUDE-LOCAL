"""
Camera parameter presets for different operating modes.

This module provides predefined camera configurations optimized for
different use cases in the PCB inspection system.
"""

from typing import Dict, Any
from core.config import CAMERA_CONFIG


class CameraPresets:
    """
    Predefined camera configurations for different operating modes.
    """
    
    # Default configuration from core config
    DEFAULT = CAMERA_CONFIG.copy()
    
    # High-speed preview mode for continuous monitoring
    PREVIEW_FAST = {
        "preview_exposure": 3000,    # Fast exposure for high FPS
        "capture_exposure": 10000,   # Standard capture exposure
        "gain": 2,                   # Slight gain increase for low light
        "pixel_format": "BayerRG8",
        "binning": 2,               # 2x2 binning for speed
        "buffer_size": 5,           # Smaller buffer for responsiveness
        "timeout": 3000,
    }
    
    # Balanced mode for normal operation
    PREVIEW_BALANCED = {
        "preview_exposure": 5000,    # Balanced exposure
        "capture_exposure": 15000,   # Higher quality capture
        "gain": 0,                   # No gain for better quality
        "pixel_format": "BayerRG8",
        "binning": 1,               # No binning
        "buffer_size": 10,          # Standard buffer
        "timeout": 5000,
    }
    
    # High-quality mode for detailed inspection
    PREVIEW_QUALITY = {
        "preview_exposure": 8000,    # Higher exposure for quality
        "capture_exposure": 20000,   # Very high quality capture
        "gain": 0,                   # No gain
        "pixel_format": "BayerRG8",
        "binning": 1,               # Full resolution
        "buffer_size": 15,          # Larger buffer
        "timeout": 8000,
    }
    
    # Low-light conditions
    LOW_LIGHT = {
        "preview_exposure": 10000,   # High exposure
        "capture_exposure": 30000,   # Very high exposure
        "gain": 5,                   # Moderate gain
        "pixel_format": "BayerRG8",
        "binning": 1,
        "buffer_size": 10,
        "timeout": 10000,
    }
    
    # Bright conditions
    BRIGHT_LIGHT = {
        "preview_exposure": 2000,    # Low exposure
        "capture_exposure": 5000,    # Low capture exposure
        "gain": 0,                   # No gain
        "pixel_format": "BayerRG8",
        "binning": 1,
        "buffer_size": 10,
        "timeout": 3000,
    }
    
    # Debug mode with verbose logging
    DEBUG = {
        "preview_exposure": 5000,
        "capture_exposure": 15000,
        "gain": 0,
        "pixel_format": "BayerRG8",
        "binning": 1,
        "buffer_size": 5,           # Small buffer for debugging
        "timeout": 5000,
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, Any]:
        """
        Get a camera preset configuration.
        
        Args:
            preset_name: Name of the preset to retrieve
            
        Returns:
            Dictionary containing camera configuration
            
        Raises:
            ValueError: If preset name is not found
        """
        presets = {
            "default": cls.DEFAULT,
            "preview_fast": cls.PREVIEW_FAST,
            "preview_balanced": cls.PREVIEW_BALANCED,
            "preview_quality": cls.PREVIEW_QUALITY,
            "low_light": cls.LOW_LIGHT,
            "bright_light": cls.BRIGHT_LIGHT,
            "debug": cls.DEBUG,
        }
        
        if preset_name.lower() not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
        
        return presets[preset_name.lower()].copy()
    
    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """
        List all available presets with descriptions.
        
        Returns:
            Dictionary mapping preset names to descriptions
        """
        return {
            "default": "Default configuration from core config",
            "preview_fast": "High-speed preview with 2x2 binning",
            "preview_balanced": "Balanced quality and speed",
            "preview_quality": "High-quality preview mode",
            "low_light": "Optimized for low-light conditions",
            "bright_light": "Optimized for bright conditions",
            "debug": "Debug mode with small buffer",
        }
    
    @classmethod
    def create_custom_preset(cls, base_preset: str = "default", **overrides) -> Dict[str, Any]:
        """
        Create a custom preset based on an existing one.
        
        Args:
            base_preset: Base preset to modify
            **overrides: Parameters to override
            
        Returns:
            Custom configuration dictionary
        """
        config = cls.get_preset(base_preset)
        config.update(overrides)
        return config
    
    @classmethod
    def get_lighting_preset(cls, lighting_condition: str) -> Dict[str, Any]:
        """
        Get preset optimized for specific lighting conditions.
        
        Args:
            lighting_condition: "low", "normal", or "bright"
            
        Returns:
            Appropriate camera configuration
        """
        lighting_map = {
            "low": "low_light",
            "normal": "preview_balanced",
            "bright": "bright_light"
        }
        
        preset_name = lighting_map.get(lighting_condition.lower())
        if not preset_name:
            raise ValueError(f"Unknown lighting condition: {lighting_condition}")
        
        return cls.get_preset(preset_name)
    
    @classmethod
    def get_speed_preset(cls, speed_priority: str) -> Dict[str, Any]:
        """
        Get preset optimized for specific speed requirements.
        
        Args:
            speed_priority: "fast", "balanced", or "quality"
            
        Returns:
            Appropriate camera configuration
        """
        speed_map = {
            "fast": "preview_fast",
            "balanced": "preview_balanced",
            "quality": "preview_quality"
        }
        
        preset_name = speed_map.get(speed_priority.lower())
        if not preset_name:
            raise ValueError(f"Unknown speed priority: {speed_priority}")
        
        return cls.get_preset(preset_name)


# Convenience functions for common operations
def get_fast_preview_config() -> Dict[str, Any]:
    """Get configuration optimized for fast preview."""
    return CameraPresets.get_preset("preview_fast")


def get_quality_config() -> Dict[str, Any]:
    """Get configuration optimized for quality."""
    return CameraPresets.get_preset("preview_quality")


def get_low_light_config() -> Dict[str, Any]:
    """Get configuration optimized for low light."""
    return CameraPresets.get_preset("low_light")


def get_debug_config() -> Dict[str, Any]:
    """Get configuration for debugging."""
    return CameraPresets.get_preset("debug")


def optimize_for_lighting(lighting_condition: str) -> Dict[str, Any]:
    """
    Get camera configuration optimized for specific lighting.
    
    Args:
        lighting_condition: "low", "normal", or "bright"
        
    Returns:
        Optimized camera configuration
    """
    return CameraPresets.get_lighting_preset(lighting_condition)


def optimize_for_speed(speed_priority: str) -> Dict[str, Any]:
    """
    Get camera configuration optimized for speed vs quality trade-off.
    
    Args:
        speed_priority: "fast", "balanced", or "quality"
        
    Returns:
        Optimized camera configuration
    """
    return CameraPresets.get_speed_preset(speed_priority)


# Configuration validation
def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate camera configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = [
        "preview_exposure", "capture_exposure", "gain", 
        "pixel_format", "binning", "buffer_size", "timeout"
    ]
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            return False
    
    # Check value ranges
    if not (100 <= config["preview_exposure"] <= 100000):
        return False
    
    if not (100 <= config["capture_exposure"] <= 100000):
        return False
    
    if not (0 <= config["gain"] <= 20):
        return False
    
    if config["binning"] not in [1, 2, 4]:
        return False
    
    if not (1 <= config["buffer_size"] <= 50):
        return False
    
    if not (1000 <= config["timeout"] <= 30000):
        return False
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # List all available presets
    print("Available camera presets:")
    for name, description in CameraPresets.list_presets().items():
        print(f"  {name}: {description}")
    
    # Test preset retrieval
    print("\nTesting preset retrieval:")
    try:
        fast_config = CameraPresets.get_preset("preview_fast")
        print(f"Fast preview config: {fast_config}")
        
        quality_config = CameraPresets.get_preset("preview_quality")
        print(f"Quality config: {quality_config}")
        
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test custom preset creation
    print("\nTesting custom preset creation:")
    custom_config = CameraPresets.create_custom_preset(
        "preview_balanced",
        preview_exposure=6000,
        gain=1
    )
    print(f"Custom config: {custom_config}")
    
    # Test configuration validation
    print("\nTesting configuration validation:")
    print(f"Fast config valid: {validate_config(fast_config)}")
    print(f"Invalid config valid: {validate_config({'invalid': 'config'})}")
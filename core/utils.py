"""
Core utilities for PCB inspection system.

This module provides common utility functions used throughout the system
including logging setup, image processing helpers, and error handling.
"""

import logging
import logging.handlers
import os
import time
import functools
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import cv2
from pathlib import Path

from .config import LOG_CONFIG, PATHS


def setup_logging(log_level: str = LOG_CONFIG["level"], 
                 log_file: str = LOG_CONFIG["file"]) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(LOG_CONFIG["format"])
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=LOG_CONFIG["max_size"],
        backupCount=LOG_CONFIG["backup_count"]
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (if enabled)
    if LOG_CONFIG["console_output"]:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_timestamp(include_timezone: bool = True) -> str:
    """
    Get current timestamp as ISO format string.
    
    Args:
        include_timezone: Whether to include timezone information
        
    Returns:
        ISO format timestamp string
    """
    if include_timezone:
        return datetime.now(timezone.utc).isoformat()
    else:
        return datetime.now().isoformat()


def get_timestamp_filename() -> str:
    """
    Get timestamp suitable for filenames (no special characters).
    
    Returns:
        Timestamp string safe for filenames
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Union[str, Path]) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: float, denominator: float, 
                default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper


def error_handler(default_return: Any = None, 
                 log_error: bool = True) -> Callable:
    """
    Decorator for graceful error handling.
    
    Args:
        default_return: Default value to return on error
        log_error: Whether to log the error
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator


def validate_image(image: np.ndarray) -> bool:
    """
    Validate if input is a valid image array.
    
    Args:
        image: Input image array
        
    Returns:
        True if valid image, False otherwise
    """
    if not isinstance(image, np.ndarray):
        return False
    
    if image.size == 0:
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False
    
    return True


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image array
        target_size: Target size (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image array
    """
    if not validate_image(image):
        raise ValueError("Invalid image input")
    
    if maintain_aspect:
        # Calculate scaling factor to maintain aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image with target size
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray, target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Normalize image to target range.
    
    Args:
        image: Input image array
        target_range: Target range (min, max)
        
    Returns:
        Normalized image array
    """
    if not validate_image(image):
        raise ValueError("Invalid image input")
    
    # Convert to float
    image_float = image.astype(np.float32)
    
    # Normalize to 0-1 range
    min_val = np.min(image_float)
    max_val = np.max(image_float)
    
    if max_val - min_val > 0:
        normalized = (image_float - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(image_float)
    
    # Scale to target range
    target_min, target_max = target_range
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled


def convert_color_space(image: np.ndarray, conversion: str) -> np.ndarray:
    """
    Convert image color space.
    
    Args:
        image: Input image array
        conversion: Color space conversion (e.g., 'BGR2RGB', 'GRAY2RGB')
        
    Returns:
        Converted image array
    """
    if not validate_image(image):
        raise ValueError("Invalid image input")
    
    conversion_map = {
        'BGR2RGB': cv2.COLOR_BGR2RGB,
        'RGB2BGR': cv2.COLOR_RGB2BGR,
        'BGR2GRAY': cv2.COLOR_BGR2GRAY,
        'RGB2GRAY': cv2.COLOR_RGB2GRAY,
        'GRAY2BGR': cv2.COLOR_GRAY2BGR,
        'GRAY2RGB': cv2.COLOR_GRAY2RGB,
        'BAYER_RG2RGB': cv2.COLOR_BAYER_RG2RGB,
        'BAYER_RG2GRAY': cv2.COLOR_BAYER_RG2GRAY,
    }
    
    if conversion not in conversion_map:
        raise ValueError(f"Unsupported conversion: {conversion}")
    
    return cv2.cvtColor(image, conversion_map[conversion])


def calculate_focus_score(image: np.ndarray, method: str = "laplacian") -> float:
    """
    Calculate focus score for an image.
    
    Args:
        image: Input image array (grayscale)
        method: Focus calculation method ('laplacian', 'sobel', 'tenengrad')
        
    Returns:
        Focus score (higher = better focus)
    """
    if not validate_image(image):
        return 0.0
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if method == "laplacian":
        # Laplacian variance method
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    elif method == "sobel":
        # Sobel gradient method
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        return sobel.mean()
    
    elif method == "tenengrad":
        # Tenengrad method
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.sum(gradient**2)
    
    else:
        raise ValueError(f"Unsupported focus calculation method: {method}")


def create_performance_monitor():
    """
    Create a simple performance monitoring context manager.
    
    Returns:
        Performance monitoring context manager
    """
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.logger = logging.getLogger(__name__)
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.logger.debug(f"Operation completed in {duration:.4f} seconds")
    
    return PerformanceMonitor()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} GB"


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import psutil
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": format_file_size(psutil.virtual_memory().total),
        "memory_available": format_file_size(psutil.virtual_memory().available),
        "disk_usage": format_file_size(psutil.disk_usage('/').total),
        "timestamp": get_timestamp(),
    }
    
    # Try to get GPU information
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = format_file_size(torch.cuda.get_device_properties(0).total_memory)
        else:
            info["gpu_available"] = False
    except ImportError:
        info["gpu_available"] = "torch_not_installed"
    
    return info


# Initialize logging on module import
logger = setup_logging()
"""
Image preprocessing module for PCB inspection system.

This module handles conversion of raw Bayer pattern images to processed
grayscale images suitable for AI detection. Includes debayering, contrast
enhancement, and noise reduction.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging
from core.interfaces import BaseProcessor
from core.config import PROCESSING_CONFIG


class ImagePreprocessor(BaseProcessor):
    """
    Image preprocessor for raw Bayer pattern images.
    
    Handles debayering, contrast enhancement, and noise reduction
    for both preview and high-quality inspection modes.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the image preprocessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or PROCESSING_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Debayer method mapping
        self.debayer_methods = {
            "bilinear": cv2.COLOR_BAYER_RG2GRAY,
            "edgesense": cv2.COLOR_BAYER_RG2GRAY,  # OpenCV uses same constant
            "vng": cv2.COLOR_BAYER_RG2GRAY,
        }
        
        # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config["clahe_clip_limit"],
            tileGridSize=self.config["clahe_tile_size"]
        )
        
        self.logger.info(f"ImagePreprocessor initialized with {self.config['debayer_method']} debayering")
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process raw Bayer image for high-quality inspection.
        
        Args:
            data: Raw Bayer pattern image
            
        Returns:
            Processed grayscale image
        """
        if data is None:
            self.logger.warning("Received None data for processing")
            return None
        
        try:
            # Full quality processing pipeline
            gray = self.debayer_full_quality(data)
            
            if self.config["contrast_enhancement"]:
                gray = self.enhance_contrast(gray)
            
            if self.config["noise_reduction"]:
                gray = self.reduce_noise(gray)
            
            return gray
            
        except Exception as e:
            self.logger.error(f"Error in image processing: {str(e)}")
            return None
    
    def process_preview(self, data: np.ndarray) -> np.ndarray:
        """
        Fast processing for preview stream.
        
        Args:
            data: Raw Bayer pattern image
            
        Returns:
            Processed grayscale image optimized for preview
        """
        if data is None:
            return None
        
        try:
            # Fast debayer for preview (just extract green channel)
            gray = self.debayer_fast(data)
            
            # Optional light enhancement for preview
            if self.config["contrast_enhancement"]:
                gray = self.enhance_contrast_light(gray)
            
            return gray
            
        except Exception as e:
            self.logger.error(f"Error in preview processing: {str(e)}")
            return None
    
    def debayer_full_quality(self, bayer_image: np.ndarray) -> np.ndarray:
        """
        Full quality debayering from Bayer pattern to grayscale.
        
        Args:
            bayer_image: Raw Bayer pattern image (BayerRG8)
            
        Returns:
            High-quality grayscale image
        """
        # Get debayer method
        debayer_code = self.debayer_methods.get(
            self.config["debayer_method"], 
            cv2.COLOR_BAYER_RG2GRAY
        )
        
        # Convert Bayer to grayscale
        gray = cv2.cvtColor(bayer_image, debayer_code)
        
        return gray
    
    def debayer_fast(self, bayer_image: np.ndarray) -> np.ndarray:
        """
        Fast debayering for preview by extracting green channel.
        
        For BayerRG8 pattern:
        R G R G ...
        G B G B ...
        
        Green pixels contain most luminance information.
        
        Args:
            bayer_image: Raw Bayer pattern image
            
        Returns:
            Fast-processed grayscale image
        """
        # Extract green channel (positions 0,1 and 1,0 in RGGB pattern)
        height, width = bayer_image.shape
        
        # Create output image
        green_channel = np.zeros((height // 2, width // 2), dtype=np.uint8)
        
        # Extract green pixels efficiently
        # For BayerRG: Green is at (0,1) and (1,0)
        green_1 = bayer_image[0::2, 1::2]  # G positions in R-G rows
        green_2 = bayer_image[1::2, 0::2]  # G positions in G-B rows
        
        # Average the two green channels
        green_channel = ((green_1.astype(np.float32) + green_2.astype(np.float32)) / 2).astype(np.uint8)
        
        # Resize back to original dimensions
        gray = cv2.resize(green_channel, (width, height), interpolation=cv2.INTER_LINEAR)
        
        return gray
    
    def process_raw(self, bayer_image: np.ndarray) -> np.ndarray:
        """
        Process raw Bayer image with full quality pipeline.
        
        This is an alias for the complete processing pipeline starting
        from raw Bayer data to enhanced grayscale output.
        
        Args:
            bayer_image: Raw Bayer pattern image
            
        Returns:
            Fully processed grayscale image
        """
        # Full quality debayering
        gray = self.debayer_full_quality(bayer_image)
        
        # Enhance contrast
        enhanced = self.enhance_contrast(gray)
        
        # Reduce noise while preserving edges
        denoised = self.reduce_noise(enhanced)
        
        return denoised
    
    def debayer_to_gray(self, bayer_image: np.ndarray) -> np.ndarray:
        """
        Simple debayer to grayscale conversion.
        
        Alias for fast debayering for backward compatibility.
        
        Args:
            bayer_image: Raw Bayer pattern image
            
        Returns:
            Grayscale image
        """
        return self.debayer_fast(bayer_image)
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Contrast-enhanced image
        """
        if image is None:
            return None
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        enhanced = self.clahe.apply(image)
        
        return enhanced
    
    def enhance_contrast_light(self, image: np.ndarray) -> np.ndarray:
        """
        Light contrast enhancement for preview.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Lightly enhanced image
        """
        if image is None:
            return None
        
        # Simple histogram stretching
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val > min_val:
            # Stretch to full range
            enhanced = ((image - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        else:
            enhanced = image
        
        return enhanced
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce noise while preserving edges using bilateral filtering.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Noise-reduced image
        """
        if image is None:
            return None
        
        # Apply bilateral filter for noise reduction
        denoised = cv2.bilateralFilter(
            image,
            d=self.config["bilateral_d"],
            sigmaColor=self.config["bilateral_sigma_color"],
            sigmaSpace=self.config["bilateral_sigma_space"]
        )
        
        return denoised
    
    def resize_for_preview(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image for preview display.
        
        Args:
            image: Input image
            
        Returns:
            Resized image for preview
        """
        if image is None:
            return None
        
        target_size = self.config["preview_resolution"]
        
        # Calculate aspect ratio preserving resize
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def prepare_for_ai(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for AI model input.
        
        Args:
            image: Processed grayscale image
            
        Returns:
            AI-ready image (normalized, etc.)
        """
        if image is None:
            return None
        
        # Normalize to 0-1 range
        normalized = image.astype(np.float32) / 255.0
        
        # Convert to 3-channel for YOLO (expects RGB input)
        if len(normalized.shape) == 2:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        
        return normalized
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """
        Get image statistics for quality assessment.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image statistics
        """
        if image is None:
            return {}
        
        stats = {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min": int(np.min(image)),
            "max": int(np.max(image)),
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "median": float(np.median(image))
        }
        
        return stats
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate image data.
        
        Args:
            image: Input image
            
        Returns:
            True if image is valid
        """
        if image is None:
            return False
        
        # Check if image has valid dimensions
        if len(image.shape) not in [2, 3]:
            return False
        
        # Check if image has reasonable size
        if image.shape[0] < 10 or image.shape[1] < 10:
            return False
        
        # Check if image has valid data type
        if image.dtype not in [np.uint8, np.uint16, np.float32]:
            return False
        
        return True


class FocusEvaluator:
    """
    Focus quality evaluator for images.
    
    Uses multiple methods to assess image focus quality for auto-trigger.
    """
    
    def __init__(self):
        """Initialize focus evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, image: np.ndarray) -> float:
        """
        Evaluate focus quality of image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Focus score (higher = better focus)
        """
        if image is None:
            return 0.0
        
        try:
            # Use Laplacian variance method
            return self.laplacian_variance(image)
        except Exception as e:
            self.logger.error(f"Focus evaluation error: {str(e)}")
            return 0.0
    
    def laplacian_variance(self, image: np.ndarray) -> float:
        """
        Calculate focus score using Laplacian variance.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Laplacian variance score
        """
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # Calculate variance
        variance = laplacian.var()
        
        return float(variance)
    
    def gradient_magnitude(self, image: np.ndarray) -> float:
        """
        Calculate focus score using gradient magnitude.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Gradient magnitude score
        """
        # Calculate gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return float(np.mean(magnitude))
    
    def is_acceptable(self, score: float, threshold: float = 100.0) -> bool:
        """
        Check if focus score is acceptable.
        
        Args:
            score: Focus score
            threshold: Minimum acceptable score
            
        Returns:
            True if focus is acceptable
        """
        return score >= threshold
    
    def get_focus_level(self, score: float) -> str:
        """
        Get focus level description.
        
        Args:
            score: Focus score
            
        Returns:
            Focus level string
        """
        if score >= 200:
            return "Excellent"
        elif score >= 150:
            return "Good"
        elif score >= 100:
            return "Acceptable"
        elif score >= 50:
            return "Poor"
        else:
            return "Very Poor"


# Utility functions for image preprocessing
def create_test_bayer_image(size: Tuple[int, int] = (1024, 768)) -> np.ndarray:
    """
    Create a test Bayer pattern image for testing.
    
    Args:
        size: Image size (width, height)
        
    Returns:
        Test Bayer pattern image
    """
    height, width = size[1], size[0]
    bayer = np.zeros((height, width), dtype=np.uint8)
    
    # Create RGGB pattern
    for y in range(height):
        for x in range(width):
            if y % 2 == 0:  # Even rows: R G R G ...
                if x % 2 == 0:
                    bayer[y, x] = 255  # Red
                else:
                    bayer[y, x] = 200  # Green
            else:  # Odd rows: G B G B ...
                if x % 2 == 0:
                    bayer[y, x] = 200  # Green
                else:
                    bayer[y, x] = 100  # Blue
    
    return bayer


def benchmark_preprocessing(preprocessor: ImagePreprocessor, 
                          test_image: np.ndarray, 
                          iterations: int = 100) -> dict:
    """
    Benchmark preprocessing performance.
    
    Args:
        preprocessor: ImagePreprocessor instance
        test_image: Test image
        iterations: Number of iterations
        
    Returns:
        Performance metrics
    """
    import time
    
    # Benchmark full quality processing
    start_time = time.time()
    for _ in range(iterations):
        _ = preprocessor.process(test_image)
    full_time = time.time() - start_time
    
    # Benchmark preview processing
    start_time = time.time()
    for _ in range(iterations):
        _ = preprocessor.process_preview(test_image)
    preview_time = time.time() - start_time
    
    results = {
        "full_processing_time": full_time / iterations,
        "preview_processing_time": preview_time / iterations,
        "full_fps": iterations / full_time,
        "preview_fps": iterations / preview_time,
        "speedup_factor": full_time / preview_time
    }
    
    return results


if __name__ == "__main__":
    # Test the preprocessor
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test image
    test_bayer = create_test_bayer_image((1024, 768))
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Test processing
    processed = preprocessor.process(test_bayer)
    preview = preprocessor.process_preview(test_bayer)
    
    # Test focus evaluation
    evaluator = FocusEvaluator()
    focus_score = evaluator.evaluate(processed)
    
    print(f"Test image shape: {test_bayer.shape}")
    print(f"Processed image shape: {processed.shape if processed is not None else 'None'}")
    print(f"Preview image shape: {preview.shape if preview is not None else 'None'}")
    print(f"Focus score: {focus_score:.2f}")
    print(f"Focus level: {evaluator.get_focus_level(focus_score)}")
    
    # Benchmark performance
    if processed is not None:
        benchmark_results = benchmark_preprocessing(preprocessor, test_bayer, 50)
        print("\nBenchmark Results:")
        for key, value in benchmark_results.items():
            print(f"  {key}: {value:.4f}")
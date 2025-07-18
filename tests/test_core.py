"""
Unit tests for Core layer components.

Tests configuration validation, utilities, and interfaces.
"""

import unittest
import tempfile
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    validate_config, ensure_directories, get_config,
    CAMERA_CONFIG, AI_CONFIG, TRIGGER_CONFIG, DB_CONFIG,
    DEFECT_CLASSES, MODEL_CLASS_MAPPING
)
from core.utils import (
    setup_logging, get_timestamp, validate_image, resize_image,
    normalize_image, calculate_focus_score, TimestampUtil, ErrorHandler
)
from core.interfaces import (
    BaseProcessor, BaseDetector, BaseAnalyzer, BaseCamera,
    PCBDetectionResult, InspectionResult
)


class TestCoreConfig(unittest.TestCase):
    """Test configuration module."""
    
    def test_config_structure(self):
        """Test that all required config sections exist."""
        self.assertIsInstance(CAMERA_CONFIG, dict)
        self.assertIsInstance(AI_CONFIG, dict)
        self.assertIsInstance(TRIGGER_CONFIG, dict)
        self.assertIsInstance(DB_CONFIG, dict)
        
        # Test required camera config keys
        required_camera_keys = ['preview_exposure', 'capture_exposure', 'gain', 'pixel_format']
        for key in required_camera_keys:
            self.assertIn(key, CAMERA_CONFIG)
        
        # Test required AI config keys
        required_ai_keys = ['model_path', 'confidence', 'device']
        for key in required_ai_keys:
            self.assertIn(key, AI_CONFIG)
    
    def test_defect_classes(self):
        """Test defect classes configuration."""
        self.assertIsInstance(DEFECT_CLASSES, list)
        self.assertEqual(len(DEFECT_CLASSES), 6)
        self.assertIn("Missing Hole", DEFECT_CLASSES)
        self.assertIn("Mouse Bite", DEFECT_CLASSES)
    
    def test_model_class_mapping(self):
        """Test model class mapping."""
        self.assertIsInstance(MODEL_CLASS_MAPPING, dict)
        self.assertEqual(len(MODEL_CLASS_MAPPING), 6)
        
        # Test that all mapped classes are in DEFECT_CLASSES
        for class_name in MODEL_CLASS_MAPPING.values():
            self.assertIn(class_name, DEFECT_CLASSES)
    
    @patch('os.path.exists')
    def test_config_validation(self, mock_exists):
        """Test configuration validation."""
        # Mock model file exists
        mock_exists.return_value = True
        
        errors = validate_config()
        self.assertIsInstance(errors, list)
        
        # With valid config, should have no errors
        if not errors:
            self.assertEqual(len(errors), 0)
    
    def test_get_config(self):
        """Test configuration getter."""
        camera_config = get_config('camera')
        self.assertEqual(camera_config, CAMERA_CONFIG)
        
        ai_config = get_config('ai')
        self.assertEqual(ai_config, AI_CONFIG)
        
        # Test invalid category
        invalid_config = get_config('invalid')
        self.assertEqual(invalid_config, {})
    
    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                ensure_directories()
                
                # Check that required directories were created
                required_dirs = ['data/images', 'data/defects', 'logs', 'weights']
                for dir_path in required_dirs:
                    self.assertTrue(os.path.exists(dir_path))
                    self.assertTrue(os.path.isdir(dir_path))
            finally:
                os.chdir(original_cwd)


class TestCoreUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')
            logger = setup_logging('TestLogger', 'INFO', log_file)
            
            self.assertIsNotNone(logger)
            logger.info("Test message")
            
            # Check log file was created
            self.assertTrue(os.path.exists(log_file))
    
    def test_timestamp_util(self):
        """Test timestamp utilities."""
        timestamp = TimestampUtil.get_current()
        self.assertIsInstance(timestamp, str)
        self.assertIn('T', timestamp)  # ISO format
        
        filename_timestamp = TimestampUtil.get_filename_safe()
        self.assertIsInstance(filename_timestamp, str)
        self.assertNotIn(':', filename_timestamp)  # Safe for filenames
    
    def test_validate_image(self):
        """Test image validation."""
        # Valid grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.assertTrue(validate_image(gray_image))
        
        # Valid color image
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertTrue(validate_image(color_image))
        
        # Invalid inputs
        self.assertFalse(validate_image(None))
        self.assertFalse(validate_image([1, 2, 3]))
        self.assertFalse(validate_image(np.array([])))
        self.assertFalse(validate_image(np.random.rand(100, 100, 100, 100)))
    
    def test_resize_image(self):
        """Test image resizing."""
        image = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        
        # Test resize without maintaining aspect ratio
        resized = resize_image(image, (150, 150), maintain_aspect=False)
        self.assertEqual(resized.shape, (150, 150))
        
        # Test resize with maintaining aspect ratio
        resized_aspect = resize_image(image, (150, 150), maintain_aspect=True)
        self.assertEqual(resized_aspect.shape, (150, 150))
        
        # Test invalid image
        with self.assertRaises(ValueError):
            resize_image(None, (100, 100))
    
    def test_normalize_image(self):
        """Test image normalization."""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Test default normalization (0-1)
        normalized = normalize_image(image)
        self.assertGreaterEqual(normalized.min(), 0)
        self.assertLessEqual(normalized.max(), 1)
        
        # Test custom range
        normalized_custom = normalize_image(image, target_range=(-1, 1))
        self.assertGreaterEqual(normalized_custom.min(), -1)
        self.assertLessEqual(normalized_custom.max(), 1)
    
    def test_calculate_focus_score(self):
        """Test focus score calculation."""
        # Create a sharp image (high frequency content)
        sharp_image = np.zeros((100, 100), dtype=np.uint8)
        sharp_image[::2, ::2] = 255  # Checkerboard pattern
        
        # Create a blurred image (low frequency content)
        blurred_image = np.ones((100, 100), dtype=np.uint8) * 128
        
        sharp_score = calculate_focus_score(sharp_image, method='laplacian')
        blurred_score = calculate_focus_score(blurred_image, method='laplacian')
        
        # Sharp image should have higher focus score
        self.assertGreater(sharp_score, blurred_score)
        
        # Test different methods
        sobel_score = calculate_focus_score(sharp_image, method='sobel')
        tenengrad_score = calculate_focus_score(sharp_image, method='tenengrad')
        
        self.assertIsInstance(sobel_score, float)
        self.assertIsInstance(tenengrad_score, float)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            calculate_focus_score(sharp_image, method='invalid')
    
    def test_error_handler(self):
        """Test error handling decorator."""
        @ErrorHandler.log_exceptions
        def test_function():
            raise ValueError("Test error")
        
        # Should not suppress the exception
        with self.assertRaises(ValueError):
            test_function()


class TestCoreInterfaces(unittest.TestCase):
    """Test interface classes."""
    
    def test_pcb_detection_result(self):
        """Test PCBDetectionResult data class."""
        result = PCBDetectionResult(
            has_pcb=True,
            position=(100, 100, 200, 200),
            is_stable=True,
            focus_score=150.0
        )
        
        self.assertTrue(result.has_pcb)
        self.assertEqual(result.position, (100, 100, 200, 200))
        self.assertTrue(result.is_stable)
        self.assertEqual(result.focus_score, 150.0)
    
    def test_inspection_result(self):
        """Test InspectionResult data class."""
        defects = ["Missing Hole", "Spur"]
        locations = [{"bbox": [10, 10, 50, 50]}, {"bbox": [100, 100, 150, 150]}]
        confidences = [0.9, 0.8]
        
        result = InspectionResult(
            defects=defects,
            locations=locations,
            confidence_scores=confidences,
            processing_time=0.5
        )
        
        self.assertEqual(result.defects, defects)
        self.assertEqual(result.locations, locations)
        self.assertEqual(result.confidence_scores, confidences)
        self.assertEqual(result.processing_time, 0.5)
        self.assertTrue(result.has_defects)
        
        # Test empty result
        empty_result = InspectionResult([], [], [], 0.1)
        self.assertFalse(empty_result.has_defects)
    
    def test_base_interfaces(self):
        """Test that base interfaces are abstract."""
        # These should raise TypeError when instantiated
        with self.assertRaises(TypeError):
            BaseProcessor()
        
        with self.assertRaises(TypeError):
            BaseDetector()
        
        with self.assertRaises(TypeError):
            BaseAnalyzer()
        
        with self.assertRaises(TypeError):
            BaseCamera()


class TestMockImplementations(unittest.TestCase):
    """Test mock implementations of interfaces for testing."""
    
    def test_mock_processor(self):
        """Test mock processor implementation."""
        class MockProcessor(BaseProcessor):
            def process(self, data):
                return data * 2
        
        processor = MockProcessor()
        result = processor.process(np.array([1, 2, 3]))
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))
    
    def test_mock_detector(self):
        """Test mock detector implementation."""
        class MockDetector(BaseDetector):
            def detect(self, image):
                return {"detections": []}
        
        detector = MockDetector()
        result = detector.detect(np.zeros((100, 100)))
        self.assertEqual(result, {"detections": []})


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    unittest.main(verbosity=2)
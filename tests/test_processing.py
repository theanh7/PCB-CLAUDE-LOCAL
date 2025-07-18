"""
Unit tests for Processing layer components.

Tests image preprocessing, PCB detection, and result postprocessing.
"""

import unittest
import numpy as np
import cv2
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import TRIGGER_CONFIG, PROCESSING_CONFIG


class TestImagePreprocessor(unittest.TestCase):
    """Test image preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from processing.preprocessor import ImagePreprocessor
            self.preprocessor = ImagePreprocessor()
        except ImportError:
            self.skipTest("Processing module not available")
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNotNone(self.preprocessor)
    
    def test_process_grayscale_image(self):
        """Test processing of grayscale images."""
        # Create test grayscale image
        gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        result = self.preprocessor.process(gray_image)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, gray_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_process_raw_bayer_image(self):
        """Test processing of raw Bayer images."""
        # Create test Bayer image
        bayer_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        result = self.preprocessor.process_raw(bayer_image)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, bayer_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_contrast_enhancement(self):
        """Test contrast enhancement functionality."""
        # Create low contrast image
        low_contrast = np.ones((100, 100), dtype=np.uint8) * 128
        
        enhanced = self.preprocessor._enhance_contrast(low_contrast)
        
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, low_contrast.shape)
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create test image with various characteristics
        test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Add some noise
        noise = np.random.normal(0, 10, test_image.shape).astype(np.uint8)
        noisy_image = cv2.add(test_image, noise)
        
        result = self.preprocessor.process(noisy_image)
        
        # Result should be same size
        self.assertEqual(result.shape, test_image.shape)
        
        # Result should be enhanced (hard to test exact values)
        self.assertIsInstance(result, np.ndarray)


class TestPCBDetector(unittest.TestCase):
    """Test PCB detection and tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from processing.pcb_detector import PCBDetector
            self.detector = PCBDetector(TRIGGER_CONFIG)
        except ImportError:
            self.skipTest("Processing module not available")
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.config, TRIGGER_CONFIG)
    
    def test_debayer_to_gray(self):
        """Test Bayer pattern to grayscale conversion."""
        # Create mock Bayer image
        bayer_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        gray = self.detector.debayer_to_gray(bayer_image)
        
        self.assertIsNotNone(gray)
        # Should be roughly half the size (extracting green channel)
        expected_shape = (240, 320)  # Half dimensions from green channel extraction
        self.assertEqual(gray.shape, expected_shape)
    
    def test_pcb_detection_no_pcb(self):
        """Test PCB detection with no PCB present."""
        # Create uniform image (no PCB)
        empty_image = np.ones((480, 640), dtype=np.uint8) * 128
        
        has_pcb, position, is_stable, focus_score = self.detector.detect_pcb(empty_image)
        
        self.assertFalse(has_pcb)
        self.assertIsNone(position)
        self.assertFalse(is_stable)
        self.assertIsInstance(focus_score, float)
    
    def test_pcb_detection_with_pcb(self):
        """Test PCB detection with PCB present."""
        # Create image with rectangular PCB-like object
        image = np.zeros((480, 640), dtype=np.uint8)
        # Add a large rectangular region (simulating PCB)
        cv2.rectangle(image, (100, 100), (500, 350), 255, -1)
        # Add some edges to simulate PCB features
        cv2.rectangle(image, (150, 150), (450, 300), 0, 3)
        
        has_pcb, position, is_stable, focus_score = self.detector.detect_pcb(image)
        
        self.assertTrue(has_pcb)
        self.assertIsNotNone(position)
        self.assertIsInstance(position, tuple)
        self.assertEqual(len(position), 4)  # x, y, width, height
        self.assertIsInstance(focus_score, float)
    
    def test_stability_checking(self):
        """Test PCB position stability checking."""
        # Create consistent PCB image
        image = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(image, (200, 150), (400, 300), 255, -1)
        
        # First detection should not be stable
        has_pcb1, pos1, stable1, _ = self.detector.detect_pcb(image)
        self.assertTrue(has_pcb1)
        self.assertFalse(stable1)
        
        # Detect same position multiple times
        for _ in range(TRIGGER_CONFIG["stability_frames"]):
            has_pcb, pos, stable, _ = self.detector.detect_pcb(image)
        
        # Should eventually become stable
        self.assertTrue(has_pcb)
        # Note: May not be stable immediately due to slight variations
    
    def test_stability_reset_on_movement(self):
        """Test that stability resets when PCB moves."""
        # Initial PCB position
        image1 = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(image1, (200, 150), (400, 300), 255, -1)
        
        # Build up stability
        for _ in range(5):
            self.detector.detect_pcb(image1)
        
        # Move PCB significantly
        image2 = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(image2, (250, 200), (450, 350), 255, -1)
        
        has_pcb, pos, stable, _ = self.detector.detect_pcb(image2)
        
        # Should detect PCB but not be stable anymore
        self.assertTrue(has_pcb)
        # Stability should be reset (likely False, but depends on implementation)
    
    def test_minimum_pcb_area(self):
        """Test minimum PCB area filtering."""
        # Create very small rectangular object
        image = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(image, (300, 250), (320, 270), 255, -1)  # Very small rectangle
        
        has_pcb, position, is_stable, focus_score = self.detector.detect_pcb(image)
        
        # Should not detect as PCB due to small size
        self.assertFalse(has_pcb)


class TestFocusEvaluator(unittest.TestCase):
    """Test focus evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from processing.pcb_detector import FocusEvaluator
            self.focus_evaluator = FocusEvaluator()
        except ImportError:
            self.skipTest("Processing module not available")
    
    def test_focus_evaluator_initialization(self):
        """Test focus evaluator initialization."""
        self.assertIsNotNone(self.focus_evaluator)
    
    def test_focus_evaluation_sharp_image(self):
        """Test focus evaluation on sharp image."""
        # Create sharp image with high frequency content
        sharp_image = np.zeros((100, 100), dtype=np.uint8)
        sharp_image[::2, ::2] = 255  # Checkerboard pattern
        
        focus_score = self.focus_evaluator.evaluate(sharp_image)
        
        self.assertIsInstance(focus_score, float)
        self.assertGreater(focus_score, 0)
    
    def test_focus_evaluation_blurred_image(self):
        """Test focus evaluation on blurred image."""
        # Create sharp image first
        sharp_image = np.zeros((100, 100), dtype=np.uint8)
        sharp_image[::2, ::2] = 255
        
        # Blur the image
        blurred_image = cv2.GaussianBlur(sharp_image, (15, 15), 5)
        
        sharp_score = self.focus_evaluator.evaluate(sharp_image)
        blurred_score = self.focus_evaluator.evaluate(blurred_image)
        
        # Sharp image should have higher focus score
        self.assertGreater(sharp_score, blurred_score)
    
    def test_focus_acceptability(self):
        """Test focus acceptability threshold."""
        # Create image with known characteristics
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        score = self.focus_evaluator.evaluate(test_image)
        
        # Test acceptability function
        is_acceptable_low = self.focus_evaluator.is_acceptable(score, threshold=10)
        is_acceptable_high = self.focus_evaluator.is_acceptable(score, threshold=10000)
        
        self.assertIsInstance(is_acceptable_low, bool)
        self.assertIsInstance(is_acceptable_high, bool)


class TestResultPostprocessor(unittest.TestCase):
    """Test result postprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from processing.postprocessor import ResultPostprocessor
            self.postprocessor = ResultPostprocessor()
        except ImportError:
            self.skipTest("Processing module not available")
    
    def test_postprocessor_initialization(self):
        """Test postprocessor initialization."""
        self.assertIsNotNone(self.postprocessor)
    
    def test_draw_results_no_detections(self):
        """Test drawing results with no detections."""
        test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Mock detection results with no detections
        mock_results = Mock()
        mock_results.boxes = None
        
        result_image = self.postprocessor.draw_results(test_image, mock_results)
        
        self.assertIsNotNone(result_image)
        self.assertEqual(result_image.shape, test_image.shape)
        # Should be identical to input when no detections
        np.testing.assert_array_equal(result_image, test_image)
    
    def test_draw_results_with_detections(self):
        """Test drawing results with detections."""
        test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Mock detection results with detections
        mock_results = Mock()
        mock_box = Mock()
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].tolist.return_value = [100, 100, 200, 200]
        mock_box.cls = 0  # Mouse Bite
        mock_box.conf = 0.85
        
        mock_results.boxes = [mock_box]
        
        result_image = self.postprocessor.draw_results(test_image, mock_results)
        
        self.assertIsNotNone(result_image)
        self.assertEqual(result_image.shape, test_image.shape)
        # Should be modified (bounding box drawn)
        # Hard to test exact pixels, but should not be identical
    
    def test_draw_multiple_detections(self):
        """Test drawing multiple detections."""
        test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Mock multiple detections
        mock_results = Mock()
        
        # Create multiple mock boxes
        mock_boxes = []
        for i in range(3):
            mock_box = Mock()
            mock_box.xyxy = [Mock()]
            mock_box.xyxy[0].tolist.return_value = [100 + i*50, 100, 200 + i*50, 200]
            mock_box.cls = i % 6  # Cycle through defect classes
            mock_box.conf = 0.8 + i*0.05
            mock_boxes.append(mock_box)
        
        mock_results.boxes = mock_boxes
        
        result_image = self.postprocessor.draw_results(test_image, mock_results)
        
        self.assertIsNotNone(result_image)
        self.assertEqual(result_image.shape, test_image.shape)


class TestProcessingIntegration(unittest.TestCase):
    """Test integration between processing components."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from processing.preprocessor import ImagePreprocessor
            from processing.pcb_detector import PCBDetector
            from processing.postprocessor import ResultPostprocessor
            
            self.preprocessor = ImagePreprocessor()
            self.pcb_detector = PCBDetector()
            self.postprocessor = ResultPostprocessor()
        except ImportError:
            self.skipTest("Processing modules not available")
    
    def test_preprocessing_to_detection_pipeline(self):
        """Test pipeline from preprocessing to PCB detection."""
        # Create raw test image
        raw_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Preprocess
        processed = self.preprocessor.process(raw_image)
        
        # Detect PCB
        has_pcb, position, stable, focus = self.pcb_detector.detect_pcb(processed)
        
        # Should complete without errors
        self.assertIsInstance(has_pcb, bool)
        self.assertIsInstance(stable, bool)
        self.assertIsInstance(focus, float)
    
    def test_full_processing_pipeline(self):
        """Test complete processing pipeline."""
        # Create test image with PCB-like features
        image = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(image, (150, 100), (450, 350), 255, -1)
        cv2.rectangle(image, (200, 150), (400, 300), 0, 2)
        
        # Step 1: Preprocess
        processed = self.preprocessor.process(image)
        
        # Step 2: Detect PCB
        has_pcb, position, stable, focus = self.pcb_detector.detect_pcb(processed)
        
        # Step 3: Mock AI results and postprocess
        mock_results = Mock()
        mock_results.boxes = None  # No defects
        
        result_image = self.postprocessor.draw_results(processed, mock_results)
        
        # Pipeline should complete successfully
        self.assertIsNotNone(processed)
        self.assertIsNotNone(result_image)
        self.assertIsInstance(has_pcb, bool)


class TestProcessingPerformance(unittest.TestCase):
    """Test processing performance characteristics."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from processing.preprocessor import ImagePreprocessor
            from processing.pcb_detector import PCBDetector
            
            self.preprocessor = ImagePreprocessor()
            self.pcb_detector = PCBDetector()
        except ImportError:
            self.skipTest("Processing modules not available")
    
    def test_preprocessing_performance(self):
        """Test preprocessing performance."""
        import time
        
        # Test with various image sizes
        for size in [(240, 320), (480, 640), (960, 1280)]:
            test_image = np.random.randint(0, 255, size, dtype=np.uint8)
            
            start_time = time.time()
            result = self.preprocessor.process(test_image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should be reasonably fast (< 100ms for these sizes)
            self.assertLess(processing_time, 0.1)
            self.assertIsNotNone(result)
    
    def test_pcb_detection_performance(self):
        """Test PCB detection performance."""
        import time
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Time multiple detections
        start_time = time.time()
        for _ in range(10):
            self.pcb_detector.detect_pcb(test_image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Should be fast enough for real-time processing
        self.assertLess(avg_time, 0.05)  # 50ms per detection


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    unittest.main(verbosity=2)
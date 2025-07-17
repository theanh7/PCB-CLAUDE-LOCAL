"""
Test suite for processing pipeline components.

This module provides comprehensive testing for the image processing pipeline
including preprocessor, PCB detector, focus evaluator, and postprocessor.
"""

import cv2
import numpy as np
import time
import unittest
from typing import Tuple, List
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.preprocessor import ImagePreprocessor, FocusEvaluator
from processing.pcb_detector import PCBDetector, AutoTriggerSystem, PCBPosition
from processing.postprocessor import ResultPostprocessor, DetectionBox
from core.config import PROCESSING_CONFIG, TRIGGER_CONFIG


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        self.test_bayer = self._create_test_bayer_image()
        self.test_gray = self._create_test_gray_image()
    
    def _create_test_bayer_image(self, size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """Create test Bayer pattern image."""
        height, width = size
        bayer = np.zeros((height, width), dtype=np.uint8)
        
        # Create RGGB pattern
        for y in range(height):
            for x in range(width):
                if y % 2 == 0:  # Even rows: R G R G
                    if x % 2 == 0:
                        bayer[y, x] = 200  # Red
                    else:
                        bayer[y, x] = 255  # Green
                else:  # Odd rows: G B G B
                    if x % 2 == 0:
                        bayer[y, x] = 255  # Green
                    else:
                        bayer[y, x] = 100  # Blue
        
        return bayer
    
    def _create_test_gray_image(self, size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """Create test grayscale image."""
        height, width = size
        return np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    def test_validate_image(self):
        """Test image validation."""
        # Valid images
        self.assertTrue(self.preprocessor.validate_image(self.test_bayer))
        self.assertTrue(self.preprocessor.validate_image(self.test_gray))
        
        # Invalid images
        self.assertFalse(self.preprocessor.validate_image(None))
        self.assertFalse(self.preprocessor.validate_image(np.array([1, 2, 3])))  # 1D
        self.assertFalse(self.preprocessor.validate_image(np.zeros((5, 5))))  # Too small
    
    def test_debayer_full_quality(self):
        """Test full quality debayering."""
        result = self.preprocessor.debayer_full_quality(self.test_bayer)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.shape), 2)  # Should be grayscale
        self.assertEqual(result.shape, self.test_bayer.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_debayer_fast(self):
        """Test fast debayering."""
        result = self.preprocessor.debayer_fast(self.test_bayer)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.shape), 2)  # Should be grayscale
        self.assertEqual(result.shape, self.test_bayer.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_process_full_vs_preview(self):
        """Test difference between full and preview processing."""
        full_result = self.preprocessor.process(self.test_bayer)
        preview_result = self.preprocessor.process_preview(self.test_bayer)
        
        self.assertIsNotNone(full_result)
        self.assertIsNotNone(preview_result)
        self.assertEqual(full_result.shape, preview_result.shape)
        
        # Full processing should be more detailed (different results)
        self.assertFalse(np.array_equal(full_result, preview_result))
    
    def test_enhance_contrast(self):
        """Test contrast enhancement."""
        # Create low contrast image
        low_contrast = np.full((100, 100), 128, dtype=np.uint8)
        enhanced = self.preprocessor.enhance_contrast(low_contrast)
        
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, low_contrast.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
    
    def test_reduce_noise(self):
        """Test noise reduction."""
        # Add noise to image
        noisy = self.test_gray.copy()
        noise = np.random.randint(-20, 20, noisy.shape, dtype=np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        denoised = self.preprocessor.reduce_noise(noisy)
        
        self.assertIsNotNone(denoised)
        self.assertEqual(denoised.shape, noisy.shape)
        self.assertEqual(denoised.dtype, np.uint8)
    
    def test_prepare_for_ai(self):
        """Test AI preparation."""
        ai_ready = self.preprocessor.prepare_for_ai(self.test_gray)
        
        self.assertIsNotNone(ai_ready)
        self.assertEqual(len(ai_ready.shape), 3)  # Should be RGB
        self.assertEqual(ai_ready.shape[2], 3)  # 3 channels
        self.assertEqual(ai_ready.dtype, np.float32)
        self.assertTrue(np.all(ai_ready >= 0) and np.all(ai_ready <= 1))  # Normalized
    
    def test_get_image_stats(self):
        """Test image statistics."""
        stats = self.preprocessor.get_image_stats(self.test_gray)
        
        self.assertIsInstance(stats, dict)
        required_keys = ["shape", "dtype", "min", "max", "mean", "std", "median"]
        for key in required_keys:
            self.assertIn(key, stats)
    
    def test_resize_for_preview(self):
        """Test preview resizing."""
        resized = self.preprocessor.resize_for_preview(self.test_gray)
        
        self.assertIsNotNone(resized)
        self.assertEqual(len(resized.shape), 2)
        
        # Should be smaller than original
        self.assertLessEqual(resized.shape[0], self.test_gray.shape[0])
        self.assertLessEqual(resized.shape[1], self.test_gray.shape[1])


class TestFocusEvaluator(unittest.TestCase):
    """Test cases for FocusEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = FocusEvaluator()
        self.sharp_image = self._create_sharp_image()
        self.blurry_image = self._create_blurry_image()
    
    def _create_sharp_image(self, size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """Create sharp test image."""
        height, width = size
        image = np.zeros((height, width), dtype=np.uint8)
        
        # Add sharp edges
        cv2.rectangle(image, (50, 50), (150, 150), 255, 2)
        cv2.line(image, (0, 100), (200, 100), 255, 1)
        cv2.line(image, (100, 0), (100, 200), 255, 1)
        
        return image
    
    def _create_blurry_image(self, size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """Create blurry test image."""
        sharp = self._create_sharp_image(size)
        # Apply heavy blur
        blurry = cv2.GaussianBlur(sharp, (15, 15), 0)
        return blurry
    
    def test_evaluate_focus(self):
        """Test focus evaluation."""
        sharp_score = self.evaluator.evaluate(self.sharp_image)
        blurry_score = self.evaluator.evaluate(self.blurry_image)
        
        self.assertIsInstance(sharp_score, float)
        self.assertIsInstance(blurry_score, float)
        self.assertGreater(sharp_score, blurry_score)
        self.assertGreater(sharp_score, 0)
    
    def test_laplacian_variance(self):
        """Test Laplacian variance method."""
        sharp_score = self.evaluator.laplacian_variance(self.sharp_image)
        blurry_score = self.evaluator.laplacian_variance(self.blurry_image)
        
        self.assertGreater(sharp_score, blurry_score)
        self.assertGreater(sharp_score, 0)
    
    def test_gradient_magnitude(self):
        """Test gradient magnitude method."""
        sharp_score = self.evaluator.gradient_magnitude(self.sharp_image)
        blurry_score = self.evaluator.gradient_magnitude(self.blurry_image)
        
        self.assertGreater(sharp_score, blurry_score)
        self.assertGreater(sharp_score, 0)
    
    def test_is_acceptable(self):
        """Test focus acceptability."""
        sharp_score = self.evaluator.evaluate(self.sharp_image)
        blurry_score = self.evaluator.evaluate(self.blurry_image)
        
        # Test with different thresholds
        self.assertTrue(self.evaluator.is_acceptable(sharp_score, 50))
        self.assertFalse(self.evaluator.is_acceptable(blurry_score, 100))
    
    def test_get_focus_level(self):
        """Test focus level description."""
        levels = [
            (250, "Excellent"),
            (175, "Good"),
            (125, "Acceptable"),
            (75, "Poor"),
            (25, "Very Poor")
        ]
        
        for score, expected_level in levels:
            level = self.evaluator.get_focus_level(score)
            self.assertEqual(level, expected_level)


class TestPCBDetector(unittest.TestCase):
    """Test cases for PCBDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = PCBDetector()
        self.test_image = self._create_test_pcb_image()
        self.empty_image = self._create_empty_image()
    
    def _create_test_pcb_image(self, size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """Create test image with PCB-like object."""
        height, width = size
        image = np.random.randint(30, 70, (height, width), dtype=np.uint8)
        
        # Add PCB rectangle
        pcb_x, pcb_y = 200, 150
        pcb_w, pcb_h = 400, 300
        
        image[pcb_y:pcb_y + pcb_h, pcb_x:pcb_x + pcb_w] = 150
        
        # Add some features
        cv2.rectangle(image, (pcb_x, pcb_y), (pcb_x + pcb_w, pcb_y + pcb_h), 255, 3)
        
        # Add holes
        for i in range(5):
            hole_x = pcb_x + 50 + i * 60
            hole_y = pcb_y + 50 + (i % 2) * 100
            cv2.circle(image, (hole_x, hole_y), 8, 0, -1)
        
        return image
    
    def _create_empty_image(self, size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """Create empty test image."""
        height, width = size
        return np.random.randint(30, 70, (height, width), dtype=np.uint8)
    
    def test_detect_pcb_found(self):
        """Test PCB detection when PCB is present."""
        result = self.detector.detect_pcb(self.test_image)
        
        self.assertTrue(result.has_pcb)
        self.assertIsNotNone(result.position)
        self.assertIsInstance(result.focus_score, float)
        self.assertGreaterEqual(result.focus_score, 0)
    
    def test_detect_pcb_not_found(self):
        """Test PCB detection when PCB is not present."""
        result = self.detector.detect_pcb(self.empty_image)
        
        self.assertFalse(result.has_pcb)
        self.assertIsNone(result.position)
        self.assertFalse(result.is_stable)
        self.assertEqual(result.focus_score, 0.0)
    
    def test_stability_tracking(self):
        """Test stability tracking over multiple frames."""
        # First detection should not be stable
        result1 = self.detector.detect_pcb(self.test_image)
        self.assertFalse(result1.is_stable)
        
        # After enough stable frames, should be stable
        for i in range(TRIGGER_CONFIG["stability_frames"]):
            result = self.detector.detect_pcb(self.test_image)
        
        self.assertTrue(result.is_stable)
    
    def test_stability_reset_on_movement(self):
        """Test stability reset when PCB moves."""
        # Get to stable state
        for i in range(TRIGGER_CONFIG["stability_frames"]):
            result = self.detector.detect_pcb(self.test_image)
        
        self.assertTrue(result.is_stable)
        
        # Create moved image
        moved_image = np.roll(self.test_image, 50, axis=1)  # Move horizontally
        
        # Should reset stability
        result = self.detector.detect_pcb(moved_image)
        self.assertFalse(result.is_stable)
    
    def test_can_trigger_inspection(self):
        """Test inspection trigger timing."""
        # Should be able to trigger initially
        self.assertTrue(self.detector.can_trigger_inspection())
        
        # After triggering, should have cooldown
        self.detector.trigger_inspection()
        self.assertFalse(self.detector.can_trigger_inspection())
        
        # After enough time, should be able to trigger again
        self.detector.last_inspection_time = time.time() - TRIGGER_CONFIG["inspection_interval"] - 1
        self.assertTrue(self.detector.can_trigger_inspection())
    
    def test_should_trigger_inspection(self):
        """Test complete trigger decision logic."""
        # Get to stable state with good focus
        for i in range(TRIGGER_CONFIG["stability_frames"]):
            result = self.detector.detect_pcb(self.test_image)
        
        # Should trigger if all conditions met
        should_trigger = self.detector.should_trigger_inspection(result)
        expected = (result.has_pcb and result.is_stable and 
                   result.focus_score >= TRIGGER_CONFIG["focus_threshold"])
        
        self.assertEqual(should_trigger, expected)
    
    def test_get_stability_info(self):
        """Test stability information."""
        info = self.detector.get_stability_info()
        
        required_keys = ["stable_frames", "required_frames", "stability_progress",
                        "position_history_count", "time_since_last_inspection"]
        
        for key in required_keys:
            self.assertIn(key, info)
    
    def test_get_detection_stats(self):
        """Test detection statistics."""
        # Run some detections
        for i in range(5):
            self.detector.detect_pcb(self.test_image)
        
        stats = self.detector.get_detection_stats()
        
        required_keys = ["total_detections", "average_area", "position_variance", "detection_rate"]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertGreater(stats["total_detections"], 0)
    
    def test_reset_detection_state(self):
        """Test detection state reset."""
        # Run some detections
        for i in range(5):
            self.detector.detect_pcb(self.test_image)
        
        # Reset
        self.detector.reset_detection_state()
        
        # Check state is reset
        self.assertIsNone(self.detector.last_position)
        self.assertEqual(len(self.detector.position_history), 0)
        self.assertEqual(self.detector.stable_frames, 0)
        self.assertEqual(self.detector.last_inspection_time, 0.0)
    
    def test_visualize_detection(self):
        """Test detection visualization."""
        result = self.detector.detect_pcb(self.test_image)
        
        if result.has_pcb:
            vis_image = self.detector.visualize_detection(self.test_image, result)
            
            self.assertIsNotNone(vis_image)
            self.assertEqual(len(vis_image.shape), 3)  # Should be color
            self.assertEqual(vis_image.shape[:2], self.test_image.shape)


class TestAutoTriggerSystem(unittest.TestCase):
    """Test cases for AutoTriggerSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pcb_detector = PCBDetector()
        self.auto_trigger = AutoTriggerSystem(self.pcb_detector)
        self.test_image = self._create_test_pcb_image()
    
    def _create_test_pcb_image(self, size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """Create test image with PCB-like object."""
        height, width = size
        image = np.random.randint(30, 70, (height, width), dtype=np.uint8)
        
        # Add PCB rectangle
        pcb_x, pcb_y = 200, 150
        pcb_w, pcb_h = 400, 300
        
        image[pcb_y:pcb_y + pcb_h, pcb_x:pcb_x + pcb_w] = 150
        cv2.rectangle(image, (pcb_x, pcb_y), (pcb_x + pcb_w, pcb_y + pcb_h), 255, 3)
        
        return image
    
    def test_process_frame(self):
        """Test frame processing."""
        detection_result, should_trigger = self.auto_trigger.process_frame(self.test_image)
        
        self.assertIsNotNone(detection_result)
        self.assertIsInstance(should_trigger, bool)
        self.assertGreater(self.auto_trigger.total_detections, 0)
    
    def test_trigger_statistics(self):
        """Test trigger statistics."""
        # Process some frames
        for i in range(10):
            self.auto_trigger.process_frame(self.test_image)
        
        stats = self.auto_trigger.get_trigger_stats()
        
        required_keys = ["total_detections", "trigger_count", "successful_triggers",
                        "runtime_hours", "detection_rate", "trigger_rate"]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertGreater(stats["total_detections"], 0)
        self.assertGreaterEqual(stats["detection_rate"], 0)


class TestResultPostprocessor(unittest.TestCase):
    """Test cases for ResultPostprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.postprocessor = ResultPostprocessor()
        self.test_image = self._create_test_image()
        self.test_results = self._create_test_results()
    
    def _create_test_image(self, size: Tuple[int, int] = (600, 800)) -> np.ndarray:
        """Create test image."""
        height, width = size
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    def _create_test_results(self):
        """Create mock detection results."""
        class MockBox:
            def __init__(self, coords, confidence, class_id):
                self.xyxy = [coords]
                self.conf = [confidence]
                self.cls = [class_id]
        
        class MockResults:
            def __init__(self):
                self.boxes = [
                    MockBox([100, 100, 200, 150], 0.85, 0),
                    MockBox([300, 200, 400, 280], 0.75, 1),
                    MockBox([150, 300, 250, 350], 0.60, 2),
                ]
        
        return MockResults()
    
    def test_process_yolo_results(self):
        """Test YOLO results processing."""
        boxes = self.postprocessor.process_yolo_results(self.test_results)
        
        self.assertEqual(len(boxes), 3)
        
        for box in boxes:
            self.assertIsInstance(box, DetectionBox)
            self.assertGreater(box.confidence, 0)
            self.assertGreaterEqual(box.class_id, 0)
            self.assertIsInstance(box.class_name, str)
    
    def test_draw_results(self):
        """Test result drawing."""
        annotated = self.postprocessor.draw_results(self.test_image, self.test_results)
        
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, self.test_image.shape)
        self.assertEqual(annotated.dtype, np.uint8)
    
    def test_create_defect_summary_image(self):
        """Test defect summary creation."""
        boxes = self.postprocessor.process_yolo_results(self.test_results)
        summary = self.postprocessor.create_defect_summary_image(boxes)
        
        self.assertIsNotNone(summary)
        self.assertEqual(len(summary.shape), 3)
        self.assertEqual(summary.shape[2], 3)
    
    def test_create_confidence_histogram(self):
        """Test confidence histogram creation."""
        boxes = self.postprocessor.process_yolo_results(self.test_results)
        histogram = self.postprocessor.create_confidence_histogram(boxes)
        
        self.assertIsNotNone(histogram)
        self.assertEqual(len(histogram.shape), 3)
        self.assertEqual(histogram.shape[2], 3)
    
    def test_create_inspection_report(self):
        """Test inspection report creation."""
        metadata = {
            "timestamp": "2024-01-01 12:00:00",
            "focus_score": 150.5,
            "processing_time": 0.05
        }
        
        report = self.postprocessor.create_inspection_report(
            self.test_image, self.test_results, metadata
        )
        
        self.assertIsNotNone(report)
        self.assertEqual(len(report.shape), 3)
        self.assertEqual(report.shape[2], 3)
    
    def test_create_thumbnail(self):
        """Test thumbnail creation."""
        thumbnail = self.postprocessor.create_thumbnail(self.test_image)
        
        self.assertIsNotNone(thumbnail)
        self.assertEqual(len(thumbnail.shape), 3)
        self.assertEqual(thumbnail.shape[2], 3)
        
        # Should be smaller than original
        self.assertLessEqual(thumbnail.shape[0], self.test_image.shape[0])
        self.assertLessEqual(thumbnail.shape[1], self.test_image.shape[1])


class TestProcessingPerformance(unittest.TestCase):
    """Performance tests for processing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        self.detector = PCBDetector()
        self.focus_evaluator = FocusEvaluator()
        self.test_image = self._create_test_image()
    
    def _create_test_image(self, size: Tuple[int, int] = (1024, 768)) -> np.ndarray:
        """Create test image."""
        height, width = size
        return np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    def test_preprocessing_performance(self):
        """Test preprocessing performance."""
        iterations = 50
        
        # Test full processing
        start_time = time.time()
        for _ in range(iterations):
            result = self.preprocessor.process(self.test_image)
        full_time = time.time() - start_time
        
        # Test preview processing
        start_time = time.time()
        for _ in range(iterations):
            result = self.preprocessor.process_preview(self.test_image)
        preview_time = time.time() - start_time
        
        full_fps = iterations / full_time
        preview_fps = iterations / preview_time
        
        print(f"Full processing: {full_fps:.2f} FPS")
        print(f"Preview processing: {preview_fps:.2f} FPS")
        
        # Preview should be faster
        self.assertGreater(preview_fps, full_fps)
        
        # Should achieve reasonable FPS
        self.assertGreater(preview_fps, 20)  # At least 20 FPS for preview
    
    def test_pcb_detection_performance(self):
        """Test PCB detection performance."""
        iterations = 100
        
        start_time = time.time()
        for _ in range(iterations):
            result = self.detector.detect_pcb(self.test_image)
        elapsed_time = time.time() - start_time
        
        fps = iterations / elapsed_time
        
        print(f"PCB detection: {fps:.2f} FPS")
        
        # Should achieve reasonable FPS
        self.assertGreater(fps, 30)  # At least 30 FPS for detection
    
    def test_focus_evaluation_performance(self):
        """Test focus evaluation performance."""
        iterations = 200
        
        start_time = time.time()
        for _ in range(iterations):
            score = self.focus_evaluator.evaluate(self.test_image)
        elapsed_time = time.time() - start_time
        
        fps = iterations / elapsed_time
        
        print(f"Focus evaluation: {fps:.2f} FPS")
        
        # Should be very fast
        self.assertGreater(fps, 100)  # At least 100 FPS for focus evaluation


def run_performance_tests():
    """Run performance tests."""
    print("Running performance tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(TestProcessingPerformance('test_preprocessing_performance'))
    suite.addTest(TestProcessingPerformance('test_pcb_detection_performance'))
    suite.addTest(TestProcessingPerformance('test_focus_evaluation_performance'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_all_tests():
    """Run all tests."""
    print("Running all processing tests...")
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_test():
    """Run integration test of complete pipeline."""
    print("Running processing pipeline integration test...")
    
    # Create components
    preprocessor = ImagePreprocessor()
    detector = PCBDetector()
    postprocessor = ResultPostprocessor()
    
    # Create test image
    test_image = np.random.randint(0, 255, (800, 600), dtype=np.uint8)
    
    # Add PCB-like object
    cv2.rectangle(test_image, (200, 150), (600, 450), 150, -1)
    cv2.rectangle(test_image, (200, 150), (600, 450), 255, 3)
    
    try:
        # Step 1: Preprocess
        print("1. Preprocessing...")
        processed = preprocessor.process(test_image)
        preview = preprocessor.process_preview(test_image)
        
        assert processed is not None
        assert preview is not None
        print("   ✓ Preprocessing successful")
        
        # Step 2: PCB Detection
        print("2. PCB Detection...")
        detection_result = detector.detect_pcb(processed)
        
        assert detection_result is not None
        print(f"   ✓ PCB detected: {detection_result.has_pcb}")
        print(f"   ✓ Focus score: {detection_result.focus_score:.2f}")
        
        # Step 3: Stability tracking
        print("3. Stability tracking...")
        for i in range(15):
            result = detector.detect_pcb(processed)
            if result.is_stable:
                print(f"   ✓ Stable after {i+1} frames")
                break
        
        # Step 4: Visualization
        print("4. Result visualization...")
        if detection_result.has_pcb:
            vis_image = detector.visualize_detection(processed, detection_result)
            assert vis_image is not None
            print("   ✓ Visualization successful")
        
        print("✓ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test processing pipeline")
    parser.add_argument("--performance", action="store_true", 
                       help="Run performance tests only")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration test only")
    
    args = parser.parse_args()
    
    success = True
    
    if args.performance:
        success = run_performance_tests()
    elif args.integration:
        success = run_integration_test()
    else:
        # Run all tests
        success = run_all_tests()
        
        # Run integration test
        if success:
            success = run_integration_test()
    
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
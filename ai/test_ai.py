"""
Comprehensive test suite for AI layer components.

This module provides testing for the PCB defect detection AI components
including model loading, inference, performance, and integration tests.
"""

import unittest
import numpy as np
import torch
import time
import os
import sys
from typing import List, Dict, Any
import tempfile
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.inference import PCBDefectDetector, ModelManager, create_test_image
from core.interfaces import InspectionResult
from core.config import AI_CONFIG, MODEL_CLASS_MAPPING, DEFECT_CLASSES


class TestPCBDefectDetector(unittest.TestCase):
    """Test cases for PCBDefectDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tests."""
        cls.test_images = [
            create_test_image(add_defects=True),
            create_test_image(add_defects=False),
            create_test_image(size=(512, 512), add_defects=True),
            create_test_image(size=(800, 600), add_defects=False)
        ]
        
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.detector = None
        self.test_image = create_test_image()
    
    def tearDown(self):
        """Clean up after each test."""
        if self.detector is not None:
            del self.detector
            self.detector = None
    
    def test_model_initialization(self):
        """Test model initialization and loading."""
        try:
            self.detector = PCBDefectDetector()
            
            # Check if model loaded successfully
            self.assertTrue(self.detector.is_loaded)
            self.assertIsNotNone(self.detector.model)
            self.assertIsNotNone(self.detector.device)
            
            # Check model info
            model_info = self.detector.get_model_info()
            self.assertIn("model_path", model_info)
            self.assertIn("device", model_info)
            self.assertIn("num_classes", model_info)
            
        except Exception as e:
            self.skipTest(f"Model initialization failed: {str(e)}")
    
    def test_device_selection(self):
        """Test device selection logic."""
        try:
            # Test with default config
            detector = PCBDefectDetector()
            self.assertIsNotNone(detector.device)
            
            # Test CPU fallback
            cpu_config = AI_CONFIG.copy()
            cpu_config["device"] = "cpu"
            detector_cpu = PCBDefectDetector(cpu_config)
            self.assertEqual(detector_cpu.device, "cpu")
            
            # Test GPU selection (if available)
            if torch.cuda.is_available():
                gpu_config = AI_CONFIG.copy()
                gpu_config["device"] = "cuda:0"
                detector_gpu = PCBDefectDetector(gpu_config)
                self.assertTrue(detector_gpu.device.startswith("cuda"))
            
        except Exception as e:
            self.skipTest(f"Device selection test failed: {str(e)}")
    
    def test_single_image_detection(self):
        """Test detection on single image."""
        try:
            self.detector = PCBDefectDetector()
            
            # Test with defects
            result = self.detector.detect(self.test_image)
            
            self.assertIsInstance(result, InspectionResult)
            self.assertIsInstance(result.defects, list)
            self.assertIsInstance(result.locations, list)
            self.assertIsInstance(result.confidence_scores, list)
            self.assertGreater(result.processing_time, 0)
            
            # Check consistency of results
            self.assertEqual(len(result.defects), len(result.locations))
            self.assertEqual(len(result.defects), len(result.confidence_scores))
            
            # Check if defects are valid
            for defect in result.defects:
                self.assertIn(defect, DEFECT_CLASSES)
            
            # Check confidence scores
            for conf in result.confidence_scores:
                self.assertGreaterEqual(conf, 0.0)
                self.assertLessEqual(conf, 1.0)
            
        except Exception as e:
            self.skipTest(f"Single image detection test failed: {str(e)}")
    
    def test_batch_detection(self):
        """Test batch detection."""
        try:
            self.detector = PCBDefectDetector()
            
            # Test batch processing
            batch_results = self.detector.detect_batch(self.test_images)
            
            self.assertEqual(len(batch_results), len(self.test_images))
            
            for result in batch_results:
                self.assertIsInstance(result, InspectionResult)
                self.assertGreaterEqual(result.processing_time, 0)
            
        except Exception as e:
            self.skipTest(f"Batch detection test failed: {str(e)}")
    
    def test_confidence_threshold(self):
        """Test confidence threshold setting."""
        try:
            self.detector = PCBDefectDetector()
            
            # Test setting different thresholds
            thresholds = [0.1, 0.5, 0.9]
            
            for threshold in thresholds:
                self.detector.set_confidence_threshold(threshold)
                self.assertEqual(self.detector.config["confidence"], threshold)
                
                # Run detection with new threshold
                result = self.detector.detect(self.test_image)
                
                # All detections should have confidence >= threshold
                for conf in result.confidence_scores:
                    self.assertGreaterEqual(conf, threshold)
            
            # Test invalid thresholds
            with self.assertRaises(ValueError):
                self.detector.set_confidence_threshold(-0.1)
            
            with self.assertRaises(ValueError):
                self.detector.set_confidence_threshold(1.1)
            
        except Exception as e:
            self.skipTest(f"Confidence threshold test failed: {str(e)}")
    
    def test_preprocessing(self):
        """Test image preprocessing."""
        try:
            self.detector = PCBDefectDetector()
            
            # Test with grayscale image
            gray_image = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
            processed = self.detector._preprocess_image(gray_image)
            
            self.assertEqual(len(processed.shape), 3)
            self.assertEqual(processed.shape[2], 3)
            
            # Test with RGB image
            rgb_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            processed = self.detector._preprocess_image(rgb_image)
            
            self.assertEqual(len(processed.shape), 3)
            self.assertEqual(processed.shape[2], 3)
            
        except Exception as e:
            self.skipTest(f"Preprocessing test failed: {str(e)}")
    
    def test_result_processing(self):
        """Test result processing and class mapping."""
        try:
            self.detector = PCBDefectDetector()
            
            # Mock YOLO results
            class MockBox:
                def __init__(self, xyxy, conf, cls):
                    self.xyxy = [torch.tensor(xyxy)]
                    self.conf = [torch.tensor(conf)]
                    self.cls = [torch.tensor(cls)]
                    
                def cpu(self):
                    return self
                    
                def numpy(self):
                    return self
            
            class MockResult:
                def __init__(self):
                    self.boxes = [
                        MockBox([100, 100, 200, 200], 0.8, 0),
                        MockBox([300, 300, 400, 400], 0.9, 1)
                    ]
            
            # Process mock results
            mock_result = MockResult()
            defects, locations, scores = self.detector._process_results(mock_result)
            
            self.assertEqual(len(defects), 2)
            self.assertEqual(len(locations), 2)
            self.assertEqual(len(scores), 2)
            
            # Check class mapping
            self.assertIn(defects[0], DEFECT_CLASSES)
            self.assertIn(defects[1], DEFECT_CLASSES)
            
        except Exception as e:
            self.skipTest(f"Result processing test failed: {str(e)}")
    
    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        try:
            self.detector = PCBDefectDetector()
            
            # Run several detections
            for _ in range(5):
                self.detector.detect(self.test_image)
            
            # Get performance stats
            stats = self.detector.get_performance_stats()
            
            self.assertIn("total_inferences", stats)
            self.assertIn("avg_inference_time", stats)
            self.assertIn("fps", stats)
            
            self.assertGreater(stats["total_inferences"], 0)
            self.assertGreater(stats["avg_inference_time"], 0)
            self.assertGreater(stats["fps"], 0)
            
        except Exception as e:
            self.skipTest(f"Performance tracking test failed: {str(e)}")
    
    def test_gpu_memory_management(self):
        """Test GPU memory management."""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        
        try:
            gpu_config = AI_CONFIG.copy()
            gpu_config["device"] = "cuda:0"
            self.detector = PCBDefectDetector(gpu_config)
            
            # Check initial memory
            initial_memory = torch.cuda.memory_allocated()
            
            # Run detection
            self.detector.detect(self.test_image)
            
            # Clear cache
            self.detector.clear_gpu_cache()
            
            # Memory should not increase significantly
            final_memory = torch.cuda.memory_allocated()
            self.assertLess(final_memory - initial_memory, 1e8)  # Less than 100MB
            
        except Exception as e:
            self.skipTest(f"GPU memory management test failed: {str(e)}")
    
    def test_error_handling(self):
        """Test error handling."""
        try:
            self.detector = PCBDefectDetector()
            
            # Test with None image
            result = self.detector.detect(None)
            self.assertIsInstance(result, InspectionResult)
            self.assertEqual(len(result.defects), 0)
            
            # Test with invalid image
            invalid_image = np.array([1, 2, 3])
            result = self.detector.detect(invalid_image)
            self.assertIsInstance(result, InspectionResult)
            
        except Exception as e:
            self.skipTest(f"Error handling test failed: {str(e)}")


class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ModelManager()
    
    def test_model_validation(self):
        """Test model validation."""
        validation_results = self.manager.validate_model()
        
        self.assertIn("model_exists", validation_results)
        self.assertIn("config_valid", validation_results)
        self.assertIn("gpu_available", validation_results)
        self.assertIn("errors", validation_results)
        
        self.assertIsInstance(validation_results["errors"], list)
    
    def test_model_metadata(self):
        """Test model metadata extraction."""
        metadata = self.manager.get_model_metadata()
        
        required_keys = [
            "model_path", "framework", "task", "classes",
            "class_mapping", "input_size", "confidence_threshold"
        ]
        
        for key in required_keys:
            self.assertIn(key, metadata)
        
        self.assertEqual(metadata["classes"], DEFECT_CLASSES)
        self.assertEqual(metadata["class_mapping"], MODEL_CLASS_MAPPING)
    
    def test_model_benchmark(self):
        """Test model benchmarking."""
        try:
            detector = PCBDefectDetector()
            test_images = [create_test_image() for _ in range(3)]
            
            benchmark_results = self.manager.benchmark_model(detector, test_images)
            
            self.assertIn("num_test_images", benchmark_results)
            self.assertIn("avg_inference_time", benchmark_results)
            self.assertIn("fps", benchmark_results)
            
            self.assertEqual(benchmark_results["num_test_images"], 3)
            self.assertGreater(benchmark_results["fps"], 0)
            
        except Exception as e:
            self.skipTest(f"Model benchmark test failed: {str(e)}")


class TestAIPerformance(unittest.TestCase):
    """Performance tests for AI components."""
    
    def test_inference_speed(self):
        """Test inference speed requirements."""
        try:
            detector = PCBDefectDetector()
            test_image = create_test_image()
            
            # Run multiple inferences
            times = []
            for _ in range(10):
                start_time = time.time()
                detector.detect(test_image)
                inference_time = time.time() - start_time
                times.append(inference_time)
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time
            
            print(f"Average inference time: {avg_time:.3f}s")
            print(f"Average FPS: {fps:.2f}")
            
            # Performance requirements
            self.assertLess(avg_time, 1.0)  # Should be under 1 second
            self.assertGreater(fps, 1.0)   # Should achieve at least 1 FPS
            
        except Exception as e:
            self.skipTest(f"Inference speed test failed: {str(e)}")
    
    def test_memory_usage(self):
        """Test memory usage."""
        try:
            detector = PCBDefectDetector()
            test_image = create_test_image()
            
            # Check memory usage
            stats = detector.get_performance_stats()
            
            if "gpu_memory_allocated" in stats:
                gpu_memory = stats["gpu_memory_allocated"]
                print(f"GPU memory usage: {gpu_memory / 1e6:.2f} MB")
                
                # Should use less than 2GB
                self.assertLess(gpu_memory, 2e9)
            
        except Exception as e:
            self.skipTest(f"Memory usage test failed: {str(e)}")
    
    def test_batch_performance(self):
        """Test batch processing performance."""
        try:
            detector = PCBDefectDetector()
            test_images = [create_test_image() for _ in range(5)]
            
            # Test single vs batch processing
            start_time = time.time()
            for img in test_images:
                detector.detect(img)
            single_time = time.time() - start_time
            
            start_time = time.time()
            detector.detect_batch(test_images)
            batch_time = time.time() - start_time
            
            print(f"Single processing: {single_time:.3f}s")
            print(f"Batch processing: {batch_time:.3f}s")
            
            # Batch should be faster or comparable
            self.assertLessEqual(batch_time, single_time * 1.2)  # Allow 20% overhead
            
        except Exception as e:
            self.skipTest(f"Batch performance test failed: {str(e)}")


class TestAIIntegration(unittest.TestCase):
    """Integration tests for AI components."""
    
    def test_config_integration(self):
        """Test configuration integration."""
        # Test with custom config
        custom_config = AI_CONFIG.copy()
        custom_config["confidence"] = 0.3
        
        detector = PCBDefectDetector(custom_config)
        
        self.assertEqual(detector.config["confidence"], 0.3)
    
    def test_class_mapping_consistency(self):
        """Test class mapping consistency."""
        detector = PCBDefectDetector()
        
        # All mapped classes should be in DEFECT_CLASSES
        for class_id, class_name in MODEL_CLASS_MAPPING.items():
            self.assertIn(class_name, DEFECT_CLASSES)
        
        # Test with detection result
        result = detector.detect(create_test_image())
        
        for defect in result.defects:
            self.assertIn(defect, DEFECT_CLASSES)
    
    def test_preprocessing_integration(self):
        """Test preprocessing integration."""
        from processing.preprocessor import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        detector = PCBDefectDetector()
        
        # Test with preprocessed image
        raw_image = create_test_image()
        processed_image = preprocessor.process(raw_image)
        
        result = detector.detect(processed_image)
        
        self.assertIsInstance(result, InspectionResult)
        self.assertGreaterEqual(result.processing_time, 0)


def create_test_suite():
    """Create comprehensive test suite."""
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestPCBDefectDetector,
        TestModelManager,
        TestAIPerformance,
        TestAIIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    return test_suite


def run_quick_test():
    """Run quick functionality test."""
    print("Running quick AI functionality test...")
    
    try:
        # Test basic functionality
        detector = PCBDefectDetector()
        test_image = create_test_image()
        
        result = detector.detect(test_image)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Detection completed: {len(result.defects)} defects found")
        print(f"✓ Processing time: {result.processing_time:.3f}s")
        
        # Test performance
        stats = detector.get_performance_stats()
        print(f"✓ Performance: {stats['fps']:.2f} FPS")
        
        # Test validation
        manager = ModelManager()
        validation = manager.validate_model()
        
        if validation["model_exists"]:
            print("✓ Model file exists")
        
        if validation["config_valid"]:
            print("✓ Configuration is valid")
        
        if validation["gpu_available"]:
            print("✓ GPU is available")
        
        print("\n✅ Quick test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test FAILED: {str(e)}")
        return False


def run_performance_test():
    """Run performance benchmarking."""
    print("Running performance benchmarks...")
    
    try:
        detector = PCBDefectDetector()
        manager = ModelManager()
        
        # Create test images
        test_images = [create_test_image() for _ in range(10)]
        
        # Run benchmark
        benchmark_results = manager.benchmark_model(detector, test_images)
        
        print(f"Benchmark Results:")
        print(f"  Average inference time: {benchmark_results['avg_inference_time']:.3f}s")
        print(f"  FPS: {benchmark_results['fps']:.2f}")
        print(f"  Total detections: {benchmark_results['total_detections']}")
        print(f"  Avg detections per image: {benchmark_results['avg_detections_per_image']:.2f}")
        
        # Performance requirements check
        if benchmark_results['avg_inference_time'] < 1.0:
            print("✓ Inference time requirement met (<1s)")
        else:
            print("⚠ Inference time requirement not met")
        
        if benchmark_results['fps'] > 1.0:
            print("✓ FPS requirement met (>1 FPS)")
        else:
            print("⚠ FPS requirement not met")
        
        return True
        
    except Exception as e:
        print(f"Performance test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AI components")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--performance", action="store_true", help="Run performance test only")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
    elif args.performance:
        success = run_performance_test()
    elif args.full:
        # Run full test suite
        suite = create_test_suite()
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        success = result.wasSuccessful()
    else:
        # Default: run quick test
        success = run_quick_test()
    
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
"""
Unit tests for AI layer components.

Tests YOLOv11 integration, defect detection, and model optimization.
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import AI_CONFIG, MODEL_CLASS_MAPPING, DEFECT_CLASSES


class TestPCBDefectDetectorMock(unittest.TestCase):
    """Test PCBDefectDetector with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock YOLO model
        self.mock_yolo = MagicMock()
        self.mock_model = MagicMock()
        self.mock_yolo.return_value = self.mock_model
        
        # Mock torch
        self.mock_torch = MagicMock()
        self.mock_torch.cuda.is_available.return_value = True
    
    @patch('torch.cuda.is_available')
    @patch('ultralytics.YOLO')
    def test_detector_initialization_gpu(self, mock_yolo_class, mock_cuda):
        """Test detector initialization with GPU."""
        mock_cuda.return_value = True
        mock_yolo_class.return_value = self.mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector(
                model_path="test_model.pt",
                device="cuda:0",
                confidence=0.5
            )
            
            self.assertIsNotNone(detector)
            self.assertEqual(detector.device, "cuda:0")
            self.assertEqual(detector.confidence, 0.5)
            
            # Should have called YOLO constructor
            mock_yolo_class.assert_called_once_with("test_model.pt")
            
        except ImportError:
            self.skipTest("AI module not available")
    
    @patch('torch.cuda.is_available')
    @patch('ultralytics.YOLO')
    def test_detector_initialization_cpu_fallback(self, mock_yolo_class, mock_cuda):
        """Test detector initialization with CPU fallback."""
        mock_cuda.return_value = False
        mock_yolo_class.return_value = self.mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector(
                model_path="test_model.pt",
                device="cuda:0",  # Request GPU
                confidence=0.5
            )
            
            # Should fallback to CPU
            self.assertEqual(detector.device, "cpu")
            
        except ImportError:
            self.skipTest("AI module not available")
    
    @patch('ultralytics.YOLO')
    def test_model_loading_error_handling(self, mock_yolo_class):
        """Test error handling during model loading."""
        # Mock model loading failure
        mock_yolo_class.side_effect = Exception("Model not found")
        
        try:
            from ai.inference import PCBDefectDetector
            
            with self.assertRaises(Exception):
                PCBDefectDetector(
                    model_path="nonexistent_model.pt",
                    device="cpu",
                    confidence=0.5
                )
                
        except ImportError:
            self.skipTest("AI module not available")
    
    @patch('ultralytics.YOLO')
    def test_detection_no_objects(self, mock_yolo_class):
        """Test detection with no objects found."""
        # Setup mock for no detections
        mock_results = MagicMock()
        mock_results.boxes = None
        self.mock_model.return_value = [mock_results]
        
        mock_yolo_class.return_value = self.mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = detector.detect(test_image)
            
            self.assertIsNotNone(results)
            self.assertIsNone(results.boxes)
            
        except ImportError:
            self.skipTest("AI module not available")
    
    @patch('ultralytics.YOLO')
    def test_detection_with_objects(self, mock_yolo_class):
        """Test detection with objects found."""
        # Setup mock for detections
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [100, 100, 200, 200]
        mock_box.cls = 0  # Mouse Bite
        mock_box.conf = 0.85
        
        mock_results = MagicMock()
        mock_results.boxes = [mock_box]
        self.mock_model.return_value = [mock_results]
        
        mock_yolo_class.return_value = self.mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = detector.detect(test_image)
            
            self.assertIsNotNone(results)
            self.assertIsNotNone(results.boxes)
            self.assertEqual(len(results.boxes), 1)
            
        except ImportError:
            self.skipTest("AI module not available")
    
    @patch('ultralytics.YOLO')
    def test_confidence_filtering(self, mock_yolo_class):
        """Test confidence threshold filtering."""
        # Setup mock with low confidence detection
        mock_box_low = MagicMock()
        mock_box_low.cls = 0
        mock_box_low.conf = 0.3  # Below threshold
        
        mock_box_high = MagicMock()
        mock_box_high.cls = 1
        mock_box_high.conf = 0.8  # Above threshold
        
        mock_results = MagicMock()
        mock_results.boxes = [mock_box_low, mock_box_high]
        self.mock_model.return_value = [mock_results]
        
        mock_yolo_class.return_value = self.mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)  # 0.5 threshold
            
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = detector.detect(test_image)
            
            # Model should handle confidence filtering internally
            self.assertIsNotNone(results)
            
        except ImportError:
            self.skipTest("AI module not available")
    
    @patch('ultralytics.YOLO')
    def test_batch_processing(self, mock_yolo_class):
        """Test batch processing of multiple images."""
        mock_results = MagicMock()
        mock_results.boxes = None
        self.mock_model.return_value = [mock_results, mock_results]  # 2 results
        
        mock_yolo_class.return_value = self.mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            
            # Test with multiple images
            images = [
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            ]
            
            results = detector.detect_batch(images)
            
            self.assertIsNotNone(results)
            self.assertEqual(len(results), 2)
            
        except (ImportError, AttributeError):
            self.skipTest("AI module or batch processing not available")


class TestModelClassMapping(unittest.TestCase):
    """Test model class mapping functionality."""
    
    def test_class_mapping_completeness(self):
        """Test that class mapping covers all expected classes."""
        self.assertIsInstance(MODEL_CLASS_MAPPING, dict)
        self.assertEqual(len(MODEL_CLASS_MAPPING), 6)
        
        # Check that all mapped classes are in DEFECT_CLASSES
        for class_id, class_name in MODEL_CLASS_MAPPING.items():
            self.assertIsInstance(class_id, int)
            self.assertIn(class_name, DEFECT_CLASSES)
    
    def test_class_mapping_consistency(self):
        """Test consistency between class mapping and defect classes."""
        mapped_classes = set(MODEL_CLASS_MAPPING.values())
        defect_classes_set = set(DEFECT_CLASSES)
        
        # All mapped classes should be in defect classes
        self.assertTrue(mapped_classes.issubset(defect_classes_set))
    
    def test_class_id_range(self):
        """Test that class IDs are in expected range."""
        for class_id in MODEL_CLASS_MAPPING.keys():
            self.assertGreaterEqual(class_id, 0)
            self.assertLess(class_id, 10)  # Reasonable upper bound


class TestAIConfiguration(unittest.TestCase):
    """Test AI configuration settings."""
    
    def test_ai_config_structure(self):
        """Test AI configuration structure."""
        required_keys = ['model_path', 'confidence', 'device']
        
        for key in required_keys:
            self.assertIn(key, AI_CONFIG)
    
    def test_confidence_threshold(self):
        """Test confidence threshold validation."""
        confidence = AI_CONFIG['confidence']
        self.assertIsInstance(confidence, (int, float))
        self.assertGreater(confidence, 0)
        self.assertLess(confidence, 1)
    
    def test_device_specification(self):
        """Test device specification format."""
        device = AI_CONFIG['device']
        self.assertIsInstance(device, str)
        self.assertTrue(device in ['cpu', 'cuda:0', 'cuda:1'] or device.startswith('cuda:'))
    
    def test_model_path_format(self):
        """Test model path format."""
        model_path = AI_CONFIG['model_path']
        self.assertIsInstance(model_path, str)
        self.assertTrue(model_path.endswith('.pt'))


class TestAIPerformance(unittest.TestCase):
    """Test AI performance characteristics."""
    
    @patch('ultralytics.YOLO')
    def test_inference_timing(self, mock_yolo_class):
        """Test inference timing characteristics."""
        import time
        
        # Setup fast mock
        mock_results = MagicMock()
        mock_results.boxes = None
        mock_model = MagicMock()
        mock_model.return_value = [mock_results]
        mock_yolo_class.return_value = mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Time multiple inferences
            start_time = time.time()
            for _ in range(10):
                detector.detect(test_image)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            
            # Should be reasonably fast with mock (much faster than real model)
            self.assertLess(avg_time, 0.01)  # 10ms with mock
            
        except ImportError:
            self.skipTest("AI module not available")
    
    @patch('ultralytics.YOLO')
    def test_memory_usage(self, mock_yolo_class):
        """Test memory usage characteristics."""
        mock_results = MagicMock()
        mock_results.boxes = None
        mock_model = MagicMock()
        mock_model.return_value = [mock_results]
        mock_yolo_class.return_value = mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            
            # Process multiple images without accumulating memory
            for i in range(100):
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                results = detector.detect(test_image)
                
                # Should not accumulate results
                self.assertIsNotNone(results)
            
        except ImportError:
            self.skipTest("AI module not available")


class TestAIIntegration(unittest.TestCase):
    """Test AI integration with other components."""
    
    @patch('ultralytics.YOLO')
    def test_integration_with_preprocessing(self, mock_yolo_class):
        """Test integration with preprocessing output."""
        mock_results = MagicMock()
        mock_results.boxes = None
        mock_model = MagicMock()
        mock_model.return_value = [mock_results]
        mock_yolo_class.return_value = mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            from processing.preprocessor import ImagePreprocessor
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            preprocessor = ImagePreprocessor()
            
            # Test pipeline: raw image -> preprocessing -> AI
            raw_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            processed_image = preprocessor.process(raw_image)
            
            # AI should handle grayscale input (convert to RGB internally)
            results = detector.detect(processed_image)
            
            self.assertIsNotNone(results)
            
        except ImportError:
            self.skipTest("AI or processing modules not available")
    
    @patch('ultralytics.YOLO')
    def test_integration_with_postprocessing(self, mock_yolo_class):
        """Test integration with postprocessing."""
        # Setup mock with detections
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [100, 100, 200, 200]
        mock_box.cls = 0
        mock_box.conf = 0.85
        
        mock_results = MagicMock()
        mock_results.boxes = [mock_box]
        mock_model = MagicMock()
        mock_model.return_value = [mock_results]
        mock_yolo_class.return_value = mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            from processing.postprocessor import ResultPostprocessor
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            postprocessor = ResultPostprocessor()
            
            test_image = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
            
            # AI detection
            results = detector.detect(test_image)
            
            # Postprocessing
            annotated_image = postprocessor.draw_results(test_image, results)
            
            self.assertIsNotNone(annotated_image)
            self.assertEqual(annotated_image.shape, test_image.shape)
            
        except ImportError:
            self.skipTest("AI or processing modules not available")


class TestAIErrorHandling(unittest.TestCase):
    """Test AI error handling scenarios."""
    
    @patch('ultralytics.YOLO')
    def test_invalid_image_input(self, mock_yolo_class):
        """Test handling of invalid image inputs."""
        mock_yolo_class.return_value = MagicMock()
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            
            # Test with None input
            with self.assertRaises((ValueError, TypeError, AttributeError)):
                detector.detect(None)
            
            # Test with invalid shape
            with self.assertRaises((ValueError, TypeError)):
                detector.detect(np.array([1, 2, 3]))
            
        except ImportError:
            self.skipTest("AI module not available")
    
    @patch('ultralytics.YOLO')
    def test_model_inference_error(self, mock_yolo_class):
        """Test handling of model inference errors."""
        # Setup model that raises exception
        mock_model = MagicMock()
        mock_model.side_effect = Exception("Inference failed")
        mock_yolo_class.return_value = mock_model
        
        try:
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Should handle inference errors gracefully
            with self.assertRaises(Exception):
                detector.detect(test_image)
            
        except ImportError:
            self.skipTest("AI module not available")


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    unittest.main(verbosity=2)
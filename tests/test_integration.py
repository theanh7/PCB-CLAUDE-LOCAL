"""
Integration tests for PCB inspection system.

Tests complete workflows and component interactions across layers.
"""

import unittest
import tempfile
import os
import sys
import numpy as np
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import CAMERA_CONFIG, AI_CONFIG, TRIGGER_CONFIG, DB_CONFIG


class TestFullInspectionPipeline(unittest.TestCase):
    """Test complete inspection pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    @patch('ultralytics.YOLO')
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_complete_inspection_workflow(self, mock_yolo_class):
        """Test complete inspection workflow from camera to database."""
        try:
            # Setup mocks
            mock_pylon = MagicMock()
            mock_camera = MagicMock()
            mock_grab_result = MagicMock()
            
            mock_pylon.TlFactory.GetInstance.return_value.CreateFirstDevice.return_value = mock_camera
            mock_grab_result.GrabSucceeded.return_value = True
            mock_grab_result.Array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            mock_camera.GrabOne.return_value = mock_grab_result
            
            # Setup AI mock
            mock_box = MagicMock()
            mock_box.xyxy = [MagicMock()]
            mock_box.xyxy[0].tolist.return_value = [100, 100, 200, 200]
            mock_box.cls = 0  # Mouse Bite
            mock_box.conf = 0.85
            
            mock_results = MagicMock()
            mock_results.boxes = [mock_box]
            mock_model = MagicMock()
            mock_model.return_value = [mock_results]
            mock_yolo_class.return_value = mock_model
            
            with patch('pypylon.pylon', mock_pylon):
                # Import components
                from hardware.camera_controller import BaslerCamera
                from processing.preprocessor import ImagePreprocessor
                from processing.pcb_detector import PCBDetector
                from processing.postprocessor import ResultPostprocessor
                from ai.inference import PCBDefectDetector
                from data.database import PCBDatabase
                from analytics.analyzer import DefectAnalyzer
                
                # Initialize components
                camera = BaslerCamera(CAMERA_CONFIG)
                preprocessor = ImagePreprocessor()
                pcb_detector = PCBDetector(TRIGGER_CONFIG)
                postprocessor = ResultPostprocessor()
                ai_detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
                database = PCBDatabase(self.db_path)
                analyzer = DefectAnalyzer(database)
                
                # Step 1: Capture image
                raw_image = camera.capture()
                self.assertIsNotNone(raw_image)
                
                # Step 2: Preprocess
                processed_image = preprocessor.process(raw_image)
                self.assertIsNotNone(processed_image)
                
                # Step 3: PCB Detection
                has_pcb, position, is_stable, focus_score = pcb_detector.detect_pcb(raw_image)
                self.assertIsInstance(has_pcb, bool)
                self.assertIsInstance(focus_score, float)
                
                # Step 4: AI Detection
                detection_results = ai_detector.detect(processed_image)
                self.assertIsNotNone(detection_results)
                
                # Step 5: Postprocess results
                display_image = postprocessor.draw_results(processed_image, detection_results)
                self.assertIsNotNone(display_image)
                
                # Step 6: Extract defects
                defects = []
                locations = []
                confidences = []
                
                if detection_results.boxes:
                    for box in detection_results.boxes:
                        defects.append("Mouse Bite")  # From mock
                        locations.append({"bbox": [100, 100, 200, 200]})
                        confidences.append(0.85)
                
                # Step 7: Save to database
                inspection_id = database.save_inspection_metadata(
                    timestamp=datetime.now(),
                    defects=defects,
                    locations=locations,
                    confidence_scores=confidences,
                    raw_image_shape=raw_image.shape,
                    focus_score=focus_score
                )
                
                self.assertIsInstance(inspection_id, int)
                self.assertGreater(inspection_id, 0)
                
                # Step 8: Analytics
                stats = analyzer.get_realtime_stats()
                self.assertIsInstance(stats, dict)
                self.assertGreater(stats['total_inspections'], 0)
                
                # Cleanup
                database.close()
                camera.close()
                
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    @patch('ultralytics.YOLO')
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_auto_trigger_workflow(self, mock_yolo_class):
        """Test auto-trigger inspection workflow."""
        try:
            # Setup mocks (similar to above)
            mock_pylon = MagicMock()
            mock_camera = MagicMock()
            mock_grab_result = MagicMock()
            
            mock_pylon.TlFactory.GetInstance.return_value.CreateFirstDevice.return_value = mock_camera
            mock_grab_result.GrabSucceeded.return_value = True
            
            # Create PCB-like image for detection
            pcb_image = np.zeros((480, 640), dtype=np.uint8)
            pcb_image[100:350, 150:500] = 255  # Large rectangular region
            mock_grab_result.Array = pcb_image
            mock_camera.GrabOne.return_value = mock_grab_result
            
            mock_yolo_class.return_value = MagicMock()
            
            with patch('pypylon.pylon', mock_pylon):
                from hardware.camera_controller import BaslerCamera
                from processing.pcb_detector import PCBDetector
                
                camera = BaslerCamera(CAMERA_CONFIG)
                pcb_detector = PCBDetector(TRIGGER_CONFIG)
                
                # Test preview workflow
                camera.start_streaming()
                
                # Simulate getting frames and detecting PCB
                stability_count = 0
                for i in range(15):  # More than stability threshold
                    frame = camera.get_preview_frame()
                    if frame is None:
                        # Simulate frame in queue
                        camera.frame_queue.put(pcb_image)
                        frame = camera.get_preview_frame()
                    
                    if frame is not None:
                        has_pcb, position, is_stable, focus_score = pcb_detector.detect_pcb(frame)
                        
                        if has_pcb and is_stable:
                            stability_count += 1
                        
                        # Should eventually become stable
                        if i > TRIGGER_CONFIG["stability_frames"]:
                            # May become stable after stability threshold
                            pass
                
                camera.stop_streaming()
                camera.close()
                
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")


class TestConcurrentOperations(unittest.TestCase):
    """Test concurrent system operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    @patch('ultralytics.YOLO')
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_concurrent_inspection_operations(self, mock_yolo_class):
        """Test concurrent inspection operations."""
        try:
            # Setup mocks
            mock_pylon = MagicMock()
            mock_camera = MagicMock()
            mock_grab_result = MagicMock()
            
            mock_pylon.TlFactory.GetInstance.return_value.CreateFirstDevice.return_value = mock_camera
            mock_grab_result.GrabSucceeded.return_value = True
            mock_grab_result.Array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            mock_camera.GrabOne.return_value = mock_grab_result
            
            mock_yolo_class.return_value = MagicMock()
            
            with patch('pypylon.pylon', mock_pylon):
                from hardware.camera_controller import BaslerCamera
                from processing.preprocessor import ImagePreprocessor
                from ai.inference import PCBDefectDetector
                from data.database import PCBDatabase
                
                # Initialize components
                camera = BaslerCamera(CAMERA_CONFIG)
                preprocessor = ImagePreprocessor()
                ai_detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
                database = PCBDatabase(self.db_path)
                
                results = []
                errors = []
                
                def inspection_worker(worker_id):
                    """Worker function for concurrent inspections."""
                    try:
                        for i in range(5):
                            # Simulate inspection workflow
                            raw_image = camera.capture()
                            processed_image = preprocessor.process(raw_image)
                            detection_results = ai_detector.detect(processed_image)
                            
                            # Save results
                            inspection_id = database.save_inspection_metadata(
                                timestamp=datetime.now(),
                                defects=[],
                                locations=[],
                                confidence_scores=[],
                                raw_image_shape=raw_image.shape,
                                focus_score=100.0
                            )
                            
                            results.append((worker_id, inspection_id))
                            time.sleep(0.01)  # Small delay
                            
                    except Exception as e:
                        errors.append((worker_id, e))
                
                # Start multiple worker threads
                threads = []
                for i in range(3):
                    thread = threading.Thread(target=inspection_worker, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for completion
                for thread in threads:
                    thread.join()
                
                # Verify results
                self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
                self.assertEqual(len(results), 15)  # 3 workers * 5 inspections
                
                # All inspection IDs should be unique
                inspection_ids = [r[1] for r in results]
                self.assertEqual(len(set(inspection_ids)), len(inspection_ids))
                
                # Cleanup
                database.close()
                camera.close()
                
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")


class TestSystemIntegrationWithMockGUI(unittest.TestCase):
    """Test system integration with GUI components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    @patch('ultralytics.YOLO')
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_gui_integration_workflow(self, mock_yolo_class):
        """Test integration between system components and GUI."""
        try:
            # Setup mocks
            mock_pylon = MagicMock()
            mock_camera = MagicMock()
            mock_grab_result = MagicMock()
            
            mock_pylon.TlFactory.GetInstance.return_value.CreateFirstDevice.return_value = mock_camera
            mock_grab_result.GrabSucceeded.return_value = True
            mock_grab_result.Array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            mock_camera.GrabOne.return_value = mock_grab_result
            
            mock_yolo_class.return_value = MagicMock()
            
            with patch('pypylon.pylon', mock_pylon):
                from hardware.camera_controller import BaslerCamera
                from processing.preprocessor import ImagePreprocessor
                from processing.pcb_detector import PCBDetector
                from ai.inference import PCBDefectDetector
                from data.database import PCBDatabase
                from analytics.analyzer import DefectAnalyzer
                
                # Initialize system components
                camera = BaslerCamera(CAMERA_CONFIG)
                preprocessor = ImagePreprocessor()
                pcb_detector = PCBDetector(TRIGGER_CONFIG)
                ai_detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
                database = PCBDatabase(self.db_path)
                analyzer = DefectAnalyzer(database)
                
                # Mock GUI interface
                gui_updates = []
                
                def mock_update_preview(image, has_pcb=False, is_stable=False, focus_score=0):
                    gui_updates.append(('preview', has_pcb, is_stable, focus_score))
                
                def mock_update_results(image, defects, locations, confidences, processing_time):
                    gui_updates.append(('results', len(defects), processing_time))
                
                def mock_update_statistics(stats):
                    gui_updates.append(('statistics', stats))
                
                # Simulate inspection workflow with GUI updates
                camera.start_streaming()
                
                # Preview updates
                for i in range(5):
                    raw_frame = camera.get_preview_frame()
                    if raw_frame is None:
                        # Simulate frame
                        test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
                        camera.frame_queue.put(test_frame)
                        raw_frame = camera.get_preview_frame()
                    
                    if raw_frame is not None:
                        has_pcb, pos, stable, focus = pcb_detector.detect_pcb(raw_frame)
                        mock_update_preview(raw_frame, has_pcb, stable, focus)
                
                # Inspection workflow
                raw_image = camera.capture_high_quality()
                processed_image = preprocessor.process(raw_image)
                results = ai_detector.detect(processed_image)
                
                # Mock defect extraction
                defects = []
                locations = []
                confidences = []
                
                # Save and get stats
                inspection_id = database.save_inspection_metadata(
                    timestamp=datetime.now(),
                    defects=defects,
                    locations=locations,
                    confidence_scores=confidences,
                    raw_image_shape=raw_image.shape,
                    focus_score=120.0,
                    processing_time=0.15
                )
                
                stats = analyzer.get_realtime_stats()
                
                # Update GUI
                mock_update_results(processed_image, defects, locations, confidences, 0.15)
                mock_update_statistics(stats)
                
                camera.stop_streaming()
                
                # Verify GUI updates occurred
                self.assertGreater(len(gui_updates), 0)
                
                # Check different types of updates
                update_types = [update[0] for update in gui_updates]
                self.assertIn('preview', update_types)
                self.assertIn('results', update_types)
                self.assertIn('statistics', update_types)
                
                # Cleanup
                database.close()
                camera.close()
                
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")


class TestErrorRecoveryIntegration(unittest.TestCase):
    """Test error recovery across system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    @patch('ultralytics.YOLO')
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_camera_failure_recovery(self, mock_yolo_class):
        """Test system recovery from camera failures."""
        try:
            # Setup camera that fails occasionally
            mock_pylon = MagicMock()
            mock_camera = MagicMock()
            mock_grab_result = MagicMock()
            
            mock_pylon.TlFactory.GetInstance.return_value.CreateFirstDevice.return_value = mock_camera
            
            # Simulate intermittent failures
            call_count = 0
            def mock_grab_one(*args):
                nonlocal call_count
                call_count += 1
                if call_count % 3 == 0:  # Fail every 3rd call
                    mock_grab_result.GrabSucceeded.return_value = False
                else:
                    mock_grab_result.GrabSucceeded.return_value = True
                    mock_grab_result.Array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
                return mock_grab_result
            
            mock_camera.GrabOne.side_effect = mock_grab_one
            mock_yolo_class.return_value = MagicMock()
            
            with patch('pypylon.pylon', mock_pylon):
                from hardware.camera_controller import BaslerCamera
                from processing.preprocessor import ImagePreprocessor
                
                camera = BaslerCamera(CAMERA_CONFIG)
                preprocessor = ImagePreprocessor()
                
                successful_captures = 0
                failed_captures = 0
                
                # Attempt multiple captures
                for i in range(10):
                    try:
                        image = camera.capture()
                        if image is not None:
                            # Try to process the image
                            processed = preprocessor.process(image)
                            successful_captures += 1
                        else:
                            failed_captures += 1
                    except Exception:
                        failed_captures += 1
                
                # Should have some successful captures despite failures
                self.assertGreater(successful_captures, 0)
                
                camera.close()
                
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    @patch('ultralytics.YOLO')
    def test_ai_model_failure_recovery(self, mock_yolo_class):
        """Test system recovery from AI model failures."""
        try:
            # Setup AI model that fails occasionally
            call_count = 0
            def mock_inference(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 4 == 0:  # Fail every 4th call
                    raise Exception("Model inference failed")
                else:
                    mock_results = MagicMock()
                    mock_results.boxes = None
                    return [mock_results]
            
            mock_model = MagicMock()
            mock_model.side_effect = mock_inference
            mock_yolo_class.return_value = mock_model
            
            from ai.inference import PCBDefectDetector
            from processing.preprocessor import ImagePreprocessor
            
            ai_detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            preprocessor = ImagePreprocessor()
            
            successful_inferences = 0
            failed_inferences = 0
            
            # Attempt multiple inferences
            for i in range(12):
                try:
                    test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
                    processed = preprocessor.process(test_image)
                    results = ai_detector.detect(processed)
                    successful_inferences += 1
                except Exception:
                    failed_inferences += 1
            
            # Should have some successful inferences despite failures
            self.assertGreater(successful_inferences, 0)
            self.assertGreater(failed_inferences, 0)  # Should also have some failures
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")


class TestPerformanceIntegration(unittest.TestCase):
    """Test system performance under integrated conditions."""
    
    @patch('ultralytics.YOLO')
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_throughput_performance(self, mock_yolo_class):
        """Test system throughput performance."""
        try:
            # Setup fast mocks
            mock_pylon = MagicMock()
            mock_camera = MagicMock()
            mock_grab_result = MagicMock()
            
            mock_pylon.TlFactory.GetInstance.return_value.CreateFirstDevice.return_value = mock_camera
            mock_grab_result.GrabSucceeded.return_value = True
            mock_grab_result.Array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            mock_camera.GrabOne.return_value = mock_grab_result
            
            mock_results = MagicMock()
            mock_results.boxes = None
            mock_model = MagicMock()
            mock_model.return_value = [mock_results]
            mock_yolo_class.return_value = mock_model
            
            with patch('pypylon.pylon', mock_pylon):
                from hardware.camera_controller import BaslerCamera
                from processing.preprocessor import ImagePreprocessor
                from processing.pcb_detector import PCBDetector
                from ai.inference import PCBDefectDetector
                
                # Initialize components
                camera = BaslerCamera(CAMERA_CONFIG)
                preprocessor = ImagePreprocessor()
                pcb_detector = PCBDetector(TRIGGER_CONFIG)
                ai_detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
                
                # Measure throughput
                start_time = time.time()
                processed_count = 0
                
                for i in range(20):  # Process 20 images
                    # Complete pipeline
                    raw_image = camera.capture()
                    processed_image = preprocessor.process(raw_image)
                    has_pcb, pos, stable, focus = pcb_detector.detect_pcb(raw_image)
                    
                    if i % 2 == 0:  # Simulate some AI processing
                        results = ai_detector.detect(processed_image)
                    
                    processed_count += 1
                
                end_time = time.time()
                total_time = end_time - start_time
                throughput = processed_count / total_time
                
                # Should achieve reasonable throughput with mocks
                self.assertGreater(throughput, 10)  # At least 10 images/second
                
                camera.close()
                
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    unittest.main(verbosity=2)
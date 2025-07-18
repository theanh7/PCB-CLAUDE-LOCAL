"""
Performance benchmarking and stress tests for PCB inspection system.

Tests system performance under various load conditions and measures
key performance indicators.
"""

import unittest
import time
import threading
import psutil
import os
import sys
import numpy as np
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import CAMERA_CONFIG, AI_CONFIG, TRIGGER_CONFIG


class PerformanceBenchmark:
    """Performance measurement utilities."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_peak = None
        self.cpu_usage = []
    
    def start(self):
        """Start performance measurement."""
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_peak = self.memory_start
        self.cpu_usage = []
    
    def update(self):
        """Update performance measurements."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_peak = max(self.memory_peak, current_memory)
        self.cpu_usage.append(psutil.cpu_percent())
    
    def stop(self):
        """Stop performance measurement and return results."""
        self.end_time = time.time()
        
        return {
            'duration': self.end_time - self.start_time,
            'memory_start': self.memory_start,
            'memory_peak': self.memory_peak,
            'memory_increase': self.memory_peak - self.memory_start,
            'avg_cpu': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'max_cpu': max(self.cpu_usage) if self.cpu_usage else 0
        }


class TestCameraPerformance(unittest.TestCase):
    """Test camera operation performance."""
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_camera_capture_performance(self):
        """Test single capture performance."""
        try:
            # Setup mock
            mock_pylon = MagicMock()
            mock_camera = MagicMock()
            mock_grab_result = MagicMock()
            
            mock_pylon.TlFactory.GetInstance.return_value.CreateFirstDevice.return_value = mock_camera
            mock_grab_result.GrabSucceeded.return_value = True
            mock_grab_result.Array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            mock_camera.GrabOne.return_value = mock_grab_result
            
            with patch('pypylon.pylon', mock_pylon):
                from hardware.camera_controller import BaslerCamera
                
                camera = BaslerCamera(CAMERA_CONFIG)
                benchmark = PerformanceBenchmark()
                
                # Warm up
                for _ in range(5):
                    camera.capture()
                
                # Benchmark single captures
                benchmark.start()
                
                capture_times = []
                for i in range(100):
                    start = time.time()
                    image = camera.capture()
                    end = time.time()
                    
                    capture_times.append(end - start)
                    benchmark.update()
                    
                    self.assertIsNotNone(image)
                
                results = benchmark.stop()
                
                # Performance assertions
                avg_capture_time = sum(capture_times) / len(capture_times)
                max_capture_time = max(capture_times)
                
                print(f"\nCamera Capture Performance:")
                print(f"  Average capture time: {avg_capture_time*1000:.2f}ms")
                print(f"  Maximum capture time: {max_capture_time*1000:.2f}ms")
                print(f"  Total duration: {results['duration']:.2f}s")
                print(f"  Memory increase: {results['memory_increase']:.1f}MB")
                
                # Should be reasonably fast (mocked operations)
                self.assertLess(avg_capture_time, 0.01)  # 10ms average
                self.assertLess(max_capture_time, 0.05)   # 50ms max
                
                camera.close()
                
        except ImportError:
            self.skipTest("Hardware module not available")
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_camera_streaming_performance(self):
        """Test streaming performance and frame rates."""
        try:
            # Setup mock
            mock_pylon = MagicMock()
            mock_camera = MagicMock()
            
            mock_pylon.TlFactory.GetInstance.return_value.CreateFirstDevice.return_value = mock_camera
            
            with patch('pypylon.pylon', mock_pylon):
                from hardware.camera_controller import BaslerCamera
                
                camera = BaslerCamera(CAMERA_CONFIG)
                benchmark = PerformanceBenchmark()
                
                benchmark.start()
                
                camera.start_streaming()
                
                # Simulate frame processing
                frame_count = 0
                frame_times = []
                
                for i in range(300):  # Simulate 10 seconds at 30 FPS
                    start = time.time()
                    
                    # Simulate frame in queue
                    test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
                    if not camera.frame_queue.full():
                        camera.frame_queue.put(test_frame)
                    
                    frame = camera.get_preview_frame()
                    if frame is not None:
                        frame_count += 1
                    
                    end = time.time()
                    frame_times.append(end - start)
                    
                    benchmark.update()
                    time.sleep(0.033)  # 30 FPS target
                
                camera.stop_streaming()
                results = benchmark.stop()
                
                fps = frame_count / results['duration']
                avg_frame_time = sum(frame_times) / len(frame_times)
                
                print(f"\nCamera Streaming Performance:")
                print(f"  Frames processed: {frame_count}")
                print(f"  Average FPS: {fps:.1f}")
                print(f"  Average frame time: {avg_frame_time*1000:.2f}ms")
                print(f"  Memory increase: {results['memory_increase']:.1f}MB")
                
                # Should achieve target frame rate
                self.assertGreater(fps, 25)  # At least 25 FPS
                
                camera.close()
                
        except ImportError:
            self.skipTest("Hardware module not available")


class TestProcessingPerformance(unittest.TestCase):
    """Test image processing performance."""
    
    def test_preprocessing_performance(self):
        """Test image preprocessing performance."""
        try:
            from processing.preprocessor import ImagePreprocessor
            
            preprocessor = ImagePreprocessor()
            benchmark = PerformanceBenchmark()
            
            # Test different image sizes
            test_sizes = [(240, 320), (480, 640), (960, 1280), (1920, 2560)]
            
            for size in test_sizes:
                test_image = np.random.randint(0, 255, size, dtype=np.uint8)
                
                benchmark.start()
                
                process_times = []
                for i in range(50):
                    start = time.time()
                    result = preprocessor.process(test_image)
                    end = time.time()
                    
                    process_times.append(end - start)
                    benchmark.update()
                    
                    self.assertIsNotNone(result)
                
                results = benchmark.stop()
                
                avg_time = sum(process_times) / len(process_times)
                throughput = len(process_times) / results['duration']
                
                print(f"\nPreprocessing Performance ({size[0]}x{size[1]}):")
                print(f"  Average time: {avg_time*1000:.2f}ms")
                print(f"  Throughput: {throughput:.1f} images/sec")
                print(f"  Memory increase: {results['memory_increase']:.1f}MB")
                
                # Performance thresholds based on image size
                if size == (480, 640):  # Standard size
                    self.assertLess(avg_time, 0.05)  # 50ms
                elif size == (1920, 2560):  # Large size
                    self.assertLess(avg_time, 0.2)   # 200ms
                
        except ImportError:
            self.skipTest("Processing module not available")
    
    def test_pcb_detection_performance(self):
        """Test PCB detection performance."""
        try:
            from processing.pcb_detector import PCBDetector
            
            detector = PCBDetector(TRIGGER_CONFIG)
            benchmark = PerformanceBenchmark()
            
            # Create test images with and without PCB
            pcb_image = np.zeros((480, 640), dtype=np.uint8)
            pcb_image[100:350, 150:500] = 255  # PCB region
            
            empty_image = np.ones((480, 640), dtype=np.uint8) * 128
            
            test_images = [pcb_image, empty_image] * 50  # 100 total
            
            benchmark.start()
            
            detection_times = []
            pcb_detected_count = 0
            
            for image in test_images:
                start = time.time()
                has_pcb, position, is_stable, focus_score = detector.detect_pcb(image)
                end = time.time()
                
                detection_times.append(end - start)
                if has_pcb:
                    pcb_detected_count += 1
                
                benchmark.update()
            
            results = benchmark.stop()
            
            avg_time = sum(detection_times) / len(detection_times)
            throughput = len(detection_times) / results['duration']
            
            print(f"\nPCB Detection Performance:")
            print(f"  Average detection time: {avg_time*1000:.2f}ms")
            print(f"  Throughput: {throughput:.1f} detections/sec")
            print(f"  PCBs detected: {pcb_detected_count}/{len(test_images)}")
            print(f"  Memory increase: {results['memory_increase']:.1f}MB")
            
            # Should be fast enough for real-time processing
            self.assertLess(avg_time, 0.03)  # 30ms per detection
            self.assertGreater(throughput, 30)  # 30+ detections/sec
            
        except ImportError:
            self.skipTest("Processing module not available")


class TestAIPerformance(unittest.TestCase):
    """Test AI inference performance."""
    
    @patch('ultralytics.YOLO')
    def test_ai_inference_performance(self, mock_yolo_class):
        """Test AI inference performance."""
        try:
            # Setup fast mock
            mock_results = MagicMock()
            mock_results.boxes = None
            mock_model = MagicMock()
            mock_model.return_value = [mock_results]
            mock_yolo_class.return_value = mock_model
            
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            benchmark = PerformanceBenchmark()
            
            # Test different image sizes
            test_sizes = [(640, 640), (1280, 1280)]
            
            for size in test_sizes:
                test_image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
                
                # Warm up
                for _ in range(5):
                    detector.detect(test_image)
                
                benchmark.start()
                
                inference_times = []
                for i in range(100):
                    start = time.time()
                    results = detector.detect(test_image)
                    end = time.time()
                    
                    inference_times.append(end - start)
                    benchmark.update()
                    
                    self.assertIsNotNone(results)
                
                benchmark_results = benchmark.stop()
                
                avg_time = sum(inference_times) / len(inference_times)
                throughput = len(inference_times) / benchmark_results['duration']
                
                print(f"\nAI Inference Performance ({size[0]}x{size[1]}):")
                print(f"  Average inference time: {avg_time*1000:.2f}ms")
                print(f"  Throughput: {throughput:.1f} inferences/sec")
                print(f"  Memory increase: {benchmark_results['memory_increase']:.1f}MB")
                
                # Should be fast with mock (real GPU inference target: <100ms)
                self.assertLess(avg_time, 0.01)  # 10ms with mock
                
        except ImportError:
            self.skipTest("AI module not available")
    
    @patch('ultralytics.YOLO')
    def test_batch_inference_performance(self, mock_yolo_class):
        """Test batch inference performance."""
        try:
            # Setup mock for batch processing
            mock_results = [MagicMock() for _ in range(4)]
            for result in mock_results:
                result.boxes = None
            
            mock_model = MagicMock()
            mock_model.return_value = mock_results
            mock_yolo_class.return_value = mock_model
            
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            
            # Test batch processing if available
            if hasattr(detector, 'detect_batch'):
                benchmark = PerformanceBenchmark()
                
                batch_sizes = [1, 2, 4, 8]
                
                for batch_size in batch_sizes:
                    images = [
                        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                        for _ in range(batch_size)
                    ]
                    
                    benchmark.start()
                    
                    batch_times = []
                    for i in range(20):
                        start = time.time()
                        results = detector.detect_batch(images)
                        end = time.time()
                        
                        batch_times.append(end - start)
                        benchmark.update()
                        
                        self.assertIsNotNone(results)
                        self.assertEqual(len(results), batch_size)
                    
                    benchmark_results = benchmark.stop()
                    
                    avg_time = sum(batch_times) / len(batch_times)
                    throughput = (len(batch_times) * batch_size) / benchmark_results['duration']
                    
                    print(f"\nBatch Inference Performance (batch_size={batch_size}):")
                    print(f"  Average batch time: {avg_time*1000:.2f}ms")
                    print(f"  Time per image: {(avg_time/batch_size)*1000:.2f}ms")
                    print(f"  Throughput: {throughput:.1f} images/sec")
                    
                    # Batch should be more efficient than individual
                    if batch_size > 1:
                        time_per_image = avg_time / batch_size
                        self.assertLess(time_per_image, 0.008)  # Should be faster per image
            else:
                self.skipTest("Batch processing not available")
                
        except ImportError:
            self.skipTest("AI module not available")


class TestDatabasePerformance(unittest.TestCase):
    """Test database performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_database_write_performance(self):
        """Test database write performance."""
        try:
            from data.database import PCBDatabase
            
            database = PCBDatabase(self.db_path)
            benchmark = PerformanceBenchmark()
            
            benchmark.start()
            
            write_times = []
            
            # Test bulk writes
            for i in range(1000):
                timestamp = datetime.now()
                defects = ["Test Defect"] if i % 5 == 0 else []
                locations = [{"bbox": [100, 100, 200, 200]}] if defects else []
                confidences = [0.8] if defects else []
                
                start = time.time()
                inspection_id = database.save_inspection_metadata(
                    timestamp=timestamp,
                    defects=defects,
                    locations=locations,
                    confidence_scores=confidences,
                    raw_image_shape=(480, 640),
                    focus_score=120.0,
                    processing_time=0.1
                )
                end = time.time()
                
                write_times.append(end - start)
                benchmark.update()
                
                self.assertIsInstance(inspection_id, int)
            
            results = benchmark.stop()
            
            avg_write_time = sum(write_times) / len(write_times)
            throughput = len(write_times) / results['duration']
            
            print(f"\nDatabase Write Performance:")
            print(f"  Average write time: {avg_write_time*1000:.2f}ms")
            print(f"  Throughput: {throughput:.1f} writes/sec")
            print(f"  Total duration: {results['duration']:.2f}s")
            print(f"  Memory increase: {results['memory_increase']:.1f}MB")
            
            # Should achieve high write throughput
            self.assertLess(avg_write_time, 0.01)  # 10ms per write
            self.assertGreater(throughput, 100)    # 100+ writes/sec
            
            database.close()
            
        except ImportError:
            self.skipTest("Data module not available")
    
    def test_database_read_performance(self):
        """Test database read performance."""
        try:
            from data.database import PCBDatabase
            
            database = PCBDatabase(self.db_path)
            
            # Populate with test data
            for i in range(500):
                timestamp = datetime.now() - timedelta(seconds=i)
                defects = [f"Defect_{i}"] if i % 3 == 0 else []
                
                database.save_inspection_metadata(
                    timestamp=timestamp,
                    defects=defects,
                    locations=[{}] if defects else [],
                    confidence_scores=[0.8] if defects else [],
                    raw_image_shape=(480, 640),
                    focus_score=100.0 + i
                )
            
            benchmark = PerformanceBenchmark()
            benchmark.start()
            
            read_times = []
            
            # Test various read operations
            for i in range(100):
                start = time.time()
                
                if i % 3 == 0:
                    # Recent inspections
                    results = database.get_recent_inspections(50)
                elif i % 3 == 1:
                    # Defect statistics
                    results = database.get_defect_statistics()
                else:
                    # Recent inspections with different limit
                    results = database.get_recent_inspections(10)
                
                end = time.time()
                
                read_times.append(end - start)
                benchmark.update()
                
                self.assertIsNotNone(results)
            
            benchmark_results = benchmark.stop()
            
            avg_read_time = sum(read_times) / len(read_times)
            throughput = len(read_times) / benchmark_results['duration']
            
            print(f"\nDatabase Read Performance:")
            print(f"  Average read time: {avg_read_time*1000:.2f}ms")
            print(f"  Throughput: {throughput:.1f} reads/sec")
            print(f"  Memory increase: {benchmark_results['memory_increase']:.1f}MB")
            
            # Should achieve fast read performance
            self.assertLess(avg_read_time, 0.05)   # 50ms per read
            self.assertGreater(throughput, 20)     # 20+ reads/sec
            
            database.close()
            
        except ImportError:
            self.skipTest("Data module not available")


class TestSystemLoadTesting(unittest.TestCase):
    """Test system performance under load."""
    
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
    def test_sustained_operation_performance(self, mock_yolo_class):
        """Test performance during sustained operation."""
        try:
            # Setup mocks
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
                from ai.inference import PCBDefectDetector
                from data.database import PCBDatabase
                
                # Initialize components
                camera = BaslerCamera(CAMERA_CONFIG)
                preprocessor = ImagePreprocessor()
                ai_detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
                database = PCBDatabase(self.db_path)
                
                benchmark = PerformanceBenchmark()
                benchmark.start()
                
                inspection_times = []
                memory_samples = []
                
                # Simulate sustained operation (5 minutes at 1 inspection/second)
                for i in range(300):  # 5 minutes worth
                    start = time.time()
                    
                    # Complete inspection workflow
                    raw_image = camera.capture()
                    processed_image = preprocessor.process(raw_image)
                    detection_results = ai_detector.detect(processed_image)
                    
                    # Save to database
                    inspection_id = database.save_inspection_metadata(
                        timestamp=datetime.now(),
                        defects=[],
                        locations=[],
                        confidence_scores=[],
                        raw_image_shape=raw_image.shape,
                        focus_score=120.0,
                        processing_time=0.1
                    )
                    
                    end = time.time()
                    inspection_times.append(end - start)
                    
                    # Sample memory every 10 inspections
                    if i % 10 == 0:
                        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        memory_samples.append(current_memory)
                        benchmark.update()
                    
                    # Target 1 inspection per second
                    elapsed = end - start
                    if elapsed < 1.0:
                        time.sleep(1.0 - elapsed)
                
                results = benchmark.stop()
                
                # Performance analysis
                avg_inspection_time = sum(inspection_times) / len(inspection_times)
                max_inspection_time = max(inspection_times)
                min_inspection_time = min(inspection_times)
                
                memory_growth = memory_samples[-1] - memory_samples[0] if len(memory_samples) > 1 else 0
                
                print(f"\nSustained Operation Performance (300 inspections):")
                print(f"  Average inspection time: {avg_inspection_time*1000:.2f}ms")
                print(f"  Min inspection time: {min_inspection_time*1000:.2f}ms")
                print(f"  Max inspection time: {max_inspection_time*1000:.2f}ms")
                print(f"  Total duration: {results['duration']:.2f}s")
                print(f"  Memory growth: {memory_growth:.1f}MB")
                print(f"  Peak memory: {results['memory_peak']:.1f}MB")
                
                # Performance assertions
                self.assertLess(avg_inspection_time, 0.5)   # Should average <500ms
                self.assertLess(max_inspection_time, 1.0)   # Should never exceed 1s
                self.assertLess(memory_growth, 50)          # Memory growth <50MB
                
                # Cleanup
                database.close()
                camera.close()
                
        except ImportError:
            self.skipTest("Required modules not available")
    
    def test_concurrent_load_performance(self):
        """Test performance under concurrent load."""
        try:
            from data.database import PCBDatabase
            from analytics.analyzer import DefectAnalyzer
            
            database = PCBDatabase(self.db_path)
            analyzer = DefectAnalyzer(database)
            
            benchmark = PerformanceBenchmark()
            benchmark.start()
            
            results = {'writes': 0, 'reads': 0, 'errors': 0}
            
            def writer_thread():
                """Thread that writes data continuously."""
                try:
                    for i in range(100):
                        timestamp = datetime.now()
                        defects = [f"Defect_{i}"] if i % 4 == 0 else []
                        
                        database.save_inspection_metadata(
                            timestamp=timestamp,
                            defects=defects,
                            locations=[{}] if defects else [],
                            confidence_scores=[0.8] if defects else [],
                            raw_image_shape=(480, 640),
                            focus_score=100.0
                        )
                        results['writes'] += 1
                        time.sleep(0.01)
                except Exception:
                    results['errors'] += 1
            
            def reader_thread():
                """Thread that reads data continuously."""
                try:
                    for i in range(50):
                        if i % 2 == 0:
                            analyzer.get_realtime_stats()
                        else:
                            database.get_recent_inspections(20)
                        results['reads'] += 1
                        time.sleep(0.02)
                except Exception:
                    results['errors'] += 1
            
            # Start concurrent threads
            threads = []
            
            # 2 writer threads
            for _ in range(2):
                thread = threading.Thread(target=writer_thread)
                threads.append(thread)
                thread.start()
            
            # 1 reader thread
            thread = threading.Thread(target=reader_thread)
            threads.append(thread)
            thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            benchmark_results = benchmark.stop()
            
            print(f"\nConcurrent Load Performance:")
            print(f"  Total writes: {results['writes']}")
            print(f"  Total reads: {results['reads']}")
            print(f"  Errors: {results['errors']}")
            print(f"  Duration: {benchmark_results['duration']:.2f}s")
            print(f"  Write throughput: {results['writes']/benchmark_results['duration']:.1f}/sec")
            print(f"  Read throughput: {results['reads']/benchmark_results['duration']:.1f}/sec")
            
            # Should handle concurrent load without errors
            self.assertEqual(results['errors'], 0)
            self.assertGreater(results['writes'], 150)  # Should complete most writes
            self.assertGreater(results['reads'], 40)    # Should complete most reads
            
            database.close()
            
        except ImportError:
            self.skipTest("Data modules not available")


class TestMemoryLeakDetection(unittest.TestCase):
    """Test for memory leaks during extended operation."""
    
    @patch('ultralytics.YOLO')
    def test_ai_memory_leak(self, mock_yolo_class):
        """Test for memory leaks in AI operations."""
        try:
            mock_results = MagicMock()
            mock_results.boxes = None
            mock_model = MagicMock()
            mock_model.return_value = [mock_results]
            mock_yolo_class.return_value = mock_model
            
            from ai.inference import PCBDefectDetector
            
            detector = PCBDefectDetector("test_model.pt", "cpu", 0.5)
            
            # Baseline memory
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Process many images
            for i in range(500):
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                results = detector.detect(test_image)
                
                # Check memory every 50 iterations
                if i % 50 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    # Memory increase should be bounded
                    self.assertLess(memory_increase, 100)  # Less than 100MB increase
            
            # Final memory check
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            total_increase = final_memory - initial_memory
            
            print(f"\nAI Memory Leak Test:")
            print(f"  Initial memory: {initial_memory:.1f}MB")
            print(f"  Final memory: {final_memory:.1f}MB")
            print(f"  Total increase: {total_increase:.1f}MB")
            
            # Should not have significant memory leak
            self.assertLess(total_increase, 50)  # Less than 50MB total increase
            
        except ImportError:
            self.skipTest("AI module not available")


def generate_performance_report():
    """Generate comprehensive performance report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
        },
        'test_results': {}
    }
    
    return report


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run performance tests
    unittest.main(verbosity=2)
    
    # Generate performance report
    report = generate_performance_report()
    
    with open('tests/performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nPerformance report saved to: tests/performance_report.json")
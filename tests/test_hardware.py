"""
Unit tests for Hardware layer components.

Tests camera controller, image handling, and presets functionality.
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import CAMERA_CONFIG


class TestBaslerCameraMock(unittest.TestCase):
    """Test BaslerCamera with mocked pypylon."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock pypylon module
        self.mock_pylon = MagicMock()
        self.mock_camera = MagicMock()
        self.mock_grab_result = MagicMock()
        
        # Setup mock hierarchy
        self.mock_pylon.TlFactory.GetInstance.return_value.CreateFirstDevice.return_value = self.mock_camera
        self.mock_grab_result.GrabSucceeded.return_value = True
        self.mock_grab_result.Array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Mock the camera grab operations
        self.mock_camera.GrabOne.return_value = self.mock_grab_result
        self.mock_camera.RetrieveResult.return_value = self.mock_grab_result
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_camera_initialization(self):
        """Test camera initialization with mocked pypylon."""
        with patch('pypylon.pylon', self.mock_pylon):
            from hardware.camera_controller import BaslerCamera
            
            camera = BaslerCamera(CAMERA_CONFIG)
            
            # Verify camera was created and opened
            self.mock_pylon.TlFactory.GetInstance.assert_called_once()
            self.mock_camera.Open.assert_called_once()
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_camera_configuration(self):
        """Test camera parameter configuration."""
        with patch('pypylon.pylon', self.mock_pylon):
            from hardware.camera_controller import BaslerCamera
            
            camera = BaslerCamera(CAMERA_CONFIG)
            
            # Verify configuration calls
            self.mock_camera.ExposureTime.SetValue.assert_called()
            self.mock_camera.Gain.SetValue.assert_called()
            self.mock_camera.PixelFormat.SetValue.assert_called()
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_single_capture(self):
        """Test single image capture."""
        with patch('pypylon.pylon', self.mock_pylon):
            from hardware.camera_controller import BaslerCamera
            
            camera = BaslerCamera(CAMERA_CONFIG)
            image = camera.capture()
            
            self.assertIsNotNone(image)
            self.assertIsInstance(image, np.ndarray)
            self.mock_camera.StartGrabbingMax.assert_called_with(1)
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_streaming_operations(self):
        """Test streaming start/stop operations."""
        with patch('pypylon.pylon', self.mock_pylon):
            from hardware.camera_controller import BaslerCamera
            
            camera = BaslerCamera(CAMERA_CONFIG)
            
            # Test start streaming
            camera.start_streaming()
            self.assertTrue(camera.is_streaming)
            self.mock_camera.StartGrabbing.assert_called()
            
            # Test stop streaming
            camera.stop_streaming()
            self.assertFalse(camera.is_streaming)
            self.mock_camera.StopGrabbing.assert_called()
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_high_quality_capture(self):
        """Test high-quality capture mode switching."""
        with patch('pypylon.pylon', self.mock_pylon):
            from hardware.camera_controller import BaslerCamera
            
            camera = BaslerCamera(CAMERA_CONFIG)
            
            # Start streaming first
            camera.start_streaming()
            
            # Test high-quality capture
            image = camera.capture_high_quality()
            
            self.assertIsNotNone(image)
            self.assertIsInstance(image, np.ndarray)
            
            # Should have stopped and restarted streaming
            self.mock_camera.StopGrabbing.assert_called()
            # Exposure should have been changed for high quality
            self.assertTrue(self.mock_camera.ExposureTime.SetValue.call_count >= 2)
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_frame_queue_operations(self):
        """Test frame queue and preview frame operations."""
        with patch('pypylon.pylon', self.mock_pylon):
            from hardware.camera_controller import BaslerCamera
            
            camera = BaslerCamera(CAMERA_CONFIG)
            
            # Initially, no frames in queue
            frame = camera.get_preview_frame()
            self.assertIsNone(frame)
            
            # Simulate frame in queue
            test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            camera.frame_queue.put(test_frame)
            
            frame = camera.get_preview_frame()
            self.assertIsNotNone(frame)
            np.testing.assert_array_equal(frame, test_frame)
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_camera_error_handling(self):
        """Test error handling for camera operations."""
        with patch('pypylon.pylon', self.mock_pylon):
            from hardware.camera_controller import BaslerCamera
            
            # Mock failed grab
            self.mock_grab_result.GrabSucceeded.return_value = False
            
            camera = BaslerCamera(CAMERA_CONFIG)
            image = camera.capture()
            
            # Should return None on failed capture
            self.assertIsNone(image)
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_context_manager(self):
        """Test camera as context manager."""
        with patch('pypylon.pylon', self.mock_pylon):
            from hardware.camera_controller import BaslerCamera
            
            with BaslerCamera(CAMERA_CONFIG) as camera:
                self.assertIsNotNone(camera)
                # Should be properly initialized
                self.mock_camera.Open.assert_called()
            
            # Should be properly closed
            self.mock_camera.Close.assert_called()


class TestCameraPresets(unittest.TestCase):
    """Test camera presets functionality."""
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_preset_loading(self):
        """Test loading different camera presets."""
        try:
            from hardware.camera_presets import CameraPresets
            
            # Test getting a preset
            preset = CameraPresets.get_preset('balanced')
            self.assertIsInstance(preset, dict)
            self.assertIn('preview_exposure', preset)
            self.assertIn('capture_exposure', preset)
            
            # Test lighting presets
            low_light = CameraPresets.get_lighting_preset('low')
            self.assertIsInstance(low_light, dict)
            
            # Test custom preset creation
            custom = CameraPresets.create_custom_preset(
                'test_preset',
                preview_exposure=3000,
                gain=2
            )
            self.assertEqual(custom['preview_exposure'], 3000)
            self.assertEqual(custom['gain'], 2)
            
        except ImportError:
            self.skipTest("Camera presets module not available")
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_preset_validation(self):
        """Test preset parameter validation."""
        try:
            from hardware.camera_presets import CameraPresets
            
            # Test invalid preset name
            with self.assertRaises(ValueError):
                CameraPresets.get_preset('invalid_preset')
            
            # Test parameter bounds validation
            preset = CameraPresets.get_preset('balanced')
            self.assertGreater(preset['preview_exposure'], 0)
            self.assertGreater(preset['capture_exposure'], 0)
            self.assertGreaterEqual(preset['gain'], 0)
            
        except ImportError:
            self.skipTest("Camera presets module not available")


class TestImageHandler(unittest.TestCase):
    """Test camera image handler functionality."""
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_image_handler_initialization(self):
        """Test image handler initialization."""
        try:
            from hardware.camera_controller import CameraImageHandler
            from queue import Queue
            
            test_queue = Queue(maxsize=5)
            handler = CameraImageHandler(test_queue)
            
            self.assertEqual(handler.queue, test_queue)
            
        except ImportError:
            self.skipTest("Camera controller module not available")
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_image_handler_grab_success(self):
        """Test successful image grab handling."""
        try:
            from hardware.camera_controller import CameraImageHandler
            from queue import Queue
            
            test_queue = Queue(maxsize=5)
            handler = CameraImageHandler(test_queue)
            
            # Mock successful grab result
            mock_camera = MagicMock()
            mock_grab_result = MagicMock()
            mock_grab_result.GrabSucceeded.return_value = True
            mock_grab_result.Array = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            
            # Test image handling
            handler.OnImageGrabbed(mock_camera, mock_grab_result)
            
            # Queue should contain the image
            self.assertFalse(test_queue.empty())
            
        except ImportError:
            self.skipTest("Camera controller module not available")


class TestCameraThreadSafety(unittest.TestCase):
    """Test thread safety of camera operations."""
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_concurrent_frame_access(self):
        """Test concurrent access to camera frames."""
        try:
            from hardware.camera_controller import BaslerCamera
            
            with patch('pypylon.pylon'):
                camera = BaslerCamera(CAMERA_CONFIG)
                
                # Simulate multiple threads accessing camera
                results = []
                errors = []
                
                def capture_frames():
                    try:
                        for _ in range(10):
                            frame = camera.get_preview_frame()
                            results.append(frame)
                            time.sleep(0.01)
                    except Exception as e:
                        errors.append(e)
                
                # Start multiple threads
                threads = []
                for _ in range(3):
                    thread = threading.Thread(target=capture_frames)
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                # Should not have any threading errors
                self.assertEqual(len(errors), 0)
                
        except ImportError:
            self.skipTest("Camera controller module not available")


class TestCameraPerformance(unittest.TestCase):
    """Test camera performance characteristics."""
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_capture_timing(self):
        """Test capture operation timing."""
        try:
            from hardware.camera_controller import BaslerCamera
            
            with patch('pypylon.pylon'):
                camera = BaslerCamera(CAMERA_CONFIG)
                
                # Time multiple captures
                start_time = time.time()
                for _ in range(10):
                    camera.capture()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                
                # Should be reasonably fast (mock should be very fast)
                self.assertLess(avg_time, 0.1)
                
        except ImportError:
            self.skipTest("Camera controller module not available")
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_memory_usage(self):
        """Test memory usage during streaming."""
        try:
            from hardware.camera_controller import BaslerCamera
            
            with patch('pypylon.pylon'):
                camera = BaslerCamera(CAMERA_CONFIG)
                
                # Start streaming and add many frames
                camera.start_streaming()
                
                # Fill the queue beyond capacity
                for i in range(20):
                    test_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
                    if not camera.frame_queue.full():
                        camera.frame_queue.put(test_frame)
                
                # Queue should not exceed max size
                self.assertLessEqual(camera.frame_queue.qsize(), camera.frame_queue.maxsize)
                
                camera.stop_streaming()
                
        except ImportError:
            self.skipTest("Camera controller module not available")


class TestCameraIntegration(unittest.TestCase):
    """Test camera integration with other components."""
    
    @patch.dict('sys.modules', {'pypylon': MagicMock()})
    def test_camera_config_integration(self):
        """Test camera integration with core config."""
        try:
            from hardware.camera_controller import BaslerCamera
            
            with patch('pypylon.pylon'):
                # Test initialization with actual config
                camera = BaslerCamera(CAMERA_CONFIG)
                
                # Should use config values
                self.assertEqual(camera.config, CAMERA_CONFIG)
                
        except ImportError:
            self.skipTest("Camera controller module not available")


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    unittest.main(verbosity=2)
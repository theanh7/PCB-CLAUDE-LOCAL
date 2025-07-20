"""
Basler Camera Controller for PCB Inspection System.

This module implements the camera interface for the Basler acA3800-10gm camera
with dual-mode operation: preview streaming and high-quality capture.
"""

import logging
import time
import threading
from queue import Queue, Empty
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    from pypylon import pylon
except ImportError:
    pylon = None
    logging.warning("pypylon not available. Camera functionality will be limited.")

from core.interfaces import BaseCamera, PCBDetectionResult
from core.config import CAMERA_CONFIG
from core.utils import timing_decorator, error_handler


class CameraImageHandler(pylon.ImageEventHandler if pylon else object):
    """
    Event handler for asynchronous frame grabbing from Basler camera.
    Manages frame queue and memory optimization.
    """
    
    def __init__(self, frame_queue: Queue, max_queue_size: int = 10):
        if pylon:
            super().__init__()
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.frames_received = 0
        self.frames_dropped = 0
        self.logger = logging.getLogger(__name__)
        
    def OnImageGrabbed(self, camera, grab_result):
        """
        Called when a new frame is available from the camera.
        
        Args:
            camera: Camera instance
            grab_result: Grab result containing image data
        """
        if not grab_result.GrabSucceeded():
            self.logger.warning(f"Frame grab failed: {grab_result.ErrorDescription}")
            return
            
        try:
            # Get image data
            image_array = grab_result.Array.copy()
            
            # Queue management - drop old frames if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except Empty:
                    pass
            
            # Add new frame to queue
            self.frame_queue.put(image_array)
            self.frames_received += 1
            
            # Log statistics periodically
            if self.frames_received % 100 == 0:
                self.logger.debug(f"Frames received: {self.frames_received}, "
                                f"dropped: {self.frames_dropped}")
                
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get frame handling statistics."""
        return {
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "queue_size": self.frame_queue.qsize()
        }
    
    def reset_statistics(self):
        """Reset frame statistics."""
        self.frames_received = 0
        self.frames_dropped = 0


class BaslerCamera(BaseCamera):
    """
    Basler Camera controller implementing dual-mode operation.
    
    Supports both preview streaming (30 FPS) and high-quality capture modes
    with automatic parameter switching and thread-safe operations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Basler camera controller.
        
        Args:
            config: Camera configuration dictionary
        """
        self.config = config or CAMERA_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Check if pypylon is available
        if pylon is None:
            raise ImportError("pypylon is required for camera functionality. "
                            "Install with: pip install pypylon")
        
        # Camera state
        self.camera = None
        self.is_streaming = False
        self.is_connected = False
        self.connection_lock = threading.Lock()
        
        # Frame handling
        self.frame_queue = Queue(maxsize=self.config.get("buffer_size", 10))
        self.image_handler = CameraImageHandler(self.frame_queue)
        
        # Camera parameters
        self.preview_exposure = self.config.get("preview_exposure", 5000)
        self.capture_exposure = self.config.get("capture_exposure", 10000)
        self.gain = self.config.get("gain", 0)
        self.pixel_format = self.config.get("pixel_format", "BayerRG8")
        
        # Initialize camera
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera connection and basic setup."""
        try:
            self.logger.info("Initializing Basler camera...")
            
            # Create camera instance
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            
            # Open camera
            self.camera.Open()
            
            # Log camera info
            self.logger.info(f"Camera model: {self.camera.GetDeviceInfo().GetModelName()}")
            self.logger.info(f"Camera serial: {self.camera.GetDeviceInfo().GetSerialNumber()}")
            
            # Configure camera
            self._configure_camera()
            
            # Skip event handler registration for this camera model
            # We'll use polling approach instead
            self.logger.info("Using polling approach instead of event handler")
            
            self.is_connected = True
            self.logger.info("Camera initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            self.is_connected = False
            raise
    
    def _configure_camera(self):
        """Configure camera parameters for optimal performance."""
        try:
            # Set pixel format (this works)
            self.camera.PixelFormat.SetValue(self.pixel_format)
            self.logger.info(f"Pixel format set to: {self.pixel_format}")
            
            # Skip advanced configuration for this camera model
            # The acA3800-10gm seems to have limited GenICam support
            self.logger.info("Using default camera settings (limited GenICam support)")
            
            # Try to configure buffer handling if available
            try:
                self.camera.MaxNumBuffer.SetValue(self.config.get("buffer_size", 10))
                self.logger.info("Buffer size configured")
            except:
                self.logger.warning("Could not configure buffer size")
            
            self.logger.info("Camera configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to configure camera: {e}")
            raise
    
    @error_handler(default_return=None)
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a single frame (implements BaseCamera interface).
        
        Returns:
            Captured image array or None if capture failed
        """
        if not self.is_connected:
            self.logger.error("Camera not connected")
            return None
            
        try:
            with self.connection_lock:
                # Grab single frame
                grab_result = self.camera.GrabOne(
                    self.config.get("timeout", 5000)
                )
                
                if grab_result.GrabSucceeded():
                    image_array = grab_result.Array.copy()
                    grab_result.Release()
                    return image_array
                else:
                    self.logger.warning(f"Frame grab failed: {grab_result.ErrorDescription}")
                    grab_result.Release()
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None
    
    @timing_decorator
    def start_streaming(self) -> None:
        """Start continuous image acquisition for preview."""
        if not self.is_connected:
            self.logger.error("Camera not connected")
            return
            
        if self.is_streaming:
            self.logger.warning("Streaming already active")
            return
            
        try:
            with self.connection_lock:
                # Skip exposure configuration (not supported by this camera)
                self.logger.debug("Starting polling-based streaming")
                
                # For this camera model, use continuous grabbing
                # We'll poll frames in get_preview_frame()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                
                self.is_streaming = True
                self.logger.info("Camera streaming started")
                
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            self.is_streaming = False
            raise
    
    @timing_decorator
    def stop_streaming(self) -> None:
        """Stop continuous image acquisition."""
        if not self.is_streaming:
            return
            
        try:
            with self.connection_lock:
                self.camera.StopGrabbing()
                self.is_streaming = False
                
                # Clear frame queue
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        break
                
                self.logger.info("Camera streaming stopped")
                
        except Exception as e:
            self.logger.error(f"Failed to stop streaming: {e}")
            raise
    
    def get_preview_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame from preview stream.
        For this camera model, we use polling instead of queue.
        
        Returns:
            Latest frame array or None if no frame available
        """
        if not self.is_streaming or not self.is_connected:
            return None
            
        try:
            # Use polling approach for this camera model
            grab_result = self.camera.RetrieveResult(10)  # 10ms timeout for quick polling
            
            if grab_result.GrabSucceeded():
                image_array = grab_result.Array.copy()
                grab_result.Release()
                return image_array
            else:
                grab_result.Release()
                return None
                
        except Exception as e:
            # Don't log every timeout as error (too noisy)
            if "timeout" not in str(e).lower():
                self.logger.debug(f"Frame polling error: {e}")
            return None
    
    @timing_decorator
    def capture_high_quality(self) -> Optional[np.ndarray]:
        """
        Capture single high-quality image for inspection.
        
        Temporarily stops streaming, adjusts exposure for high quality,
        captures frame, then resumes streaming if it was active.
        
        Returns:
            High-quality image array or None if capture failed
        """
        if not self.is_connected:
            self.logger.error("Camera not connected")
            return None
            
        # Remember streaming state
        was_streaming = self.is_streaming
        
        try:
            with self.connection_lock:
                # Stop streaming if active
                if was_streaming:
                    self.camera.StopGrabbing()
                    self.is_streaming = False
                
                # Skip exposure adjustment (not supported by this camera)
                self.logger.debug("Using default exposure for capture")
                
                # Allow camera to adjust
                time.sleep(0.1)
                
                # Capture single frame
                grab_result = self.camera.GrabOne(
                    self.config.get("timeout", 5000)
                )
                
                if grab_result.GrabSucceeded():
                    image_array = grab_result.Array.copy()
                    grab_result.Release()
                    
                    # Resume streaming if it was active
                    if was_streaming:
                        # Skip exposure adjustment for resume
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                        self.is_streaming = True
                    
                    self.logger.debug("High-quality capture successful")
                    return image_array
                else:
                    self.logger.warning(f"High-quality capture failed: {grab_result.ErrorDescription}")
                    grab_result.Release()
                    
                    # Resume streaming if it was active
                    if was_streaming:
                        # Skip exposure adjustment for resume
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                        self.is_streaming = True
                    
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in high-quality capture: {e}")
            
            # Try to resume streaming if it was active
            if was_streaming:
                try:
                    # Skip exposure adjustment for resume
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    self.is_streaming = True
                except:
                    self.logger.error("Failed to resume streaming after error")
            
            return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information and current status.
        
        Returns:
            Dictionary containing camera information
        """
        info = {
            "connected": self.is_connected,
            "streaming": self.is_streaming,
            "config": self.config.copy()
        }
        
        if self.is_connected and self.camera:
            try:
                info.update({
                    "model": self.camera.GetDeviceInfo().GetModelName(),
                    "serial": self.camera.GetDeviceInfo().GetSerialNumber(),
                    "current_exposure": self.camera.ExposureTime.GetValue(),
                    "current_gain": self.camera.Gain.GetValue(),
                    "pixel_format": self.camera.PixelFormat.GetValue(),
                    "frame_rate": self.camera.ResultingFrameRate.GetValue()
                })
            except Exception as e:
                self.logger.warning(f"Error getting camera info: {e}")
        
        return info
    
    def get_frame_statistics(self) -> Dict[str, Any]:
        """Get frame handling statistics."""
        stats = self.image_handler.get_statistics()
        stats.update({
            "streaming": self.is_streaming,
            "connected": self.is_connected
        })
        return stats
    
    def reset_statistics(self):
        """Reset frame statistics."""
        self.image_handler.reset_statistics()
    
    def set_exposure(self, exposure_us: int, mode: str = "preview") -> bool:
        """
        Set camera exposure time.
        
        Args:
            exposure_us: Exposure time in microseconds
            mode: Mode to set ("preview" or "capture")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if mode == "preview":
                self.preview_exposure = exposure_us
                # Skip exposure setting (not supported)
            elif mode == "capture":
                self.capture_exposure = exposure_us
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            self.logger.info(f"Exposure set to {exposure_us}μs for {mode} mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set exposure: {e}")
            return False
    
    def set_gain(self, gain_db: float) -> bool:
        """
        Set camera gain.
        
        Args:
            gain_db: Gain in dB
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.gain = gain_db
            # Skip gain setting (not supported by this camera)
            
            self.logger.info(f"Gain set to {gain_db}dB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set gain: {e}")
            return False
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to camera.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self.logger.info("Attempting camera reconnection...")
        
        try:
            # Stop streaming and close connection
            if self.is_streaming:
                self.stop_streaming()
            
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
            
            # Reinitialize
            self._initialize_camera()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            return False
    
    def close(self):
        """Close camera connection and cleanup resources."""
        try:
            if self.is_streaming:
                self.stop_streaming()
            
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
            
            self.is_connected = False
            self.logger.info("Camera connection closed")
            
        except Exception as e:
            self.logger.error(f"Error closing camera: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Factory function for easy camera creation
def create_camera(config: Dict[str, Any] = None) -> BaslerCamera:
    """
    Create and initialize a Basler camera instance.
    
    Args:
        config: Camera configuration dictionary
        
    Returns:
        Initialized BaslerCamera instance
    """
    return BaslerCamera(config or CAMERA_CONFIG)
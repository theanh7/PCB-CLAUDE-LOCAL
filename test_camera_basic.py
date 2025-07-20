#!/usr/bin/env python3
"""
Basic camera test to verify connection and capture.
Minimal configuration to test camera functionality.
"""

from pypylon import pylon
import numpy as np
import cv2
import time

def test_basic_camera():
    """Test basic camera functionality with minimal configuration."""
    try:
        print("Testing basic camera functionality...")
        
        # Create camera instance
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        
        # Open camera
        camera.Open()
        
        print(f"Camera Model: {camera.GetDeviceInfo().GetModelName()}")
        print(f"Camera Serial: {camera.GetDeviceInfo().GetSerialNumber()}")
        
        # Minimal configuration
        camera.PixelFormat.SetValue("Mono8")
        camera.ExposureTime.SetValue(10000)  # 10ms
        
        print("Camera configured successfully")
        
        # Test single capture
        print("Testing single frame capture...")
        camera.StartGrabbingMax(1)
        grab_result = camera.RetrieveResult(5000)  # 5 second timeout
        
        if grab_result.GrabSucceeded():
            image = grab_result.Array
            print(f"Captured image: {image.shape}, dtype: {image.dtype}")
            print(f"Image stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}")
            
            # Save test image
            cv2.imwrite("test_capture.jpg", image)
            print("Test image saved as 'test_capture.jpg'")
            
        else:
            print(f"Capture failed: {grab_result.ErrorDescription}")
            
        grab_result.Release()
        
        # Test streaming
        print("\nTesting streaming for 5 seconds...")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5.0:
            grab_result = camera.RetrieveResult(1000)
            if grab_result.GrabSucceeded():
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Captured {frame_count} frames")
            grab_result.Release()
            
        camera.StopGrabbing()
        
        fps = frame_count / 5.0
        print(f"Streaming test complete: {frame_count} frames in 5 seconds ({fps:.1f} FPS)")
        
        # Close camera
        camera.Close()
        print("Camera test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Camera test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_camera()
    if success:
        print("\n✓ Camera is working correctly")
    else:
        print("\n✗ Camera test failed")
#!/usr/bin/env python3
"""
Test camera with polling instead of event handler.
"""

from pypylon import pylon
import cv2
import numpy as np
import time

def test_camera_polling():
    """Test camera with direct polling approach."""
    try:
        print("Testing camera with polling approach...")
        
        # Create camera
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        
        print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")
        
        # Set pixel format
        camera.PixelFormat.SetValue("Mono8")
        
        # Start grabbing
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print("Camera grabbing started")
        
        frame_count = 0
        start_time = time.time()
        
        # Test for 5 seconds
        while time.time() - start_time < 5.0:
            grab_result = camera.RetrieveResult(100)  # 100ms timeout
            
            if grab_result.GrabSucceeded():
                frame_count += 1
                image = grab_result.Array
                
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: {image.shape}, min={image.min()}, max={image.max()}")
                    
                # Save first frame for testing
                if frame_count == 1:
                    cv2.imwrite("polling_test_frame.jpg", image)
                    print("First frame saved as polling_test_frame.jpg")
            
            grab_result.Release()
        
        camera.StopGrabbing()
        camera.Close()
        
        fps = frame_count / 5.0
        print(f"\nPolling test complete: {frame_count} frames in 5 seconds ({fps:.1f} FPS)")
        
        return frame_count > 0
        
    except Exception as e:
        print(f"Polling test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_camera_polling()
    if success:
        print("SUCCESS: Camera polling works")
    else:
        print("FAILED: Camera polling failed")
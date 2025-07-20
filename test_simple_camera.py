#!/usr/bin/env python3
"""
Test camera with minimal configuration.
"""

from pypylon import pylon
import cv2
import numpy as np

def test_minimal_camera():
    """Test camera with absolutely minimal configuration."""
    try:
        print("Testing minimal camera setup...")
        
        # Create camera
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        
        # Open camera
        camera.Open()
        
        print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")
        
        # Try to set only pixel format, skip everything else
        try:
            camera.PixelFormat.SetValue("Mono8")
            print("Pixel format set to Mono8")
        except Exception as e:
            print(f"Could not set pixel format: {e}")
        
        # Try basic capture without any other configuration
        print("Attempting capture...")
        camera.StartGrabbingMax(1)
        grab_result = camera.RetrieveResult(5000)
        
        if grab_result.GrabSucceeded():
            image = grab_result.Array
            print(f"SUCCESS! Captured image: {image.shape}")
            
            # Save image
            cv2.imwrite("minimal_test.jpg", image)
            print("Image saved as minimal_test.jpg")
            
        grab_result.Release()
        camera.Close()
        
        return True
        
    except Exception as e:
        print(f"Minimal test failed: {e}")
        return False

if __name__ == "__main__":
    test_minimal_camera()
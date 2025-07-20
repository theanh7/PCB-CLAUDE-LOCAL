#!/usr/bin/env python3
"""
Test camera with different trigger modes.
"""

from pypylon import pylon
import cv2
import time

def test_camera_trigger_modes():
    """Test different camera trigger modes."""
    try:
        print("Testing camera trigger modes...")
        
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        
        print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")
        
        # Set basic parameters
        camera.PixelFormat.SetValue("Mono8")
        
        # Test different acquisition modes
        print("\nTesting acquisition configurations...")
        
        # Try to check if camera needs special configuration
        try:
            # Check if trigger mode exists and what values are available
            node_map = camera.GetNodeMap()
            
            # Check AcquisitionMode
            try:
                acq_mode_node = node_map.GetNode("AcquisitionMode")
                if acq_mode_node.IsAvailable():
                    print("AcquisitionMode available")
                    # Try different acquisition modes
                    modes_to_try = ["Continuous", "SingleFrame"]
                    for mode in modes_to_try:
                        try:
                            camera.AcquisitionMode.SetValue(mode)
                            print(f"  Set AcquisitionMode to: {mode}")
                            break
                        except:
                            print(f"  Failed to set AcquisitionMode to: {mode}")
                else:
                    print("AcquisitionMode not available")
            except:
                print("Cannot access AcquisitionMode")
            
            # Check TriggerMode
            try:
                trigger_mode_node = node_map.GetNode("TriggerMode")
                if trigger_mode_node.IsAvailable():
                    print("TriggerMode available")
                    try:
                        camera.TriggerMode.SetValue("Off")
                        print("  Set TriggerMode to: Off")
                    except:
                        print("  Failed to set TriggerMode")
                else:
                    print("TriggerMode not available")
            except:
                print("Cannot access TriggerMode")
                
        except Exception as e:
            print(f"Error checking camera configuration: {e}")
        
        # Try simple grabbing
        print("\nTesting simple frame grabbing...")
        
        try:
            # Try single frame grab first
            camera.StartGrabbingMax(1)
            grab_result = camera.RetrieveResult(5000)  # 5 second timeout
            
            if grab_result.GrabSucceeded():
                image = grab_result.Array
                print(f"SUCCESS: Single frame captured: {image.shape}")
                cv2.imwrite("trigger_test_frame.jpg", image)
                print("Frame saved as trigger_test_frame.jpg")
                
                grab_result.Release()
                return True
            else:
                print(f"Single frame grab failed: {grab_result.ErrorDescription}")
                grab_result.Release()
                
        except Exception as e:
            print(f"Single frame grab error: {e}")
        
        camera.Close()
        return False
        
    except Exception as e:
        print(f"Camera trigger test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_camera_trigger_modes()
    if success:
        print("\nSUCCESS: Camera can capture frames")
    else:
        print("\nFAILED: Camera cannot capture frames")
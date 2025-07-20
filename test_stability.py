#!/usr/bin/env python3
"""
Test PCB stability detection with the same detector instance.
"""

import cv2
from processing.pcb_detector import PCBDetector
from core.config import TRIGGER_CONFIG

def test_stability():
    """Test PCB stability by running detection multiple times on same image."""
    
    # Load test image
    image = cv2.imread("trigger_test_frame.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Could not load test image")
        return
    
    print(f"Testing stability with image: {image.shape}")
    
    # Initialize detector (use the same instance for all detections)
    detector = PCBDetector(TRIGGER_CONFIG)
    
    # Run detection multiple times to achieve stability
    print("Running detections to achieve stability...")
    
    for i in range(15):  # More than the 10 required for stability
        result = detector.detect_pcb(image)
        
        print(f"Detection {i+1:2d}: has_pcb={result.has_pcb}, "
              f"stable={result.is_stable}, focus={result.focus_score:.1f}")
        
        if result.has_pcb and result.position:
            x, y, w, h = result.position
            print(f"              position=({x},{y},{w},{h}), "
                  f"stable_frames={detector.stable_frames}")
        
        # Check if we can trigger inspection
        can_trigger = detector.can_trigger_inspection()
        should_trigger = detector.should_trigger_inspection(result)
        
        print(f"              can_trigger={can_trigger}, should_trigger={should_trigger}")
        
        if should_trigger:
            print(f"*** INSPECTION TRIGGERED ON FRAME {i+1} ***")
            detector.trigger_inspection()
            break
    
    # Get final stats
    stability_info = detector.get_stability_info()
    detection_stats = detector.get_detection_stats()
    
    print(f"\nFinal stability info: {stability_info}")
    print(f"Final detection stats: {detection_stats}")
    
    # Test visualization of stable detection
    if result.has_pcb and result.is_stable:
        vis_image = detector.visualize_detection(image, result)
        cv2.imwrite("test_stable_detection.jpg", vis_image)
        print("Stable detection visualization saved as test_stable_detection.jpg")
        return True
    
    return False

if __name__ == "__main__":
    success = test_stability()
    if success:
        print("\nSUCCESS: Achieved stable PCB detection and trigger!")
    else:
        print("\nFAILED: Could not achieve stable detection")
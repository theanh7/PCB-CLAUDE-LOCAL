#!/usr/bin/env python3
"""
Test manual inspection functionality to verify the fix.
"""

import cv2
import time
from main import PCBInspectionSystem
from hardware.camera_controller import BaslerCamera
from ai.inference import PCBDefectDetector
from core.config import CAMERA_CONFIG, AI_CONFIG

def test_manual_inspection():
    """Test manual inspection without GUI."""
    
    print("Testing manual inspection functionality...")
    
    try:
        # Initialize components
        print("1. Initializing camera...")
        camera = BaslerCamera(CAMERA_CONFIG)
        
        print("2. Initializing AI detector...")
        ai_detector = PCBDefectDetector(config=AI_CONFIG)
        
        print("3. Starting camera streaming...")
        camera.start_streaming()
        
        # Give camera time to stabilize
        time.sleep(1.0)
        
        print("4. Capturing high-quality image...")
        # Capture image like manual inspection would
        hq_image = camera.capture_high_quality()
        
        if hq_image is None:
            print("ERROR: Failed to capture image")
            return False
        
        print(f"5. Captured image: {hq_image.shape}")
        
        # Convert to RGB for AI if needed
        if len(hq_image.shape) == 2:
            hq_rgb = cv2.cvtColor(hq_image, cv2.COLOR_GRAY2RGB)
        else:
            hq_rgb = hq_image
        
        print(f"6. RGB image for AI: {hq_rgb.shape}")
        
        print("7. Running AI inference...")
        # This is where the error was occurring
        detection_results = ai_detector.detect(hq_rgb)
        
        print(f"8. AI inference completed successfully!")
        print(f"   Result type: {type(detection_results)}")
        print(f"   Has defects: {hasattr(detection_results, 'has_defects') and detection_results.has_defects}")
        
        if hasattr(detection_results, 'defects'):
            print(f"   Defects found: {len(detection_results.defects)}")
            for i, defect in enumerate(detection_results.defects):
                confidence = detection_results.confidence_scores[i] if i < len(detection_results.confidence_scores) else 0.0
                print(f"     {i+1}. {defect} (confidence: {confidence:.3f})")
        
        # Test the _extract_results method that was failing
        print("9. Testing result extraction...")
        
        # Create a mock system to test _extract_results
        system = PCBInspectionSystem()
        
        try:
            defects, locations, confidences = system._extract_results(detection_results)
            print(f"   Extraction successful!")
            print(f"   Extracted: {len(defects)} defects, {len(locations)} locations, {len(confidences)} confidences")
            
        except Exception as e:
            print(f"   ERROR in extraction: {e}")
            return False
        
        # Cleanup
        camera.stop_streaming()
        
        print("✅ Manual inspection test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Manual inspection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_manual_inspection()
    if success:
        print("\n🎉 Manual inspection is working correctly!")
    else:
        print("\n💥 Manual inspection still has issues!")
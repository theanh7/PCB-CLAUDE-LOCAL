#!/usr/bin/env python3
"""
Test preview loop functionality to debug GUI issues.
"""

import time
import logging
from core.config import CAMERA_CONFIG, TRIGGER_CONFIG
from hardware.camera_controller import BaslerCamera
from processing.pcb_detector import PCBDetector

def test_preview_loop():
    """Test the preview loop components individually."""
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    try:
        print("Testing preview loop components...")
        
        # Initialize camera
        print("1. Initializing camera...")
        camera = BaslerCamera(CAMERA_CONFIG)
        
        # Initialize PCB detector
        print("2. Initializing PCB detector...")
        pcb_detector = PCBDetector(TRIGGER_CONFIG)
        
        # Start streaming
        print("3. Starting camera streaming...")
        camera.start_streaming()
        
        # Test frame capture for 10 seconds
        print("4. Testing frame capture...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10.0:
            # Get frame
            raw_frame = camera.get_preview_frame()
            
            if raw_frame is not None:
                frame_count += 1
                
                # Test PCB detection every 10 frames
                if frame_count % 10 == 0:
                    detection_result = pcb_detector.detect_pcb(raw_frame)
                    preview_gray = pcb_detector.debayer_to_gray(raw_frame)
                    
                    print(f"Frame {frame_count}: "
                          f"raw_shape={raw_frame.shape}, "
                          f"preview_shape={preview_gray.shape if preview_gray is not None else 'None'}, "
                          f"has_pcb={detection_result.has_pcb}, "
                          f"focus={detection_result.focus_score:.1f}")
            
            time.sleep(0.033)  # ~30 FPS
        
        # Stop streaming
        camera.stop_streaming()
        
        print(f"\nTest completed: {frame_count} frames captured in 10 seconds")
        print(f"Average FPS: {frame_count / 10.0:.1f}")
        
        if frame_count == 0:
            print("ERROR: No frames captured!")
            return False
        else:
            print("SUCCESS: Frame capture working")
            return True
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_preview_loop()
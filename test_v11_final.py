#!/usr/bin/env python3
"""
Final test for v1.1 optimized auto-trigger with position smoothing
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from processing.pcb_detector_v11 import PCBDetectorV11
from hardware.camera_controller import BaslerCamera
from core.config import CAMERA_CONFIG, TRIGGER_CONFIG

def test_v11_final():
    """Test v1.1 với position smoothing"""
    
    print("="*60)
    print("PCB DETECTOR v1.1 - FINAL TEST")
    print("Position Smoothing + Enhanced Stability")
    print("="*60)
    
    print("v1.1 Features:")
    print("  + Position smoothing (5-frame history)")
    print("  + Enhanced edge detection")
    print("  + Confidence-based selection")
    print("  + Adaptive thresholding")
    print("  + Better noise handling")
    
    print("\\nOptimized thresholds:")
    for key, value in TRIGGER_CONFIG.items():
        print(f"  {key}: {value}")
    
    # Test với camera thật
    try:
        print("\\n" + "-"*50)
        print("REAL CAMERA TEST - v1.1")
        print("-"*50)
        
        camera = BaslerCamera(CAMERA_CONFIG)
        detector = PCBDetectorV11()
        
        camera.start_streaming()
        print("Camera streaming started...")
        
        # Test parameters
        test_duration = 30  # 30 seconds for thorough test
        trigger_conditions_met = 0
        actual_triggers = 0
        
        stats = {
            "frames": 0,
            "pcb_detected": 0,
            "stable": 0,
            "focus_ok": 0,
            "focus_scores": []
        }
        
        print(f"Testing for {test_duration} seconds...")
        print("Frame |  PCB  | Stable | Focus | Trigger | Debug Info")
        print("-" * 65)
        
        last_trigger_time = 0
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            frame = camera.get_preview_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            stats["frames"] += 1
            result = detector.detect_pcb(frame)
            
            # Track statistics
            if result.has_pcb:
                stats["pcb_detected"] += 1
                stats["focus_scores"].append(result.focus_score)
                
                if result.is_stable:
                    stats["stable"] += 1
                
                if result.focus_score >= TRIGGER_CONFIG["focus_threshold"]:
                    stats["focus_ok"] += 1
            
            # Check full trigger conditions
            current_time = time.time()
            time_since_last = current_time - last_trigger_time
            
            would_trigger = (result.has_pcb and 
                           result.is_stable and 
                           result.focus_score >= TRIGGER_CONFIG["focus_threshold"] and
                           time_since_last >= TRIGGER_CONFIG["inspection_interval"])
            
            if would_trigger:
                trigger_conditions_met += 1
                last_trigger_time = current_time
                actual_triggers += 1
            
            # Print status every 50 frames
            if stats["frames"] % 50 == 0:
                trigger_str = "YES!" if would_trigger else "no"
                debug_info = detector.get_debug_info()
                stable_rate = debug_info['stability_rate']
                
                print(f"{stats['frames']:5d} | {result.has_pcb!s:5s} | {result.is_stable!s:6s} | "
                      f"{result.focus_score:5.1f} | {trigger_str:7s} | "
                      f"Stable:{stable_rate:.1f}% Hist:{debug_info['position_history_size']}")
            
            time.sleep(0.05)  # 20 FPS for testing
        
        camera.stop_streaming()
        camera.close()
        
        # Final results
        print("\\n" + "="*60)
        print("v1.1 FINAL TEST RESULTS")
        print("="*60)
        
        detection_rate = stats["pcb_detected"] / max(stats["frames"], 1) * 100
        stability_rate = stats["stable"] / max(stats["pcb_detected"], 1) * 100
        focus_rate = stats["focus_ok"] / max(stats["pcb_detected"], 1) * 100
        
        print(f"Duration: {test_duration}s")
        print(f"Total frames: {stats['frames']}")
        print(f"PCB detection: {stats['pcb_detected']} ({detection_rate:.1f}%)")
        print(f"Stable frames: {stats['stable']} ({stability_rate:.1f}%)")
        print(f"Good focus: {stats['focus_ok']} ({focus_rate:.1f}%)")
        print(f"Trigger conditions met: {trigger_conditions_met}")
        print(f"Actual triggers: {actual_triggers}")
        
        if stats["focus_scores"]:
            avg_focus = sum(stats["focus_scores"]) / len(stats["focus_scores"])
            max_focus = max(stats["focus_scores"])
            print(f"Focus scores: avg={avg_focus:.1f}, max={max_focus:.1f}")
        
        # Get detector debug info
        debug_info = detector.get_debug_info()
        print(f"\\nDetector Performance:")
        print(f"  Detection rate: {debug_info['detection_rate']:.1f}%")
        print(f"  Stability rate: {debug_info['stability_rate']:.1f}%")
        print(f"  Position smoothing active: {debug_info['position_history_size']}/5 frames")
        
        # Success evaluation
        print(f"\\n" + "="*60)
        if actual_triggers > 0:
            print("SUCCESS! v1.1 AUTO-TRIGGER IS WORKING!")
            print(f"Achieved {actual_triggers} triggers in {test_duration}s")
            triggers_per_minute = actual_triggers / test_duration * 60
            print(f"Trigger rate: {triggers_per_minute:.1f} triggers/minute")
            
            if stability_rate > 0:
                print(f"MAJOR IMPROVEMENT: Stability rate {stability_rate:.1f}% (was 0% in v1.0)")
            
        elif stability_rate > 0:
            print("PARTIAL SUCCESS: Stability detection working!")
            print(f"Stability rate: {stability_rate:.1f}% (was 0% in v1.0)")
            print("Focus or timing conditions preventing triggers")
            
        else:
            print("NEEDS MORE TUNING: Still no stability detected")
            print("Position smoothing may need further adjustment")
        
        # Comparison with v1.0
        print(f"\\nv1.0 vs v1.1 Improvement:")
        print(f"  v1.0 stability rate: 0%")
        print(f"  v1.1 stability rate: {stability_rate:.1f}%")
        if stability_rate > 0:
            print(f"  IMPROVEMENT: {stability_rate:.1f}% better stability!")
        
        print(f"\\nv1.1 Key Features Validated:")
        print(f"  Position smoothing: {debug_info['position_history_size']}/5 frames")
        print(f"  Enhanced detection: {debug_info['detection_rate']:.1f}% rate")
        print(f"  Improved stability: {debug_info['stability_rate']:.1f}% success")
        
    except Exception as e:
        print(f"Camera test failed: {str(e)}")
        print("Testing with synthetic data...")
        test_synthetic_v11()

def test_synthetic_v11():
    """Test v1.1 với synthetic data"""
    
    print("\\nTesting v1.1 with synthetic data...")
    detector = PCBDetectorV11()
    
    # Create test image
    test_img = np.ones((600, 800), dtype=np.uint8) * 180
    cv2.rectangle(test_img, (200, 150), (600, 450), 80, -1)
    
    trigger_count = 0
    
    for i in range(20):
        result = detector.detect_pcb(test_img)
        debug = detector.get_debug_info()
        
        would_trigger = (result.has_pcb and result.is_stable and 
                        result.focus_score >= TRIGGER_CONFIG["focus_threshold"])
        
        if would_trigger:
            trigger_count += 1
            
        print(f"Frame {i+1:2d}: PCB={result.has_pcb}, Stable={result.is_stable}, "
              f"Focus={result.focus_score:.1f}, Smoothing={debug['position_history_size']}/5")
        
        if would_trigger:
            print(f"  -> TRIGGER #{trigger_count}!")
    
    print(f"\\nSynthetic test: {trigger_count}/20 triggers")

def main():
    """Main test function"""
    
    print("v1.1 ENHANCED AUTO-TRIGGER - FINAL TEST")
    print("Testing position smoothing and enhanced stability...")
    
    test_v11_final()
    
    print("\\n" + "="*60)
    print("v1.1 TEST COMPLETE")
    print("="*60)
    print("v1.1 implements:")
    print("  1. Position smoothing (5-frame moving average)")
    print("  2. Enhanced detection algorithms") 
    print("  3. Confidence-based candidate selection")
    print("  4. Better noise handling")
    print("  5. Adaptive thresholds")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
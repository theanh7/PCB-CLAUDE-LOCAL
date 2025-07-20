#!/usr/bin/env python3
"""
Test script cho v1.1 optimized auto-trigger
Kiểm tra xem các thay đổi có cải thiện auto-trigger không
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from processing.pcb_detector import PCBDetector
from hardware.camera_controller import BaslerCamera
from core.config import CAMERA_CONFIG, TRIGGER_CONFIG

def test_improved_trigger():
    """Test v1.1 optimized trigger settings"""
    
    print("="*60)
    print("TESTING v1.1 OPTIMIZED AUTO-TRIGGER")
    print("="*60)
    
    print("v1.1 Optimized TRIGGER_CONFIG:")
    for key, value in TRIGGER_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\\nChanges from v1.0:")
    print("  stability_frames: 10 -> 3 (300% easier)")
    print("  focus_threshold: 100 -> 50 (50% easier)")
    print("  movement_threshold: 5 -> 15 (300% more tolerant)")
    print("  min_pcb_area: 0.1 -> 0.05 (50% smaller PCBs allowed)")
    print("  inspection_interval: 2.0 -> 1.5 (faster inspection)")
    
    # Test with synthetic images first
    print("\\n" + "-"*40)
    print("Testing with synthetic images...")
    print("-"*40)
    
    detector = PCBDetector()
    
    # Create test image
    test_img = np.ones((600, 800), dtype=np.uint8) * 200
    cv2.rectangle(test_img, (200, 150), (600, 450), 80, -1)
    
    # Test stability tracking over multiple frames
    trigger_count = 0
    total_frames = 30
    
    for frame_num in range(total_frames):
        result = detector.detect_pcb(test_img)
        
        print(f"Frame {frame_num+1:2d}: PCB={result.has_pcb}, "
              f"Stable={result.is_stable}, Focus={result.focus_score:.1f}, "
              f"Stable_frames={detector.stable_frames}")
        
        # Check if would trigger
        would_trigger = (result.has_pcb and result.is_stable and 
                        result.focus_score >= TRIGGER_CONFIG["focus_threshold"])
        
        if would_trigger:
            trigger_count += 1
            print(f"  -> WOULD TRIGGER! (trigger #{trigger_count})")
            
            # Reset for next trigger test
            detector.last_inspection_time = time.time()
        
        time.sleep(0.1)  # Simulate frame rate
    
    print(f"\\nSynthetic Test Results:")
    print(f"  Total frames: {total_frames}")
    print(f"  Triggers: {trigger_count}")
    print(f"  Trigger rate: {trigger_count/total_frames*100:.1f}%")
    
    if trigger_count > 0:
        print("  SUCCESS: Auto-trigger is working!")
    else:
        print("  FAILED: Still no triggers")

def test_real_camera_v11():
    """Test v1.1 với camera thật"""
    
    print("\\n" + "="*60)
    print("TESTING v1.1 WITH REAL CAMERA")
    print("="*60)
    
    try:
        camera = BaslerCamera(CAMERA_CONFIG)
        detector = PCBDetector()
        
        print("Starting camera stream for v1.1 test...")
        camera.start_streaming()
        
        stats = {
            "frames": 0,
            "pcb_detected": 0,
            "stable": 0,
            "triggers": 0,
            "focus_scores": []
        }
        
        start_time = time.time()
        test_duration = 20  # 20 seconds test
        
        print(f"Running {test_duration}s test...")
        print("Frame | PCB | Stable | Focus | Stable_Frames | Trigger")
        print("-" * 55)
        
        while time.time() - start_time < test_duration:
            frame = camera.get_preview_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            stats["frames"] += 1
            result = detector.detect_pcb(frame)
            
            if result.has_pcb:
                stats["pcb_detected"] += 1
                stats["focus_scores"].append(result.focus_score)
                
                if result.is_stable:
                    stats["stable"] += 1
            
            # Check trigger condition
            would_trigger = (result.has_pcb and result.is_stable and 
                           result.focus_score >= TRIGGER_CONFIG["focus_threshold"] and
                           time.time() - detector.last_inspection_time >= TRIGGER_CONFIG["inspection_interval"])
            
            if would_trigger:
                stats["triggers"] += 1
                detector.last_inspection_time = time.time()
                
            # Print every 10th frame
            if stats["frames"] % 10 == 0:
                trigger_str = "YES" if would_trigger else "no"
                print(f"{stats['frames']:5d} | {result.has_pcb!s:3s} | {result.is_stable!s:6s} | "
                      f"{result.focus_score:5.1f} | {detector.stable_frames:13d} | {trigger_str}")
            
            time.sleep(0.1)  # ~10 FPS for testing
        
        camera.stop_streaming()
        camera.close()
        
        # Final results
        print("\\n" + "="*60)
        print("v1.1 REAL CAMERA TEST RESULTS")
        print("="*60)
        
        print(f"Test duration: {test_duration}s")
        print(f"Total frames: {stats['frames']}")
        print(f"PCB detected: {stats['pcb_detected']} ({stats['pcb_detected']/max(stats['frames'],1)*100:.1f}%)")
        print(f"Stable frames: {stats['stable']} ({stats['stable']/max(stats['frames'],1)*100:.1f}%)")
        print(f"Triggers: {stats['triggers']}")
        
        if stats['focus_scores']:
            avg_focus = sum(stats['focus_scores']) / len(stats['focus_scores'])
            max_focus = max(stats['focus_scores'])
            print(f"Focus scores: avg={avg_focus:.1f}, max={max_focus:.1f}")
        
        # Performance evaluation
        if stats['triggers'] > 0:
            print("\\nSUCCESS: v1.1 auto-trigger is working!")
            print(f"Trigger rate: {stats['triggers']/test_duration*60:.1f} triggers/minute")
        else:
            print("\\nNEEDS MORE TUNING: Still no triggers detected")
            
        # Comparison with v1.0
        stable_improvement = stats['stable'] / max(stats['frames'], 1) * 100
        print(f"\\nv1.0 vs v1.1 Comparison:")
        print(f"  v1.0 stability rate: 0%")
        print(f"  v1.1 stability rate: {stable_improvement:.1f}%")
        
        if stable_improvement > 0:
            print(f"  IMPROVEMENT: {stable_improvement:.1f}% better!")
        
    except Exception as e:
        print(f"Real camera test failed: {str(e)}")
        print("This is normal if no camera is connected")

def main():
    """Main test function"""
    
    print("v1.1 AUTO-TRIGGER OPTIMIZATION TEST")
    print("Testing improved trigger sensitivity...")
    
    # Test 1: Synthetic image stability
    test_improved_trigger()
    
    # Test 2: Real camera if available
    test_real_camera_v11()
    
    print("\\n" + "="*60)
    print("v1.1 TEST COMPLETE")
    print("="*60)
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Debug false positives trong v1.1
Phân tích tại sao background được detect và over-triggering
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

def test_background_detection():
    """Test detection với background (không có PCB)"""
    
    print("="*60)
    print("DEBUG FALSE POSITIVE DETECTION")
    print("="*60)
    
    # Tạo empty background images
    backgrounds = [
        np.ones((600, 800), dtype=np.uint8) * 180,  # Light gray
        np.ones((600, 800), dtype=np.uint8) * 120,  # Medium gray  
        np.ones((600, 800), dtype=np.uint8) * 200,  # Light background
        np.random.randint(150, 200, (600, 800), dtype=np.uint8),  # Noisy background
    ]
    
    detector = PCBDetectorV11()
    
    for i, bg_image in enumerate(backgrounds):
        print(f"\\nTesting background {i+1}:")
        print("-" * 40)
        
        detection_count = 0
        stable_count = 0
        
        # Test 10 frames cho mỗi background
        for frame in range(10):
            result = detector.detect_pcb(bg_image)
            
            if result.has_pcb:
                detection_count += 1
                print(f"  Frame {frame+1}: DETECTED PCB! Position: {result.position}, Focus: {result.focus_score:.1f}")
                
                if result.is_stable:
                    stable_count += 1
                    print(f"    -> STABLE! (this would trigger inspection)")
        
        print(f"\\nBackground {i+1} Results:")
        print(f"  False detections: {detection_count}/10 ({detection_count*10}%)")
        print(f"  False stables: {stable_count}/10 ({stable_count*10}%)")
        
        if detection_count > 0:
            print(f"  PROBLEM: Background incorrectly detected as PCB")
        else:
            print(f"  OK: Background correctly ignored")

def test_over_triggering():
    """Test over-triggering với PCB đứng yên"""
    
    print("\\n" + "="*60)
    print("DEBUG OVER-TRIGGERING")
    print("="*60)
    
    # Tạo PCB image giống hệt nhau
    pcb_image = np.ones((600, 800), dtype=np.uint8) * 180
    cv2.rectangle(pcb_image, (200, 150), (600, 450), 80, -1)  # Dark PCB
    
    detector = PCBDetectorV11()
    trigger_times = []
    
    print("Testing with identical PCB images (should only trigger once)...")
    print("Frame | PCB | Stable | Focus | Would Trigger | Time Since Last")
    print("-" * 70)
    
    start_time = time.time()
    last_trigger_time = 0
    
    for frame in range(20):
        current_time = time.time()
        result = detector.detect_pcb(pcb_image)
        
        # Check trigger condition
        time_since_last = current_time - last_trigger_time
        would_trigger = (result.has_pcb and 
                        result.is_stable and 
                        result.focus_score >= TRIGGER_CONFIG["focus_threshold"] and
                        time_since_last >= TRIGGER_CONFIG["inspection_interval"])
        
        if would_trigger:
            trigger_times.append(current_time - start_time)
            last_trigger_time = current_time
            
        trigger_str = "YES!" if would_trigger else "no"
        
        print(f"{frame+1:5d} | {result.has_pcb!s:3s} | {result.is_stable!s:6s} | "
              f"{result.focus_score:5.1f} | {trigger_str:11s} | {time_since_last:5.2f}s")
        
        time.sleep(0.2)  # Simulate frame rate
    
    print(f"\\nOver-triggering Analysis:")
    print(f"  Total triggers: {len(trigger_times)}")
    print(f"  Expected: 1-2 (initial + maybe 1 more after cooldown)")
    print(f"  Trigger times: {[f'{t:.1f}s' for t in trigger_times]}")
    
    if len(trigger_times) > 2:
        print(f"  PROBLEM: Too many triggers for stationary PCB")
        print(f"  SOLUTION: Need longer cooldown or same-PCB detection")
    else:
        print(f"  OK: Reasonable trigger count")

def test_real_camera_issues():
    """Test với camera thật để tìm false positives"""
    
    print("\\n" + "="*60)
    print("DEBUG REAL CAMERA FALSE POSITIVES")
    print("="*60)
    
    try:
        camera = BaslerCamera(CAMERA_CONFIG)
        detector = PCBDetectorV11()
        
        camera.start_streaming()
        print("Testing real camera for false positives...")
        
        # Phase 1: Background only (no PCB)
        print("\\nPhase 1: Background detection test (10 seconds)")
        print("Please ensure NO PCB is in view...")
        time.sleep(3)  # Wait for user to remove PCB
        
        bg_detections = 0
        bg_stables = 0
        bg_triggers = 0
        
        start_time = time.time()
        last_trigger = 0
        
        while time.time() - start_time < 10:
            frame = camera.get_preview_frame()
            if frame is None:
                continue
                
            result = detector.detect_pcb(frame)
            
            if result.has_pcb:
                bg_detections += 1
                
                if result.is_stable:
                    bg_stables += 1
                    
                    # Check trigger
                    if (result.focus_score >= TRIGGER_CONFIG["focus_threshold"] and
                        time.time() - last_trigger >= TRIGGER_CONFIG["inspection_interval"]):
                        bg_triggers += 1
                        last_trigger = time.time()
                        print(f"  FALSE TRIGGER! Focus: {result.focus_score:.1f}")
            
            time.sleep(0.1)
        
        print(f"\\nBackground Test Results:")
        print(f"  False detections: {bg_detections}")
        print(f"  False stables: {bg_stables}")  
        print(f"  False triggers: {bg_triggers}")
        
        # Phase 2: PCB stationary test
        print("\\nPhase 2: Place PCB and keep it stationary (15 seconds)")
        input("Press Enter when PCB is in position...")
        
        pcb_triggers = 0
        trigger_times = []
        last_trigger = time.time()
        start_time = time.time()
        
        while time.time() - start_time < 15:
            frame = camera.get_preview_frame()
            if frame is None:
                continue
                
            result = detector.detect_pcb(frame)
            current_time = time.time()
            
            # Check trigger
            if (result.has_pcb and result.is_stable and
                result.focus_score >= TRIGGER_CONFIG["focus_threshold"] and
                current_time - last_trigger >= TRIGGER_CONFIG["inspection_interval"]):
                
                pcb_triggers += 1
                trigger_times.append(current_time - start_time)
                last_trigger = current_time
                print(f"  TRIGGER #{pcb_triggers} at {current_time - start_time:.1f}s")
            
            time.sleep(0.1)
        
        print(f"\\nStationary PCB Results:")
        print(f"  Total triggers: {pcb_triggers}")
        print(f"  Trigger times: {[f'{t:.1f}s' for t in trigger_times]}")
        print(f"  Expected: 1-2 triggers max")
        
        camera.stop_streaming()
        camera.close()
        
        # Analysis
        print(f"\\n" + "="*60)
        print("ANALYSIS & RECOMMENDATIONS")
        print("="*60)
        
        if bg_triggers > 0:
            print("Issue 1: False positive triggers on background")
            print("   Solution: Stricter PCB validation criteria")
            
        if pcb_triggers > 3:
            print("Issue 2: Over-triggering on stationary PCB")
            print("   Solution: Longer cooldown or same-position detection")
            
        if bg_triggers == 0 and pcb_triggers <= 2:
            print("No major issues detected")
            
    except Exception as e:
        print(f"Camera test failed: {str(e)}")

def main():
    """Main debug function"""
    
    print("v1.1 FALSE POSITIVE DEBUG")
    print("Investigating background detection and over-triggering...")
    
    # Test 1: Background false positives
    test_background_detection()
    
    # Test 2: Over-triggering with identical images
    test_over_triggering()
    
    # Test 3: Real camera issues
    test_real_camera_issues()
    
    print("\\n" + "="*60)
    print("RECOMMENDATIONS FOR v1.2:")
    print("="*60)
    print("1. Add stricter PCB validation (area, shape, contrast)")
    print("2. Implement inspection cooldown for same position")
    print("3. Add minimum PCB size threshold")
    print("4. Better background rejection algorithm")
    print("5. Add 'PCB removed' detection to reset trigger state")
    
    input("\\nPress Enter to exit...")

if __name__ == "__main__":
    main()
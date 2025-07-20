#!/usr/bin/env python3
"""
Test v1.2 fixes for over-triggering and false positives
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

def test_v12_over_trigger_fix():
    """Test v1.2 fix for over-triggering"""
    
    print("="*60)
    print("v1.2 OVER-TRIGGER FIX TEST")
    print("="*60)
    
    print("v1.2 New Settings:")
    print(f"  inspection_interval: {TRIGGER_CONFIG['inspection_interval']}s (was 1.5s)")
    print(f"  same_position_threshold: {TRIGGER_CONFIG['same_position_threshold']} pixels")
    print(f"  min_pcb_change_required: {TRIGGER_CONFIG['min_pcb_change_required']}")
    
    try:
        camera = BaslerCamera(CAMERA_CONFIG)
        detector = PCBDetectorV11()
        
        camera.start_streaming()
        print("\\nTesting with real camera...")
        print("Place PCB and keep it stationary for 30 seconds")
        
        # Simulate main system trigger logic
        last_inspection_time = 0
        last_inspection_position = None
        triggers = []
        
        print("\\nTime | PCB | Stable | Focus | Same Pos | Cooldown | Trigger")
        print("-" * 65)
        
        start_time = time.time()
        
        while time.time() - start_time < 30:
            frame = camera.get_preview_frame()
            if frame is None:
                continue
                
            current_time = time.time() - start_time
            result = detector.detect_pcb(frame)
            
            # Check all trigger conditions
            has_pcb = result.has_pcb
            is_stable = result.is_stable
            focus_ok = result.focus_score >= TRIGGER_CONFIG["focus_threshold"]
            
            # Time cooldown check
            time_since_last = time.time() - last_inspection_time
            cooldown_ok = time_since_last >= TRIGGER_CONFIG["inspection_interval"]
            
            # Position change check (v1.2 NEW)
            position_changed = True
            if TRIGGER_CONFIG.get("min_pcb_change_required", False) and last_inspection_position:
                current_position = detector.smoothed_position
                if current_position:
                    distance = current_position.distance_to(last_inspection_position)
                    position_changed = distance >= TRIGGER_CONFIG["same_position_threshold"]
            
            # Final trigger decision
            would_trigger = (has_pcb and is_stable and focus_ok and 
                           cooldown_ok and position_changed)
            
            if would_trigger:
                triggers.append(current_time)
                last_inspection_time = time.time()
                if detector.smoothed_position:
                    last_inspection_position = detector.smoothed_position
            
            # Log every 5 seconds
            if int(current_time) % 5 == 0 and int(current_time * 10) % 50 == 0:
                same_pos_str = "NO" if position_changed else "YES"
                cooldown_str = "OK" if cooldown_ok else f"{TRIGGER_CONFIG['inspection_interval'] - time_since_last:.1f}s"
                trigger_str = "YES!" if would_trigger else "no"
                
                print(f"{current_time:4.0f}s| {has_pcb!s:3s} | {is_stable!s:6s} | "
                      f"{focus_ok!s:5s} | {same_pos_str:8s} | {cooldown_str:8s} | {trigger_str}")
            
            time.sleep(0.1)
        
        camera.stop_streaming()
        camera.close()
        
        # Results analysis
        print(f"\\n" + "="*60)
        print("v1.2 TEST RESULTS")
        print("="*60)
        
        print(f"Test duration: 30s")
        print(f"Total triggers: {len(triggers)}")
        print(f"Trigger times: {[f'{t:.1f}s' for t in triggers]}")
        
        if len(triggers) <= 2:
            print("SUCCESS: Over-triggering fixed!")
            print(f"  Expected: 1-2 triggers max")
            print(f"  Actual: {len(triggers)} triggers")
            
        elif len(triggers) <= 4:
            print("IMPROVED: Reduced over-triggering")
            print(f"  v1.1 would have: 20+ triggers")
            print(f"  v1.2 achieved: {len(triggers)} triggers")
            
        else:
            print("STILL NEEDS WORK: Over-triggering persists")
            print(f"  {len(triggers)} triggers in 30s is still too many")
        
        # Trigger rate analysis
        if len(triggers) > 0:
            triggers_per_minute = len(triggers) / 30 * 60
            print(f"\\nTrigger rate: {triggers_per_minute:.1f} triggers/minute")
            print(f"  v1.1 rate: 30+ triggers/minute")
            print(f"  v1.2 rate: {triggers_per_minute:.1f} triggers/minute")
            
            if triggers_per_minute <= 6:  # Max 1 every 10 seconds
                print("  EXCELLENT: Very reasonable trigger rate")
            elif triggers_per_minute <= 12:
                print("  GOOD: Acceptable trigger rate")
            else:
                print("  NEEDS IMPROVEMENT: Still too frequent")
        
    except Exception as e:
        print(f"Camera test failed: {str(e)}")
        test_synthetic_v12()

def test_synthetic_v12():
    """Test v1.2 với synthetic data"""
    
    print("\\nTesting v1.2 with synthetic data...")
    print("-" * 40)
    
    detector = PCBDetectorV11()
    
    # Simulate identical PCB frames
    pcb_image = np.ones((600, 800), dtype=np.uint8) * 180
    cv2.rectangle(pcb_image, (200, 150), (600, 450), 80, -1)
    
    last_inspection_time = 0
    last_position = None
    triggers = []
    
    for i in range(20):
        current_time = time.time()
        result = detector.detect_pcb(pcb_image)
        
        # Check trigger conditions
        time_since_last = current_time - last_inspection_time
        cooldown_ok = time_since_last >= TRIGGER_CONFIG["inspection_interval"]
        
        # Position check
        position_changed = True
        if last_position and detector.smoothed_position:
            distance = detector.smoothed_position.distance_to(last_position)
            position_changed = distance >= TRIGGER_CONFIG["same_position_threshold"]
        
        would_trigger = (result.has_pcb and result.is_stable and 
                        result.focus_score >= TRIGGER_CONFIG["focus_threshold"] and
                        cooldown_ok and position_changed)
        
        if would_trigger:
            triggers.append(i)
            last_inspection_time = current_time
            if detector.smoothed_position:
                last_position = detector.smoothed_position
        
        pos_str = "same" if not position_changed else "new"
        trigger_str = "TRIGGER!" if would_trigger else "no"
        
        print(f"Frame {i+1:2d}: PCB={result.has_pcb}, Stable={result.is_stable}, "
              f"Position={pos_str}, Cooldown={cooldown_ok}, -> {trigger_str}")
        
        time.sleep(0.2)
    
    print(f"\\nSynthetic Test Results:")
    print(f"  Total triggers: {len(triggers)}")
    print(f"  Trigger frames: {triggers}")
    print(f"  Expected: 1-2 max for identical images")

def test_background_rejection():
    """Test background rejection improvements"""
    
    print("\\n" + "="*60)
    print("BACKGROUND REJECTION TEST")
    print("="*60)
    
    detector = PCBDetectorV11()
    
    # Various background types
    backgrounds = [
        ("Empty light", np.ones((600, 800), dtype=np.uint8) * 180),
        ("Empty dark", np.ones((600, 800), dtype=np.uint8) * 100),
        ("Noisy", np.random.randint(140, 200, (600, 800), dtype=np.uint8)),
        ("Gradient", np.linspace(100, 200, 600*800).reshape(600, 800).astype(np.uint8))
    ]
    
    for name, bg_image in backgrounds:
        print(f"\\nTesting: {name} background")
        detections = 0
        
        for frame in range(5):
            result = detector.detect_pcb(bg_image)
            if result.has_pcb:
                detections += 1
                
        print(f"  False detections: {detections}/5")
        if detections == 0:
            print("  OK: Background correctly rejected")
        else:
            print("  ISSUE: Background incorrectly detected as PCB")

def main():
    """Main test function"""
    
    print("v1.2 OVER-TRIGGER FIX VALIDATION")
    print("Testing improved trigger logic...")
    
    # Test 1: Over-trigger fix
    test_v12_over_trigger_fix()
    
    # Test 2: Synthetic validation
    test_synthetic_v12()
    
    # Test 3: Background rejection
    test_background_rejection()
    
    print("\\n" + "="*60)
    print("v1.2 VALIDATION COMPLETE")
    print("="*60)
    
    print("v1.2 Improvements:")
    print("  1. Increased inspection_interval: 1.5s -> 5.0s")
    print("  2. Added same-position detection")
    print("  3. Require PCB movement for new trigger")
    print("  4. Better cooldown management")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
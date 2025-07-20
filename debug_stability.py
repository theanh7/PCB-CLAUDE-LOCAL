#!/usr/bin/env python3
"""
Debug stability logic chi tiết
Tìm hiểu tại sao stable_frames luôn = 0
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from processing.pcb_detector import PCBDetector, PCBPosition
from hardware.camera_controller import BaslerCamera
from core.config import CAMERA_CONFIG, TRIGGER_CONFIG

def debug_stability_calculation():
    """Debug tính toán stability chi tiết"""
    
    print("="*60)
    print("DEBUG STABILITY CALCULATION")
    print("="*60)
    
    detector = PCBDetector()
    
    # Test với camera thật
    try:
        camera = BaslerCamera(CAMERA_CONFIG)
        camera.start_streaming()
        
        print("Collecting frames for stability analysis...")
        frames_data = []
        
        # Thu thập 10 frames liên tiếp
        for i in range(10):
            frame = camera.get_preview_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            result = detector.detect_pcb(frame)
            if result.has_pcb:
                frames_data.append({
                    'frame': i+1,
                    'position': result.position,
                    'stable_frames': detector.stable_frames,
                    'is_stable': result.is_stable
                })
                
                if len(frames_data) >= 2:
                    # Tính distance giữa 2 frame liên tiếp
                    curr_pos = result.position
                    prev_pos = frames_data[-2]['position']
                    
                    # Tạo PCBPosition objects để tính distance
                    curr_pcb = PCBPosition(curr_pos[0], curr_pos[1], curr_pos[2], curr_pos[3], 
                                         curr_pos[2]*curr_pos[3], time.time())
                    prev_pcb = PCBPosition(prev_pos[0], prev_pos[1], prev_pos[2], prev_pos[3], 
                                         prev_pos[2]*prev_pos[3], time.time())
                    
                    distance = curr_pcb.distance_to(prev_pcb)
                    size_diff = curr_pcb.size_difference(prev_pcb)
                    
                    print(f"Frame {i+1}:")
                    print(f"  Position: ({curr_pos[0]}, {curr_pos[1]}, {curr_pos[2]}, {curr_pos[3]})")
                    print(f"  Distance from prev: {distance:.1f} pixels (threshold: {TRIGGER_CONFIG['movement_threshold']})")
                    print(f"  Size diff from prev: {size_diff:.1f} pixels (threshold: {TRIGGER_CONFIG['movement_threshold']*2})")
                    print(f"  Stable frames: {detector.stable_frames}")
                    print(f"  Is stable: {result.is_stable}")
                    print(f"  Movement OK: {distance <= TRIGGER_CONFIG['movement_threshold']}")
                    print(f"  Size OK: {size_diff <= TRIGGER_CONFIG['movement_threshold']*2}")
                    print("-" * 40)
                else:
                    print(f"Frame {i+1}: Position: ({curr_pos[0]}, {curr_pos[1]}, {curr_pos[2]}, {curr_pos[3]}) - First frame")
            
            time.sleep(0.2)  # 5 FPS for debugging
        
        camera.stop_streaming()
        camera.close()
        
        # Phân tích kết quả
        print("\\nSTABILITY ANALYSIS:")
        if len(frames_data) >= 2:
            total_frames = len(frames_data)
            stable_count = sum(1 for f in frames_data if f['is_stable'])
            max_stable_frames = max(f['stable_frames'] for f in frames_data)
            
            print(f"Total frames analyzed: {total_frames}")
            print(f"Stable frames: {stable_count}")
            print(f"Max consecutive stable: {max_stable_frames}")
            print(f"Required for trigger: {TRIGGER_CONFIG['stability_frames']}")
            
            if max_stable_frames == 0:
                print("\\nISSUE: PCB position varies too much between frames")
                print("SOLUTION: Increase movement_threshold or implement position averaging")
            elif max_stable_frames < TRIGGER_CONFIG['stability_frames']:
                print(f"\\nISSUE: Max stable ({max_stable_frames}) < required ({TRIGGER_CONFIG['stability_frames']})")
                print("SOLUTION: Reduce stability_frames requirement")
        
    except Exception as e:
        print(f"Camera test failed: {str(e)}")
        print("Testing with synthetic data...")
        test_synthetic_stability()

def test_synthetic_stability():
    """Test stability với synthetic data để hiểu logic"""
    
    print("\\nTesting stability logic with synthetic data...")
    print("-" * 50)
    
    detector = PCBDetector()
    
    # Test case 1: Hoàn toàn giống nhau
    print("Test 1: Identical positions")
    base_pos = (100, 100, 200, 150)
    
    for i in range(5):
        # Tạo fake result
        class FakeResult:
            def __init__(self, pos):
                self.has_pcb = True
                self.position = pos
                self.is_stable = False
                self.focus_score = 100
        
        result = FakeResult(base_pos)
        
        # Manually test stability check
        current_pcb = PCBPosition(base_pos[0], base_pos[1], base_pos[2], base_pos[3], 
                                base_pos[2]*base_pos[3], time.time())
        
        is_stable = detector._check_stability(current_pcb)
        
        print(f"  Frame {i+1}: stable_frames={detector.stable_frames}, is_stable={is_stable}")
    
    # Reset detector
    detector = PCBDetector()
    
    # Test case 2: Slight movement (within threshold)
    print("\\nTest 2: Slight movement (within threshold)")
    positions = [
        (100, 100, 200, 150),
        (102, 101, 201, 149),  # 2-3 pixel difference
        (101, 102, 199, 151),  # Small variation
        (100, 100, 200, 150),  # Back to original
        (103, 99, 200, 150),   # Small variation
    ]
    
    for i, pos in enumerate(positions):
        current_pcb = PCBPosition(pos[0], pos[1], pos[2], pos[3], 
                                pos[2]*pos[3], time.time())
        
        is_stable = detector._check_stability(current_pcb)
        
        if i > 0:
            prev_pos = positions[i-1]
            distance = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
            print(f"  Frame {i+1}: pos={pos}, distance={distance:.1f}, stable_frames={detector.stable_frames}, is_stable={is_stable}")
        else:
            print(f"  Frame {i+1}: pos={pos}, stable_frames={detector.stable_frames}, is_stable={is_stable}")
    
    # Test case 3: Large movement (exceeds threshold)
    print("\\nTest 3: Large movement (exceeds threshold)")
    detector = PCBDetector()
    
    positions = [
        (100, 100, 200, 150),
        (120, 120, 200, 150),  # 20+ pixel movement
        (125, 125, 200, 150),
        (120, 120, 200, 150),
        (119, 119, 200, 150),
    ]
    
    for i, pos in enumerate(positions):
        current_pcb = PCBPosition(pos[0], pos[1], pos[2], pos[3], 
                                pos[2]*pos[3], time.time())
        
        is_stable = detector._check_stability(current_pcb)
        
        if i > 0:
            prev_pos = positions[i-1]
            distance = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
            print(f"  Frame {i+1}: pos={pos}, distance={distance:.1f}, stable_frames={detector.stable_frames}, is_stable={is_stable}")
        else:
            print(f"  Frame {i+1}: pos={pos}, stable_frames={detector.stable_frames}, is_stable={is_stable}")

def main():
    """Main debug function"""
    
    print("STABILITY LOGIC DEBUG")
    print("Investigating why stable_frames always = 0...")
    
    # Debug real camera stability
    debug_stability_calculation()
    
    # Test synthetic stability logic
    test_synthetic_stability()
    
    print("\\n" + "="*60)
    print("CONCLUSIONS:")
    print("="*60)
    print("1. Check if PCB detection is too sensitive (position varies each frame)")
    print("2. Camera noise/vibration might prevent stability")
    print("3. May need position smoothing or averaging")
    print("4. Movement threshold might be too strict")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
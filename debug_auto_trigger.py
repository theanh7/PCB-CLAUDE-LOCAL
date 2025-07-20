#!/usr/bin/env python3
"""
Debug script để phân tích vấn đề auto-trigger PCB detection
Tìm hiểu tại sao auto-trigger không hoạt động trong v1.0
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
from processing.preprocessor import ImagePreprocessor
from hardware.camera_controller import BaslerCamera
from core.config import CAMERA_CONFIG, TRIGGER_CONFIG

def create_test_images():
    """Tạo test images để debug PCB detection"""
    
    # Test image 1: Dark PCB on light background (giống thực tế)
    img1 = np.ones((600, 800), dtype=np.uint8) * 200  # Light background
    cv2.rectangle(img1, (200, 150), (600, 450), 80, -1)  # Dark PCB
    cv2.imwrite("test_dark_pcb.jpg", img1)
    
    # Test image 2: Light PCB on dark background
    img2 = np.ones((600, 800), dtype=np.uint8) * 50   # Dark background
    cv2.rectangle(img2, (200, 150), (600, 450), 180, -1)  # Light PCB
    cv2.imwrite("test_light_pcb.jpg", img2)
    
    # Test image 3: Noisy image with PCB
    img3 = np.random.randint(150, 200, (600, 800), dtype=np.uint8)  # Noisy background
    cv2.rectangle(img3, (200, 150), (600, 450), 70, -1)  # Dark PCB
    noise = np.random.randint(-30, 30, img3.shape, dtype=np.int16)
    img3 = np.clip(img3.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite("test_noisy_pcb.jpg", img3)
    
    # Test image 4: Low contrast PCB
    img4 = np.ones((600, 800), dtype=np.uint8) * 130  # Gray background
    cv2.rectangle(img4, (200, 150), (600, 450), 110, -1)  # Slightly darker PCB
    cv2.imwrite("test_low_contrast_pcb.jpg", img4)
    
    print("Created test images: test_dark_pcb.jpg, test_light_pcb.jpg, test_noisy_pcb.jpg, test_low_contrast_pcb.jpg")

def debug_pcb_detection():
    """Debug PCB detection với các test images"""
    
    detector = PCBDetector()
    preprocessor = ImagePreprocessor()
    
    test_images = [
        "test_dark_pcb.jpg",
        "test_light_pcb.jpg", 
        "test_noisy_pcb.jpg",
        "test_low_contrast_pcb.jpg"
    ]
    
    print("\\n" + "="*60)
    print("DEBUG PCB DETECTION SENSITIVITY")
    print("="*60)
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
            
        print(f"\\nTesting: {img_path}")
        print("-" * 40)
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("❌ Could not load image")
            continue
            
        # Test detection multiple times to check consistency
        results = []
        for i in range(5):
            result = detector.detect_pcb(image)
            results.append(result)
            
        # Analyze results
        detected_count = sum(1 for r in results if r.has_pcb)
        stable_count = sum(1 for r in results if r.is_stable)
        focus_scores = [r.focus_score for r in results if r.focus_score > 0]
        
        print(f"Detection Rate: {detected_count}/5 ({detected_count*20}%)")
        print(f"Stability Rate: {stable_count}/5 ({stable_count*20}%)")
        
        if focus_scores:
            avg_focus = sum(focus_scores) / len(focus_scores)
            print(f"Average Focus: {avg_focus:.1f} (threshold: {TRIGGER_CONFIG['focus_threshold']})")
        else:
            print("Focus: No valid scores")
            
        # Show detailed info for first detection
        if results[0].has_pcb:
            pos = results[0].position
            print(f"Position: ({pos[0]}, {pos[1]}, {pos[2]}, {pos[3]})")
            print(f"Area: {pos[2] * pos[3]} pixels")
        else:
            print("No PCB detected")
            
        # Create debug visualization
        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if results[0].has_pcb and results[0].position:
            x, y, w, h = results[0].position
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, f"Focus: {results[0].focus_score:.1f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        debug_path = f"debug_{img_path}"
        cv2.imwrite(debug_path, debug_img)
        print(f"Saved debug image: {debug_path}")

def debug_trigger_conditions():
    """Debug trigger conditions để hiểu tại sao auto-trigger không hoạt động"""
    
    print("\\n" + "="*60)
    print("DEBUG TRIGGER CONDITIONS")
    print("="*60)
    
    print(f"Current TRIGGER_CONFIG:")
    for key, value in TRIGGER_CONFIG.items():
        print(f"  {key}: {value}")
        
    print(f"\\nTrigger conditions required:")
    print(f"  1. has_pcb = True")
    print(f"  2. is_stable = True (need {TRIGGER_CONFIG['stability_frames']} stable frames)")
    print(f"  3. focus_score >= {TRIGGER_CONFIG['focus_threshold']}")
    print(f"  4. auto_mode = True")
    print(f"  5. time since last inspection >= {TRIGGER_CONFIG['inspection_interval']}s")
    
    # Test with easier thresholds
    print(f"\\nSuggested easier thresholds for testing:")
    print(f"  stability_frames: {TRIGGER_CONFIG['stability_frames']} -> 3 (reduce from 10)")
    print(f"  focus_threshold: {TRIGGER_CONFIG['focus_threshold']} -> 50 (reduce from 100)")
    print(f"  min_pcb_area: {TRIGGER_CONFIG['min_pcb_area']} -> 0.05 (reduce from 0.1)")

def debug_real_camera():
    """Debug với camera thật nếu có"""
    
    print("\\n" + "="*60)
    print("DEBUG REAL CAMERA (if available)")
    print("="*60)
    
    try:
        print("Attempting to connect to camera...")
        camera = BaslerCamera(CAMERA_CONFIG)
        detector = PCBDetector()
        
        print("Camera connected, starting debug stream...")
        camera.start_streaming()
        
        detection_stats = {"detected": 0, "stable": 0, "total": 0}
        focus_scores = []
        
        for i in range(30):  # Test 30 frames
            frame = camera.get_preview_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            detection_stats["total"] += 1
            result = detector.detect_pcb(frame)
            
            if result.has_pcb:
                detection_stats["detected"] += 1
                focus_scores.append(result.focus_score)
                
                if result.is_stable:
                    detection_stats["stable"] += 1
                    
            # Print status every 10 frames
            if i % 10 == 0:
                print(f"Frame {i}/30: Detection={detection_stats['detected']}/{detection_stats['total']}, "
                      f"Stable={detection_stats['stable']}")
                      
            time.sleep(0.1)
        
        camera.stop_streaming()
        camera.close()
        
        # Final stats
        print(f"\\nCamera Debug Results:")
        print(f"  Total frames: {detection_stats['total']}")
        print(f"  PCB detected: {detection_stats['detected']} ({detection_stats['detected']/max(detection_stats['total'],1)*100:.1f}%)")
        print(f"  Stable: {detection_stats['stable']} ({detection_stats['stable']/max(detection_stats['total'],1)*100:.1f}%)")
        
        if focus_scores:
            avg_focus = sum(focus_scores) / len(focus_scores)
            max_focus = max(focus_scores)
            print(f"  Focus scores: avg={avg_focus:.1f}, max={max_focus:.1f}")
        else:
            print(f"  Focus scores: No valid scores")
            
    except Exception as e:
        print(f"Camera debug failed: {str(e)}")
        print("This is normal if no camera is connected")

def main():
    """Main debug function"""
    
    print("PCB Auto-Trigger Debug Tool")
    print("Analyzing why auto-trigger doesn't work in v1.0...")
    
    # 1. Create and test with synthetic images
    create_test_images()
    debug_pcb_detection()
    
    # 2. Analyze trigger conditions
    debug_trigger_conditions()
    
    # 3. Test with real camera if available
    debug_real_camera()
    
    print("\\n" + "="*60)
    print("RECOMMENDATIONS FOR v1.1:")
    print("="*60)
    print("1. Lower thresholds: stability_frames=3, focus_threshold=50")
    print("2. Add debug mode with visual feedback")
    print("3. Add detection statistics to GUI")
    print("4. Add configurable trigger sensitivity")
    print("5. Add manual PCB detection area selection")
    
    input("\\nPress Enter to exit...")

if __name__ == "__main__":
    main()
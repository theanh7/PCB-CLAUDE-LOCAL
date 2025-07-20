#!/usr/bin/env python3
"""
Test the improved PCB detection methods individually.
"""

import cv2
import numpy as np
from processing.pcb_detector import PCBDetector
from core.config import TRIGGER_CONFIG

def test_pcb_detection_methods():
    """Test each PCB detection method individually."""
    
    # Load test image
    image = cv2.imread("trigger_test_frame.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Could not load test image")
        return
    
    print(f"Testing image: {image.shape}")
    print(f"Image stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}")
    
    # Initialize detector
    detector = PCBDetector(TRIGGER_CONFIG)
    
    # Test method 1: Edge detection
    print("\n1. Testing edge detection method...")
    pcb_pos = detector._detect_by_edges(image)
    if pcb_pos:
        print(f"  Edge detection found PCB: ({pcb_pos.x}, {pcb_pos.y}, {pcb_pos.width}, {pcb_pos.height})")
        print(f"  Area: {pcb_pos.area}, Timestamp: {pcb_pos.timestamp}")
    else:
        print("  Edge detection: No PCB found")
    
    # Test method 2: Dark object detection
    print("\n2. Testing dark object detection method...")
    pcb_pos = detector._detect_dark_object(image)
    if pcb_pos:
        print(f"  Dark object detection found PCB: ({pcb_pos.x}, {pcb_pos.y}, {pcb_pos.width}, {pcb_pos.height})")
        print(f"  Area: {pcb_pos.area}")
        
        # Save visualization
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_image, (pcb_pos.x, pcb_pos.y), 
                     (pcb_pos.x + pcb_pos.width, pcb_pos.y + pcb_pos.height), 
                     (0, 255, 0), 3)
        cv2.putText(vis_image, "Dark Object Detection", (pcb_pos.x, pcb_pos.y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite("test_dark_detection.jpg", vis_image)
        print("  Visualization saved as test_dark_detection.jpg")
    else:
        print("  Dark object detection: No PCB found")
    
    # Test method 3: Contrast detection
    print("\n3. Testing contrast detection method...")
    pcb_pos = detector._detect_by_contrast(image)
    if pcb_pos:
        print(f"  Contrast detection found PCB: ({pcb_pos.x}, {pcb_pos.y}, {pcb_pos.width}, {pcb_pos.height})")
        print(f"  Area: {pcb_pos.area}")
        
        # Save visualization
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_image, (pcb_pos.x, pcb_pos.y), 
                     (pcb_pos.x + pcb_pos.width, pcb_pos.y + pcb_pos.height), 
                     (255, 0, 0), 3)
        cv2.putText(vis_image, "Contrast Detection", (pcb_pos.x, pcb_pos.y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite("test_contrast_detection.jpg", vis_image)
        print("  Visualization saved as test_contrast_detection.jpg")
    else:
        print("  Contrast detection: No PCB found")
    
    # Test full detection
    print("\n4. Testing full detection (all methods)...")
    result = detector.detect_pcb(image)
    if result.has_pcb:
        print(f"  Full detection found PCB: {result.position}")
        print(f"  Stable: {result.is_stable}, Focus: {result.focus_score:.1f}")
    else:
        print("  Full detection: No PCB found")
    
    # Let's also test our manual analysis approach
    print("\n5. Testing manual approach (from analyze_pcb_image.py)...")
    manual_detect(image)

def manual_detect(image):
    """Test the manual detection approach that worked."""
    
    # Enhance contrast first
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Invert image to make dark objects bright
    inverted = 255 - enhanced
    
    # Apply threshold to isolate dark regions
    _, binary = cv2.threshold(inverted, 80, 255, cv2.THRESH_BINARY)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"  Manual method found {len(contours)} contours")
    
    if contours:
        # Filter by size
        total_area = image.shape[0] * image.shape[1]
        min_area = total_area * 0.01  # At least 1% of image
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        print(f"  Large contours (>1% of image): {len(large_contours)}")
        
        for i, contour in enumerate(large_contours[:3]):  # Show top 3
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = area / total_area
            aspect_ratio = max(w, h) / min(w, h)
            
            print(f"    Contour {i+1}: area={area}, bbox=({x},{y},{w},{h})")
            print(f"               ratio={area_ratio:.3f}, aspect={aspect_ratio:.1f}")
            
            # Check if this looks like a PCB
            if (0.05 <= area_ratio <= 0.8 and  # Between 5% and 80% of image
                1.2 <= aspect_ratio <= 5.0):    # Reasonable rectangle
                print(f"    >>> This looks like a PCB!")
                
                # Save visualization
                vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(vis_image, f"Manual PCB Detection {i+1}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(f"test_manual_detection_{i+1}.jpg", vis_image)
                print(f"    Visualization saved as test_manual_detection_{i+1}.jpg")
                
                return True
    
    return False

if __name__ == "__main__":
    test_pcb_detection_methods()
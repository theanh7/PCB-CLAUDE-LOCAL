#!/usr/bin/env python3
"""
Debug dark object detection step by step.
"""

import cv2
import numpy as np

def debug_dark_detection():
    """Debug the dark object detection step by step."""
    
    # Load test image
    image = cv2.imread("trigger_test_frame.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Could not load test image")
        return
    
    print(f"Original image: {image.shape}")
    print(f"Stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}")
    
    # Step 1: Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    print(f"Enhanced image stats: min={enhanced.min()}, max={enhanced.max()}, mean={enhanced.mean():.1f}")
    cv2.imwrite("debug_step1_enhanced.jpg", enhanced)
    
    # Step 2: Invert image
    inverted = 255 - enhanced
    
    print(f"Inverted image stats: min={inverted.min()}, max={inverted.max()}, mean={inverted.mean():.1f}")
    cv2.imwrite("debug_step2_inverted.jpg", inverted)
    
    # Step 3: Try different thresholds
    thresholds = [30, 50, 70, 90, 110, 130]
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold {threshold} ---")
        
        # Apply threshold
        _, binary = cv2.threshold(inverted, threshold, 255, cv2.THRESH_BINARY)
        
        # Count white pixels
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.shape[0] * binary.shape[1]
        white_ratio = white_pixels / total_pixels
        
        print(f"White pixels: {white_pixels} ({white_ratio:.3f} of image)")
        
        # Save binary image
        cv2.imwrite(f"debug_step3_binary_{threshold}.jpg", binary)
        
        # Apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Count white pixels after morphology
        white_pixels_cleaned = np.sum(cleaned == 255)
        white_ratio_cleaned = white_pixels_cleaned / total_pixels
        
        print(f"After morphology: {white_pixels_cleaned} ({white_ratio_cleaned:.3f} of image)")
        
        # Save cleaned image
        cv2.imwrite(f"debug_step4_cleaned_{threshold}.jpg", cleaned)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        # Analyze contours
        if contours:
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                area_ratio = (w * h) / total_pixels
                aspect_ratio = max(w, h) / min(w, h)
                
                print(f"  Contour {i+1}: area={area}, bbox=({x},{y},{w},{h})")
                print(f"             area_ratio={area_ratio:.3f}, aspect={aspect_ratio:.1f}")
                
                # Check PCB criteria
                if (0.05 <= area_ratio <= 0.8 and  # Between 5% and 80% of image
                    1.2 <= aspect_ratio <= 5.0):    # Reasonable rectangle
                    print(f"  >>> POTENTIAL PCB DETECTED!")
                    
                    # Visualize this detection
                    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(vis_image, f"PCB T={threshold}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(f"debug_pcb_detected_{threshold}_{i+1}.jpg", vis_image)
                    print(f"  Saved visualization: debug_pcb_detected_{threshold}_{i+1}.jpg")
    
    # Try adaptive threshold approach
    print(f"\n--- Testing adaptive threshold ---")
    
    # Different adaptive methods
    adaptive_methods = [
        (cv2.ADAPTIVE_THRESH_MEAN_C, "MEAN"),
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "GAUSSIAN")
    ]
    
    for method, method_name in adaptive_methods:
        print(f"\nAdaptive method: {method_name}")
        
        # Apply adaptive threshold on inverted image
        adaptive = cv2.adaptiveThreshold(inverted, 255, method, cv2.THRESH_BINARY, 11, 2)
        
        # Count white pixels
        white_pixels = np.sum(adaptive == 255)
        white_ratio = white_pixels / total_pixels
        print(f"White pixels: {white_pixels} ({white_ratio:.3f} of image)")
        
        # Save adaptive result
        cv2.imwrite(f"debug_adaptive_{method_name.lower()}.jpg", adaptive)
        
        # Apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours")
        
        # Analyze best contours
        if contours:
            # Sort by area
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for i, contour in enumerate(contours_sorted[:5]):  # Top 5
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                area_ratio = (w * h) / total_pixels
                aspect_ratio = max(w, h) / min(w, h)
                
                print(f"  Contour {i+1}: area={area}, bbox=({x},{y},{w},{h})")
                print(f"             area_ratio={area_ratio:.3f}, aspect={aspect_ratio:.1f}")
                
                # Check PCB criteria
                if (0.05 <= area_ratio <= 0.8 and  # Between 5% and 80% of image
                    1.2 <= aspect_ratio <= 5.0):    # Reasonable rectangle
                    print(f"  >>> POTENTIAL PCB DETECTED!")
                    
                    # Visualize this detection
                    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
                    cv2.putText(vis_image, f"PCB {method_name}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.imwrite(f"debug_adaptive_pcb_{method_name.lower()}_{i+1}.jpg", vis_image)
                    print(f"  Saved: debug_adaptive_pcb_{method_name.lower()}_{i+1}.jpg")

if __name__ == "__main__":
    debug_dark_detection()
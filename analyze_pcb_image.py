#!/usr/bin/env python3
"""
Analyze the captured PCB image to debug detection issues.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_image():
    """Analyze the captured image for PCB detection debugging."""
    
    # Load the image
    image = cv2.imread("trigger_test_frame.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Could not load trigger_test_frame.jpg")
        return
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.1f}")
    
    # Test edge detection parameters
    print("\nTesting edge detection...")
    
    # Apply blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Try different edge detection thresholds
    thresholds = [(50, 150), (30, 100), (100, 200), (20, 80)]
    
    for i, (t1, t2) in enumerate(thresholds):
        edges = cv2.Canny(blurred, t1, t2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Threshold {t1}-{t2}: Found {len(contours)} contours")
        
        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            x, y, w, h = cv2.boundingRect(largest)
            
            total_area = image.shape[0] * image.shape[1]
            area_ratio = area / total_area
            
            print(f"  Largest contour: area={area}, bbox=({x},{y},{w},{h}), ratio={area_ratio:.3f}")
            
            # Save visualization
            vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(f"debug_edges_{t1}_{t2}.jpg", vis)
    
    # Test histogram analysis
    print("\nHistogram analysis:")
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Find peaks
    hist_smooth = cv2.GaussianBlur(hist.ravel(), (1, 15), 0)
    
    # Basic stats
    dark_pixels = np.sum(image < 50)
    bright_pixels = np.sum(image > 200)
    mid_pixels = np.sum((image >= 50) & (image <= 200))
    
    total_pixels = image.shape[0] * image.shape[1]
    
    print(f"Dark pixels (<50): {dark_pixels} ({100*dark_pixels/total_pixels:.1f}%)")
    print(f"Mid pixels (50-200): {mid_pixels} ({100*mid_pixels/total_pixels:.1f}%)")
    print(f"Bright pixels (>200): {bright_pixels} ({100*bright_pixels/total_pixels:.1f}%)")
    
    # Test adaptive thresholding
    print("\nTesting adaptive thresholding...")
    
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours_adaptive, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Adaptive threshold found {len(contours_adaptive)} contours")
    
    if contours_adaptive:
        largest = max(contours_adaptive, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)
        area_ratio = area / total_pixels
        
        print(f"  Largest: area={area}, bbox=({x},{y},{w},{h}), ratio={area_ratio:.3f}")
        
        # Save adaptive result
        vis_adaptive = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_adaptive, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite("debug_adaptive_threshold.jpg", vis_adaptive)
    
    # Try to detect the dark PCB object
    print("\nTrying to detect dark objects...")
    
    # Invert image to make dark objects bright
    inverted = 255 - image
    
    # Apply threshold
    _, binary = cv2.threshold(inverted, 80, 255, cv2.THRESH_BINARY)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    contours_dark, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Dark object detection found {len(contours_dark)} contours")
    
    if contours_dark:
        # Filter by size
        min_area = total_pixels * 0.01  # At least 1% of image
        large_contours = [c for c in contours_dark if cv2.contourArea(c) > min_area]
        
        print(f"Large contours (>1% of image): {len(large_contours)}")
        
        for i, contour in enumerate(large_contours[:5]):  # Show top 5
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = area / total_pixels
            aspect_ratio = max(w, h) / min(w, h)
            
            print(f"  Contour {i+1}: area={area}, bbox=({x},{y},{w},{h}), "
                  f"ratio={area_ratio:.3f}, aspect={aspect_ratio:.1f}")
        
        if large_contours:
            # Visualize best detection
            best_contour = max(large_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(best_contour)
            
            vis_dark = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis_dark, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(vis_dark, "Detected PCB", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite("debug_dark_object_detection.jpg", vis_dark)
            
            print("Dark object detection visualization saved")
            
            return True
    
    print("\nNo suitable PCB detected with current methods")
    return False

if __name__ == "__main__":
    analyze_image()
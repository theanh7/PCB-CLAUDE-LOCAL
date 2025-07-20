#!/usr/bin/env python3
"""
Debug script to test the complete inference pipeline step by step.
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_focus_calculation():
    """Test focus calculation with actual camera image."""
    print("=" * 60)
    print("TESTING FOCUS CALCULATION")
    print("=" * 60)
    
    try:
        from processing.preprocessor import FocusEvaluator
        
        # Load the actual camera image
        image_path = "trigger_test_frame.jpg"
        if not Path(image_path).exists():
            print(f"ERROR: Test image {image_path} not found!")
            return False
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"Loaded image: {image.shape}, dtype: {image.dtype}")
        print(f"Image stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}")
        
        # Test focus evaluator
        focus_evaluator = FocusEvaluator()
        focus_score = focus_evaluator.evaluate(image)
        
        print(f"Focus score: {focus_score:.2f}")
        print(f"Focus level: {focus_evaluator.get_focus_level(focus_score)}")
        print(f"Is acceptable (threshold=100): {focus_evaluator.is_acceptable(focus_score, 100)}")
        
        # Test with a region of the image (simulating PCB region)
        h, w = image.shape
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        region_focus = focus_evaluator.evaluate(center_region)
        
        print(f"Center region focus: {region_focus:.2f}")
        
        return focus_score > 0
        
    except Exception as e:
        print(f"Focus calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pcb_detection():
    """Test PCB detection with actual camera image."""
    print("\n" + "=" * 60)
    print("TESTING PCB DETECTION")
    print("=" * 60)
    
    try:
        from processing.pcb_detector import PCBDetector
        from core.config import TRIGGER_CONFIG
        
        # Load test image
        image_path = "trigger_test_frame.jpg"
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"Test image loaded: {image.shape}")
        
        # Initialize PCB detector
        detector = PCBDetector(TRIGGER_CONFIG)
        
        # Test detection multiple times to check stability
        print("Testing PCB detection stability...")
        for i in range(5):
            result = detector.detect_pcb(image)
            print(f"Detection {i+1}: has_pcb={result.has_pcb}, "
                  f"position={result.position}, stable={result.is_stable}, "
                  f"focus={result.focus_score:.1f}")
        
        # Get detection stats
        stats = detector.get_detection_stats()
        print(f"Detection stats: {stats}")
        
        # Test visualization
        if result.has_pcb:
            vis_image = detector.visualize_detection(image, result)
            cv2.imwrite("debug_pcb_detection.jpg", vis_image)
            print("Detection visualization saved as debug_pcb_detection.jpg")
        
        return result.has_pcb
        
    except Exception as e:
        print(f"PCB detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_inference():
    """Test AI inference with actual camera image."""
    print("\n" + "=" * 60)
    print("TESTING AI INFERENCE")
    print("=" * 60)
    
    try:
        from ai.inference import PCBDefectDetector
        from core.config import AI_CONFIG
        
        # Check if model exists
        model_path = AI_CONFIG["model_path"]
        if not Path(model_path).exists():
            print(f"ERROR: Model file {model_path} not found!")
            return False
        
        print(f"Loading model from: {model_path}")
        
        # Initialize detector
        detector = PCBDefectDetector(config=AI_CONFIG)
        
        # Load test image
        image_path = "trigger_test_frame.jpg"
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"Test image loaded: {image.shape}")
        
        # Convert to RGB for YOLO
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        print(f"RGB image for AI: {image_rgb.shape}")
        
        # Test inference
        print("Running AI inference...")
        start_time = time.time()
        results = detector.detect(image_rgb)
        inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time*1000:.1f}ms")
        print(f"Results type: {type(results)}")
        
        # Extract detection results
        defects = []
        locations = []
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            print(f"Found {len(results.boxes)} detections")
            
            for i, box in enumerate(results.boxes):
                class_id = int(box.cls)
                confidence = float(box.conf)
                bbox = box.xyxy[0].tolist()
                
                print(f"Detection {i+1}: class_id={class_id}, confidence={confidence:.3f}, bbox={bbox}")
                
                # Map class ID to defect name (if mapping exists)
                try:
                    from core.config import MODEL_CLASS_MAPPING
                    defect_name = MODEL_CLASS_MAPPING.get(class_id, f"Unknown_{class_id}")
                except ImportError:
                    defect_name = f"Defect_{class_id}"
                
                defects.append(defect_name)
                locations.append(bbox)
        else:
            print("No detections found")
        
        print(f"Final results: {len(defects)} defects detected")
        for i, defect in enumerate(defects):
            print(f"  {i+1}. {defect} at {locations[i]}")
        
        # Save visualization if detections found
        if defects:
            from processing.postprocessor import ResultPostprocessor
            postprocessor = ResultPostprocessor()
            vis_image = postprocessor.draw_results(image_rgb, results)
            cv2.imwrite("debug_ai_results.jpg", vis_image)
            print("AI results visualization saved as debug_ai_results.jpg")
        
        return True
        
    except Exception as e:
        print(f"AI inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_pipeline():
    """Test the complete pipeline from camera to results."""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE PIPELINE")
    print("=" * 60)
    
    try:
        from hardware.camera_controller import BaslerCamera
        from processing.pcb_detector import PCBDetector
        from ai.inference import PCBDefectDetector
        from core.config import CAMERA_CONFIG, TRIGGER_CONFIG, AI_CONFIG
        
        # Initialize components
        print("Initializing camera...")
        camera = BaslerCamera(CAMERA_CONFIG)
        
        print("Initializing PCB detector...")
        pcb_detector = PCBDetector(TRIGGER_CONFIG)
        
        print("Initializing AI detector...")
        ai_detector = PCBDefectDetector(config=AI_CONFIG)
        
        # Start camera streaming
        print("Starting camera streaming...")
        camera.start_streaming()
        
        # Test pipeline for 5 seconds
        print("Testing pipeline for 5 seconds...")
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        
        while time.time() - start_time < 5.0:
            # Get frame
            raw_frame = camera.get_preview_frame()
            
            if raw_frame is not None:
                frame_count += 1
                
                # PCB detection
                detection_result = pcb_detector.detect_pcb(raw_frame)
                
                if detection_result.has_pcb and detection_result.is_stable:
                    detection_count += 1
                    print(f"Frame {frame_count}: PCB detected and stable, focus={detection_result.focus_score:.1f}")
                    
                    # High quality capture
                    hq_image = camera.capture_high_quality()
                    if hq_image is not None:
                        # Convert to RGB for AI
                        if len(hq_image.shape) == 2:
                            hq_rgb = cv2.cvtColor(hq_image, cv2.COLOR_GRAY2RGB)
                        else:
                            hq_rgb = hq_image
                        
                        # AI inference
                        ai_results = ai_detector.detect(hq_rgb)
                        
                        defect_count = 0
                        if hasattr(ai_results, 'boxes') and ai_results.boxes is not None:
                            defect_count = len(ai_results.boxes)
                        
                        print(f"  AI inference: {defect_count} defects detected")
                        
                        # Save this frame for analysis
                        cv2.imwrite("debug_pipeline_frame.jpg", hq_image)
                        print("  Pipeline test frame saved")
                        
                        break  # Found one good detection, that's enough
            
            time.sleep(0.033)  # ~30 FPS
        
        # Stop camera
        camera.stop_streaming()
        
        print(f"Pipeline test complete: {frame_count} frames, {detection_count} stable detections")
        
        return frame_count > 0 and detection_count > 0
        
    except Exception as e:
        print(f"Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests."""
    print("STARTING COMPREHENSIVE INFERENCE DEBUG")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Focus calculation
    results['focus'] = test_focus_calculation()
    
    # Test 2: PCB detection
    results['pcb_detection'] = test_pcb_detection()
    
    # Test 3: AI inference
    results['ai_inference'] = test_ai_inference()
    
    # Test 4: Complete pipeline
    results['complete_pipeline'] = test_complete_pipeline()
    
    # Summary
    print("\n" + "=" * 80)
    print("DEBUG RESULTS SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    overall_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"\nOVERALL: {overall_status}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
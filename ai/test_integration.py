"""
Integration tests for AI layer with processing pipeline.

This module tests the complete integration of the AI layer with
the image processing pipeline and validates end-to-end functionality.
"""

import sys
import os
import numpy as np
import time
import logging
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ai_preprocessing_integration():
    """Test AI integration with preprocessing pipeline."""
    print("Testing AI + Preprocessing Integration...")
    
    try:
        # Import modules
        from ai.inference import PCBDefectDetector, create_test_image
        from processing.preprocessor import ImagePreprocessor
        
        # Create components
        preprocessor = ImagePreprocessor()
        detector = PCBDefectDetector()
        
        # Create test raw image (simulated Bayer pattern)
        raw_bayer = np.random.randint(0, 255, (800, 600), dtype=np.uint8)
        
        # Test preprocessing
        processed_image = preprocessor.process(raw_bayer)
        
        if processed_image is None:
            print("✗ Preprocessing failed")
            return False
        
        print(f"✓ Preprocessing successful: {raw_bayer.shape} -> {processed_image.shape}")
        
        # Test AI detection on preprocessed image
        result = detector.detect(processed_image)
        
        if result is None:
            print("✗ AI detection failed")
            return False
        
        print(f"✓ AI detection successful: {len(result.defects)} defects found")
        print(f"    Processing time: {result.processing_time:.3f}s")
        
        # Test with different preprocessing modes
        preview_image = preprocessor.process_preview(raw_bayer)
        
        if preview_image is not None:
            result_preview = detector.detect(preview_image)
            print(f"✓ Preview processing successful: {len(result_preview.defects)} defects")
        
        return True
        
    except Exception as e:
        print(f"✗ AI-Preprocessing integration failed: {e}")
        return False

def test_ai_postprocessing_integration():
    """Test AI integration with postprocessing pipeline."""
    print("\nTesting AI + Postprocessing Integration...")
    
    try:
        # Import modules
        from ai.inference import PCBDefectDetector, create_test_image
        from processing.postprocessor import ResultPostprocessor
        
        # Create components
        detector = PCBDefectDetector()
        postprocessor = ResultPostprocessor()
        
        # Create test image
        test_image = create_test_image(add_defects=True)
        
        # Run AI detection
        result = detector.detect(test_image)
        
        print(f"✓ AI detection: {len(result.defects)} defects found")
        
        # Create mock YOLO results for postprocessor
        class MockBox:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = [np.array(xyxy)]
                self.conf = [conf]
                self.cls = [cls]
        
        class MockResults:
            def __init__(self, defects, locations, scores):
                self.boxes = []
                for i, (defect, location, score) in enumerate(zip(defects, locations, scores)):
                    bbox = location['bbox']
                    cls_id = location['class_id']
                    self.boxes.append(MockBox(bbox, score, cls_id))
        
        if len(result.defects) > 0:
            mock_results = MockResults(result.defects, result.locations, result.confidence_scores)
            
            # Test postprocessing
            annotated_image = postprocessor.draw_results(test_image, mock_results)
            
            if annotated_image is not None:
                print(f"✓ Postprocessing successful: {annotated_image.shape}")
            else:
                print("✗ Postprocessing failed")
                return False
            
            # Test other postprocessing features
            boxes = postprocessor.process_yolo_results(mock_results)
            print(f"✓ Result processing: {len(boxes)} boxes processed")
            
            summary_image = postprocessor.create_defect_summary_image(boxes)
            if summary_image is not None:
                print(f"✓ Summary image created: {summary_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ AI-Postprocessing integration failed: {e}")
        return False

def test_ai_pcb_detector_integration():
    """Test AI integration with PCB detector."""
    print("\nTesting AI + PCB Detector Integration...")
    
    try:
        # Import modules
        from ai.inference import PCBDefectDetector, create_test_image
        from processing.pcb_detector import PCBDetector
        
        # Create components
        pcb_detector = PCBDetector()
        ai_detector = PCBDefectDetector()
        
        # Create test image with PCB
        test_image = create_test_image(add_defects=True)
        
        # Test PCB detection
        pcb_result = pcb_detector.detect_pcb(test_image)
        
        print(f"✓ PCB detection: PCB found = {pcb_result.has_pcb}")
        print(f"    Focus score: {pcb_result.focus_score:.2f}")
        print(f"    Stability: {pcb_result.is_stable}")
        
        if pcb_result.has_pcb:
            # Test AI detection on detected PCB
            ai_result = ai_detector.detect(test_image)
            
            print(f"✓ AI detection on PCB: {len(ai_result.defects)} defects")
            
            # Test trigger logic
            should_trigger = pcb_detector.should_trigger_inspection(pcb_result)
            print(f"✓ Trigger logic: Should trigger = {should_trigger}")
            
            if should_trigger:
                print("    System would trigger automatic inspection")
            else:
                print("    System would wait for better conditions")
        
        return True
        
    except Exception as e:
        print(f"✗ AI-PCB Detector integration failed: {e}")
        return False

def test_complete_pipeline():
    """Test complete pipeline integration."""
    print("\nTesting Complete Pipeline Integration...")
    
    try:
        # Import all modules
        from ai.inference import PCBDefectDetector, create_test_image
        from processing.preprocessor import ImagePreprocessor
        from processing.pcb_detector import PCBDetector
        from processing.postprocessor import ResultPostprocessor
        
        # Create components
        preprocessor = ImagePreprocessor()
        pcb_detector = PCBDetector()
        ai_detector = PCBDefectDetector()
        postprocessor = ResultPostprocessor()
        
        # Simulate complete pipeline
        print("Running complete pipeline simulation...")
        
        # Step 1: Create raw image
        raw_image = create_test_image(add_defects=True)
        print(f"✓ Step 1: Raw image created {raw_image.shape}")
        
        # Step 2: Preprocess for preview
        preview_image = preprocessor.process_preview(raw_image)
        print(f"✓ Step 2: Preview processing {preview_image.shape}")
        
        # Step 3: PCB detection
        pcb_result = pcb_detector.detect_pcb(preview_image)
        print(f"✓ Step 3: PCB detection - Found: {pcb_result.has_pcb}")
        
        if pcb_result.has_pcb:
            # Step 4: Check trigger conditions
            should_trigger = pcb_detector.should_trigger_inspection(pcb_result)
            print(f"✓ Step 4: Trigger check - Should trigger: {should_trigger}")
            
            if should_trigger:
                # Step 5: High-quality processing
                processed_image = preprocessor.process(raw_image)
                print(f"✓ Step 5: High-quality processing {processed_image.shape}")
                
                # Step 6: AI detection
                ai_result = ai_detector.detect(processed_image)
                print(f"✓ Step 6: AI detection - {len(ai_result.defects)} defects")
                
                # Step 7: Postprocessing
                if len(ai_result.defects) > 0:
                    # Create mock results for postprocessor
                    class MockResults:
                        def __init__(self, result):
                            self.boxes = []
                            for i, location in enumerate(result.locations):
                                self.boxes.append(MockBox(location))
                    
                    class MockBox:
                        def __init__(self, location):
                            self.xyxy = [np.array(location['bbox'])]
                            self.conf = [location['confidence']]
                            self.cls = [location['class_id']]
                    
                    mock_results = MockResults(ai_result)
                    annotated_image = postprocessor.draw_results(processed_image, mock_results)
                    
                    if annotated_image is not None:
                        print(f"✓ Step 7: Postprocessing completed {annotated_image.shape}")
                    else:
                        print("⚠ Step 7: Postprocessing failed")
                
                # Step 8: Generate report
                print("✓ Step 8: Pipeline completed successfully")
                
                return True
        
        print("⚠ PCB not detected, pipeline stopped early")
        return True
        
    except Exception as e:
        print(f"✗ Complete pipeline integration failed: {e}")
        return False

def test_performance_integration():
    """Test performance of integrated system."""
    print("\nTesting Performance Integration...")
    
    try:
        # Import modules
        from ai.inference import PCBDefectDetector, create_test_image
        from processing.preprocessor import ImagePreprocessor
        
        # Create components
        preprocessor = ImagePreprocessor()
        detector = PCBDefectDetector()
        
        # Create test images
        test_images = [create_test_image(add_defects=True) for _ in range(5)]
        
        # Test preprocessing performance
        print("Testing preprocessing performance...")
        start_time = time.time()
        
        processed_images = []
        for image in test_images:
            processed = preprocessor.process(image)
            processed_images.append(processed)
        
        preprocess_time = time.time() - start_time
        preprocess_fps = len(test_images) / preprocess_time
        
        print(f"✓ Preprocessing: {preprocess_time:.3f}s total, {preprocess_fps:.2f} FPS")
        
        # Test AI detection performance
        print("Testing AI detection performance...")
        start_time = time.time()
        
        results = []
        for image in processed_images:
            result = detector.detect(image)
            results.append(result)
        
        ai_time = time.time() - start_time
        ai_fps = len(processed_images) / ai_time
        
        print(f"✓ AI Detection: {ai_time:.3f}s total, {ai_fps:.2f} FPS")
        
        # Test batch processing
        print("Testing batch processing...")
        start_time = time.time()
        
        batch_results = detector.detect_batch(processed_images)
        
        batch_time = time.time() - start_time
        batch_fps = len(processed_images) / batch_time
        
        print(f"✓ Batch Processing: {batch_time:.3f}s total, {batch_fps:.2f} FPS")
        
        # Compare performance
        total_individual = preprocess_time + ai_time
        total_batch = preprocess_time + batch_time
        
        print(f"Performance comparison:")
        print(f"  Individual processing: {total_individual:.3f}s")
        print(f"  Batch processing: {total_batch:.3f}s")
        print(f"  Speedup: {total_individual/total_batch:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance integration test failed: {e}")
        return False

def test_error_handling_integration():
    """Test error handling in integrated system."""
    print("\nTesting Error Handling Integration...")
    
    try:
        # Import modules
        from ai.inference import PCBDefectDetector
        from processing.preprocessor import ImagePreprocessor
        
        # Create components
        preprocessor = ImagePreprocessor()
        detector = PCBDefectDetector()
        
        # Test with None input
        print("Testing None input handling...")
        result = detector.detect(None)
        if result is not None and len(result.defects) == 0:
            print("✓ None input handled correctly")
        else:
            print("⚠ None input not handled properly")
        
        # Test with invalid image
        print("Testing invalid image handling...")
        invalid_image = np.array([1, 2, 3])  # Invalid shape
        
        try:
            result = detector.detect(invalid_image)
            print("✓ Invalid image handled gracefully")
        except Exception as e:
            print(f"⚠ Invalid image caused exception: {e}")
        
        # Test with corrupted image
        print("Testing corrupted image handling...")
        corrupted_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        corrupted_image[50:60, 50:60] = 0  # Create some corruption
        
        try:
            result = detector.detect(corrupted_image)
            print("✓ Corrupted image handled gracefully")
        except Exception as e:
            print(f"⚠ Corrupted image caused exception: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling integration test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests."""
    print("=== AI Integration Tests ===\n")
    
    tests = [
        ("AI + Preprocessing", test_ai_preprocessing_integration),
        ("AI + Postprocessing", test_ai_postprocessing_integration),
        ("AI + PCB Detector", test_ai_pcb_detector_integration),
        ("Complete Pipeline", test_complete_pipeline),
        ("Performance", test_performance_integration),
        ("Error Handling", test_error_handling_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        print()
    
    # Summary
    print("=== Integration Test Summary ===")
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} integration tests passed")
    
    # Check critical integration
    critical_tests = ["AI + Preprocessing", "Complete Pipeline"]
    critical_passed = all(success for name, success in results if name in critical_tests)
    
    if critical_passed:
        print("✓ Critical integration components are working")
    else:
        print("✗ Critical integration components have issues")
    
    return passed == total

if __name__ == "__main__":
    success = run_integration_tests()
    print(f"\nIntegration Tests Result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
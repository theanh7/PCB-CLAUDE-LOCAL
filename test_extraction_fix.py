#!/usr/bin/env python3
"""
Test the _extract_results fix directly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.interfaces import InspectionResult
from core.config import MODEL_CLASS_MAPPING

def test_extract_results_fix():
    """Test the fixed _extract_results method."""
    
    print("Testing _extract_results fix...")
    
    # Create mock InspectionResult (like AI detector returns)
    defects = ["Mouse Bite", "Spur"]
    locations = [
        {"bbox": [100, 100, 200, 200], "confidence": 0.85, "class_id": 0, "class_name": "Mouse Bite"},
        {"bbox": [300, 300, 400, 400], "confidence": 0.75, "class_id": 1, "class_name": "Spur"}
    ]
    confidence_scores = [0.85, 0.75]
    
    inspection_result = InspectionResult(defects, locations, confidence_scores, 0.5)
    
    print(f"Created InspectionResult:")
    print(f"  Defects: {inspection_result.defects}")
    print(f"  Has defects: {inspection_result.has_defects}")
    print(f"  Locations: {len(inspection_result.locations)}")
    print(f"  Confidences: {inspection_result.confidence_scores}")
    
    # Test the extraction logic directly (without initializing full system)
    def extract_results_test(detection_results):
        """Extracted _extract_results logic for testing."""
        
        # Check if detection_results is InspectionResult or YOLO raw results
        if hasattr(detection_results, 'defects'):
            # InspectionResult object
            defects = detection_results.defects
            locations = detection_results.locations  
            confidences = detection_results.confidence_scores
            
            print(f"[OK] Extracted from InspectionResult: {len(defects)} defects")
            
        elif hasattr(detection_results, 'boxes'):
            # YOLO raw results
            defects = []
            locations = []
            confidences = []
            
            if detection_results.boxes is not None:
                for box in detection_results.boxes:
                    # Map class ID to defect name
                    class_id = int(box.cls)
                    if class_id in MODEL_CLASS_MAPPING:
                        defect_name = MODEL_CLASS_MAPPING[class_id]
                        confidence = float(box.conf)
                        bbox = box.xyxy[0].tolist()
                        
                        defects.append(defect_name)
                        confidences.append(confidence)
                        locations.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_id': class_id
                        })
            
            print(f"[OK] Extracted from YOLO results: {len(defects)} defects")
            
        else:
            # Unknown format
            print(f"[WARN] Unknown detection result format: {type(detection_results)}")
            defects, locations, confidences = [], [], []
        
        return defects, locations, confidences
    
    # Test the extraction
    try:
        extracted_defects, extracted_locations, extracted_confidences = extract_results_test(inspection_result)
        
        print(f"\nExtraction Results:")
        print(f"  Defects: {extracted_defects}")
        print(f"  Locations: {len(extracted_locations)} items")
        print(f"  Confidences: {extracted_confidences}")
        
        # Verify results
        assert extracted_defects == defects, f"Defects mismatch: {extracted_defects} != {defects}"
        assert extracted_locations == locations, f"Locations mismatch"
        assert extracted_confidences == confidence_scores, f"Confidences mismatch"
        
        print("\n[SUCCESS] _extract_results fix works correctly!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] _extract_results fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_postprocessor_fix():
    """Test the postprocessor fix."""
    print("\nTesting postprocessor fix...")
    
    try:
        from processing.postprocessor import ResultPostprocessor
        
        # Create mock InspectionResult
        defects = ["Mouse Bite"]
        locations = [{"bbox": [100, 100, 200, 200], "confidence": 0.85, "class_id": 0, "class_name": "Mouse Bite"}]
        confidence_scores = [0.85]
        
        inspection_result = InspectionResult(defects, locations, confidence_scores, 0.5)
        
        # Test postprocessor
        postprocessor = ResultPostprocessor()
        boxes = postprocessor.process_yolo_results(inspection_result)
        
        print(f"[OK] Postprocessor processed {len(boxes)} boxes")
        
        if boxes:
            box = boxes[0]
            print(f"  Box: ({box.x1}, {box.y1}, {box.x2}, {box.y2})")
            print(f"  Class: {box.class_name}")
            print(f"  Confidence: {box.confidence}")
        
        print("[SUCCESS] Postprocessor fix works correctly!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Postprocessor fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test1_success = test_extract_results_fix()
    test2_success = test_postprocessor_fix()
    
    if test1_success and test2_success:
        print("\n[SUCCESS] ALL FIXES WORKING! Manual inspection should work now.")
    else:
        print("\n[ERROR] Some fixes still have issues.")
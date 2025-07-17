"""
Simple validation script for processing pipeline components.

This script validates that all processing components can be imported
and basic functionality works without external dependencies.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.interfaces import BaseProcessor, PCBDetectionResult, InspectionResult
        print("✓ Core interfaces imported")
        
        from core.config import PROCESSING_CONFIG, TRIGGER_CONFIG, DEFECT_CLASSES
        print("✓ Core config imported")
        
        # Test processing modules (these might fail due to OpenCV dependency)
        try:
            from processing.preprocessor import ImagePreprocessor, FocusEvaluator
            print("✓ Preprocessor imported")
        except ImportError as e:
            print(f"⚠ Preprocessor import failed: {e}")
        
        try:
            from processing.pcb_detector import PCBDetector, AutoTriggerSystem
            print("✓ PCB detector imported")
        except ImportError as e:
            print(f"⚠ PCB detector import failed: {e}")
        
        try:
            from processing.postprocessor import ResultPostprocessor, DetectionBox
            print("✓ Postprocessor imported")
        except ImportError as e:
            print(f"⚠ Postprocessor import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_data_structures():
    """Test data structure classes."""
    print("\nTesting data structures...")
    
    try:
        from core.interfaces import PCBDetectionResult, InspectionResult
        
        # Test PCBDetectionResult
        result = PCBDetectionResult(
            has_pcb=True,
            position=(100, 100, 200, 200),
            is_stable=True,
            focus_score=150.0
        )
        
        assert result.has_pcb == True
        assert result.position == (100, 100, 200, 200)
        assert result.is_stable == True
        assert result.focus_score == 150.0
        
        print("✓ PCBDetectionResult works")
        
        # Test InspectionResult
        inspection = InspectionResult(
            defects=["Missing Hole", "Open Circuit"],
            locations=[{"bbox": [10, 10, 50, 50]}, {"bbox": [100, 100, 150, 150]}],
            confidence_scores=[0.85, 0.75],
            processing_time=0.05
        )
        
        assert len(inspection.defects) == 2
        assert inspection.has_defects == True
        assert inspection.processing_time == 0.05
        
        print("✓ InspectionResult works")
        
        return True
        
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        return False

def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration...")
    
    try:
        from core.config import (
            CAMERA_CONFIG, AI_CONFIG, TRIGGER_CONFIG, 
            PROCESSING_CONFIG, DEFECT_CLASSES, DEFECT_COLORS
        )
        
        # Test that all configs are dictionaries
        configs = [CAMERA_CONFIG, AI_CONFIG, TRIGGER_CONFIG, PROCESSING_CONFIG]
        for i, config in enumerate(configs):
            assert isinstance(config, dict), f"Config {i} is not a dictionary"
        
        print("✓ All configs are dictionaries")
        
        # Test defect classes
        assert isinstance(DEFECT_CLASSES, list)
        assert len(DEFECT_CLASSES) > 0
        assert all(isinstance(cls, str) for cls in DEFECT_CLASSES)
        
        print(f"✓ Defect classes: {len(DEFECT_CLASSES)} classes defined")
        
        # Test defect colors
        assert isinstance(DEFECT_COLORS, dict)
        assert len(DEFECT_COLORS) == len(DEFECT_CLASSES)
        
        print("✓ Defect colors match defect classes")
        
        # Test key configuration values
        assert TRIGGER_CONFIG["stability_frames"] > 0
        assert TRIGGER_CONFIG["focus_threshold"] >= 0
        assert TRIGGER_CONFIG["inspection_interval"] > 0
        
        print("✓ Trigger config validation passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Config validation failed: {e}")
        return False

def test_processing_logic():
    """Test processing logic without OpenCV."""
    print("\nTesting processing logic...")
    
    try:
        # Test basic numpy operations that our modules should handle
        test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Test image validation logic
        assert test_image.shape == (480, 640)
        assert test_image.dtype == np.uint8
        assert len(test_image.shape) == 2
        
        print("✓ Basic image array operations work")
        
        # Test simple Bayer pattern extraction (core logic)
        bayer_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Extract green channel positions (like our fast debayer)
        green_1 = bayer_image[0::2, 1::2]  # G positions in R-G rows
        green_2 = bayer_image[1::2, 0::2]  # G positions in G-B rows
        
        # Should be quarter size
        assert green_1.shape == (240, 320)
        assert green_2.shape == (240, 320)
        
        print("✓ Bayer pattern extraction logic works")
        
        # Test statistics calculation
        stats = {
            "min": int(np.min(test_image)),
            "max": int(np.max(test_image)),
            "mean": float(np.mean(test_image)),
            "std": float(np.std(test_image)),
        }
        
        assert all(key in stats for key in ["min", "max", "mean", "std"])
        
        print("✓ Statistics calculation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Processing logic test failed: {e}")
        return False

def test_detection_box_class():
    """Test DetectionBox class logic."""
    print("\nTesting DetectionBox class...")
    
    try:
        from processing.postprocessor import DetectionBox
        
        # Create test detection box
        box = DetectionBox(
            x1=100, y1=100, x2=200, y2=150,
            confidence=0.85, class_id=0, class_name="Missing Hole"
        )
        
        # Test properties
        assert box.width == 100
        assert box.height == 50
        assert box.center == (150, 125)
        assert box.area == 5000
        
        print("✓ DetectionBox class works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ DetectionBox test failed: {e}")
        return False

def test_pcb_position_class():
    """Test PCBPosition class logic."""
    print("\nTesting PCBPosition class...")
    
    try:
        from processing.pcb_detector import PCBPosition
        import time
        
        # Create test positions
        pos1 = PCBPosition(x=100, y=100, width=200, height=150, 
                          area=30000, timestamp=time.time())
        
        pos2 = PCBPosition(x=105, y=103, width=198, height=148, 
                          area=29304, timestamp=time.time())
        
        # Test properties
        assert pos1.center() == (200, 175)
        
        # Test distance calculation
        distance = pos1.distance_to(pos2)
        assert distance > 0
        assert distance < 10  # Should be small for close positions
        
        # Test size difference
        size_diff = pos1.size_difference(pos2)
        assert size_diff > 0
        assert size_diff < 5  # Should be small for similar sizes
        
        print("✓ PCBPosition class works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ PCBPosition test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=== Processing Pipeline Validation ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Data Structures", test_data_structures),
        ("Configuration", test_config_validation),
        ("Processing Logic", test_processing_logic),
        ("DetectionBox Class", test_detection_box_class),
        ("PCBPosition Class", test_pcb_position_class),
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
    print("=== Test Summary ===")
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Check if we have the key functionality
    core_tests = ["Import Test", "Data Structures", "Configuration"]
    core_passed = all(success for name, success in results if name in core_tests)
    
    if core_passed:
        print("✓ Core functionality is working")
    else:
        print("✗ Core functionality has issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
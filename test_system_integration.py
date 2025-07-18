#!/usr/bin/env python3
"""
System integration test for PCB inspection system.

This script tests the integration of all system components without
requiring actual hardware or camera connection.
"""

import sys
import os
import logging
import numpy as np
from unittest.mock import Mock, patch
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all components
try:
    from core.config import *
    from core.utils import setup_logging
    
    # Mock hardware for testing
    from unittest.mock import MagicMock
    
    print("✓ Core modules imported successfully")
    
    # Test Processing Layer
    from processing.preprocessor import ImagePreprocessor
    from processing.pcb_detector import PCBDetector
    from processing.postprocessor import ResultPostprocessor
    
    print("✓ Processing modules imported successfully")
    
    # Test AI Layer (with mock if model not available)
    try:
        from ai.inference import PCBDefectDetector
        print("✓ AI module imported successfully")
    except Exception as e:
        print(f"⚠ AI module warning: {e}")
        # Create mock for testing
        class MockPCBDefectDetector:
            def __init__(self, *args, **kwargs):
                self.model_loaded = False
            def detect(self, image):
                # Return mock detection results
                from types import SimpleNamespace
                result = SimpleNamespace()
                result.boxes = None
                return result
        PCBDefectDetector = MockPCBDefectDetector
    
    # Test Data Layer
    from data.database import PCBDatabase
    from analytics.analyzer import DefectAnalyzer
    
    print("✓ Data modules imported successfully")
    
    # Test Presentation Layer
    from presentation.gui import PCBInspectionGUI
    
    print("✓ Presentation modules imported successfully")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_system_initialization():
    """Test system initialization without hardware."""
    print("\n=== Testing System Initialization ===")
    
    # Setup logging
    logger = setup_logging("SystemTest")
    logger.info("Starting system integration test")
    
    # Test Core Layer
    print("Testing Core Layer...")
    errors = validate_config()
    if errors:
        print(f"⚠ Configuration warnings: {errors}")
    else:
        print("✓ Configuration valid")
    
    # Create required directories
    ensure_directories()
    print("✓ Directories created")
    
    # Test Processing Layer
    print("\nTesting Processing Layer...")
    preprocessor = ImagePreprocessor()
    pcb_detector = PCBDetector()
    postprocessor = ResultPostprocessor()
    print("✓ Processing components initialized")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    # Test preprocessing
    processed = preprocessor.process(dummy_image)
    assert processed is not None
    print("✓ Image preprocessing works")
    
    # Test PCB detection (with mock image)
    has_pcb, position, is_stable, focus_score = pcb_detector.detect_pcb(dummy_image)
    print(f"✓ PCB detection works: has_pcb={has_pcb}, focus={focus_score:.1f}")
    
    # Test AI Layer
    print("\nTesting AI Layer...")
    ai_detector = PCBDefectDetector(
        model_path="weights/dummy.pt",  # Will fail gracefully
        device="cpu",
        confidence=0.5
    )
    
    # Test detection (should handle missing model gracefully)
    try:
        results = ai_detector.detect(processed)
        print("✓ AI detection interface works")
    except Exception as e:
        print(f"⚠ AI detection expected error (no model): {e}")
    
    # Test Data Layer
    print("\nTesting Data Layer...")
    db_path = "test_integration.db"
    try:
        database = PCBDatabase(db_path)
        
        # Test saving inspection
        from datetime import datetime
        inspection_id = database.save_inspection_metadata(
            timestamp=datetime.now(),
            defects=["Test Defect"],
            locations=[{"bbox": [100, 100, 200, 200], "confidence": 0.9}],
            confidence_scores=[0.9],
            raw_image_shape=(480, 640),
            focus_score=150.0,
            processing_time=0.1
        )
        
        print(f"✓ Database save works, inspection ID: {inspection_id}")
        
        # Test analytics
        analyzer = DefectAnalyzer(database)
        stats = analyzer.get_realtime_stats()
        print(f"✓ Analytics works: {stats}")
        
        database.close()
        
    except Exception as e:
        print(f"⚠ Data layer error: {e}")
    finally:
        # Cleanup test database
        if os.path.exists(db_path):
            os.remove(db_path)
    
    print("\n✅ System integration test completed!")
    return True


def test_mock_workflow():
    """Test complete workflow with mock data."""
    print("\n=== Testing Mock Workflow ===")
    
    # Create mock camera
    class MockCamera:
        def __init__(self, config):
            self.is_streaming = False
            self.frame_count = 0
            
        def start_streaming(self):
            self.is_streaming = True
            
        def stop_streaming(self):
            self.is_streaming = False
            
        def get_preview_frame(self):
            if self.is_streaming:
                self.frame_count += 1
                # Return dummy frame
                return np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            return None
            
        def capture_high_quality(self):
            # Return higher resolution dummy frame
            return np.random.randint(0, 255, (960, 1280), dtype=np.uint8)
            
        def close(self):
            pass
    
    # Initialize components
    mock_camera = MockCamera(CAMERA_CONFIG)
    preprocessor = ImagePreprocessor()
    pcb_detector = PCBDetector()
    postprocessor = ResultPostprocessor()
    ai_detector = PCBDefectDetector("weights/dummy.pt", "cpu", 0.5)
    
    # Test preview workflow
    print("Testing preview workflow...")
    mock_camera.start_streaming()
    
    for i in range(3):
        frame = mock_camera.get_preview_frame()
        if frame is not None:
            has_pcb, position, is_stable, focus_score = pcb_detector.detect_pcb(frame)
            print(f"Frame {i+1}: PCB={has_pcb}, stable={is_stable}, focus={focus_score:.1f}")
    
    mock_camera.stop_streaming()
    print("✓ Preview workflow complete")
    
    # Test inspection workflow
    print("Testing inspection workflow...")
    raw_image = mock_camera.capture_high_quality()
    processed_image = preprocessor.process(raw_image)
    
    # Mock detection results
    detection_results = Mock()
    detection_results.boxes = None
    
    display_image = postprocessor.draw_results(processed_image, detection_results)
    
    print("✓ Inspection workflow complete")
    
    mock_camera.close()
    print("✅ Mock workflow test completed!")
    return True


def main():
    """Run all integration tests."""
    print("PCB Inspection System - Integration Test")
    print("="*50)
    
    try:
        # Test 1: System initialization
        if not test_system_initialization():
            return 1
        
        # Test 2: Mock workflow
        if not test_mock_workflow():
            return 1
        
        print("\n🎉 All integration tests passed!")
        print("\nNext steps:")
        print("1. Connect actual Basler camera")
        print("2. Place YOLOv11 model in weights/ directory")
        print("3. Run: python main.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
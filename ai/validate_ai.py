"""
AI validation script for PCB defect detection system.

This script validates the AI components can be loaded and basic functionality
works without requiring the actual trained model for testing.
"""

import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all AI modules can be imported."""
    print("Testing AI imports...")
    
    try:
        from core.config import AI_CONFIG, MODEL_CLASS_MAPPING, DEFECT_CLASSES
        print("✓ Core config imported")
        
        from core.interfaces import InspectionResult
        print("✓ Core interfaces imported")
        
        # Test AI modules
        try:
            from ai.inference import PCBDefectDetector, ModelManager, create_test_image
            print("✓ AI inference module imported")
        except ImportError as e:
            print(f"⚠ AI inference import failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_configuration():
    """Test AI configuration."""
    print("\nTesting AI configuration...")
    
    try:
        from core.config import AI_CONFIG, MODEL_CLASS_MAPPING, DEFECT_CLASSES
        
        # Test AI config structure
        required_keys = ["model_path", "confidence", "device", "imgsz"]
        for key in required_keys:
            if key not in AI_CONFIG:
                print(f"✗ Missing config key: {key}")
                return False
        
        print("✓ AI config structure is valid")
        
        # Test model path
        model_path = AI_CONFIG["model_path"]
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path)
            print(f"✓ Model file exists: {model_path} ({model_size / 1e6:.1f} MB)")
        else:
            print(f"⚠ Model file not found: {model_path}")
            # Don't fail the test, as model might not be available during development
        
        # Test class mapping
        if len(MODEL_CLASS_MAPPING) != len(DEFECT_CLASSES):
            print("⚠ Class mapping size mismatch")
        else:
            print("✓ Class mapping is consistent")
        
        # Test that all mapped classes are in DEFECT_CLASSES
        for class_id, class_name in MODEL_CLASS_MAPPING.items():
            if class_name not in DEFECT_CLASSES:
                print(f"✗ Mapped class not in DEFECT_CLASSES: {class_name}")
                return False
        
        print("✓ All mapped classes are valid")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_model_manager():
    """Test ModelManager functionality."""
    print("\nTesting ModelManager...")
    
    try:
        from ai.inference import ModelManager
        
        manager = ModelManager()
        
        # Test validation
        validation_result = manager.validate_model()
        
        required_keys = ["model_exists", "config_valid", "gpu_available", "errors"]
        for key in required_keys:
            if key not in validation_result:
                print(f"✗ Missing validation key: {key}")
                return False
        
        print("✓ Model validation structure is correct")
        
        if validation_result["gpu_available"]:
            print("✓ GPU is available")
        else:
            print("⚠ GPU not available, will use CPU")
        
        if validation_result["config_valid"]:
            print("✓ Configuration is valid")
        else:
            print("⚠ Configuration issues found:")
            for error in validation_result["errors"]:
                print(f"    - {error}")
        
        # Test metadata
        metadata = manager.get_model_metadata()
        
        required_metadata = ["model_path", "framework", "classes", "class_mapping"]
        for key in required_metadata:
            if key not in metadata:
                print(f"✗ Missing metadata key: {key}")
                return False
        
        print("✓ Model metadata is complete")
        
        return True
        
    except Exception as e:
        print(f"✗ ModelManager test failed: {e}")
        return False

def test_torch_integration():
    """Test PyTorch integration."""
    print("\nTesting PyTorch integration...")
    
    try:
        import torch
        
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                print(f"    Device {i}: {device_name}")
        else:
            print("⚠ CUDA not available, will use CPU")
        
        # Test basic tensor operations
        x = torch.rand(5, 3)
        y = torch.rand(3, 5)
        z = torch.mm(x, y)
        
        print("✓ Basic tensor operations work")
        
        # Test GPU operations if available
        if torch.cuda.is_available():
            try:
                x_gpu = x.cuda()
                y_gpu = y.cuda()
                z_gpu = torch.mm(x_gpu, y_gpu)
                print("✓ GPU tensor operations work")
            except Exception as e:
                print(f"⚠ GPU operations failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch integration test failed: {e}")
        return False

def test_ultralytics_integration():
    """Test Ultralytics integration."""
    print("\nTesting Ultralytics integration...")
    
    try:
        from ultralytics import YOLO
        
        print("✓ Ultralytics imported successfully")
        
        # Test with a dummy model (if available)
        model_path = "weights/best.pt"
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
                print("✓ Model loaded successfully")
                
                # Test model info
                print(f"    Model type: {type(model)}")
                
                # Test prediction on dummy image
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                start_time = time.time()
                results = model(dummy_image, verbose=False)
                inference_time = time.time() - start_time
                
                print(f"✓ Inference successful ({inference_time:.3f}s)")
                
                # Test result structure
                if results and len(results) > 0:
                    result = results[0]
                    print(f"✓ Result structure: {type(result)}")
                    
                    if hasattr(result, 'boxes'):
                        print(f"    Boxes: {result.boxes}")
                    
                else:
                    print("⚠ No results returned")
                
            except Exception as e:
                print(f"⚠ Model loading failed: {e}")
                print("    This is expected if the model file is not available")
        else:
            print(f"⚠ Model file not found: {model_path}")
            print("    This is expected during development")
        
        return True
        
    except Exception as e:
        print(f"✗ Ultralytics integration test failed: {e}")
        return False

def test_inference_class():
    """Test PCBDefectDetector class structure."""
    print("\nTesting PCBDefectDetector class...")
    
    try:
        from ai.inference import PCBDefectDetector, create_test_image
        
        # Test class structure (without loading model)
        print("✓ PCBDefectDetector class imported")
        
        # Test test image creation
        test_image = create_test_image()
        
        if test_image is not None:
            print(f"✓ Test image created: {test_image.shape}")
            
            # Test image properties
            if len(test_image.shape) == 3 and test_image.shape[2] == 3:
                print("✓ Test image has correct format (RGB)")
            else:
                print("⚠ Test image format may be incorrect")
        
        # Test different image sizes
        test_image_small = create_test_image(size=(512, 512))
        test_image_large = create_test_image(size=(800, 600))
        
        print("✓ Test images with different sizes created")
        
        # Test with and without defects
        test_image_defects = create_test_image(add_defects=True)
        test_image_clean = create_test_image(add_defects=False)
        
        print("✓ Test images with/without defects created")
        
        return True
        
    except Exception as e:
        print(f"✗ PCBDefectDetector test failed: {e}")
        return False

def test_integration_readiness():
    """Test readiness for integration with other modules."""
    print("\nTesting integration readiness...")
    
    try:
        # Test interface compatibility
        from core.interfaces import InspectionResult
        
        # Test that we can create InspectionResult
        result = InspectionResult(
            defects=["Missing Hole", "Open Circuit"],
            locations=[{"bbox": [100, 100, 200, 200]}, {"bbox": [300, 300, 400, 400]}],
            confidence_scores=[0.85, 0.75],
            processing_time=0.05
        )
        
        print("✓ InspectionResult can be created")
        print(f"    Has defects: {result.has_defects}")
        print(f"    Defect count: {len(result.defects)}")
        
        # Test preprocessing integration
        try:
            from processing.preprocessor import ImagePreprocessor
            
            preprocessor = ImagePreprocessor()
            test_image = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
            
            # Test preprocessing
            processed = preprocessor.process(test_image)
            
            if processed is not None:
                print("✓ Preprocessing integration ready")
            else:
                print("⚠ Preprocessing returned None")
            
        except ImportError:
            print("⚠ Preprocessing module not available")
        
        # Test postprocessing integration
        try:
            from processing.postprocessor import ResultPostprocessor
            
            postprocessor = ResultPostprocessor()
            print("✓ Postprocessing integration ready")
            
        except ImportError:
            print("⚠ Postprocessing module not available")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration readiness test failed: {e}")
        return False

def run_comprehensive_validation():
    """Run comprehensive validation."""
    print("=== AI Layer Validation ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("ModelManager Test", test_model_manager),
        ("PyTorch Integration", test_torch_integration),
        ("Ultralytics Integration", test_ultralytics_integration),
        ("Inference Class Test", test_inference_class),
        ("Integration Readiness", test_integration_readiness),
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
    print("=== Validation Summary ===")
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Check critical components
    critical_tests = ["Import Test", "Configuration Test", "PyTorch Integration"]
    critical_passed = all(success for name, success in results if name in critical_tests)
    
    if critical_passed:
        print("✓ Critical components are working")
    else:
        print("✗ Critical components have issues")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
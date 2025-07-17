"""
AI inference module for PCB defect detection.

This module implements the YOLOv11 integration for automated PCB defect detection
using the pre-trained model on PCB defect dataset.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import os
from pathlib import Path
import cv2

from ultralytics import YOLO
from core.interfaces import BaseDetector, InspectionResult
from core.config import AI_CONFIG, MODEL_CLASS_MAPPING, DEFECT_CLASSES


class PCBDefectDetector(BaseDetector):
    """
    PCB defect detection using YOLOv11 model.
    
    This class handles model loading, inference, and result processing
    for PCB defect detection with GPU acceleration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PCB defect detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or AI_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Model state
        self.model = None
        self.device = None
        self.is_loaded = False
        self.model_info = {}
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        self.gpu_available = torch.cuda.is_available()
        
        # Initialize model
        self._initialize_model()
        
        self.logger.info(f"PCBDefectDetector initialized on device: {self.device}")
    
    def _initialize_model(self):
        """Initialize and load the YOLOv11 model."""
        try:
            # Check if model file exists
            model_path = self.config["model_path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Set device
            self.device = self._get_device()
            
            # Load model
            self.logger.info(f"Loading YOLOv11 model from {model_path}")
            self.model = YOLO(model_path)
            
            # Move model to device
            self.model.to(self.device)
            
            # Get model info
            self.model_info = self._get_model_info()
            
            # Warmup model if enabled
            if self.config.get("warmup", True):
                self._warmup_model()
            
            self.is_loaded = True
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            self.is_loaded = False
            raise
    
    def _get_device(self) -> str:
        """
        Get the appropriate device for inference.
        
        Returns:
            Device string (cuda:0, cpu, etc.)
        """
        requested_device = self.config.get("device", "cuda:0")
        
        if requested_device.startswith("cuda") and self.gpu_available:
            # Check if specific GPU is available
            if ":" in requested_device:
                gpu_id = int(requested_device.split(":")[1])
                if gpu_id < torch.cuda.device_count():
                    return requested_device
                else:
                    self.logger.warning(f"GPU {gpu_id} not available, using cuda:0")
                    return "cuda:0"
            else:
                return "cuda:0"
        elif requested_device.startswith("cuda") and not self.gpu_available:
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        else:
            return "cpu"
    
    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": self.config["model_path"],
            "device": self.device,
            "input_size": self.config["imgsz"],
            "num_classes": len(DEFECT_CLASSES),
            "class_names": DEFECT_CLASSES,
            "model_type": "YOLOv11",
            "framework": "Ultralytics"
        }
        
        # Add GPU info if available
        if self.gpu_available and self.device.startswith("cuda"):
            gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
            info["gpu_name"] = torch.cuda.get_device_name(gpu_id)
            info["gpu_memory"] = torch.cuda.get_device_properties(gpu_id).total_memory
        
        return info
    
    def _warmup_model(self):
        """Warmup model for consistent performance."""
        try:
            self.logger.info("Warming up model...")
            
            # Create dummy input
            dummy_input = np.random.randint(
                0, 255, 
                (self.config["imgsz"], self.config["imgsz"], 3), 
                dtype=np.uint8
            )
            
            # Run a few warmup inferences
            for i in range(3):
                _ = self.model(dummy_input, 
                             device=self.device,
                             verbose=False,
                             conf=self.config["confidence"])
            
            self.logger.info("Model warmup completed")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {str(e)}")
    
    def detect(self, image: np.ndarray) -> InspectionResult:
        """
        Detect defects in PCB image.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            InspectionResult with detected defects
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        if image is None:
            return InspectionResult([], [], [], 0.0)
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run inference
            results = self.model(processed_image,
                               device=self.device,
                               imgsz=self.config["imgsz"],
                               conf=self.config["confidence"],
                               max_det=self.config["max_det"],
                               agnostic_nms=self.config["agnostic_nms"],
                               augment=self.config["augment"],
                               half=self.config["half"],
                               verbose=False)
            
            # Process results
            defects, locations, confidence_scores = self._process_results(results[0])
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_statistics(processing_time)
            
            # Create inspection result
            inspection_result = InspectionResult(
                defects=defects,
                locations=locations,
                confidence_scores=confidence_scores,
                processing_time=processing_time
            )
            
            self.logger.debug(f"Detected {len(defects)} defects in {processing_time:.3f}s")
            
            return inspection_result
            
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            processing_time = time.time() - start_time
            return InspectionResult([], [], [], processing_time)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure RGB (not BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume input is RGB, YOLO expects RGB
            pass
        
        return image
    
    def _process_results(self, results: Any) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Process YOLO detection results.
        
        Args:
            results: YOLO detection results
            
        Returns:
            Tuple of (defects, locations, confidence_scores)
        """
        defects = []
        locations = []
        confidence_scores = []
        
        if results.boxes is None:
            return defects, locations, confidence_scores
        
        for box in results.boxes:
            # Extract data
            xyxy = box.xyxy[0].cpu().numpy()  # Bounding box
            conf = float(box.conf[0].cpu().numpy())  # Confidence
            cls = int(box.cls[0].cpu().numpy())  # Class ID
            
            # Map class ID to defect name
            defect_name = MODEL_CLASS_MAPPING.get(cls, f"Unknown_{cls}")
            
            # Create location dict
            location = {
                "bbox": [float(x) for x in xyxy],  # [x1, y1, x2, y2]
                "confidence": conf,
                "class_id": cls,
                "class_name": defect_name
            }
            
            # Add to results
            defects.append(defect_name)
            locations.append(location)
            confidence_scores.append(conf)
        
        return defects, locations, confidence_scores
    
    def _update_statistics(self, inference_time: float):
        """Update performance statistics."""
        self.inference_times.append(inference_time)
        self.total_inferences += 1
        
        # Keep only recent times (last 100)
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
    
    def detect_batch(self, images: List[np.ndarray]) -> List[InspectionResult]:
        """
        Detect defects in batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of InspectionResult objects
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        results = []
        
        try:
            # Preprocess all images
            processed_images = [self._preprocess_image(img) for img in images]
            
            # Run batch inference
            start_time = time.time()
            
            batch_results = self.model(processed_images,
                                     device=self.device,
                                     imgsz=self.config["imgsz"],
                                     conf=self.config["confidence"],
                                     max_det=self.config["max_det"],
                                     agnostic_nms=self.config["agnostic_nms"],
                                     augment=self.config["augment"],
                                     half=self.config["half"],
                                     verbose=False)
            
            total_time = time.time() - start_time
            avg_time = total_time / len(images)
            
            # Process each result
            for i, result in enumerate(batch_results):
                defects, locations, confidence_scores = self._process_results(result)
                
                inspection_result = InspectionResult(
                    defects=defects,
                    locations=locations,
                    confidence_scores=confidence_scores,
                    processing_time=avg_time
                )
                
                results.append(inspection_result)
                
                # Update statistics
                self._update_statistics(avg_time)
            
            self.logger.info(f"Batch inference completed: {len(images)} images in {total_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Batch inference failed: {str(e)}")
            # Return empty results for all images
            results = [InspectionResult([], [], [], 0.0) for _ in images]
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.model_info.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {
                "total_inferences": 0,
                "avg_inference_time": 0.0,
                "min_inference_time": 0.0,
                "max_inference_time": 0.0,
                "fps": 0.0
            }
        
        stats = {
            "total_inferences": self.total_inferences,
            "avg_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "fps": 1.0 / np.mean(self.inference_times),
            "gpu_available": self.gpu_available,
            "device": self.device
        }
        
        # Add GPU memory info if available
        if self.gpu_available and self.device.startswith("cuda"):
            try:
                gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
                stats["gpu_memory_allocated"] = torch.cuda.memory_allocated(gpu_id)
                stats["gpu_memory_reserved"] = torch.cuda.memory_reserved(gpu_id)
                stats["gpu_memory_cached"] = torch.cuda.memory_cached(gpu_id)
            except Exception:
                pass
        
        return stats
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        if self.gpu_available and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            self.logger.info("GPU cache cleared")
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set confidence threshold for detections.
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.config["confidence"] = threshold
            self.logger.info(f"Confidence threshold set to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def reload_model(self):
        """Reload the model (useful for model updates)."""
        self.logger.info("Reloading model...")
        self.is_loaded = False
        self._initialize_model()
    
    def __del__(self):
        """Cleanup on object destruction."""
        if hasattr(self, 'gpu_available') and self.gpu_available:
            self.clear_gpu_cache()


class ModelManager:
    """
    Model management utilities for PCB defect detection.
    
    Handles model validation, metadata, and performance monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or AI_CONFIG
        self.logger = logging.getLogger(__name__)
    
    def validate_model(self) -> Dict[str, Any]:
        """
        Validate model file and configuration.
        
        Returns:
            Validation results dictionary
        """
        results = {
            "model_exists": False,
            "model_size": 0,
            "config_valid": False,
            "gpu_available": torch.cuda.is_available(),
            "errors": []
        }
        
        try:
            # Check model file
            model_path = self.config["model_path"]
            if os.path.exists(model_path):
                results["model_exists"] = True
                results["model_size"] = os.path.getsize(model_path)
            else:
                results["errors"].append(f"Model file not found: {model_path}")
            
            # Validate configuration
            if not 0 < self.config["confidence"] < 1:
                results["errors"].append("Invalid confidence threshold")
            
            if self.config["imgsz"] <= 0:
                results["errors"].append("Invalid image size")
            
            if self.config["max_det"] <= 0:
                results["errors"].append("Invalid max detections")
            
            results["config_valid"] = len(results["errors"]) == 0
            
        except Exception as e:
            results["errors"].append(f"Validation error: {str(e)}")
        
        return results
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Model metadata dictionary
        """
        metadata = {
            "model_path": self.config["model_path"],
            "framework": "YOLOv11 (Ultralytics)",
            "task": "PCB Defect Detection",
            "classes": DEFECT_CLASSES,
            "class_mapping": MODEL_CLASS_MAPPING,
            "input_size": self.config["imgsz"],
            "confidence_threshold": self.config["confidence"]
        }
        
        # Add file info if model exists
        if os.path.exists(self.config["model_path"]):
            stat = os.stat(self.config["model_path"])
            metadata["file_size"] = stat.st_size
            metadata["modified_time"] = stat.st_mtime
        
        return metadata
    
    def benchmark_model(self, detector: PCBDefectDetector, 
                       test_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Benchmark model performance.
        
        Args:
            detector: PCBDefectDetector instance
            test_images: List of test images
            
        Returns:
            Benchmark results
        """
        if not detector.is_loaded:
            raise RuntimeError("Model not loaded")
        
        results = {
            "num_test_images": len(test_images),
            "inference_times": [],
            "total_detections": 0,
            "avg_inference_time": 0.0,
            "fps": 0.0
        }
        
        try:
            # Run inference on all test images
            start_time = time.time()
            
            for image in test_images:
                inference_start = time.time()
                result = detector.detect(image)
                inference_time = time.time() - inference_start
                
                results["inference_times"].append(inference_time)
                results["total_detections"] += len(result.defects)
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            results["avg_inference_time"] = np.mean(results["inference_times"])
            results["fps"] = len(test_images) / total_time
            results["total_time"] = total_time
            results["avg_detections_per_image"] = results["total_detections"] / len(test_images)
            
            self.logger.info(f"Benchmark completed: {results['fps']:.2f} FPS average")
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {str(e)}")
            results["error"] = str(e)
        
        return results


# Utility functions for testing and validation
def create_test_image(size: Tuple[int, int] = (640, 640), 
                     add_defects: bool = True) -> np.ndarray:
    """
    Create a test PCB image for testing.
    
    Args:
        size: Image size (width, height)
        add_defects: Whether to add simulated defects
        
    Returns:
        Test image array
    """
    height, width = size
    
    # Create base PCB texture
    image = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)
    
    # Add PCB features
    # Traces
    for i in range(10):
        start = (np.random.randint(0, width), np.random.randint(0, height))
        end = (np.random.randint(0, width), np.random.randint(0, height))
        cv2.line(image, start, end, (80, 80, 80), 2)
    
    # Pads
    for i in range(20):
        center = (np.random.randint(50, width-50), np.random.randint(50, height-50))
        cv2.circle(image, center, 15, (60, 60, 60), -1)
    
    # Add simulated defects if requested
    if add_defects:
        # Missing hole (white circle)
        cv2.circle(image, (200, 200), 10, (255, 255, 255), -1)
        
        # Short circuit (dark line connecting traces)
        cv2.line(image, (300, 300), (350, 350), (0, 0, 0), 3)
        
        # Open circuit (break in trace)
        cv2.rectangle(image, (400, 400), (420, 410), (150, 150, 150), -1)
    
    return image


def test_detector():
    """Test the PCB defect detector."""
    try:
        # Create detector
        detector = PCBDefectDetector()
        
        # Create test image
        test_image = create_test_image()
        
        # Run detection
        result = detector.detect(test_image)
        
        # Print results
        print(f"Detected {len(result.defects)} defects:")
        for i, (defect, conf) in enumerate(zip(result.defects, result.confidence_scores)):
            print(f"  {i+1}. {defect} (confidence: {conf:.2f})")
        
        print(f"Processing time: {result.processing_time:.3f}s")
        
        # Get performance stats
        stats = detector.get_performance_stats()
        print(f"Performance: {stats['fps']:.2f} FPS")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the detector
    success = test_detector()
    print(f"Test {'PASSED' if success else 'FAILED'}")
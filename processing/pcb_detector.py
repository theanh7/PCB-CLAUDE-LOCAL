"""
PCB detection and auto-trigger module for PCB inspection system.

This module handles real-time PCB detection, position tracking, stability
checking, and auto-trigger logic for the inspection system.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
import time
from dataclasses import dataclass
from core.interfaces import PCBDetectionResult
from core.config import TRIGGER_CONFIG, PROCESSING_CONFIG
from processing.preprocessor import FocusEvaluator


@dataclass
class PCBPosition:
    """Data class for PCB position information."""
    x: int
    y: int
    width: int
    height: int
    area: int
    timestamp: float
    
    def center(self) -> Tuple[int, int]:
        """Get center point of PCB."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def distance_to(self, other: 'PCBPosition') -> float:
        """Calculate distance to another PCB position."""
        if other is None:
            return float('inf')
        
        center1 = self.center()
        center2 = other.center()
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def size_difference(self, other: 'PCBPosition') -> float:
        """Calculate size difference with another PCB position."""
        if other is None:
            return float('inf')
        
        width_diff = abs(self.width - other.width)
        height_diff = abs(self.height - other.height)
        
        return max(width_diff, height_diff)


class PCBDetector:
    """
    PCB detection and tracking system with auto-trigger capability.
    
    Detects PCB presence, tracks position stability, and triggers
    automatic inspection when conditions are met.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize PCB detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or TRIGGER_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters
        self.min_area_ratio = self.config["min_pcb_area"]
        self.stability_threshold = self.config["stability_frames"]
        self.movement_threshold = self.config["movement_threshold"]
        self.focus_threshold = self.config["focus_threshold"]
        self.inspection_interval = self.config["inspection_interval"]
        
        # State tracking
        self.last_position: Optional[PCBPosition] = None
        self.position_history: List[PCBPosition] = []
        self.stable_frames = 0
        self.last_inspection_time = 0.0
        
        # Focus evaluator
        self.focus_evaluator = FocusEvaluator()
        
        # Edge detection parameters
        self.edge_threshold1 = 50
        self.edge_threshold2 = 150
        self.blur_kernel = (5, 5)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        self.logger.info(f"PCBDetector initialized with {self.stability_threshold} stability frames")
    
    def detect_pcb(self, image: np.ndarray) -> PCBDetectionResult:
        """
        Detect PCB in image and check stability.
        
        Args:
            image: Input grayscale image
            
        Returns:
            PCBDetectionResult with detection and stability information
        """
        if image is None:
            return PCBDetectionResult(False, None, False, 0.0)
        
        try:
            # Find PCB in image
            pcb_position = self._find_pcb_position(image)
            
            if pcb_position is None:
                # No PCB found, reset stability counter
                self.stable_frames = 0
                self.last_position = None
                return PCBDetectionResult(False, None, False, 0.0)
            
            # Check stability
            is_stable = self._check_stability(pcb_position)
            
            # Evaluate focus if stable
            focus_score = 0.0
            if is_stable:
                # Extract PCB region for focus evaluation
                pcb_region = image[pcb_position.y:pcb_position.y + pcb_position.height,
                                 pcb_position.x:pcb_position.x + pcb_position.width]
                focus_score = self.focus_evaluator.evaluate(pcb_region)
            
            # Update position history
            self._update_position_history(pcb_position)
            
            # Create result
            position_tuple = (pcb_position.x, pcb_position.y, 
                            pcb_position.width, pcb_position.height)
            
            return PCBDetectionResult(
                has_pcb=True,
                position=position_tuple,
                is_stable=is_stable,
                focus_score=focus_score
            )
            
        except Exception as e:
            self.logger.error(f"Error in PCB detection: {str(e)}")
            return PCBDetectionResult(False, None, False, 0.0)
    
    def _find_pcb_position(self, image: np.ndarray) -> Optional[PCBPosition]:
        """
        Find PCB position in image using edge detection.
        
        Args:
            image: Input grayscale image
            
        Returns:
            PCBPosition if found, None otherwise
        """
        # Image preprocessing for edge detection
        blurred = cv2.GaussianBlur(image, self.blur_kernel, 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.edge_threshold1, self.edge_threshold2)
        
        # Morphological operations to clean up edges
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.morph_kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest rectangular contour
        best_contour = None
        best_area = 0
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if contour is rectangular (4 vertices)
            if len(approx) >= 4:
                area = cv2.contourArea(contour)
                if area > best_area:
                    best_area = area
                    best_contour = contour
        
        if best_contour is None:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Check if area is sufficient
        total_area = image.shape[0] * image.shape[1]
        area_ratio = (w * h) / total_area
        
        if area_ratio < self.min_area_ratio:
            return None
        
        # Check aspect ratio (PCBs are typically rectangular)
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 10:  # Too elongated
            return None
        
        # Create PCB position
        pcb_position = PCBPosition(
            x=x, y=y, width=w, height=h,
            area=int(w * h),
            timestamp=time.time()
        )
        
        return pcb_position
    
    def _check_stability(self, current_position: PCBPosition) -> bool:
        """
        Check if PCB position is stable.
        
        Args:
            current_position: Current PCB position
            
        Returns:
            True if position is stable
        """
        if self.last_position is None:
            self.last_position = current_position
            self.stable_frames = 1
            return False
        
        # Calculate position and size differences
        position_diff = current_position.distance_to(self.last_position)
        size_diff = current_position.size_difference(self.last_position)
        
        # Check if within tolerance
        is_stable_position = position_diff <= self.movement_threshold
        is_stable_size = size_diff <= self.movement_threshold * 2  # More tolerance for size
        
        if is_stable_position and is_stable_size:
            self.stable_frames += 1
        else:
            self.stable_frames = 0
        
        # Update last position
        self.last_position = current_position
        
        # Return true if stable for required frames
        return self.stable_frames >= self.stability_threshold
    
    def _update_position_history(self, position: PCBPosition):
        """
        Update position history for trend analysis.
        
        Args:
            position: Current PCB position
        """
        self.position_history.append(position)
        
        # Keep only recent history (last 100 positions)
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]
    
    def can_trigger_inspection(self) -> bool:
        """
        Check if inspection can be triggered based on timing constraints.
        
        Returns:
            True if inspection can be triggered
        """
        current_time = time.time()
        time_since_last = current_time - self.last_inspection_time
        
        return time_since_last >= self.inspection_interval
    
    def should_trigger_inspection(self, detection_result: PCBDetectionResult) -> bool:
        """
        Determine if inspection should be triggered.
        
        Args:
            detection_result: PCB detection result
            
        Returns:
            True if inspection should be triggered
        """
        # Check all conditions for auto-trigger
        conditions = [
            detection_result.has_pcb,
            detection_result.is_stable,
            detection_result.focus_score >= self.focus_threshold,
            self.can_trigger_inspection()
        ]
        
        return all(conditions)
    
    def trigger_inspection(self) -> None:
        """Mark that inspection has been triggered."""
        self.last_inspection_time = time.time()
        self.logger.info("Inspection triggered")
    
    def get_stability_info(self) -> dict:
        """
        Get detailed stability information.
        
        Returns:
            Dictionary with stability metrics
        """
        return {
            "stable_frames": self.stable_frames,
            "required_frames": self.stability_threshold,
            "stability_progress": min(self.stable_frames / self.stability_threshold, 1.0),
            "last_position": self.last_position.__dict__ if self.last_position else None,
            "position_history_count": len(self.position_history),
            "time_since_last_inspection": time.time() - self.last_inspection_time
        }
    
    def get_detection_stats(self) -> dict:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with detection statistics
        """
        if not self.position_history:
            return {
                "total_detections": 0,
                "average_area": 0,
                "position_variance": 0,
                "detection_rate": 0
            }
        
        areas = [pos.area for pos in self.position_history]
        centers = [pos.center() for pos in self.position_history]
        
        # Calculate position variance
        if len(centers) > 1:
            centers_array = np.array(centers)
            position_variance = np.var(centers_array, axis=0).sum()
        else:
            position_variance = 0
        
        return {
            "total_detections": len(self.position_history),
            "average_area": np.mean(areas),
            "position_variance": position_variance,
            "detection_rate": len(self.position_history) / max(1, time.time() - self.position_history[0].timestamp)
        }
    
    def reset_detection_state(self):
        """Reset detection state (useful for testing or recalibration)."""
        self.last_position = None
        self.position_history.clear()
        self.stable_frames = 0
        self.last_inspection_time = 0.0
        self.logger.info("Detection state reset")
    
    def visualize_detection(self, image: np.ndarray, 
                          detection_result: PCBDetectionResult) -> np.ndarray:
        """
        Visualize detection results on image.
        
        Args:
            image: Input image
            detection_result: Detection result to visualize
            
        Returns:
            Image with detection visualization
        """
        if image is None:
            return None
        
        # Convert to color for visualization
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        if not detection_result.has_pcb or detection_result.position is None:
            return vis_image
        
        x, y, w, h = detection_result.position
        
        # Choose color based on stability
        if detection_result.is_stable:
            color = (0, 255, 0)  # Green for stable
            thickness = 3
        else:
            color = (255, 165, 0)  # Orange for unstable
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw center point
        center = (x + w // 2, y + h // 2)
        cv2.circle(vis_image, center, 5, color, -1)
        
        # Add text information
        info_text = f"PCB: {'Stable' if detection_result.is_stable else 'Unstable'}"
        cv2.putText(vis_image, info_text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        focus_text = f"Focus: {detection_result.focus_score:.1f}"
        cv2.putText(vis_image, focus_text, (x, y + h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add stability progress bar
        if not detection_result.is_stable:
            progress = min(self.stable_frames / self.stability_threshold, 1.0)
            bar_width = 200
            bar_height = 10
            bar_x = x
            bar_y = y + h + 40
            
            # Background
            cv2.rectangle(vis_image, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (128, 128, 128), -1)
            
            # Progress
            progress_width = int(bar_width * progress)
            cv2.rectangle(vis_image, (bar_x, bar_y), 
                         (bar_x + progress_width, bar_y + bar_height), 
                         (0, 255, 255), -1)
            
            # Text
            progress_text = f"Stability: {self.stable_frames}/{self.stability_threshold}"
            cv2.putText(vis_image, progress_text, (bar_x, bar_y + bar_height + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image


class AutoTriggerSystem:
    """
    Auto-trigger system that combines PCB detection with inspection triggering.
    
    Manages the complete auto-trigger pipeline including timing, conditions,
    and trigger events.
    """
    
    def __init__(self, pcb_detector: PCBDetector, config: Optional[dict] = None):
        """
        Initialize auto-trigger system.
        
        Args:
            pcb_detector: PCB detector instance
            config: Optional configuration dictionary
        """
        self.pcb_detector = pcb_detector
        self.config = config or TRIGGER_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Trigger statistics
        self.trigger_count = 0
        self.total_detections = 0
        self.successful_triggers = 0
        self.start_time = time.time()
        
        # Rate limiting
        self.max_inspection_rate = self.config["max_inspection_rate"]  # per hour
        self.inspection_times = []
        
        self.logger.info("AutoTriggerSystem initialized")
    
    def process_frame(self, image: np.ndarray) -> Tuple[PCBDetectionResult, bool]:
        """
        Process a single frame for auto-trigger.
        
        Args:
            image: Input image frame
            
        Returns:
            Tuple of (detection_result, should_trigger)
        """
        # Detect PCB
        detection_result = self.pcb_detector.detect_pcb(image)
        self.total_detections += 1
        
        # Check if should trigger
        should_trigger = False
        
        if detection_result.has_pcb:
            # Check trigger conditions
            if self.pcb_detector.should_trigger_inspection(detection_result):
                # Check rate limiting
                if self._check_rate_limit():
                    should_trigger = True
                    self.pcb_detector.trigger_inspection()
                    self._record_trigger()
        
        return detection_result, should_trigger
    
    def _check_rate_limit(self) -> bool:
        """
        Check if rate limit allows triggering.
        
        Returns:
            True if rate limit allows triggering
        """
        current_time = time.time()
        
        # Remove old trigger times (older than 1 hour)
        hour_ago = current_time - 3600
        self.inspection_times = [t for t in self.inspection_times if t > hour_ago]
        
        # Check if under rate limit
        return len(self.inspection_times) < self.max_inspection_rate
    
    def _record_trigger(self):
        """Record a trigger event."""
        self.trigger_count += 1
        self.successful_triggers += 1
        self.inspection_times.append(time.time())
        
        self.logger.info(f"Trigger #{self.trigger_count} recorded")
    
    def get_trigger_stats(self) -> dict:
        """
        Get trigger statistics.
        
        Returns:
            Dictionary with trigger statistics
        """
        runtime = time.time() - self.start_time
        
        stats = {
            "total_detections": self.total_detections,
            "trigger_count": self.trigger_count,
            "successful_triggers": self.successful_triggers,
            "runtime_hours": runtime / 3600,
            "detection_rate": self.total_detections / max(runtime, 1),
            "trigger_rate": self.trigger_count / max(runtime, 1),
            "trigger_efficiency": self.successful_triggers / max(self.trigger_count, 1),
            "current_hour_triggers": len(self.inspection_times)
        }
        
        return stats


# Utility functions for testing and validation
def create_test_pcb_image(size: Tuple[int, int] = (1024, 768), 
                         pcb_size: Tuple[int, int] = (400, 300)) -> np.ndarray:
    """
    Create a test image with a PCB-like rectangular object.
    
    Args:
        size: Image size (width, height)
        pcb_size: PCB size (width, height)
        
    Returns:
        Test image with PCB-like object
    """
    width, height = size
    pcb_width, pcb_height = pcb_size
    
    # Create background
    image = np.random.randint(50, 100, (height, width), dtype=np.uint8)
    
    # Add PCB rectangle
    pcb_x = (width - pcb_width) // 2
    pcb_y = (height - pcb_height) // 2
    
    # PCB body
    image[pcb_y:pcb_y + pcb_height, pcb_x:pcb_x + pcb_width] = 150
    
    # Add some features (holes, traces)
    for i in range(10):
        hole_x = np.random.randint(pcb_x + 20, pcb_x + pcb_width - 20)
        hole_y = np.random.randint(pcb_y + 20, pcb_y + pcb_height - 20)
        cv2.circle(image, (hole_x, hole_y), 5, 0, -1)
    
    return image


def test_pcb_detector():
    """Test PCB detector functionality."""
    # Create detector
    detector = PCBDetector()
    
    # Create test image
    test_image = create_test_pcb_image()
    
    # Test detection
    result = detector.detect_pcb(test_image)
    
    print(f"PCB detected: {result.has_pcb}")
    print(f"Position: {result.position}")
    print(f"Stable: {result.is_stable}")
    print(f"Focus score: {result.focus_score:.2f}")
    
    # Test stability tracking
    for i in range(15):
        # Simulate stable frames
        result = detector.detect_pcb(test_image)
        print(f"Frame {i}: Stable frames: {detector.stable_frames}, "
              f"Is stable: {result.is_stable}")
    
    # Get statistics
    stats = detector.get_detection_stats()
    print(f"Detection stats: {stats}")


if __name__ == "__main__":
    # Test the detector
    test_pcb_detector()
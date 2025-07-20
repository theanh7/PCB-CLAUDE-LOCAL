"""
PCB detection v1.1 with position smoothing and stability improvements.

This version addresses the main issue in v1.0: PCB positions vary too much
between frames, preventing stability detection. Solution: position averaging
and smoothing to handle camera noise and detection variations.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Deque
import logging
import time
from dataclasses import dataclass
from collections import deque
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
    confidence: float = 1.0  # New: detection confidence
    
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
    
    def average_with(self, other: 'PCBPosition', weight: float = 0.5) -> 'PCBPosition':
        """Create averaged position with another position."""
        return PCBPosition(
            x=int(self.x * (1-weight) + other.x * weight),
            y=int(self.y * (1-weight) + other.y * weight),
            width=int(self.width * (1-weight) + other.width * weight),
            height=int(self.height * (1-weight) + other.height * weight),
            area=int(self.area * (1-weight) + other.area * weight),
            timestamp=time.time(),
            confidence=max(self.confidence, other.confidence)
        )


class PCBDetectorV11:
    """
    Enhanced PCB detector v1.1 with position smoothing and stability improvements.
    
    Key improvements over v1.0:
    1. Position smoothing using moving average
    2. Confidence-based detection filtering
    3. Adaptive thresholds based on detection history
    4. Better noise handling and vibration compensation
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize enhanced PCB detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or TRIGGER_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters (more lenient than v1.0)
        self.min_area_ratio = self.config["min_pcb_area"]
        self.stability_threshold = self.config["stability_frames"] 
        self.movement_threshold = self.config["movement_threshold"]
        self.focus_threshold = self.config["focus_threshold"]
        self.inspection_interval = self.config["inspection_interval"]
        
        # NEW: Position smoothing parameters
        self.position_history_size = 5  # Keep last 5 positions for averaging
        self.position_history: Deque[PCBPosition] = deque(maxlen=self.position_history_size)
        self.smoothed_position: Optional[PCBPosition] = None
        
        # Enhanced state tracking
        self.last_position: Optional[PCBPosition] = None
        self.stable_frames = 0
        self.last_inspection_time = 0.0
        self.detection_confidence_history: Deque[float] = deque(maxlen=10)
        
        # Focus evaluator
        self.focus_evaluator = FocusEvaluator()
        
        # Adaptive detection parameters
        self.edge_threshold1 = 30  # More sensitive than v1.0
        self.edge_threshold2 = 100
        self.blur_kernel = (3, 3)  # Smaller kernel for better edge preservation
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Detection performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.stability_count = 0
        
        self.logger.info(f"PCBDetectorV11 initialized with position smoothing (history={self.position_history_size})")
    
    def detect_pcb(self, image: np.ndarray) -> PCBDetectionResult:
        """
        Enhanced PCB detection with position smoothing.
        
        Args:
            image: Input grayscale image
            
        Returns:
            PCBDetectionResult with improved stability detection
        """
        if image is None:
            return PCBDetectionResult(False, None, False, 0.0)
        
        self.frame_count += 1
        
        try:
            # Find PCB position using enhanced detection
            raw_position = self._find_pcb_position_enhanced(image)
            
            if raw_position is None:
                # No PCB found, reset stability but keep position history for smoothing
                self.stable_frames = 0
                return PCBDetectionResult(False, None, False, 0.0)
            
            self.detection_count += 1
            
            # Add to position history for smoothing
            self.position_history.append(raw_position)
            
            # Calculate smoothed position
            smoothed_position = self._calculate_smoothed_position()
            
            # Check stability using smoothed position
            is_stable = self._check_stability_enhanced(smoothed_position)
            
            if is_stable:
                self.stability_count += 1
            
            # Evaluate focus on smoothed region
            focus_score = self._evaluate_focus_enhanced(image, smoothed_position)
            
            # Create result using smoothed position
            position_tuple = (smoothed_position.x, smoothed_position.y, 
                            smoothed_position.width, smoothed_position.height)
            
            # Debug logging every 30 frames
            if self.frame_count % 30 == 0:
                detection_rate = self.detection_count / self.frame_count * 100
                stability_rate = self.stability_count / max(self.detection_count, 1) * 100
                self.logger.debug(f"V1.1 Stats - Detection: {detection_rate:.1f}%, "
                                f"Stability: {stability_rate:.1f}%, "
                                f"Focus: {focus_score:.1f}")
            
            return PCBDetectionResult(
                has_pcb=True,
                position=position_tuple,
                is_stable=is_stable,
                focus_score=focus_score
            )
            
        except Exception as e:
            self.logger.error(f"Error in enhanced PCB detection: {str(e)}")
            return PCBDetectionResult(False, None, False, 0.0)
    
    def _find_pcb_position_enhanced(self, image: np.ndarray) -> Optional[PCBPosition]:
        """
        Enhanced PCB position detection with multiple methods and confidence scoring.
        """
        # Try multiple detection methods with confidence scoring
        candidates = []
        
        # Method 1: Enhanced edge detection
        edge_result = self._detect_by_edges_enhanced(image)
        if edge_result:
            candidates.append((edge_result, 0.8))  # High confidence for edge detection
        
        # Method 2: Improved dark object detection  
        dark_result = self._detect_dark_object_enhanced(image)
        if dark_result:
            candidates.append((dark_result, 0.9))  # Higher confidence for dark detection
        
        # Method 3: Contour-based detection
        contour_result = self._detect_by_contours_enhanced(image)
        if contour_result:
            candidates.append((contour_result, 0.7))  # Medium confidence
        
        if not candidates:
            return None
        
        # Select best candidate based on confidence and consistency with history
        best_candidate = self._select_best_candidate(candidates)
        return best_candidate
    
    def _detect_by_edges_enhanced(self, image: np.ndarray) -> Optional[PCBPosition]:
        """Enhanced edge detection with better preprocessing."""
        
        # Enhanced preprocessing
        # 1. Bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. Adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Multiple edge detection thresholds
        edge_candidates = []
        
        for t1, t2 in [(20, 60), (30, 90), (40, 120)]:
            edges = cv2.Canny(enhanced, t1, t2)
            
            # Light morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                position = self._evaluate_contours_enhanced(contours, image.shape)
                if position:
                    edge_candidates.append(position)
        
        # Return most consistent candidate
        return self._select_most_consistent_position(edge_candidates)
    
    def _detect_dark_object_enhanced(self, image: np.ndarray) -> Optional[PCBPosition]:
        """Enhanced dark object detection with better noise handling."""
        
        # Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Invert for dark object detection
        inverted = 255 - enhanced
        
        # Adaptive thresholding instead of fixed thresholds
        candidates = []
        
        # Method 1: Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 21, 10)
        
        # Method 2: Multiple fixed thresholds (more conservative)
        for threshold in [100, 120, 140]:
            _, binary = cv2.threshold(inverted, threshold, 255, cv2.THRESH_BINARY)
            
            # Combine with adaptive result
            combined = cv2.bitwise_and(binary, adaptive_thresh)
            
            # Light cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                position = self._evaluate_contours_enhanced(contours, image.shape)
                if position:
                    candidates.append(position)
        
        return self._select_most_consistent_position(candidates)
    
    def _detect_by_contours_enhanced(self, image: np.ndarray) -> Optional[PCBPosition]:
        """Additional contour-based detection method."""
        
        # Gaussian blur for smooth contours
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Multiple threshold levels
        candidates = []
        
        for thresh_val in [60, 80, 100, 120, 140]:
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                position = self._evaluate_contours_enhanced(contours, image.shape)
                if position:
                    candidates.append(position)
        
        return self._select_most_consistent_position(candidates)
    
    def _evaluate_contours_enhanced(self, contours: List, image_shape: Tuple) -> Optional[PCBPosition]:
        """Enhanced contour evaluation with better scoring."""
        
        if not contours:
            return None
        
        total_area = image_shape[0] * image_shape[1]
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if touches borders (noise)
            if (x <= 2 or y <= 2 or 
                x + w >= image_shape[1] - 2 or 
                y + h >= image_shape[0] - 2):
                continue
            
            # Calculate enhanced metrics
            area_ratio = (w * h) / total_area
            aspect_ratio = max(w, h) / min(w, h)
            fill_ratio = area / max(w * h, 1)
            perimeter = cv2.arcLength(contour, True)
            compactness = 4 * np.pi * area / max(perimeter * perimeter, 1)
            
            # Enhanced scoring system
            score = 0
            
            # Area score (more flexible range)
            if 0.03 <= area_ratio <= 0.7:  # 3% to 70% of image
                score += 3.0
            elif 0.01 <= area_ratio <= 0.9:
                score += 1.5
            
            # Aspect ratio score (prefer rectangles)
            if 1.1 <= aspect_ratio <= 4.0:
                score += 2.0
            elif aspect_ratio <= 6.0:
                score += 1.0
            
            # Fill ratio score
            if fill_ratio >= 0.6:
                score += 1.5
            elif fill_ratio >= 0.4:
                score += 1.0
            
            # Size score (prefer reasonable absolute sizes)
            if w >= 80 and h >= 80:
                score += 1.0
            
            # Compactness score (prefer more rectangular shapes)
            if compactness >= 0.3:
                score += 1.0
            
            # History consistency bonus
            if self.position_history and len(self.position_history) > 0:
                avg_x = sum(pos.x for pos in self.position_history) / len(self.position_history)
                avg_y = sum(pos.y for pos in self.position_history) / len(self.position_history)
                distance_to_history = np.sqrt((x - avg_x)**2 + (y - avg_y)**2)
                
                if distance_to_history <= 50:  # Close to historical average
                    score += 2.0
                elif distance_to_history <= 100:
                    score += 1.0
            
            if score > best_score and score >= 3.0:  # Lower threshold than v1.0
                best_score = score
                best_contour = contour
        
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            return PCBPosition(
                x=x, y=y, width=w, height=h,
                area=int(w * h),
                timestamp=time.time(),
                confidence=min(best_score / 8.0, 1.0)  # Normalize confidence
            )
        
        return None
    
    def _select_best_candidate(self, candidates: List[Tuple[PCBPosition, float]]) -> Optional[PCBPosition]:
        """Select the best detection candidate based on confidence and consistency."""
        
        if not candidates:
            return None
        
        # If only one candidate, return it
        if len(candidates) == 1:
            pos, conf = candidates[0]
            pos.confidence = conf
            return pos
        
        # Multiple candidates - select based on combined scoring
        best_candidate = None
        best_score = 0
        
        for position, confidence in candidates:
            score = confidence
            
            # Bonus for consistency with position history
            if self.position_history and len(self.position_history) > 0:
                avg_position = self._calculate_average_position(list(self.position_history))
                if avg_position:
                    consistency = 1.0 / (1.0 + position.distance_to(avg_position) / 100.0)
                    score += consistency * 0.5
            
            if score > best_score:
                best_score = score
                best_candidate = position
        
        if best_candidate:
            best_candidate.confidence = best_score
        
        return best_candidate
    
    def _select_most_consistent_position(self, candidates: List[PCBPosition]) -> Optional[PCBPosition]:
        """Select the most consistent position from candidates."""
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # If we have position history, select most consistent
        if self.position_history and len(self.position_history) > 0:
            avg_position = self._calculate_average_position(list(self.position_history))
            if avg_position:
                best_candidate = None
                min_distance = float('inf')
                
                for candidate in candidates:
                    distance = candidate.distance_to(avg_position)
                    if distance < min_distance:
                        min_distance = distance
                        best_candidate = candidate
                
                return best_candidate
        
        # No history - select largest area (most likely to be PCB)
        return max(candidates, key=lambda pos: pos.area)
    
    def _calculate_smoothed_position(self) -> PCBPosition:
        """Calculate smoothed position using position history."""
        
        if not self.position_history:
            return None
        
        if len(self.position_history) == 1:
            return self.position_history[0]
        
        # Weighted average with more recent positions having higher weight
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        weighted_w = 0
        weighted_h = 0
        max_confidence = 0
        
        for i, position in enumerate(self.position_history):
            # More recent positions get higher weight
            weight = (i + 1) * position.confidence
            total_weight += weight
            
            weighted_x += position.x * weight
            weighted_y += position.y * weight
            weighted_w += position.width * weight
            weighted_h += position.height * weight
            max_confidence = max(max_confidence, position.confidence)
        
        if total_weight == 0:
            return self.position_history[-1]
        
        smoothed = PCBPosition(
            x=int(weighted_x / total_weight),
            y=int(weighted_y / total_weight),
            width=int(weighted_w / total_weight),
            height=int(weighted_h / total_weight),
            area=int((weighted_w / total_weight) * (weighted_h / total_weight)),
            timestamp=time.time(),
            confidence=max_confidence
        )
        
        self.smoothed_position = smoothed
        return smoothed
    
    def _calculate_average_position(self, positions: List[PCBPosition]) -> Optional[PCBPosition]:
        """Calculate simple average of positions."""
        
        if not positions:
            return None
        
        avg_x = sum(pos.x for pos in positions) / len(positions)
        avg_y = sum(pos.y for pos in positions) / len(positions)
        avg_w = sum(pos.width for pos in positions) / len(positions)
        avg_h = sum(pos.height for pos in positions) / len(positions)
        
        return PCBPosition(
            x=int(avg_x), y=int(avg_y),
            width=int(avg_w), height=int(avg_h),
            area=int(avg_w * avg_h),
            timestamp=time.time()
        )
    
    def _check_stability_enhanced(self, current_position: PCBPosition) -> bool:
        """Enhanced stability check using smoothed positions."""
        
        if self.last_position is None:
            self.last_position = current_position
            self.stable_frames = 1
            return False
        
        # Calculate differences using smoothed positions
        position_diff = current_position.distance_to(self.last_position)
        size_diff = current_position.size_difference(self.last_position)
        
        # More lenient thresholds for smoothed positions
        is_stable_position = position_diff <= self.movement_threshold
        is_stable_size = size_diff <= self.movement_threshold * 1.5  # More tolerance for size
        
        if is_stable_position and is_stable_size:
            self.stable_frames += 1
        else:
            self.stable_frames = 0
        
        # Update last position
        self.last_position = current_position
        
        # Return true if stable for required frames
        return self.stable_frames >= self.stability_threshold
    
    def _evaluate_focus_enhanced(self, image: np.ndarray, position: PCBPosition) -> float:
        """Enhanced focus evaluation with error handling."""
        
        try:
            # Extract PCB region with padding
            padding = 5
            x1 = max(0, position.x - padding)
            y1 = max(0, position.y - padding)
            x2 = min(image.shape[1], position.x + position.width + padding)
            y2 = min(image.shape[0], position.y + position.height + padding)
            
            pcb_region = image[y1:y2, x1:x2]
            
            if pcb_region.size == 0:
                return 0.0
            
            return self.focus_evaluator.evaluate(pcb_region)
            
        except Exception as e:
            self.logger.debug(f"Focus evaluation failed: {e}")
            return 0.0
    
    def get_debug_info(self) -> dict:
        """Get debug information about detection performance."""
        
        detection_rate = self.detection_count / max(self.frame_count, 1) * 100
        stability_rate = self.stability_count / max(self.detection_count, 1) * 100
        
        return {
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'detection_rate': detection_rate,
            'stability_count': self.stability_count,
            'stability_rate': stability_rate,
            'stable_frames': self.stable_frames,
            'position_history_size': len(self.position_history),
            'smoothed_position': self.smoothed_position
        }
    
    # Compatibility methods for v1.0 interface
    def debayer_to_gray(self, raw_bayer):
        """Compatibility method for v1.0 interface."""
        if len(raw_bayer.shape) == 2:
            return raw_bayer  # Already grayscale
        
        # Simple debayering - extract green channel
        return raw_bayer[1::2, 0::2] if raw_bayer.shape[0] > 1 and raw_bayer.shape[1] > 1 else raw_bayer
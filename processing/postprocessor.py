"""
Result postprocessing module for PCB inspection system.

This module handles visualization of AI detection results, including
bounding box drawing, confidence scores, and result overlays.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from core.config import DEFECT_CLASSES, DEFECT_COLORS
from core.interfaces import InspectionResult


@dataclass
class DetectionBox:
    """Data class for detection bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height


class ResultPostprocessor:
    """
    Postprocessor for AI detection results visualization.
    
    Handles drawing bounding boxes, labels, confidence scores, and
    creating annotated result images for display.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize result postprocessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Visualization settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.box_thickness = 2
        self.text_thickness = 1
        
        # Colors
        self.defect_colors = DEFECT_COLORS
        self.default_color = (0, 255, 0)  # Green
        self.text_bg_color = (0, 0, 0)    # Black
        self.text_color = (255, 255, 255)  # White
        
        # Confidence thresholds for different visualizations
        self.high_confidence = 0.8
        self.medium_confidence = 0.5
        
        self.logger.info("ResultPostprocessor initialized")
    
    def process_yolo_results(self, detection_results: Any) -> List[DetectionBox]:
        """
        Process YOLO detection results into DetectionBox objects.
        
        Args:
            detection_results: YOLO detection results
            
        Returns:
            List of DetectionBox objects
        """
        boxes = []
        
        # Handle both InspectionResult and YOLO raw results
        if hasattr(detection_results, 'defects'):
            # InspectionResult format
            for i, defect in enumerate(detection_results.defects):
                location = detection_results.locations[i]
                confidence = detection_results.confidence_scores[i]
                
                # Extract bbox
                bbox = location['bbox']  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Get class info
                class_id = location.get('class_id', 0)
                class_name = location.get('class_name', defect)
                
                detection_box = DetectionBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                
                boxes.append(detection_box)
                
            return boxes
        
        # YOLO raw results format
        if not hasattr(detection_results, 'boxes') or detection_results.boxes is None:
            return boxes
        
        for box in detection_results.boxes:
            # Extract coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Get class name
            class_name = DEFECT_CLASSES[class_id] if class_id < len(DEFECT_CLASSES) else "Unknown"
            
            # Create detection box
            detection_box = DetectionBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name
            )
            
            boxes.append(detection_box)
        
        return boxes
    
    def draw_results(self, image: np.ndarray, detection_results: Any, 
                    show_confidence: bool = True, show_labels: bool = True) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image
            detection_results: YOLO detection results
            show_confidence: Whether to show confidence scores
            show_labels: Whether to show class labels
            
        Returns:
            Annotated image
        """
        if image is None:
            return None
        
        # Convert to color if grayscale
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()
        
        # Process detection results
        boxes = self.process_yolo_results(detection_results)
        
        # Draw each detection
        for box in boxes:
            self._draw_detection_box(annotated, box, show_confidence, show_labels)
        
        # Add summary information
        self._draw_summary(annotated, boxes)
        
        return annotated
    
    def _draw_detection_box(self, image: np.ndarray, box: DetectionBox, 
                           show_confidence: bool, show_labels: bool):
        """
        Draw a single detection box on image.
        
        Args:
            image: Image to draw on
            box: Detection box to draw
            show_confidence: Whether to show confidence
            show_labels: Whether to show labels
        """
        # Get color for this defect type
        color = self.defect_colors.get(box.class_name, self.default_color)
        
        # Adjust thickness based on confidence
        thickness = self.box_thickness
        if box.confidence >= self.high_confidence:
            thickness = 3
        elif box.confidence < self.medium_confidence:
            thickness = 1
        
        # Draw bounding box
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
        
        # Prepare label text
        label_parts = []
        if show_labels:
            label_parts.append(box.class_name)
        if show_confidence:
            label_parts.append(f"{box.confidence:.2f}")
        
        if label_parts:
            label = ": ".join(label_parts)
            self._draw_label(image, box, label, color)
    
    def _draw_label(self, image: np.ndarray, box: DetectionBox, 
                   label: str, color: Tuple[int, int, int]):
        """
        Draw label for detection box.
        
        Args:
            image: Image to draw on
            box: Detection box
            label: Label text
            color: Box color
        """
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.text_thickness
        )
        
        # Calculate label position
        label_x = box.x1
        label_y = box.y1 - 5
        
        # Ensure label stays within image bounds
        if label_y - text_height - 5 < 0:
            label_y = box.y1 + text_height + 5
        
        # Draw label background
        bg_x1 = label_x
        bg_y1 = label_y - text_height - 5
        bg_x2 = label_x + text_width + 5
        bg_y2 = label_y + 5
        
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (label_x + 2, label_y - 2),
                   self.font, self.font_scale, self.text_color, self.text_thickness)
    
    def _draw_summary(self, image: np.ndarray, boxes: List[DetectionBox]):
        """
        Draw summary information on image.
        
        Args:
            image: Image to draw on
            boxes: List of detection boxes
        """
        if not boxes:
            # No defects found
            summary = "✓ NO DEFECTS FOUND"
            color = (0, 255, 0)  # Green
        else:
            # Defects found
            summary = f"⚠ {len(boxes)} DEFECTS FOUND"
            color = (0, 0, 255)  # Red
        
        # Draw summary at top of image
        text_size = cv2.getTextSize(summary, self.font, self.font_scale * 1.2, 2)[0]
        
        # Background
        cv2.rectangle(image, (10, 10), (text_size[0] + 20, 40), (0, 0, 0), -1)
        
        # Text
        cv2.putText(image, summary, (15, 30), self.font, self.font_scale * 1.2, 
                   color, 2)
    
    def create_result_overlay(self, image: np.ndarray, boxes: List[DetectionBox], 
                             transparency: float = 0.3) -> np.ndarray:
        """
        Create semi-transparent overlay for defect regions.
        
        Args:
            image: Base image
            boxes: Detection boxes
            transparency: Overlay transparency (0-1)
            
        Returns:
            Image with overlay
        """
        if image is None or not boxes:
            return image
        
        # Create overlay
        overlay = image.copy()
        
        for box in boxes:
            color = self.defect_colors.get(box.class_name, self.default_color)
            
            # Fill detection region
            cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), color, -1)
        
        # Blend with original image
        result = cv2.addWeighted(image, 1 - transparency, overlay, transparency, 0)
        
        return result
    
    def create_defect_summary_image(self, boxes: List[DetectionBox], 
                                   image_size: Tuple[int, int] = (400, 300)) -> np.ndarray:
        """
        Create summary image showing defect statistics.
        
        Args:
            boxes: Detection boxes
            image_size: Summary image size (width, height)
            
        Returns:
            Summary image
        """
        width, height = image_size
        summary_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not boxes:
            # No defects
            text = "NO DEFECTS FOUND"
            color = (0, 255, 0)
            
            text_size = cv2.getTextSize(text, self.font, self.font_scale, 2)[0]
            x = (width - text_size[0]) // 2
            y = height // 2
            
            cv2.putText(summary_img, text, (x, y), self.font, self.font_scale, 
                       color, 2)
        else:
            # Count defects by type
            defect_counts = {}
            for box in boxes:
                defect_counts[box.class_name] = defect_counts.get(box.class_name, 0) + 1
            
            # Draw defect counts
            y_offset = 30
            for defect_type, count in defect_counts.items():
                color = self.defect_colors.get(defect_type, self.default_color)
                text = f"{defect_type}: {count}"
                
                cv2.putText(summary_img, text, (10, y_offset), self.font, 
                           self.font_scale, color, 2)
                y_offset += 25
        
        return summary_img
    
    def create_confidence_histogram(self, boxes: List[DetectionBox], 
                                  image_size: Tuple[int, int] = (400, 200)) -> np.ndarray:
        """
        Create confidence score histogram.
        
        Args:
            boxes: Detection boxes
            image_size: Histogram image size (width, height)
            
        Returns:
            Histogram image
        """
        width, height = image_size
        hist_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not boxes:
            return hist_img
        
        # Extract confidence scores
        confidences = [box.confidence for box in boxes]
        
        # Create histogram bins
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        hist, _ = np.histogram(confidences, bins=bins)
        
        # Normalize histogram
        max_count = max(hist) if max(hist) > 0 else 1
        
        # Draw histogram bars
        bar_width = width // len(hist)
        
        for i, count in enumerate(hist):
            if count > 0:
                bar_height = int((count / max_count) * (height - 40))
                x1 = i * bar_width
                x2 = (i + 1) * bar_width
                y1 = height - 20
                y2 = y1 - bar_height
                
                cv2.rectangle(hist_img, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.rectangle(hist_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                
                # Add count label
                if count > 0:
                    cv2.putText(hist_img, str(count), (x1 + 5, y2 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return hist_img
    
    def create_inspection_report(self, image: np.ndarray, 
                               detection_results: Any,
                               metadata: Dict[str, Any]) -> np.ndarray:
        """
        Create comprehensive inspection report image.
        
        Args:
            image: Original image
            detection_results: Detection results
            metadata: Additional metadata
            
        Returns:
            Report image
        """
        boxes = self.process_yolo_results(detection_results)
        
        # Create annotated image
        annotated = self.draw_results(image, detection_results)
        
        # Create summary panels
        summary_img = self.create_defect_summary_image(boxes)
        confidence_hist = self.create_confidence_histogram(boxes)
        
        # Combine into report layout
        report = self._create_report_layout(
            annotated, summary_img, confidence_hist, metadata
        )
        
        return report
    
    def _create_report_layout(self, main_image: np.ndarray, 
                            summary_image: np.ndarray,
                            histogram: np.ndarray,
                            metadata: Dict[str, Any]) -> np.ndarray:
        """
        Create report layout combining multiple images.
        
        Args:
            main_image: Main annotated image
            summary_image: Defect summary image
            histogram: Confidence histogram
            metadata: Report metadata
            
        Returns:
            Combined report image
        """
        # Calculate layout dimensions
        main_h, main_w = main_image.shape[:2]
        panel_w = 400
        panel_h = 300
        
        # Create report canvas
        report_w = main_w + panel_w
        report_h = max(main_h, panel_h * 2)
        report = np.zeros((report_h, report_w, 3), dtype=np.uint8)
        
        # Place main image
        report[:main_h, :main_w] = main_image
        
        # Place summary image
        summary_h, summary_w = summary_image.shape[:2]
        report[:summary_h, main_w:main_w + summary_w] = summary_image
        
        # Place histogram
        hist_h, hist_w = histogram.shape[:2]
        y_offset = summary_h + 20
        if y_offset + hist_h <= report_h:
            report[y_offset:y_offset + hist_h, main_w:main_w + hist_w] = histogram
        
        # Add metadata text
        self._add_metadata_text(report, metadata, main_w, y_offset + hist_h + 20)
        
        return report
    
    def _add_metadata_text(self, image: np.ndarray, metadata: Dict[str, Any], 
                          x_offset: int, y_offset: int):
        """
        Add metadata text to image.
        
        Args:
            image: Image to add text to
            metadata: Metadata dictionary
            x_offset: X position for text
            y_offset: Y position for text
        """
        y_pos = y_offset
        
        for key, value in metadata.items():
            if y_pos >= image.shape[0] - 20:
                break
            
            text = f"{key}: {value}"
            cv2.putText(image, text, (x_offset + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
    
    def save_annotated_image(self, image: np.ndarray, 
                           detection_results: Any,
                           filepath: str) -> bool:
        """
        Save annotated image to file.
        
        Args:
            image: Original image
            detection_results: Detection results
            filepath: Output file path
            
        Returns:
            True if saved successfully
        """
        try:
            annotated = self.draw_results(image, detection_results)
            success = cv2.imwrite(filepath, annotated)
            
            if success:
                self.logger.info(f"Annotated image saved to {filepath}")
            else:
                self.logger.error(f"Failed to save image to {filepath}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving annotated image: {str(e)}")
            return False
    
    def create_thumbnail(self, image: np.ndarray, 
                        size: Tuple[int, int] = (200, 150)) -> np.ndarray:
        """
        Create thumbnail of image.
        
        Args:
            image: Input image
            size: Thumbnail size (width, height)
            
        Returns:
            Thumbnail image
        """
        if image is None:
            return None
        
        # Calculate aspect ratio preserving resize
        h, w = image.shape[:2]
        target_w, target_h = size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center thumbnail
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        if len(thumbnail.shape) == 2:
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2BGR)
        
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = thumbnail
        
        return canvas


# Utility functions
def create_test_results():
    """Create test detection results for testing."""
    class MockBox:
        def __init__(self, coords, confidence, class_id):
            self.xyxy = [coords]
            self.conf = [confidence]
            self.cls = [class_id]
    
    class MockResults:
        def __init__(self):
            self.boxes = [
                MockBox([100, 100, 200, 150], 0.85, 0),  # Missing Hole
                MockBox([300, 200, 400, 280], 0.75, 1),  # Mouse Bite
                MockBox([150, 300, 250, 350], 0.60, 2),  # Open Circuit
            ]
    
    return MockResults()


def test_postprocessor():
    """Test postprocessor functionality."""
    # Create test image
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    
    # Create postprocessor
    postprocessor = ResultPostprocessor()
    
    # Create test results
    test_results = create_test_results()
    
    # Test visualization
    annotated = postprocessor.draw_results(test_image, test_results)
    
    # Test other functions
    boxes = postprocessor.process_yolo_results(test_results)
    summary = postprocessor.create_defect_summary_image(boxes)
    histogram = postprocessor.create_confidence_histogram(boxes)
    
    print(f"Processed {len(boxes)} detections")
    print(f"Annotated image shape: {annotated.shape}")
    print(f"Summary image shape: {summary.shape}")
    print(f"Histogram shape: {histogram.shape}")
    
    # Test report creation
    metadata = {
        "Timestamp": "2024-01-01 12:00:00",
        "Focus Score": "150.5",
        "Processing Time": "0.05s"
    }
    
    report = postprocessor.create_inspection_report(test_image, test_results, metadata)
    print(f"Report image shape: {report.shape}")


if __name__ == "__main__":
    test_postprocessor()
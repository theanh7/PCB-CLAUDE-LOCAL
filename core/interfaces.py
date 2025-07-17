"""
Core interfaces for PCB inspection system.

This module defines the abstract base classes that all components must implement
to ensure consistent interfaces across the system layers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseProcessor(ABC):
    """Abstract base class for image processors."""
    
    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process input data and return processed result.
        
        Args:
            data: Input data (typically image array)
            
        Returns:
            Processed data array
        """
        pass


class BaseDetector(ABC):
    """Abstract base class for detection systems."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Any:
        """
        Detect objects/defects in the input image.
        
        Args:
            image: Input image array
            
        Returns:
            Detection results (format depends on implementation)
        """
        pass


class BaseAnalyzer(ABC):
    """Abstract base class for analytics systems."""
    
    @abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze input data and return statistics.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        pass


class BaseCamera(ABC):
    """Abstract base class for camera controllers."""
    
    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a single image.
        
        Returns:
            Captured image array or None if capture failed
        """
        pass
    
    @abstractmethod
    def start_streaming(self) -> None:
        """Start continuous image streaming."""
        pass
    
    @abstractmethod
    def stop_streaming(self) -> None:
        """Stop continuous image streaming."""
        pass


class BaseDatabase(ABC):
    """Abstract base class for database operations."""
    
    @abstractmethod
    def save_inspection(self, timestamp: str, defects: List[str], 
                       locations: List[Dict], **kwargs) -> int:
        """
        Save inspection results to database.
        
        Args:
            timestamp: Inspection timestamp
            defects: List of detected defects
            locations: List of defect locations
            **kwargs: Additional metadata
            
        Returns:
            Inspection ID
        """
        pass
    
    @abstractmethod
    def get_recent_inspections(self, limit: int = 50) -> List[Dict]:
        """
        Get recent inspection records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of inspection records
        """
        pass


class BaseGUI(ABC):
    """Abstract base class for GUI components."""
    
    @abstractmethod
    def update_display(self, data: Any) -> None:
        """
        Update GUI display with new data.
        
        Args:
            data: Data to display
        """
        pass
    
    @abstractmethod
    def show_error(self, message: str) -> None:
        """
        Show error message to user.
        
        Args:
            message: Error message to display
        """
        pass


class PCBDetectionResult:
    """Data class for PCB detection results."""
    
    def __init__(self, has_pcb: bool, position: Optional[Tuple[int, int, int, int]] = None,
                 is_stable: bool = False, focus_score: float = 0.0):
        self.has_pcb = has_pcb
        self.position = position  # (x, y, width, height)
        self.is_stable = is_stable
        self.focus_score = focus_score


class InspectionResult:
    """Data class for inspection results."""
    
    def __init__(self, defects: List[str], locations: List[Dict], 
                 confidence_scores: List[float], processing_time: float = 0.0):
        self.defects = defects
        self.locations = locations
        self.confidence_scores = confidence_scores
        self.processing_time = processing_time
        self.has_defects = len(defects) > 0
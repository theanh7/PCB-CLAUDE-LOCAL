# PCB Auto-Inspection System - API Documentation

## Version 1.0 | December 2024

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Core Layer APIs](#core-layer-apis)
3. [Hardware Layer APIs](#hardware-layer-apis)
4. [Processing Layer APIs](#processing-layer-apis)
5. [AI Layer APIs](#ai-layer-apis)
6. [Data Layer APIs](#data-layer-apis)
7. [Analytics Layer APIs](#analytics-layer-apis)
8. [Presentation Layer APIs](#presentation-layer-apis)
9. [Integration Patterns](#integration-patterns)
10. [Error Handling](#error-handling)
11. [Performance Considerations](#performance-considerations)
12. [Configuration Reference](#configuration-reference)

---

## System Architecture Overview

### Layered Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐   │
│  │   Main GUI      │ │ Analytics GUI   │ │ History GUI   │   │
│  │  (gui.py)       │ │ (analytics_     │ │ (history_     │   │
│  │                 │ │  viewer.py)     │ │  browser.py)  │   │
│  └─────────────────┘ └─────────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     DATA & ANALYTICS LAYER                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐   │
│  │   Database      │ │    Analytics    │ │   Reporting   │   │
│  │ (database.py)   │ │  (analyzer.py)  │ │  (reports)    │   │
│  │                 │ │                 │ │               │   │
│  └─────────────────┘ └─────────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                        AI LAYER                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐   │
│  │  Defect Model   │ │   Inference     │ │ Class Mapping │   │
│  │  (YOLOv11)      │ │ (inference.py)  │ │  (config.py)  │   │
│  │                 │ │                 │ │               │   │
│  └─────────────────┘ └─────────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐   │
│  │  Preprocessor   │ │  PCB Detector   │ │ Postprocessor │   │
│  │(preprocessor.py)│ │(pcb_detector.py)│ │(postprocessor │   │
│  │                 │ │                 │ │     .py)      │   │
│  └─────────────────┘ └─────────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     HARDWARE LAYER                          │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐   │
│  │ Camera Control  │ │ Camera Presets  │ │ Image Handler │   │
│  │(camera_         │ │(camera_presets  │ │ (async frame  │   │
│  │ controller.py)  │ │      .py)       │ │   grabbing)   │   │
│  └─────────────────┘ └─────────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      CORE LAYER                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐   │
│  │  Interfaces     │ │  Configuration  │ │   Utilities   │   │
│  │(interfaces.py)  │ │  (config.py)    │ │  (utils.py)   │   │
│  │                 │ │                 │ │               │   │
│  └─────────────────┘ └─────────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Camera    │────│ Raw Image   │────│ Processing  │
│  Hardware   │    │   Buffer    │    │   Pipeline  │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ GUI Display │◄───│ AI Results  │◄───│ AI Inference│
│  & Controls │    │   & Stats   │    │   Engine    │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
                                      ┌─────────────┐
                                      │  Database   │
                                      │   Storage   │
                                      └─────────────┘
```

---

## Core Layer APIs

### Configuration API

**Module:** `core/config.py`

#### Camera Configuration

```python
CAMERA_CONFIG = {
    "model": str,              # Camera model identifier
    "preview_exposure": int,   # Preview mode exposure (μs)
    "capture_exposure": int,   # High-quality capture exposure (μs)
    "gain": int,              # Camera gain setting
    "pixel_format": str,      # Pixel format (e.g., "BayerRG8")
    "binning": int,           # Pixel binning factor
    "trigger_mode": str       # Trigger mode setting
}
```

#### AI Configuration

```python
AI_CONFIG = {
    "model_path": str,        # Path to YOLOv11 model file
    "confidence": float,      # Detection confidence threshold (0.0-1.0)
    "device": str            # Inference device ("cuda:0", "cpu")
}
```

#### Auto-Trigger Configuration

```python
TRIGGER_CONFIG = {
    "stability_frames": int,     # Frames for stability check
    "focus_threshold": float,    # Minimum focus score
    "movement_threshold": int,   # Pixel tolerance for stability
    "min_pcb_area": float,      # Minimum PCB area ratio (0.0-1.0)
    "inspection_interval": float # Minimum seconds between inspections
}
```

### Interfaces API

**Module:** `core/interfaces.py`

#### BaseCamera Interface

```python
class BaseCamera(ABC):
    """Abstract base class for camera implementations."""
    
    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """Capture a single high-quality image.
        
        Returns:
            np.ndarray: Captured image or None if failed
        """
        pass
    
    @abstractmethod
    def start_streaming(self) -> bool:
        """Start continuous image streaming.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_streaming(self) -> bool:
        """Stop image streaming.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get latest preview frame.
        
        Returns:
            np.ndarray: Latest frame or None if not available
        """
        pass
```

#### BaseProcessor Interface

```python
class BaseProcessor(ABC):
    """Abstract base class for image processors."""
    
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process an input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            np.ndarray: Processed image
        """
        pass
```

#### BaseDetector Interface

```python
class BaseDetector(ABC):
    """Abstract base class for detection algorithms."""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Any:
        """Perform detection on input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Detection results (format varies by implementation)
        """
        pass
```

### Utilities API

**Module:** `core/utils.py`

#### Logging Functions

```python
def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None) -> logging.Logger:
    """Set up application logging.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional log file path
        
    Returns:
        logging.Logger: Configured logger instance
    """
    pass

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    pass
```

#### Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor system performance metrics."""
    
    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""
        pass
    
    def end_timing(self, operation: str) -> float:
        """End timing and return duration in seconds."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        pass
```

---

## Hardware Layer APIs

### Camera Controller API

**Module:** `hardware/camera_controller.py`

#### BaslerCamera Class

```python
class BaslerCamera(BaseCamera):
    """Basler camera implementation using pypylon."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize camera with configuration.
        
        Args:
            config: Camera configuration dictionary
        """
        pass
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture high-quality image for inspection.
        
        Returns:
            np.ndarray: Raw image data (Bayer pattern) or None
            
        Raises:
            CameraError: If capture fails
        """
        pass
    
    def capture_high_quality(self) -> Optional[np.ndarray]:
        """Capture with optimized settings for inspection.
        
        Automatically adjusts exposure and stops streaming if active.
        
        Returns:
            np.ndarray: High-quality raw image or None
        """
        pass
    
    def start_streaming(self) -> bool:
        """Start continuous preview streaming.
        
        Uses preview exposure settings for real-time operation.
        
        Returns:
            bool: True if streaming started successfully
        """
        pass
    
    def stop_streaming(self) -> bool:
        """Stop preview streaming.
        
        Returns:
            bool: True if streaming stopped successfully
        """
        pass
    
    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get latest preview frame from queue.
        
        Non-blocking operation that returns immediately.
        
        Returns:
            np.ndarray: Latest frame or None if queue empty
        """
        pass
    
    def is_connected(self) -> bool:
        """Check if camera is connected and responsive.
        
        Returns:
            bool: True if camera is connected
        """
        pass
    
    def close(self) -> None:
        """Close camera and cleanup resources."""
        pass
```

#### CameraImageHandler Class

```python
class CameraImageHandler(pylon.ImageEventHandler):
    """Async image event handler for continuous streaming."""
    
    def __init__(self, frame_queue: Queue):
        """Initialize handler with frame queue.
        
        Args:
            frame_queue: Thread-safe queue for frames
        """
        pass
    
    def OnImageGrabbed(self, camera, grab_result) -> None:
        """Handle new image from camera.
        
        Called automatically by pypylon when new image is available.
        
        Args:
            camera: Camera instance
            grab_result: Grab result containing image data
        """
        pass
```

### Camera Presets API

**Module:** `hardware/camera_presets.py`

#### CameraPresets Class

```python
class CameraPresets:
    """Predefined camera configurations for different scenarios."""
    
    @staticmethod
    def get_fast_preview() -> Dict[str, Any]:
        """Get configuration for fast preview (low quality, high FPS).
        
        Returns:
            Dict: Configuration dictionary
        """
        pass
    
    @staticmethod
    def get_balanced() -> Dict[str, Any]:
        """Get balanced configuration (medium quality and speed).
        
        Returns:
            Dict: Configuration dictionary
        """
        pass
    
    @staticmethod
    def get_high_quality() -> Dict[str, Any]:
        """Get high-quality configuration (best quality, slower).
        
        Returns:
            Dict: Configuration dictionary
        """
        pass
    
    @staticmethod
    def get_lighting_preset(lighting_condition: str) -> Dict[str, Any]:
        """Get preset optimized for lighting conditions.
        
        Args:
            lighting_condition: "bright", "normal", "dim", or "low"
            
        Returns:
            Dict: Optimized configuration
        """
        pass
    
    @staticmethod
    def optimize_for_speed(base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for maximum speed.
        
        Args:
            base_config: Base configuration to modify
            
        Returns:
            Dict: Speed-optimized configuration
        """
        pass
```

---

## Processing Layer APIs

### Image Preprocessor API

**Module:** `processing/preprocessor.py`

#### ImagePreprocessor Class

```python
class ImagePreprocessor(BaseProcessor):
    """Image preprocessing for PCB inspection."""
    
    def __init__(self):
        """Initialize preprocessor with default settings."""
        pass
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process input image for AI inference.
        
        Args:
            image: Input image (raw Bayer or grayscale)
            
        Returns:
            np.ndarray: Processed grayscale image
        """
        pass
    
    def process_raw(self, raw_bayer_data: np.ndarray) -> np.ndarray:
        """Process raw Bayer pattern data to enhanced grayscale.
        
        Args:
            raw_bayer_data: Raw Bayer pattern from camera
            
        Returns:
            np.ndarray: Enhanced grayscale image
        """
        pass
    
    def debayer_to_gray(self, bayer_image: np.ndarray) -> np.ndarray:
        """Convert Bayer pattern to grayscale efficiently.
        
        Args:
            bayer_image: Raw Bayer pattern image
            
        Returns:
            np.ndarray: Grayscale image
        """
        pass
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization.
        
        Args:
            image: Input grayscale image
            
        Returns:
            np.ndarray: Contrast-enhanced image
        """
        pass
```

### PCB Detector API

**Module:** `processing/pcb_detector.py`

#### PCBDetector Class

```python
class PCBDetector:
    """Real-time PCB detection and tracking for auto-trigger."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize detector with trigger configuration.
        
        Args:
            config: TRIGGER_CONFIG dictionary
        """
        pass
    
    def detect_pcb(self, raw_image: np.ndarray) -> Tuple[bool, Optional[Tuple], bool, float]:
        """Detect PCB presence and evaluate trigger conditions.
        
        Args:
            raw_image: Raw image from camera
            
        Returns:
            Tuple containing:
            - has_pcb (bool): Whether PCB is detected
            - pcb_region (Tuple): PCB bounding box (x, y, w, h) or None
            - is_stable (bool): Whether PCB position is stable
            - focus_score (float): Image focus quality score
        """
        pass
    
    def check_stability(self, current_position: Tuple[int, int, int, int]) -> bool:
        """Check if PCB position is stable over time.
        
        Args:
            current_position: Current PCB bounding box (x, y, w, h)
            
        Returns:
            bool: True if position is stable
        """
        pass
    
    def reset_stability(self) -> None:
        """Reset stability tracking."""
        pass
```

#### FocusEvaluator Class

```python
class FocusEvaluator:
    """Evaluate image focus quality using various metrics."""
    
    def evaluate(self, image: np.ndarray) -> float:
        """Calculate focus score using Laplacian variance.
        
        Args:
            image: Input grayscale image
            
        Returns:
            float: Focus score (higher = better focus)
        """
        pass
    
    def is_acceptable(self, score: float, threshold: float = 100.0) -> bool:
        """Check if focus score meets threshold.
        
        Args:
            score: Focus score to evaluate
            threshold: Minimum acceptable score
            
        Returns:
            bool: True if focus is acceptable
        """
        pass
```

### Result Postprocessor API

**Module:** `processing/postprocessor.py`

#### ResultPostprocessor Class

```python
class ResultPostprocessor:
    """Process AI detection results for visualization."""
    
    def draw_results(self, image: np.ndarray, detection_results: Any) -> np.ndarray:
        """Draw detection results on image.
        
        Args:
            image: Input image
            detection_results: YOLOv11 detection results
            
        Returns:
            np.ndarray: Image with drawn bounding boxes and labels
        """
        pass
    
    def extract_defects(self, detection_results: Any) -> Tuple[List[str], List[Dict], List[float]]:
        """Extract defect information from detection results.
        
        Args:
            detection_results: YOLOv11 results object
            
        Returns:
            Tuple containing:
            - defects (List[str]): List of defect class names
            - locations (List[Dict]): List of bounding box dictionaries
            - confidences (List[float]): List of confidence scores
        """
        pass
```

---

## AI Layer APIs

### Defect Detection API

**Module:** `ai/inference.py`

#### PCBDefectDetector Class

```python
class PCBDefectDetector(BaseDetector):
    """YOLOv11-based PCB defect detection."""
    
    def __init__(self, model_path: str, device: str = "cuda:0", 
                 confidence_threshold: float = 0.5, half_precision: bool = True):
        """Initialize detector with model and settings.
        
        Args:
            model_path: Path to YOLOv11 model file (.pt)
            device: Inference device ("cuda:0", "cpu")
            confidence_threshold: Minimum detection confidence (0.0-1.0)
            half_precision: Use FP16 inference for speed
        """
        pass
    
    def detect(self, image: np.ndarray) -> Any:
        """Perform defect detection on image.
        
        Args:
            image: Preprocessed grayscale or RGB image
            
        Returns:
            YOLOv11 Results object containing detections
            
        Raises:
            InferenceError: If detection fails
        """
        pass
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Any]:
        """Perform batch detection on multiple images.
        
        Args:
            images: List of preprocessed images
            
        Returns:
            List of YOLOv11 Results objects
        """
        pass
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics.
        
        Returns:
            Dict containing model metadata
        """
        pass
    
    def warm_up(self, image_size: Tuple[int, int] = (640, 640)) -> None:
        """Warm up model for consistent inference timing.
        
        Args:
            image_size: Size for warmup image (width, height)
        """
        pass
```

#### Model Class Mapping

```python
MODEL_CLASS_MAPPING = {
    0: "Mouse Bite",        # mouse_bite
    1: "Spur",             # spur  
    2: "Missing Hole",     # missing_hole
    3: "Short Circuit",    # short
    4: "Open Circuit",     # open_circuit
    5: "Spurious Copper"   # spurious_copper
}
```

---

## Data Layer APIs

### Database API

**Module:** `data/database.py`

#### PCBDatabase Class

```python
class PCBDatabase:
    """SQLite database for PCB inspection data."""
    
    def __init__(self, db_path: str):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        pass
    
    def save_inspection_metadata(self, timestamp: datetime, defects: List[str],
                               locations: List[Dict], confidence_scores: List[float],
                               raw_image_shape: Tuple[int, int], focus_score: float,
                               processing_time: Optional[float] = None,
                               save_image: Optional[np.ndarray] = None) -> int:
        """Save inspection metadata to database.
        
        Args:
            timestamp: Inspection timestamp
            defects: List of detected defect class names
            locations: List of defect location dictionaries
            confidence_scores: List of confidence scores
            raw_image_shape: Shape of original image (height, width)
            focus_score: Image focus quality score
            processing_time: Time taken for processing (seconds)
            save_image: Optional processed image to save (only if defects found)
            
        Returns:
            int: Inspection ID
        """
        pass
    
    def get_recent_inspections(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent inspection records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of inspection dictionaries
        """
        pass
    
    def get_defect_statistics(self) -> List[Dict[str, Any]]:
        """Get defect type statistics.
        
        Returns:
            List of dictionaries with defect type and count
        """
        pass
    
    def get_inspection_by_id(self, inspection_id: int) -> Optional[Dict[str, Any]]:
        """Get specific inspection by ID.
        
        Args:
            inspection_id: Inspection ID to retrieve
            
        Returns:
            Inspection dictionary or None if not found
        """
        pass
    
    def get_inspections_by_date_range(self, start_date: datetime, 
                                     end_date: datetime) -> List[Dict[str, Any]]:
        """Get inspections within date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of inspection dictionaries
        """
        pass
    
    def close(self) -> None:
        """Close database connection."""
        pass
```

#### Database Schema

**Inspections Table:**
```sql
CREATE TABLE inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    unix_timestamp REAL NOT NULL,
    has_defects BOOLEAN NOT NULL,
    defect_count INTEGER NOT NULL,
    defects TEXT,                -- JSON serialized
    defect_locations TEXT,       -- JSON serialized  
    confidence_scores TEXT,      -- JSON serialized
    focus_score REAL,
    processing_time REAL,
    image_path TEXT,
    pcb_area INTEGER,
    trigger_type TEXT,
    session_id TEXT
);
```

**Defect Statistics Table:**
```sql
CREATE TABLE defect_statistics (
    defect_type TEXT PRIMARY KEY,
    total_count INTEGER DEFAULT 0,
    last_seen TEXT
);
```

---

## Analytics Layer APIs

### Defect Analyzer API

**Module:** `analytics/analyzer.py`

#### DefectAnalyzer Class

```python
class DefectAnalyzer:
    """Analytics and reporting for PCB inspection data."""
    
    def __init__(self, database: PCBDatabase):
        """Initialize analyzer with database connection.
        
        Args:
            database: PCBDatabase instance
        """
        pass
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time inspection statistics.
        
        Uses caching for performance (5-minute cache timeout).
        
        Returns:
            Dict containing:
            - total_inspections (int): Total inspection count
            - total_defects (int): Total defect count
            - pass_rate (float): Percentage of passing inspections
            - defect_rate (float): Average defects per inspection
            - recent_trend (str): "increasing", "decreasing", or "stable"
        """
        pass
    
    def get_defect_frequency(self) -> Dict[str, int]:
        """Get frequency count for each defect type.
        
        Returns:
            Dict mapping defect type to occurrence count
        """
        pass
    
    def get_time_based_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Get time-based analysis over specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict containing:
            - daily_counts (List): Daily inspection counts
            - daily_defect_rates (List): Daily defect rates
            - trend_analysis (Dict): Trend information
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics.
        
        Returns:
            Dict containing:
            - avg_processing_time (float): Average processing time
            - inspection_rate (float): Inspections per hour
            - system_uptime (float): System uptime percentage
        """
        pass
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report.
        
        Returns:
            Dict containing all available analytics
        """
        pass
    
    def export_report(self, format: str = "json", 
                     output_path: Optional[str] = None) -> str:
        """Export analytics report to file.
        
        Args:
            format: Export format ("json", "csv", "html")
            output_path: Optional output file path
            
        Returns:
            str: Path to exported file
        """
        pass
```

---

## Presentation Layer APIs

### Main GUI API

**Module:** `presentation/gui.py`

#### PCBInspectionGUI Class

```python
class PCBInspectionGUI:
    """Main application GUI using tkinter."""
    
    def __init__(self):
        """Initialize GUI components and layout."""
        pass
    
    def update_preview(self, image: np.ndarray, has_pcb: bool = False,
                      is_stable: bool = False, focus_score: float = 0) -> None:
        """Update live preview display.
        
        Args:
            image: Preview image to display
            has_pcb: Whether PCB is detected
            is_stable: Whether PCB position is stable
            focus_score: Image focus quality score
        """
        pass
    
    def update_inspection_display(self, image: np.ndarray, defects: List[str],
                                 stats: Dict[str, Any], inspection_id: int) -> None:
        """Update inspection results display.
        
        Args:
            image: Processed inspection image with annotations
            defects: List of detected defects
            stats: Current system statistics
            inspection_id: ID of completed inspection
        """
        pass
    
    def update_mode_display(self, is_auto: bool) -> None:
        """Update mode indicator display.
        
        Args:
            is_auto: True if in AUTO mode, False if MANUAL
        """
        pass
    
    def show_error(self, message: str) -> None:
        """Display error message to user.
        
        Args:
            message: Error message text
        """
        pass
    
    def show_message(self, message: str) -> None:
        """Display information message to user.
        
        Args:
            message: Information message text
        """
        pass
    
    # Callback properties (set by main application)
    toggle_auto_mode: Optional[Callable]     # Toggle AUTO/MANUAL mode
    manual_inspect: Optional[Callable]       # Trigger manual inspection
    view_analytics: Optional[Callable]       # Show analytics window
    view_history: Optional[Callable]         # Show history window
```

### Analytics Viewer API

**Module:** `presentation/analytics_viewer.py`

#### AnalyticsViewer Class

```python
class AnalyticsViewer:
    """Advanced analytics visualization window."""
    
    def __init__(self, parent, analyzer: DefectAnalyzer):
        """Initialize analytics viewer.
        
        Args:
            parent: Parent tkinter window
            analyzer: DefectAnalyzer instance
        """
        pass
    
    def show(self) -> None:
        """Show analytics window."""
        pass
    
    def update_charts(self) -> None:
        """Update all charts with latest data."""
        pass
    
    def export_charts(self, format: str = "png") -> None:
        """Export charts to image files.
        
        Args:
            format: Image format ("png", "pdf", "svg")
        """
        pass
```

### History Browser API

**Module:** `presentation/history_browser.py`

#### HistoryBrowser Class

```python
class HistoryBrowser:
    """Inspection history browser and search interface."""
    
    def __init__(self, parent, database: PCBDatabase):
        """Initialize history browser.
        
        Args:
            parent: Parent tkinter window  
            database: PCBDatabase instance
        """
        pass
    
    def show(self) -> None:
        """Show history browser window."""
        pass
    
    def search_inspections(self, criteria: Dict[str, Any]) -> None:
        """Search inspections by criteria.
        
        Args:
            criteria: Search criteria dictionary
        """
        pass
    
    def export_results(self, format: str = "csv") -> None:
        """Export search results.
        
        Args:
            format: Export format ("csv", "json", "excel")
        """
        pass
```

---

## Integration Patterns

### Main Application Integration

**Module:** `main.py`

#### PCBInspectionSystem Class

```python
class PCBInspectionSystem:
    """Main system orchestrator integrating all layers."""
    
    def __init__(self):
        """Initialize all system components."""
        pass
    
    def start_preview_stream(self) -> None:
        """Start camera preview and auto-detection thread."""
        pass
    
    def trigger_inspection(self) -> None:
        """Trigger high-quality inspection workflow."""
        pass
    
    def toggle_auto_mode(self) -> None:
        """Toggle between AUTO and MANUAL modes."""
        pass
    
    def run(self) -> None:
        """Start the application main loop."""
        pass
```

### Component Communication Patterns

#### Observer Pattern for GUI Updates

```python
class InspectionObserver:
    """Observer interface for inspection events."""
    
    def on_preview_update(self, image: np.ndarray, pcb_status: Dict) -> None:
        """Called when preview frame is updated."""
        pass
    
    def on_inspection_complete(self, results: Dict[str, Any]) -> None:
        """Called when inspection is completed."""
        pass
    
    def on_error(self, error: Exception) -> None:
        """Called when error occurs."""
        pass
```

#### Callback Pattern for System Control

```python
# GUI → System callbacks
system.toggle_auto_mode = gui.toggle_auto_mode
system.manual_inspect = gui.manual_inspect
system.show_analytics = gui.view_analytics

# System → GUI callbacks  
gui.update_preview = system.update_preview_display
gui.update_results = system.update_results_display
gui.show_error = system.handle_error
```

---

## Error Handling

### Exception Hierarchy

```python
class PCBInspectionError(Exception):
    """Base exception for PCB inspection system."""
    pass

class CameraError(PCBInspectionError):
    """Camera-related errors."""
    pass

class ProcessingError(PCBInspectionError):
    """Image processing errors."""
    pass

class InferenceError(PCBInspectionError):
    """AI inference errors."""
    pass

class DatabaseError(PCBInspectionError):
    """Database operation errors."""
    pass
```

### Error Recovery Patterns

```python
def safe_camera_operation(operation):
    """Decorator for safe camera operations with retry."""
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except CameraError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1)
    return wrapper

def graceful_degradation(primary_operation, fallback_operation):
    """Execute primary operation with fallback on failure."""
    try:
        return primary_operation()
    except Exception as e:
        logger.warning(f"Primary operation failed: {e}, using fallback")
        return fallback_operation()
```

---

## Performance Considerations

### Threading Architecture

```python
# Main Thread: GUI and system coordination
# Preview Thread: Camera streaming and PCB detection  
# Inspection Thread: High-quality capture and AI inference
# Database Thread: Async database operations
# Analytics Thread: Background analytics computation
```

### Memory Management

```python
# Image Buffer Management
class ImageBuffer:
    """Fixed-size buffer for camera frames."""
    
    def __init__(self, max_size: int = 10):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray) -> None:
        """Add frame, automatically dropping oldest if full."""
        with self.lock:
            self.buffer.append(frame.copy())
    
    def get_latest(self) -> Optional[np.ndarray]:
        """Get most recent frame without removing it."""
        with self.lock:
            return self.buffer[-1] if self.buffer else None
```

### Performance Targets

- **Camera Capture**: <50ms per high-quality image
- **Preview Stream**: 30+ FPS sustained operation
- **AI Inference**: <100ms on Tesla P4 GPU
- **Database Write**: <10ms per inspection record
- **GUI Update**: <100ms for complete refresh
- **Memory Usage**: <2GB total system memory

---

## Configuration Reference

### Environment Variables

```bash
# Camera Settings
PCB_CAMERA_MODEL=Basler_acA3800-10gm
PCB_CAMERA_EXPOSURE=10000
PCB_CAMERA_GAIN=0

# AI Settings  
PCB_MODEL_PATH=weights/best.pt
PCB_AI_DEVICE=cuda:0
PCB_CONFIDENCE=0.5

# Database Settings
PCB_DB_PATH=data/pcb_inspection.db
PCB_BACKUP_INTERVAL=3600

# Performance Settings
PCB_PREVIEW_FPS=30
PCB_BUFFER_SIZE=10
PCB_WORKER_THREADS=4
```

### Configuration File Format

```json
{
  "camera": {
    "model": "Basler_acA3800-10gm",
    "preview_exposure": 5000,
    "capture_exposure": 10000,
    "gain": 0,
    "pixel_format": "BayerRG8"
  },
  "ai": {
    "model_path": "weights/best.pt", 
    "confidence": 0.5,
    "device": "cuda:0"
  },
  "trigger": {
    "stability_frames": 10,
    "focus_threshold": 100,
    "movement_threshold": 5,
    "min_pcb_area": 0.1,
    "inspection_interval": 2.0
  },
  "database": {
    "path": "data/pcb_inspection.db",
    "save_raw_images": false,
    "save_processed_images": true
  }
}
```

---

## Version History

**Version 1.0** (December 2024)
- Initial API documentation
- Complete system architecture coverage
- All layer APIs documented
- Integration patterns defined
- Performance guidelines established

---

**Document Prepared by:** PCB Inspection System Development Team  
**Last Updated:** December 2024  
**API Version:** 1.0

For technical support or API questions, contact the development team.
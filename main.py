# main.py - PCB Inspection System Main Orchestrator
"""
PCB Auto-Inspection System
Main application that coordinates all system layers:
- Hardware: Camera streaming and high-quality capture
- Processing: PCB detection and auto-trigger logic
- AI: YOLOv11 defect detection
- Data: SQLite storage and metadata management
- Analytics: Real-time statistics and reporting
- Presentation: GUI with live preview and results

Flow: Camera Stream → PCB Detection → Auto Trigger → AI → Display
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

# Core imports
from core.config import (
    CAMERA_CONFIG, AI_CONFIG, DB_CONFIG, TRIGGER_CONFIG,
    DEFECT_CLASSES, MODEL_CLASS_MAPPING
)
from core.utils import setup_logging, TimestampUtil, ErrorHandler

# Layer imports
from hardware.camera_controller import BaslerCamera
from processing.preprocessor import ImagePreprocessor
from processing.pcb_detector import PCBDetector
from processing.postprocessor import ResultPostprocessor
from ai.inference import PCBDefectDetector
from data.database import PCBDatabase
from analytics.analyzer import DefectAnalyzer
from presentation.gui import PCBInspectionGUI


class PCBInspectionSystem:
    """
    Main system orchestrator with auto-trigger functionality.
    
    Coordinates all system layers and manages the complete inspection workflow:
    1. Camera streaming for real-time preview
    2. PCB detection and stability monitoring
    3. Auto-trigger when conditions are met
    4. High-quality image capture and AI inference
    5. Result processing and database storage
    6. GUI updates and user interaction
    """
    
    def __init__(self):
        """Initialize the PCB inspection system."""
        # Setup logging
        self.logger = setup_logging("PCBInspectionSystem")
        self.logger.info("="*60)
        self.logger.info("PCB Auto-Inspection System Starting...")
        self.logger.info("="*60)
        
        # System state
        self.is_running = False
        self.auto_mode = True
        self.last_inspection_time = 0
        self.min_inspection_interval = TRIGGER_CONFIG["inspection_interval"]
        
        # Threading components
        self.preview_thread: Optional[threading.Thread] = None
        self.inspection_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # System components (initialized in _initialize_system)
        self.camera: Optional[BaslerCamera] = None
        self.preprocessor: Optional[ImagePreprocessor] = None
        self.pcb_detector: Optional[PCBDetector] = None
        self.postprocessor: Optional[ResultPostprocessor] = None
        self.ai_detector: Optional[PCBDefectDetector] = None
        self.database: Optional[PCBDatabase] = None
        self.analyzer: Optional[DefectAnalyzer] = None
        self.gui: Optional[PCBInspectionGUI] = None
        
        # Performance monitoring
        self.inspection_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Initialize all system components
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize all system layers in proper order."""
        try:
            self.logger.info("Initializing Core Layer...")
            # Core layer is imported modules, no initialization needed
            
            self.logger.info("Initializing Hardware Layer...")
            self.camera = BaslerCamera(CAMERA_CONFIG)
            self.logger.info("Camera initialized successfully")
            
            self.logger.info("Initializing Processing Layer...")
            self.preprocessor = ImagePreprocessor()
            self.pcb_detector = PCBDetector(TRIGGER_CONFIG)
            self.postprocessor = ResultPostprocessor()
            self.logger.info("Processing layer initialized")
            
            self.logger.info("Initializing AI Layer...")
            self.ai_detector = PCBDefectDetector(AI_CONFIG)
            self.logger.info("AI detection model loaded")
            
            self.logger.info("Initializing Data Layer...")
            self.database = PCBDatabase(DB_CONFIG["path"])
            self.logger.info("Database initialized")
            
            self.logger.info("Initializing Analytics Layer...")
            self.analyzer = DefectAnalyzer(self.database)
            self.logger.info("Analytics engine ready")
            
            self.logger.info("Initializing Presentation Layer...")
            self.gui = PCBInspectionGUI()
            self._setup_gui_callbacks()
            self.logger.info("GUI interface created")
            
            self.logger.info("System initialization completed successfully!")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize PCB inspection system: {str(e)}")
    
    def _setup_gui_callbacks(self) -> None:
        """Setup all GUI callback functions."""
        if not self.gui:
            return
            
        # Bind callback functions to GUI events
        self.gui.set_callbacks(
            toggle_auto_mode=self.toggle_auto_mode,
            manual_inspect=self.manual_inspect,
            view_analytics=self.show_analytics,
            view_history=self.show_history
        )
        
        self.logger.info("GUI callbacks configured")
    
    def start_preview_stream(self) -> None:
        """Start camera preview stream and auto-detection thread."""
        try:
            if not self.camera:
                raise RuntimeError("Camera not initialized")
                
            self.is_running = True
            self.shutdown_event.clear()
            
            # Start camera streaming
            self.camera.start_streaming()
            self.logger.info("Camera streaming started")
            
            # Start preview processing thread
            self.preview_thread = threading.Thread(
                target=self._preview_loop,
                name="PreviewThread",
                daemon=True
            )
            self.preview_thread.start()
            self.logger.info("Preview thread started")
            
            # Update GUI status
            if self.gui:
                self.gui.update_status("System Active - Monitoring for PCBs")
                
        except Exception as e:
            self.logger.error(f"Failed to start preview stream: {str(e)}")
            self.is_running = False
            raise
    
    def stop_preview_stream(self) -> None:
        """Stop camera preview stream and processing."""
        self.logger.info("Stopping preview stream...")
        
        # Signal shutdown
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop camera
        if self.camera:
            self.camera.stop_streaming()
            
        # Wait for preview thread to finish
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=5.0)
            
        self.logger.info("Preview stream stopped")
    
    def _preview_loop(self) -> None:
        """
        Main preview loop with auto-trigger logic.
        Runs continuously while system is active.
        """
        self.logger.info("Preview loop started")
        frame_count = 0
        
        # FPS tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        fps_update_interval = 1.0  # Update FPS every second
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get latest frame from camera
                raw_frame = self.camera.get_preview_frame()
                if raw_frame is None:
                    time.sleep(0.01)  # Short wait if no frame available
                    continue
                
                frame_count += 1
                fps_frame_count += 1
                
                # Calculate FPS every second
                current_time = time.time()
                if current_time - fps_start_time >= fps_update_interval:
                    current_fps = fps_frame_count / (current_time - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = current_time
                
                # Detect PCB and check conditions
                detection_result = self.pcb_detector.detect_pcb(raw_frame)
                has_pcb = detection_result.has_pcb
                pcb_region = detection_result.position
                is_stable = detection_result.is_stable
                focus_score = detection_result.focus_score
                
                # Quick preview processing for display
                preview_gray = self.pcb_detector.debayer_to_gray(raw_frame)
                
                # Update preview display
                if self.gui:
                    # Debug logging every 30 frames
                    if frame_count % 30 == 0:
                        self.logger.debug(f"Frame {frame_count}: preview_gray shape={preview_gray.shape if preview_gray is not None else 'None'}, has_pcb={has_pcb}, focus={focus_score:.1f}")
                    
                    self.gui.update_preview(
                        image=preview_gray,
                        has_pcb=has_pcb,
                        is_stable=is_stable,
                        focus_score=focus_score,
                        fps=current_fps
                    )
                
                # Auto-trigger logic
                if self._should_trigger_inspection(has_pcb, is_stable, focus_score):
                    self.logger.info(
                        f"Auto-trigger activated: PCB detected, "
                        f"stable={is_stable}, focus={focus_score:.1f}"
                    )
                    self._trigger_inspection()
                
                # Performance monitoring (every 100 frames)
                if frame_count % 100 == 0:
                    self.logger.debug(f"Preview processed {frame_count} frames")
                
            except Exception as e:
                self.logger.error(f"Preview loop error: {str(e)}")
                self.error_count += 1
                
                # If too many errors, pause briefly
                if self.error_count > 10:
                    time.sleep(1.0)
                    self.error_count = 0
            
            # Control loop frequency (~30 FPS)
            time.sleep(0.033)
        
        self.logger.info("Preview loop terminated")
    
    def _should_trigger_inspection(self, has_pcb: bool, is_stable: bool, focus_score: float) -> bool:
        """
        Determine if inspection should be triggered based on all conditions.
        
        Args:
            has_pcb: PCB detected in frame
            is_stable: PCB position is stable
            focus_score: Image focus quality score
            
        Returns:
            True if inspection should be triggered
        """
        if not self.auto_mode:
            return False
            
        if not has_pcb:
            return False
            
        if not is_stable:
            return False
            
        if focus_score < TRIGGER_CONFIG["focus_threshold"]:
            return False
            
        if not self._can_inspect():
            return False
            
        return True
    
    def _can_inspect(self) -> bool:
        """Check if enough time has passed since last inspection."""
        current_time = time.time()
        return (current_time - self.last_inspection_time) > self.min_inspection_interval
    
    def _trigger_inspection(self) -> None:
        """Trigger automatic inspection in separate thread."""
        with self.inspection_lock:
            self.last_inspection_time = time.time()
            
            # Run inspection in separate thread to not block preview
            inspection_thread = threading.Thread(
                target=self._perform_inspection,
                name=f"InspectionThread-{self.inspection_count + 1}",
                daemon=True
            )
            inspection_thread.start()
    
    @ErrorHandler.log_exceptions
    def _perform_inspection(self) -> None:
        """
        Perform complete inspection workflow:
        1. Capture high-quality image
        2. Preprocess for AI
        3. Run defect detection
        4. Post-process results
        5. Save to database
        6. Update GUI
        """
        inspection_start_time = time.time()
        self.inspection_count += 1
        
        try:
            self.logger.info(f"Starting inspection #{self.inspection_count}")
            
            # Update GUI status
            if self.gui:
                self.gui.update_status(f"Performing inspection #{self.inspection_count}...")
            
            # Step 1: Capture high-quality raw image
            raw_image = self.camera.capture_high_quality()
            if raw_image is None:
                raise RuntimeError("Failed to capture high-quality image")
            
            capture_time = time.time()
            self.logger.debug(f"Image captured in {(capture_time - inspection_start_time)*1000:.1f}ms")
            
            # Step 2: Full preprocessing
            processed_image = self.preprocessor.process_raw(raw_image)
            
            process_time = time.time()
            self.logger.debug(f"Image processed in {(process_time - capture_time)*1000:.1f}ms")
            
            # Step 3: AI Detection
            detection_results = self.ai_detector.detect(processed_image)
            
            ai_time = time.time()
            self.logger.debug(f"AI inference in {(ai_time - process_time)*1000:.1f}ms")
            
            # Step 4: Extract and process results
            defects, locations, confidences = self._extract_results(detection_results)
            self.logger.info(f"Detected {len(defects)} defects: {defects}")
            
            # Step 5: Post-process for display
            display_image = self.postprocessor.draw_results(
                processed_image, detection_results
            )
            
            # Step 6: Calculate metrics
            focus_score = self.pcb_detector.focus_evaluator.evaluate(processed_image)
            processing_time = time.time() - inspection_start_time
            
            # Step 7: Save results to database
            timestamp = datetime.now()
            save_image = display_image if defects else None  # Only save defect images
            
            inspection_id = self.database.save_inspection_metadata(
                timestamp=timestamp,
                defects=defects,
                locations=locations,
                confidence_scores=confidences,
                raw_image_shape=raw_image.shape,
                focus_score=focus_score,
                processing_time=processing_time,
                save_image=save_image,
                trigger_type="auto",
                session_id=self._get_session_id()
            )
            
            # Step 8: Update analytics
            current_stats = self.analyzer.get_realtime_stats()
            
            # Step 9: Update GUI with results
            if self.gui:
                self.gui.update_inspection_results(
                    image=display_image,
                    defects=defects,
                    locations=locations,
                    confidence_scores=confidences,
                    inspection_id=inspection_id,
                    processing_time=processing_time
                )
                
                self.gui.update_statistics(current_stats)
                
                # Update status
                status = f"Inspection #{inspection_id} complete"
                if defects:
                    status += f" - {len(defects)} defects found"
                else:
                    status += " - PASS"
                self.gui.update_status(status)
            
            self.logger.info(
                f"Inspection #{inspection_id} completed in {processing_time*1000:.1f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Inspection #{self.inspection_count} failed: {str(e)}")
            if self.gui:
                self.gui.show_error(f"Inspection failed: {str(e)}")
    
    def _extract_results(self, detection_results) -> tuple[List[str], List[Dict], List[float]]:
        """Extract defects, locations, and confidences from AI results."""
        
        # Check if detection_results is InspectionResult or YOLO raw results
        if hasattr(detection_results, 'defects'):
            # InspectionResult object
            defects = detection_results.defects
            locations = detection_results.locations  
            confidences = detection_results.confidence_scores
            
            self.logger.debug(f"Extracted from InspectionResult: {len(defects)} defects")
            
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
            
            self.logger.debug(f"Extracted from YOLO results: {len(defects)} defects")
            
        else:
            # Unknown format
            self.logger.warning(f"Unknown detection result format: {type(detection_results)}")
            defects, locations, confidences = [], [], []
        
        return defects, locations, confidences
    
    def _get_session_id(self) -> str:
        """Get current session ID for tracking."""
        return f"session_{int(self.start_time)}"
    
    # Public interface methods
    
    def toggle_auto_mode(self) -> None:
        """Toggle between automatic and manual inspection modes."""
        self.auto_mode = not self.auto_mode
        mode = "AUTO" if self.auto_mode else "MANUAL"
        self.logger.info(f"Switched to {mode} mode")
        
        if self.gui:
            self.gui.update_mode_display(self.auto_mode)
            status = f"Mode: {mode} - " + ("Monitoring for PCBs" if self.auto_mode else "Manual control")
            self.gui.update_status(status)
    
    def manual_inspect(self) -> None:
        """Manually trigger inspection if conditions allow."""
        if self._can_inspect():
            self.logger.info("Manual inspection triggered")
            self._trigger_inspection()
        else:
            wait_time = self.min_inspection_interval - (time.time() - self.last_inspection_time)
            message = f"Please wait {wait_time:.1f}s before next inspection"
            self.logger.warning(message)
            if self.gui:
                self.gui.show_info(message)
    
    def show_analytics(self) -> None:
        """Display analytics dashboard."""
        try:
            if not self.analyzer or not self.gui:
                return
                
            stats = self.analyzer.get_comprehensive_report()
            # For now, show analytics in a simple dialog
            # TODO: Implement full analytics window
            analytics_text = f"Analytics Summary:\n"
            analytics_text += f"Total Inspections: {stats.get('total_inspections', 0)}\n"
            analytics_text += f"Defects Found: {stats.get('total_defects', 0)}\n"
            analytics_text += f"Pass Rate: {stats.get('pass_rate', 0):.1f}%"
            
            self.gui.show_info(analytics_text)
            self.logger.info("Analytics dashboard opened")
        except Exception as e:
            self.logger.error(f"Failed to show analytics: {str(e)}")
            if self.gui:
                self.gui.show_error(f"Analytics error: {str(e)}")
    
    def show_history(self) -> None:
        """Display inspection history browser."""
        try:
            if not self.database or not self.gui:
                return
                
            recent_inspections = self.database.get_recent_inspections(10)
            # For now, show history in a simple dialog
            # TODO: Implement full history browser window
            history_text = "Recent Inspections:\n\n"
            for inspection in recent_inspections[:5]:
                history_text += f"ID: {inspection.get('id', 'N/A')}, "
                history_text += f"Time: {inspection.get('timestamp', 'N/A')[:19]}, "
                history_text += f"Defects: {inspection.get('defect_count', 0)}\n"
            
            self.gui.show_info(history_text)
            self.logger.info("History browser opened")
        except Exception as e:
            self.logger.error(f"Failed to show history: {str(e)}")
            if self.gui:
                self.gui.show_error(f"History error: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        uptime = time.time() - self.start_time
        return {
            'is_running': self.is_running,
            'auto_mode': self.auto_mode,
            'inspection_count': self.inspection_count,
            'error_count': self.error_count,
            'uptime_seconds': uptime,
            'last_inspection_time': self.last_inspection_time,
            'can_inspect': self._can_inspect()
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        self.logger.info("Initiating system shutdown...")
        
        try:
            # Stop preview stream
            self.stop_preview_stream()
            
            # Close camera connection
            if self.camera:
                self.camera.close()
                
            # Close database connection
            if self.database:
                self.database.close()
                
            # Log final statistics
            uptime = time.time() - self.start_time
            self.logger.info(f"System shutdown complete. Uptime: {uptime:.1f}s, "
                           f"Inspections: {self.inspection_count}, Errors: {self.error_count}")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    def run(self) -> None:
        """
        Start the complete PCB inspection system.
        This is the main entry point for the application.
        """
        try:
            self.logger.info("Starting PCB Inspection System...")
            
            # Start preview stream
            self.start_preview_stream()
            
            # Log system ready
            self.logger.info("="*60)
            self.logger.info("PCB Auto-Inspection System READY")
            self.logger.info("Mode: AUTO - Monitoring for PCBs")
            self.logger.info("="*60)
            
            # Run GUI main loop (blocks until window is closed)
            if self.gui:
                self.gui.run()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"System error: {str(e)}")
            raise
        finally:
            # Ensure cleanup
            self.shutdown()


def main():
    """Main entry point for the PCB inspection system."""
    try:
        # Create and run the system
        system = PCBInspectionSystem()
        system.run()
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        print(f"System failed to start: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
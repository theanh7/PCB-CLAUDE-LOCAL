"""
Main GUI implementation for PCB inspection system.

This module provides the primary graphical user interface with live preview,
inspection results, statistics, and controls for the PCB inspection system.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import logging
import os

from core.config import GUI_CONFIG, DEFECT_COLORS
from core.utils import setup_logging

class PCBInspectionGUI:
    """
    Main GUI class for PCB inspection system.
    
    Features:
    - Live camera preview with detection overlays
    - Dual-mode operation (AUTO/MANUAL)
    - Real-time inspection results display
    - Statistics dashboard
    - Analytics viewer
    - History browser
    """
    
    def __init__(self):
        """Initialize the GUI application."""
        self.logger = setup_logging(__name__)
        
        # Initialize state variables
        self.auto_mode = True
        self.is_running = False
        self.current_image = None
        self.current_results = None
        
        # Callback functions (to be set by main system)
        self.toggle_auto_mode_callback = None
        self.manual_inspect_callback = None
        self.view_analytics_callback = None
        self.view_history_callback = None
        
        # Create main window
        self.root = tk.Tk()
        self._setup_main_window()
        
        # Create GUI layout
        self._create_layout()
        
        # Initialize image display
        self._initialize_image_display()
        
        self.logger.info("GUI initialized successfully")
    
    def _setup_main_window(self):
        """Setup main window properties."""
        self.root.title(GUI_CONFIG["window_title"])
        self.root.geometry(f"{GUI_CONFIG['window_size'][0]}x{GUI_CONFIG['window_size'][1]}")
        self.root.minsize(1200, 700)
        
        # Configure window style
        self.root.configure(bg='#f0f0f0')
        
        # Set window icon if available
        try:
            icon_path = "assets/icon.ico"
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass  # Icon not available
        
        # Configure window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_layout(self):
        """Create the main GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create main sections
        self._create_control_panel(main_frame)
        self._create_preview_section(main_frame)
        self._create_results_section(main_frame)
        self._create_status_bar(main_frame)
    
    def _create_control_panel(self, parent):
        """Create the control panel with buttons and status."""
        control_frame = ttk.LabelFrame(parent, text="System Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Mode control
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 20))
        
        ttk.Label(mode_frame, text="Mode:").grid(row=0, column=0, padx=(0, 5))
        
        self.mode_button = ttk.Button(
            mode_frame,
            text="AUTO",
            command=self._on_mode_toggle,
            width=10
        )
        self.mode_button.grid(row=0, column=1, padx=(0, 10))
        
        # Manual inspect button
        self.inspect_button = ttk.Button(
            mode_frame,
            text="Manual Inspect",
            command=self._on_manual_inspect,
            width=15,
            state=tk.DISABLED
        )
        self.inspect_button.grid(row=0, column=2, padx=(0, 10))
        
        # System status
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(20, 0))
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, padx=(0, 5))
        
        self.status_label = ttk.Label(
            status_frame,
            text="System Ready",
            foreground="green"
        )
        self.status_label.grid(row=0, column=1, padx=(0, 20))
        
        # Additional controls
        controls_frame = ttk.Frame(control_frame)
        controls_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(
            controls_frame,
            text="View Analytics",
            command=self._on_view_analytics,
            width=15
        ).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="View History",
            command=self._on_view_history,
            width=15
        ).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="Settings",
            command=self._on_settings,
            width=15
        ).grid(row=0, column=2, padx=(0, 10))
        
        # Configure grid weights
        control_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(3, weight=1)
    
    def _create_preview_section(self, parent):
        """Create the live preview section."""
        preview_frame = ttk.LabelFrame(parent, text="Live Preview", padding="10")
        preview_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Preview image display
        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=GUI_CONFIG["preview_size"][0],
            height=GUI_CONFIG["preview_size"][1],
            bg='black'
        )
        self.preview_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Preview info panel
        preview_info_frame = ttk.Frame(preview_frame)
        preview_info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # PCB detection status
        self.pcb_status_label = ttk.Label(
            preview_info_frame,
            text="PCB: Not detected",
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.pcb_status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Focus score display
        self.focus_label = ttk.Label(
            preview_info_frame,
            text="Focus: --",
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.focus_label.grid(row=1, column=0, sticky=tk.W)
        
        # Stability status
        self.stability_label = ttk.Label(
            preview_info_frame,
            text="Stability: --",
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.stability_label.grid(row=2, column=0, sticky=tk.W)
        
        # FPS counter
        self.fps_label = ttk.Label(
            preview_info_frame,
            text="FPS: --",
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.fps_label.grid(row=3, column=0, sticky=tk.W)
        
        # Configure grid weights
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
    
    def _create_results_section(self, parent):
        """Create the inspection results section."""
        results_frame = ttk.LabelFrame(parent, text="Inspection Results", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results image display
        self.results_canvas = tk.Canvas(
            results_frame,
            width=GUI_CONFIG["result_size"][0],
            height=GUI_CONFIG["result_size"][1],
            bg='black'
        )
        self.results_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results info panel
        results_info_frame = ttk.Frame(results_frame)
        results_info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Inspection details
        details_frame = ttk.LabelFrame(results_info_frame, text="Inspection Details", padding="5")
        details_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.inspection_id_label = ttk.Label(
            details_frame,
            text="Inspection ID: --",
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.inspection_id_label.grid(row=0, column=0, sticky=tk.W)
        
        self.timestamp_label = ttk.Label(
            details_frame,
            text="Timestamp: --",
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.timestamp_label.grid(row=1, column=0, sticky=tk.W)
        
        self.processing_time_label = ttk.Label(
            details_frame,
            text="Processing Time: --",
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.processing_time_label.grid(row=2, column=0, sticky=tk.W)
        
        # Defects display
        defects_frame = ttk.LabelFrame(results_info_frame, text="Detected Defects", padding="5")
        defects_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Defects listbox with scrollbar
        listbox_frame = ttk.Frame(defects_frame)
        listbox_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.defects_listbox = tk.Listbox(
            listbox_frame,
            height=8,
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.defects_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        defects_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        defects_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.defects_listbox.config(yscrollcommand=defects_scrollbar.set)
        defects_scrollbar.config(command=self.defects_listbox.yview)
        
        # Configure grid weights
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        results_info_frame.columnconfigure(0, weight=1)
        defects_frame.columnconfigure(0, weight=1)
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)
    
    def _create_status_bar(self, parent):
        """Create the status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Statistics display
        self.stats_label = ttk.Label(
            status_frame,
            text="Total Inspections: 0 | Defects Found: 0 | Pass Rate: --",
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.stats_label.grid(row=0, column=0, sticky=tk.W)
        
        # System time
        self.time_label = ttk.Label(
            status_frame,
            text="",
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"])
        )
        self.time_label.grid(row=0, column=1, sticky=tk.E)
        
        # Configure grid weights
        status_frame.columnconfigure(0, weight=1)
        
        # Start time update
        self._update_time()
    
    def _initialize_image_display(self):
        """Initialize image display with placeholder."""
        # Create placeholder images
        self.placeholder_preview = self._create_placeholder_image(
            GUI_CONFIG["preview_size"], "Camera Preview"
        )
        self.placeholder_results = self._create_placeholder_image(
            GUI_CONFIG["result_size"], "Inspection Results"
        )
        
        # Display placeholders
        self.preview_canvas.create_image(
            GUI_CONFIG["preview_size"][0]//2,
            GUI_CONFIG["preview_size"][1]//2,
            image=self.placeholder_preview
        )
        
        self.results_canvas.create_image(
            GUI_CONFIG["result_size"][0]//2,
            GUI_CONFIG["result_size"][1]//2,
            image=self.placeholder_results
        )
    
    def _create_placeholder_image(self, size, text):
        """Create a placeholder image."""
        img = Image.new('RGB', size, color='#404040')
        
        # Add text if PIL supports it
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (size[0] - text_width) // 2
            y = (size[1] - text_height) // 2
            
            draw.text((x, y), text, fill='white', font=font)
            
        except ImportError:
            # PIL not available, just return plain image
            pass
        
        return ImageTk.PhotoImage(img)
    
    def _update_time(self):
        """Update the time display."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        
        # Schedule next update
        self.root.after(1000, self._update_time)
    
    # Event handlers
    def _on_mode_toggle(self):
        """Handle mode toggle button."""
        if self.toggle_auto_mode_callback:
            self.toggle_auto_mode_callback()
    
    def _on_manual_inspect(self):
        """Handle manual inspect button."""
        if self.manual_inspect_callback:
            self.manual_inspect_callback()
    
    def _on_view_analytics(self):
        """Handle view analytics button."""
        if self.view_analytics_callback:
            self.view_analytics_callback()
        else:
            self._show_analytics_window()
    
    def _on_view_history(self):
        """Handle view history button."""
        if self.view_history_callback:
            self.view_history_callback()
        else:
            self._show_history_window()
    
    def _on_settings(self):
        """Handle settings button."""
        self._show_settings_window()
    
    def _on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.is_running = False
            self.root.destroy()
    
    # Public methods for updating GUI
    def update_preview(self, image: np.ndarray, has_pcb: bool = False, 
                      is_stable: bool = False, focus_score: float = 0.0,
                      fps: float = 0.0):
        """
        Update preview display.
        
        Args:
            image: Preview image
            has_pcb: Whether PCB is detected
            is_stable: Whether PCB is stable
            focus_score: Focus quality score
            fps: Current FPS
        """
        try:
            # Convert and resize image for display
            if len(image.shape) == 2:
                # Grayscale to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # Assume BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Draw PCB detection indicator
            if has_pcb:
                color = (0, 255, 0) if is_stable else (255, 165, 0)
                cv2.putText(image_rgb, "PCB DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Resize to fit preview area
            preview_size = GUI_CONFIG["preview_size"]
            image_resized = cv2.resize(image_rgb, preview_size)
            
            # Convert to PhotoImage
            image_pil = Image.fromarray(image_resized)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Update canvas
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                preview_size[0]//2,
                preview_size[1]//2,
                image=image_tk
            )
            
            # Keep reference to prevent garbage collection
            self.current_preview_image = image_tk
            
            # Update status labels
            self._update_preview_status(has_pcb, is_stable, focus_score, fps)
            
        except Exception as e:
            self.logger.error(f"Error updating preview: {str(e)}")
    
    def _update_preview_status(self, has_pcb: bool, is_stable: bool, 
                              focus_score: float, fps: float):
        """Update preview status labels."""
        # PCB status
        pcb_text = "PCB: Detected" if has_pcb else "PCB: Not detected"
        self.pcb_status_label.config(text=pcb_text)
        
        # Focus score
        focus_text = f"Focus: {focus_score:.1f}"
        focus_color = "green" if focus_score > 100 else "red"
        self.focus_label.config(text=focus_text, foreground=focus_color)
        
        # Stability
        stability_text = "Stability: OK" if is_stable else "Stability: Waiting..."
        self.stability_label.config(text=stability_text)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        self.fps_label.config(text=fps_text)
    
    def update_inspection_results(self, image: np.ndarray, defects: List[str],
                                 locations: List[Dict], confidence_scores: List[float],
                                 inspection_id: int, processing_time: float = 0.0):
        """
        Update inspection results display.
        
        Args:
            image: Inspection result image with overlays
            defects: List of detected defects
            locations: List of defect locations
            confidence_scores: List of confidence scores
            inspection_id: Inspection ID
            processing_time: Processing time in seconds
        """
        try:
            # Convert and resize image for display
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to fit results area
            result_size = GUI_CONFIG["result_size"]
            image_resized = cv2.resize(image_rgb, result_size)
            
            # Convert to PhotoImage
            image_pil = Image.fromarray(image_resized)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Update canvas
            self.results_canvas.delete("all")
            self.results_canvas.create_image(
                result_size[0]//2,
                result_size[1]//2,
                image=image_tk
            )
            
            # Keep reference
            self.current_results_image = image_tk
            
            # Update inspection details
            self._update_inspection_details(inspection_id, processing_time)
            
            # Update defects list
            self._update_defects_list(defects, confidence_scores)
            
        except Exception as e:
            self.logger.error(f"Error updating inspection results: {str(e)}")
    
    def _update_inspection_details(self, inspection_id: int, processing_time: float):
        """Update inspection details."""
        self.inspection_id_label.config(text=f"Inspection ID: {inspection_id}")
        self.timestamp_label.config(text=f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
        self.processing_time_label.config(text=f"Processing Time: {processing_time:.3f}s")
    
    def _update_defects_list(self, defects: List[str], confidence_scores: List[float]):
        """Update defects list display."""
        self.defects_listbox.delete(0, tk.END)
        
        if not defects:
            self.defects_listbox.insert(tk.END, "✓ No defects detected")
            self.defects_listbox.insert(tk.END, "PCB PASSED")
        else:
            for i, defect in enumerate(defects):
                confidence = confidence_scores[i] if i < len(confidence_scores) else 0.0
                self.defects_listbox.insert(tk.END, f"• {defect} ({confidence:.1%})")
    
    def update_statistics(self, stats: Dict[str, Any]):
        """
        Update statistics display.
        
        Args:
            stats: Statistics dictionary
        """
        try:
            total_inspections = stats.get('total_inspections', 0)
            total_defects = stats.get('total_defects', 0)
            pass_rate = stats.get('pass_rate', 0.0)
            
            stats_text = (f"Total Inspections: {total_inspections} | "
                         f"Defects Found: {total_defects} | "
                         f"Pass Rate: {pass_rate:.1%}")
            
            self.stats_label.config(text=stats_text)
            
        except Exception as e:
            self.logger.error(f"Error updating statistics: {str(e)}")
    
    def update_mode_display(self, is_auto: bool):
        """
        Update mode display.
        
        Args:
            is_auto: Whether in auto mode
        """
        self.auto_mode = is_auto
        
        mode_text = "AUTO" if is_auto else "MANUAL"
        self.mode_button.config(text=mode_text)
        
        # Enable/disable manual inspect button
        inspect_state = tk.DISABLED if is_auto else tk.NORMAL
        self.inspect_button.config(state=inspect_state)
    
    def update_status(self, status: str, color: str = "black"):
        """
        Update system status display.
        
        Args:
            status: Status message
            color: Status color
        """
        self.status_label.config(text=status, foreground=color)
    
    def show_error(self, message: str):
        """Show error message."""
        messagebox.showerror("Error", message)
    
    def show_info(self, message: str):
        """Show info message."""
        messagebox.showinfo("Information", message)
    
    def show_warning(self, message: str):
        """Show warning message."""
        messagebox.showwarning("Warning", message)
    
    # Dialog windows
    def _show_analytics_window(self):
        """Show analytics window."""
        analytics_window = tk.Toplevel(self.root)
        analytics_window.title("Analytics Dashboard")
        analytics_window.geometry("800x600")
        analytics_window.transient(self.root)
        analytics_window.grab_set()
        
        # Create placeholder for analytics
        ttk.Label(
            analytics_window,
            text="Analytics Dashboard\n\n(Implementation in progress)",
            font=(GUI_CONFIG["font_family"], 14),
            justify=tk.CENTER
        ).pack(expand=True)
        
        ttk.Button(
            analytics_window,
            text="Close",
            command=analytics_window.destroy
        ).pack(pady=10)
    
    def _show_history_window(self):
        """Show history window."""
        history_window = tk.Toplevel(self.root)
        history_window.title("Inspection History")
        history_window.geometry("900x600")
        history_window.transient(self.root)
        history_window.grab_set()
        
        # Create placeholder for history
        ttk.Label(
            history_window,
            text="Inspection History\n\n(Implementation in progress)",
            font=(GUI_CONFIG["font_family"], 14),
            justify=tk.CENTER
        ).pack(expand=True)
        
        ttk.Button(
            history_window,
            text="Close",
            command=history_window.destroy
        ).pack(pady=10)
    
    def _show_settings_window(self):
        """Show settings window."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("System Settings")
        settings_window.geometry("600x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Create placeholder for settings
        ttk.Label(
            settings_window,
            text="System Settings\n\n(Implementation in progress)",
            font=(GUI_CONFIG["font_family"], 14),
            justify=tk.CENTER
        ).pack(expand=True)
        
        ttk.Button(
            settings_window,
            text="Close",
            command=settings_window.destroy
        ).pack(pady=10)
    
    # Callback registration
    def set_callbacks(self, **callbacks):
        """
        Set callback functions.
        
        Args:
            **callbacks: Callback functions
        """
        self.toggle_auto_mode_callback = callbacks.get('toggle_auto_mode')
        self.manual_inspect_callback = callbacks.get('manual_inspect')
        self.view_analytics_callback = callbacks.get('view_analytics')
        self.view_history_callback = callbacks.get('view_history')
    
    def run(self):
        """Start the GUI main loop."""
        self.is_running = True
        self.logger.info("Starting GUI main loop")
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.logger.info("GUI interrupted by user")
        finally:
            self.is_running = False
            self.logger.info("GUI terminated")
    
    def stop(self):
        """Stop the GUI."""
        self.is_running = False
        self.root.quit()


# Test GUI independently
if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create test GUI
    app = PCBInspectionGUI()
    
    # Test with sample data
    def test_preview_update():
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        app.update_preview(test_image, has_pcb=True, is_stable=True, focus_score=150.0, fps=30.0)
        
        # Schedule next update
        app.root.after(1000, test_preview_update)
    
    def test_results_update():
        # Create test result image
        test_image = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
        defects = ["Missing Hole", "Open Circuit"]
        locations = [{"bbox": [100, 100, 200, 200]}, {"bbox": [300, 300, 400, 400]}]
        confidence_scores = [0.85, 0.75]
        
        app.update_inspection_results(test_image, defects, locations, confidence_scores, 123, 0.05)
        
        # Schedule next update
        app.root.after(5000, test_results_update)
    
    def test_stats_update():
        stats = {
            'total_inspections': 42,
            'total_defects': 8,
            'pass_rate': 0.81
        }
        app.update_statistics(stats)
        
        # Schedule next update
        app.root.after(2000, test_stats_update)
    
    # Start test updates
    app.root.after(100, test_preview_update)
    app.root.after(2000, test_results_update)
    app.root.after(500, test_stats_update)
    
    # Run GUI
    app.run()
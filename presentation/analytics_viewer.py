"""
Analytics viewer for PCB inspection system.

This module provides advanced analytics visualization including
charts, trends, and detailed statistics for the inspection system.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from core.config import GUI_CONFIG, DEFECT_CLASSES, DEFECT_COLORS
from core.utils import setup_logging

class AnalyticsViewer:
    """
    Advanced analytics viewer with charts and statistics.
    
    Features:
    - Real-time metrics dashboard
    - Defect trend charts
    - Performance analytics
    - Quality metrics
    - Comparative analysis
    """
    
    def __init__(self, parent=None):
        """
        Initialize analytics viewer.
        
        Args:
            parent: Parent window (optional)
        """
        self.logger = setup_logging(__name__)
        self.parent = parent
        self.analyzer = None  # Will be set by main system
        
        # Create analytics window
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self._setup_window()
        
        # Create GUI components
        self._create_layout()
        
        # Initialize charts
        self._initialize_charts()
        
        self.logger.info("Analytics viewer initialized")
    
    def _setup_window(self):
        """Setup analytics window."""
        self.window.title("PCB Inspection Analytics Dashboard")
        self.window.geometry("1200x800")
        self.window.minsize(1000, 600)
        
        if self.parent:
            self.window.transient(self.parent)
            self.window.grab_set()
    
    def _create_layout(self):
        """Create analytics layout."""
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create sections
        self._create_controls(main_frame)
        self._create_charts_section(main_frame)
        self._create_stats_section(main_frame)
    
    def _create_controls(self, parent):
        """Create control panel."""
        controls_frame = ttk.LabelFrame(parent, text="Analytics Controls", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Time period selection
        ttk.Label(controls_frame, text="Time Period:").grid(row=0, column=0, padx=(0, 5))
        
        self.period_var = tk.StringVar(value="7d")
        period_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.period_var,
            values=["1d", "7d", "30d", "90d"],
            state="readonly",
            width=10
        )
        period_combo.grid(row=0, column=1, padx=(0, 20))
        period_combo.bind("<<ComboboxSelected>>", self._on_period_change)
        
        # Refresh button
        ttk.Button(
            controls_frame,
            text="Refresh",
            command=self._refresh_analytics,
            width=10
        ).grid(row=0, column=2, padx=(0, 20))
        
        # Export button
        ttk.Button(
            controls_frame,
            text="Export Report",
            command=self._export_report,
            width=15
        ).grid(row=0, column=3, padx=(0, 20))
        
        # Status label
        self.status_label = ttk.Label(controls_frame, text="Ready")
        self.status_label.grid(row=0, column=4, padx=(20, 0))
    
    def _create_charts_section(self, parent):
        """Create charts section."""
        charts_frame = ttk.LabelFrame(parent, text="Analytics Charts", padding="10")
        charts_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create notebook for different chart types
        self.notebook = ttk.Notebook(charts_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        
        # Create chart tabs
        self._create_overview_tab()
        self._create_defects_tab()
        self._create_performance_tab()
        self._create_quality_tab()
    
    def _create_overview_tab(self):
        """Create overview tab."""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="Overview")
        
        # Create matplotlib figure
        self.overview_fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.overview_fig.tight_layout(pad=3.0)
        
        # Create canvas
        self.overview_canvas = FigureCanvasTkAgg(self.overview_fig, overview_frame)
        self.overview_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        overview_frame.columnconfigure(0, weight=1)
        overview_frame.rowconfigure(0, weight=1)
    
    def _create_defects_tab(self):
        """Create defects analysis tab."""
        defects_frame = ttk.Frame(self.notebook)
        self.notebook.add(defects_frame, text="Defects")
        
        # Create matplotlib figure
        self.defects_fig, (self.ax_defects1, self.ax_defects2) = plt.subplots(1, 2, figsize=(12, 6))
        self.defects_fig.tight_layout(pad=3.0)
        
        # Create canvas
        self.defects_canvas = FigureCanvasTkAgg(self.defects_fig, defects_frame)
        self.defects_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        defects_frame.columnconfigure(0, weight=1)
        defects_frame.rowconfigure(0, weight=1)
    
    def _create_performance_tab(self):
        """Create performance analysis tab."""
        performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(performance_frame, text="Performance")
        
        # Create matplotlib figure
        self.performance_fig, (self.ax_perf1, self.ax_perf2) = plt.subplots(1, 2, figsize=(12, 6))
        self.performance_fig.tight_layout(pad=3.0)
        
        # Create canvas
        self.performance_canvas = FigureCanvasTkAgg(self.performance_fig, performance_frame)
        self.performance_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        performance_frame.columnconfigure(0, weight=1)
        performance_frame.rowconfigure(0, weight=1)
    
    def _create_quality_tab(self):
        """Create quality analysis tab."""
        quality_frame = ttk.Frame(self.notebook)
        self.notebook.add(quality_frame, text="Quality")
        
        # Create matplotlib figure
        self.quality_fig, (self.ax_quality1, self.ax_quality2) = plt.subplots(1, 2, figsize=(12, 6))
        self.quality_fig.tight_layout(pad=3.0)
        
        # Create canvas
        self.quality_canvas = FigureCanvasTkAgg(self.quality_fig, quality_frame)
        self.quality_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        quality_frame.columnconfigure(0, weight=1)
        quality_frame.rowconfigure(0, weight=1)
    
    def _create_stats_section(self, parent):
        """Create statistics section."""
        stats_frame = ttk.LabelFrame(parent, text="Key Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Create statistics display
        self.stats_text = tk.Text(
            stats_frame,
            height=6,
            width=80,
            font=(GUI_CONFIG["font_family"], GUI_CONFIG["font_size"]),
            state=tk.DISABLED
        )
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Add scrollbar
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL)
        stats_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
        stats_scrollbar.config(command=self.stats_text.yview)
        
        # Configure grid weights
        stats_frame.columnconfigure(0, weight=1)
    
    def _initialize_charts(self):
        """Initialize charts with placeholder data."""
        # Overview charts
        self.ax1.set_title("Daily Inspections")
        self.ax1.set_xlabel("Date")
        self.ax1.set_ylabel("Count")
        
        self.ax2.set_title("Defect Rate")
        self.ax2.set_xlabel("Date")
        self.ax2.set_ylabel("Rate (%)")
        
        self.ax3.set_title("Processing Time")
        self.ax3.set_xlabel("Date")
        self.ax3.set_ylabel("Time (s)")
        
        self.ax4.set_title("Focus Score")
        self.ax4.set_xlabel("Date")
        self.ax4.set_ylabel("Score")
        
        # Defects charts
        self.ax_defects1.set_title("Defect Distribution")
        self.ax_defects2.set_title("Defect Trends")
        
        # Performance charts
        self.ax_perf1.set_title("Processing Performance")
        self.ax_perf2.set_title("System Performance")
        
        # Quality charts
        self.ax_quality1.set_title("Quality Metrics")
        self.ax_quality2.set_title("Quality Trends")
        
        # Initial draw
        self._draw_placeholder_charts()
    
    def _draw_placeholder_charts(self):
        """Draw placeholder charts."""
        # Generate sample data
        dates = [datetime.now() - timedelta(days=i) for i in range(7, 0, -1)]
        
        # Overview charts
        self.ax1.plot(dates, [20, 25, 30, 28, 35, 32, 38], 'b-o')
        self.ax1.tick_params(axis='x', rotation=45)
        
        self.ax2.plot(dates, [10, 8, 12, 15, 9, 11, 7], 'r-o')
        self.ax2.tick_params(axis='x', rotation=45)
        
        self.ax3.plot(dates, [0.05, 0.048, 0.052, 0.055, 0.049, 0.051, 0.047], 'g-o')
        self.ax3.tick_params(axis='x', rotation=45)
        
        self.ax4.plot(dates, [150, 155, 148, 152, 158, 154, 160], 'm-o')
        self.ax4.tick_params(axis='x', rotation=45)
        
        # Defects charts
        defect_counts = [15, 12, 8, 6, 4, 3]
        self.ax_defects1.pie(defect_counts, labels=DEFECT_CLASSES, autopct='%1.1f%%')
        
        # Performance charts
        self.ax_perf1.bar(['Inference', 'Preprocessing', 'Postprocessing'], [0.03, 0.015, 0.005])
        
        # Quality charts
        quality_grades = ['A', 'B', 'C', 'D', 'F']
        quality_counts = [40, 30, 20, 8, 2]
        self.ax_quality1.bar(quality_grades, quality_counts)
        
        # Refresh canvases
        self.overview_canvas.draw()
        self.defects_canvas.draw()
        self.performance_canvas.draw()
        self.quality_canvas.draw()
    
    def update_analytics(self, analytics_data: Dict[str, Any]):
        """
        Update analytics display with new data.
        
        Args:
            analytics_data: Analytics data from analyzer
        """
        try:
            self.status_label.config(text="Updating charts...")
            
            # Clear previous charts
            self._clear_charts()
            
            # Update overview charts
            self._update_overview_charts(analytics_data)
            
            # Update defects charts
            self._update_defects_charts(analytics_data)
            
            # Update performance charts
            self._update_performance_charts(analytics_data)
            
            # Update quality charts
            self._update_quality_charts(analytics_data)
            
            # Update statistics text
            self._update_statistics_text(analytics_data)
            
            self.status_label.config(text="Charts updated")
            
        except Exception as e:
            self.logger.error(f"Error updating analytics: {str(e)}")
            self.status_label.config(text="Error updating charts")
    
    def _clear_charts(self):
        """Clear all charts."""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        self.ax_defects1.clear()
        self.ax_defects2.clear()
        self.ax_perf1.clear()
        self.ax_perf2.clear()
        self.ax_quality1.clear()
        self.ax_quality2.clear()
    
    def _update_overview_charts(self, data: Dict[str, Any]):
        """Update overview charts."""
        # Implementation depends on analytics data structure
        daily_trends = data.get('daily_trends', [])
        
        if daily_trends:
            dates = [datetime.fromisoformat(trend['date']) for trend in daily_trends]
            inspections = [trend['total_inspections'] for trend in daily_trends]
            defect_rates = [trend['defect_rate'] * 100 for trend in daily_trends]
            processing_times = [trend['avg_processing_time'] for trend in daily_trends]
            focus_scores = [trend['avg_focus_score'] for trend in daily_trends]
            
            self.ax1.plot(dates, inspections, 'b-o', label='Inspections')
            self.ax1.set_title("Daily Inspections")
            self.ax1.set_xlabel("Date")
            self.ax1.set_ylabel("Count")
            self.ax1.tick_params(axis='x', rotation=45)
            
            self.ax2.plot(dates, defect_rates, 'r-o', label='Defect Rate')
            self.ax2.set_title("Defect Rate")
            self.ax2.set_xlabel("Date")
            self.ax2.set_ylabel("Rate (%)")
            self.ax2.tick_params(axis='x', rotation=45)
            
            self.ax3.plot(dates, processing_times, 'g-o', label='Processing Time')
            self.ax3.set_title("Processing Time")
            self.ax3.set_xlabel("Date")
            self.ax3.set_ylabel("Time (s)")
            self.ax3.tick_params(axis='x', rotation=45)
            
            self.ax4.plot(dates, focus_scores, 'm-o', label='Focus Score')
            self.ax4.set_title("Focus Score")
            self.ax4.set_xlabel("Date")
            self.ax4.set_ylabel("Score")
            self.ax4.tick_params(axis='x', rotation=45)
        
        self.overview_canvas.draw()
    
    def _update_defects_charts(self, data: Dict[str, Any]):
        """Update defects charts."""
        defect_metrics = data.get('defect_metrics', [])
        
        if defect_metrics:
            # Defect distribution pie chart
            defect_types = [metric['defect_type'] for metric in defect_metrics]
            defect_counts = [metric['total_count'] for metric in defect_metrics]
            
            if sum(defect_counts) > 0:
                self.ax_defects1.pie(defect_counts, labels=defect_types, autopct='%1.1f%%')
                self.ax_defects1.set_title("Defect Distribution")
            
            # Defect trends
            defect_trends = data.get('defect_trends', {})
            if defect_trends:
                for defect_type, trend_data in defect_trends.items():
                    dates = [datetime.fromisoformat(point['date']) for point in trend_data]
                    counts = [point['count'] for point in trend_data]
                    self.ax_defects2.plot(dates, counts, label=defect_type, marker='o')
                
                self.ax_defects2.set_title("Defect Trends")
                self.ax_defects2.set_xlabel("Date")
                self.ax_defects2.set_ylabel("Count")
                self.ax_defects2.legend()
                self.ax_defects2.tick_params(axis='x', rotation=45)
        
        self.defects_canvas.draw()
    
    def _update_performance_charts(self, data: Dict[str, Any]):
        """Update performance charts."""
        performance_metrics = data.get('performance_metrics', {})
        
        if performance_metrics:
            # Processing time breakdown
            components = ['Inference', 'Preprocessing', 'Postprocessing']
            times = [
                performance_metrics.get('avg_inference_time', 0),
                performance_metrics.get('avg_preprocessing_time', 0),
                performance_metrics.get('avg_total_time', 0) - performance_metrics.get('avg_inference_time', 0)
            ]
            
            self.ax_perf1.bar(components, times)
            self.ax_perf1.set_title("Processing Time Breakdown")
            self.ax_perf1.set_ylabel("Time (s)")
            
            # System performance metrics
            metrics = ['FPS', 'Memory (MB)', 'GPU Util (%)']
            values = [
                performance_metrics.get('fps', 0),
                performance_metrics.get('memory_usage_mb', 0),
                performance_metrics.get('gpu_utilization', 0)
            ]
            
            self.ax_perf2.bar(metrics, values)
            self.ax_perf2.set_title("System Performance")
        
        self.performance_canvas.draw()
    
    def _update_quality_charts(self, data: Dict[str, Any]):
        """Update quality charts."""
        quality_metrics = data.get('quality_metrics', {})
        
        if quality_metrics:
            # Focus score distribution
            focus_dist = quality_metrics.get('focus_score_distribution', {})
            if focus_dist:
                categories = list(focus_dist.keys())
                counts = list(focus_dist.values())
                
                self.ax_quality1.bar(categories, counts)
                self.ax_quality1.set_title("Focus Score Distribution")
                self.ax_quality1.set_ylabel("Count")
                self.ax_quality1.tick_params(axis='x', rotation=45)
            
            # Quality trends
            quality_trends = data.get('quality_trends', [])
            if quality_trends:
                dates = [datetime.fromisoformat(trend['date']) for trend in quality_trends]
                pass_rates = [trend['pass_rate'] * 100 for trend in quality_trends]
                
                self.ax_quality2.plot(dates, pass_rates, 'g-o', label='Pass Rate')
                self.ax_quality2.set_title("Quality Trends")
                self.ax_quality2.set_xlabel("Date")
                self.ax_quality2.set_ylabel("Pass Rate (%)")
                self.ax_quality2.tick_params(axis='x', rotation=45)
        
        self.quality_canvas.draw()
    
    def _update_statistics_text(self, data: Dict[str, Any]):
        """Update statistics text display."""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        # Format statistics
        stats_text = "=== PCB INSPECTION ANALYTICS REPORT ===\n\n"
        
        # Overview statistics
        inspection_metrics = data.get('inspection_metrics', {})
        if inspection_metrics:
            stats_text += f"Total Inspections: {inspection_metrics.get('total_inspections', 0)}\n"
            stats_text += f"Defect Rate: {inspection_metrics.get('defect_rate', 0):.1%}\n"
            stats_text += f"Average Processing Time: {inspection_metrics.get('avg_processing_time', 0):.3f}s\n"
            stats_text += f"Average Focus Score: {inspection_metrics.get('avg_focus_score', 0):.1f}\n"
            stats_text += f"Throughput: {inspection_metrics.get('throughput_per_hour', 0):.1f} inspections/hour\n\n"
        
        # Top defects
        defect_metrics = data.get('defect_metrics', [])
        if defect_metrics:
            stats_text += "Top Defects:\n"
            for i, metric in enumerate(defect_metrics[:5], 1):
                stats_text += f"  {i}. {metric['defect_type']}: {metric['total_count']} ({metric['percentage']:.1f}%)\n"
            stats_text += "\n"
        
        # System health
        system_health = data.get('system_health', {})
        if system_health:
            stats_text += f"System Health: {system_health.get('status', 'Unknown').upper()}\n"
            stats_text += f"Health Score: {system_health.get('score', 0):.1f}/100\n"
            
            issues = system_health.get('issues', [])
            if issues:
                stats_text += "Issues:\n"
                for issue in issues:
                    stats_text += f"  - {issue}\n"
        
        self.stats_text.insert(tk.END, stats_text)
        self.stats_text.config(state=tk.DISABLED)
    
    def _on_period_change(self, event):
        """Handle period change."""
        self._refresh_analytics()
    
    def _refresh_analytics(self):
        """Refresh analytics data."""
        if self.analyzer:
            try:
                period = self.period_var.get()
                analytics_data = self.analyzer.get_time_period_analysis(period)
                self.update_analytics(analytics_data)
            except Exception as e:
                self.logger.error(f"Error refreshing analytics: {str(e)}")
                self.status_label.config(text="Error refreshing data")
    
    def _export_report(self):
        """Export analytics report."""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                if self.analyzer:
                    period = self.period_var.get()
                    if filename.endswith('.html'):
                        report = self.analyzer.generate_report(period, 'html')
                    else:
                        report = self.analyzer.generate_report(period, 'json')
                    
                    with open(filename, 'w') as f:
                        if isinstance(report, dict):
                            import json
                            json.dump(report, f, indent=2)
                        else:
                            f.write(report)
                    
                    self.status_label.config(text=f"Report exported to {filename}")
                else:
                    self.status_label.config(text="No analyzer available")
        
        except Exception as e:
            self.logger.error(f"Error exporting report: {str(e)}")
            self.status_label.config(text="Error exporting report")
    
    def set_analyzer(self, analyzer):
        """Set the analyzer instance."""
        self.analyzer = analyzer
        self._refresh_analytics()
    
    def show(self):
        """Show the analytics window."""
        self.window.deiconify()
        self.window.focus_set()
    
    def hide(self):
        """Hide the analytics window."""
        self.window.withdraw()
    
    def destroy(self):
        """Destroy the analytics window."""
        self.window.destroy()


# Test analytics viewer
if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create test analytics viewer
    viewer = AnalyticsViewer()
    
    # Test with sample data
    sample_data = {
        'daily_trends': [
            {'date': '2024-01-15', 'total_inspections': 25, 'defect_rate': 0.1, 'avg_processing_time': 0.05, 'avg_focus_score': 150},
            {'date': '2024-01-16', 'total_inspections': 30, 'defect_rate': 0.08, 'avg_processing_time': 0.048, 'avg_focus_score': 155},
            {'date': '2024-01-17', 'total_inspections': 28, 'defect_rate': 0.12, 'avg_processing_time': 0.052, 'avg_focus_score': 148}
        ],
        'defect_metrics': [
            {'defect_type': 'Missing Hole', 'total_count': 15, 'percentage': 35.0},
            {'defect_type': 'Open Circuit', 'total_count': 12, 'percentage': 28.0},
            {'defect_type': 'Spur', 'total_count': 8, 'percentage': 19.0}
        ],
        'performance_metrics': {
            'avg_inference_time': 0.03,
            'avg_preprocessing_time': 0.015,
            'avg_total_time': 0.05,
            'fps': 20,
            'memory_usage_mb': 1500,
            'gpu_utilization': 75
        },
        'quality_metrics': {
            'focus_score_distribution': {'excellent': 40, 'good': 30, 'acceptable': 20, 'poor': 8, 'very_poor': 2},
        },
        'inspection_metrics': {
            'total_inspections': 100,
            'defect_rate': 0.1,
            'avg_processing_time': 0.05,
            'avg_focus_score': 150,
            'throughput_per_hour': 720
        },
        'system_health': {
            'status': 'good',
            'score': 85,
            'issues': ['Low focus quality detected']
        }
    }
    
    viewer.update_analytics(sample_data)
    viewer.window.mainloop()
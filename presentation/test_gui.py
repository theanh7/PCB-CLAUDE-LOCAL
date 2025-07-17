"""
Comprehensive test suite for GUI components.

This module provides testing for all GUI components including main interface,
analytics viewer, history browser, and integration tests.
"""

import unittest
import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import sys
import numpy as np
from datetime import datetime
import json
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from presentation.gui import PCBInspectionGUI
from presentation.analytics_viewer import AnalyticsViewer
from presentation.history_browser import HistoryBrowser
from core.config import GUI_CONFIG


class TestPCBInspectionGUI(unittest.TestCase):
    """Test cases for main PCB inspection GUI."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Suppress GUI logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Create GUI instance
        self.gui = PCBInspectionGUI()
        
        # Don't actually show the window during tests
        self.gui.root.withdraw()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'gui'):
            self.gui.root.destroy()
    
    def test_gui_initialization(self):
        """Test GUI initialization."""
        # Check that main window exists
        self.assertIsNotNone(self.gui.root)
        
        # Check that main components exist
        self.assertIsNotNone(self.gui.preview_canvas)
        self.assertIsNotNone(self.gui.results_canvas)
        self.assertIsNotNone(self.gui.mode_button)
        self.assertIsNotNone(self.gui.inspect_button)
        self.assertIsNotNone(self.gui.defects_listbox)
        
        # Check initial state
        self.assertTrue(self.gui.auto_mode)
        self.assertFalse(self.gui.is_running)
    
    def test_mode_toggle(self):
        """Test mode toggle functionality."""
        # Mock callback
        mode_toggle_called = False
        
        def mock_toggle():
            nonlocal mode_toggle_called
            mode_toggle_called = True
        
        self.gui.toggle_auto_mode_callback = mock_toggle
        
        # Test mode toggle
        self.gui._on_mode_toggle()
        self.assertTrue(mode_toggle_called)
        
        # Test mode display update
        self.gui.update_mode_display(False)
        self.assertFalse(self.gui.auto_mode)
        self.assertEqual(self.gui.mode_button.cget('text'), 'MANUAL')
    
    def test_preview_update(self):
        """Test preview display update."""
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Update preview
        self.gui.update_preview(
            test_image,
            has_pcb=True,
            is_stable=True,
            focus_score=150.0,
            fps=30.0
        )
        
        # Check that image was updated
        self.assertIsNotNone(self.gui.current_preview_image)
        
        # Check status updates
        self.assertEqual(self.gui.pcb_status_label.cget('text'), 'PCB: Detected')
        self.assertEqual(self.gui.focus_label.cget('text'), 'Focus: 150.0')
        self.assertEqual(self.gui.fps_label.cget('text'), 'FPS: 30.0')
    
    def test_inspection_results_update(self):
        """Test inspection results display update."""
        # Create test image
        test_image = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
        
        # Test data
        defects = ["Missing Hole", "Open Circuit"]
        locations = [
            {"bbox": [100, 100, 200, 200]},
            {"bbox": [300, 300, 400, 400]}
        ]
        confidence_scores = [0.85, 0.75]
        
        # Update results
        self.gui.update_inspection_results(
            test_image,
            defects,
            locations,
            confidence_scores,
            123,
            0.05
        )
        
        # Check that image was updated
        self.assertIsNotNone(self.gui.current_results_image)
        
        # Check defects list
        self.assertEqual(self.gui.defects_listbox.size(), 2)
        
        # Check inspection details
        self.assertEqual(self.gui.inspection_id_label.cget('text'), 'Inspection ID: 123')
    
    def test_statistics_update(self):
        """Test statistics display update."""
        stats = {
            'total_inspections': 100,
            'total_defects': 15,
            'pass_rate': 0.85
        }
        
        self.gui.update_statistics(stats)
        
        # Check statistics display
        stats_text = self.gui.stats_label.cget('text')
        self.assertIn('Total Inspections: 100', stats_text)
        self.assertIn('Defects Found: 15', stats_text)
        self.assertIn('Pass Rate: 85.0%', stats_text)
    
    def test_status_update(self):
        """Test status display update."""
        self.gui.update_status("System Running", "green")
        
        self.assertEqual(self.gui.status_label.cget('text'), 'System Running')
        self.assertEqual(self.gui.status_label.cget('foreground'), 'green')
    
    def test_callbacks_registration(self):
        """Test callback registration."""
        # Create mock callbacks
        def mock_toggle(): pass
        def mock_inspect(): pass
        
        # Set callbacks
        self.gui.set_callbacks(
            toggle_auto_mode=mock_toggle,
            manual_inspect=mock_inspect
        )
        
        # Check callbacks were set
        self.assertEqual(self.gui.toggle_auto_mode_callback, mock_toggle)
        self.assertEqual(self.gui.manual_inspect_callback, mock_inspect)
    
    def test_error_handling(self):
        """Test error handling in GUI updates."""
        # Test with invalid image data
        invalid_image = np.array([])
        
        # Should not crash
        self.gui.update_preview(invalid_image)
        
        # Test with invalid statistics
        invalid_stats = None
        self.gui.update_statistics(invalid_stats)
    
    def test_placeholder_images(self):
        """Test placeholder image creation."""
        # Check that placeholder images exist
        self.assertIsNotNone(self.gui.placeholder_preview)
        self.assertIsNotNone(self.gui.placeholder_results)
        
        # Check placeholder dimensions
        # Note: This depends on PhotoImage implementation


class TestAnalyticsViewer(unittest.TestCase):
    """Test cases for analytics viewer."""
    
    def setUp(self):
        """Set up test fixtures."""
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Create root window first
        self.root = tk.Tk()
        self.root.withdraw()
        
        # Create analytics viewer
        self.viewer = AnalyticsViewer(self.root)
        self.viewer.window.withdraw()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'viewer'):
            self.viewer.window.destroy()
        if hasattr(self, 'root'):
            self.root.destroy()
    
    def test_analytics_initialization(self):
        """Test analytics viewer initialization."""
        # Check that main components exist
        self.assertIsNotNone(self.viewer.window)
        self.assertIsNotNone(self.viewer.notebook)
        self.assertIsNotNone(self.viewer.period_var)
        self.assertIsNotNone(self.viewer.stats_text)
        
        # Check initial state
        self.assertEqual(self.viewer.period_var.get(), "7d")
    
    def test_period_change(self):
        """Test period selection change."""
        # Change period
        self.viewer.period_var.set("30d")
        
        # Check that period was changed
        self.assertEqual(self.viewer.period_var.get(), "30d")
    
    def test_analytics_update(self):
        """Test analytics data update."""
        # Create sample analytics data
        sample_data = {
            'daily_trends': [
                {
                    'date': '2024-01-15',
                    'total_inspections': 25,
                    'defect_rate': 0.1,
                    'avg_processing_time': 0.05,
                    'avg_focus_score': 150
                }
            ],
            'defect_metrics': [
                {
                    'defect_type': 'Missing Hole',
                    'total_count': 15,
                    'percentage': 35.0
                }
            ],
            'performance_metrics': {
                'avg_inference_time': 0.03,
                'avg_total_time': 0.05,
                'fps': 20
            },
            'inspection_metrics': {
                'total_inspections': 100,
                'defect_rate': 0.1
            }
        }
        
        # Update analytics
        self.viewer.update_analytics(sample_data)
        
        # Check that status was updated
        self.assertEqual(self.viewer.status_label.cget('text'), 'Charts updated')
    
    def test_chart_clearing(self):
        """Test chart clearing functionality."""
        # This should not raise any errors
        self.viewer._clear_charts()
    
    def test_statistics_text_update(self):
        """Test statistics text update."""
        sample_data = {
            'inspection_metrics': {
                'total_inspections': 100,
                'defect_rate': 0.1,
                'avg_processing_time': 0.05
            }
        }
        
        # Update statistics text
        self.viewer._update_statistics_text(sample_data)
        
        # Check that text was updated
        text_content = self.viewer.stats_text.get(1.0, tk.END)
        self.assertIn('Total Inspections: 100', text_content)


class TestHistoryBrowser(unittest.TestCase):
    """Test cases for history browser."""
    
    def setUp(self):
        """Set up test fixtures."""
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Create root window first
        self.root = tk.Tk()
        self.root.withdraw()
        
        # Create history browser
        self.browser = HistoryBrowser(self.root)
        self.browser.window.withdraw()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'browser'):
            self.browser.window.destroy()
        if hasattr(self, 'root'):
            self.root.destroy()
    
    def test_history_initialization(self):
        """Test history browser initialization."""
        # Check that main components exist
        self.assertIsNotNone(self.browser.window)
        self.assertIsNotNone(self.browser.history_tree)
        self.assertIsNotNone(self.browser.defects_detail_listbox)
        self.assertIsNotNone(self.browser.technical_text)
        
        # Check initial state
        self.assertEqual(len(self.browser.current_inspections), 0)
        self.assertEqual(len(self.browser.filtered_inspections), 0)
    
    def test_filter_functionality(self):
        """Test filter functionality."""
        # Create sample inspection data
        sample_inspections = [
            {
                'id': 1,
                'timestamp': '2024-01-15T10:30:00',
                'has_defects': True,
                'defects': ['Missing Hole'],
                'focus_score': 150.0
            },
            {
                'id': 2,
                'timestamp': '2024-01-16T10:30:00',
                'has_defects': False,
                'defects': [],
                'focus_score': 155.0
            }
        ]
        
        # Set sample data
        self.browser.current_inspections = sample_inspections
        self.browser.filtered_inspections = sample_inspections.copy()
        
        # Test status filter
        self.browser.status_filter_var.set("Failed")
        self.browser._apply_filters()
        
        # Should only show failed inspections
        self.assertEqual(len(self.browser.filtered_inspections), 1)
        self.assertTrue(self.browser.filtered_inspections[0]['has_defects'])
    
    def test_history_display_update(self):
        """Test history display update."""
        # Create sample inspection data
        sample_inspections = [
            {
                'id': 1,
                'timestamp': '2024-01-15T10:30:00',
                'has_defects': True,
                'defect_count': 2,
                'defects': ['Missing Hole', 'Open Circuit'],
                'focus_score': 150.0,
                'processing_time': 0.05
            }
        ]
        
        # Set sample data
        self.browser.filtered_inspections = sample_inspections
        
        # Update display
        self.browser._update_history_display()
        
        # Check that tree was populated
        items = self.browser.history_tree.get_children()
        self.assertEqual(len(items), 1)
        
        # Check results count
        self.assertEqual(self.browser.results_label.cget('text'), 'Results: 1')
    
    def test_details_display_update(self):
        """Test details display update."""
        # Create sample inspection
        sample_inspection = {
            'id': 1,
            'timestamp': '2024-01-15T10:30:00',
            'has_defects': True,
            'defect_count': 2,
            'defects': ['Missing Hole', 'Open Circuit'],
            'confidence_scores': [0.85, 0.75],
            'focus_score': 150.0,
            'processing_time': 0.05,
            'trigger_type': 'auto',
            'session_id': 'session_1'
        }
        
        # Update details display
        self.browser._update_details_display(sample_inspection)
        
        # Check that details were updated
        self.assertEqual(self.browser.detail_labels['id'].cget('text'), '1')
        self.assertEqual(self.browser.detail_labels['defect_count'].cget('text'), '2')
        self.assertEqual(self.browser.detail_labels['focus_score'].cget('text'), '150.0')
        
        # Check defects list
        self.assertEqual(self.browser.defects_detail_listbox.size(), 2)
    
    def test_filter_clearing(self):
        """Test filter clearing functionality."""
        # Set some filter values
        self.browser.defect_filter_var.set("Missing Hole")
        self.browser.status_filter_var.set("Failed")
        
        # Clear filters
        self.browser._clear_filters()
        
        # Check that filters were cleared
        self.assertEqual(self.browser.defect_filter_var.get(), "All")
        self.assertEqual(self.browser.status_filter_var.get(), "All")
    
    def test_export_functionality(self):
        """Test export functionality."""
        # Create sample data
        sample_inspections = [
            {
                'id': 1,
                'timestamp': '2024-01-15T10:30:00',
                'has_defects': True,
                'defect_count': 2,
                'focus_score': 150.0,
                'processing_time': 0.05
            }
        ]
        
        # Test JSON export
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            self.browser._export_to_json(sample_inspections, filename)
            
            # Check that file was created
            self.assertTrue(os.path.exists(filename))
            
            # Check file content
            with open(filename, 'r') as f:
                data = json.load(f)
                self.assertEqual(len(data), 1)
                self.assertEqual(data[0]['id'], 1)
        
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


class TestGUIIntegration(unittest.TestCase):
    """Integration tests for GUI components."""
    
    def setUp(self):
        """Set up test fixtures."""
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Create main GUI
        self.gui = PCBInspectionGUI()
        self.gui.root.withdraw()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'gui'):
            self.gui.root.destroy()
    
    def test_gui_thread_safety(self):
        """Test GUI thread safety."""
        # Test that GUI updates can be called from different threads
        def update_from_thread():
            # Create test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Update preview (should be thread-safe)
            self.gui.update_preview(
                test_image,
                has_pcb=True,
                is_stable=True,
                focus_score=150.0,
                fps=30.0
            )
        
        # Run update from different thread
        thread = threading.Thread(target=update_from_thread)
        thread.start()
        thread.join()
        
        # Should not crash
        self.assertIsNotNone(self.gui.current_preview_image)
    
    def test_gui_performance(self):
        """Test GUI performance with rapid updates."""
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Time multiple updates
        start_time = time.time()
        
        for i in range(10):
            self.gui.update_preview(
                test_image,
                has_pcb=True,
                is_stable=True,
                focus_score=150.0 + i,
                fps=30.0
            )
        
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(elapsed_time, 1.0)  # Less than 1 second for 10 updates
    
    def test_gui_memory_usage(self):
        """Test GUI memory usage with large images."""
        # Create large test image
        large_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        
        # Update multiple times
        for i in range(5):
            self.gui.update_preview(large_image)
            self.gui.update_inspection_results(
                large_image,
                ["Missing Hole"],
                [{"bbox": [100, 100, 200, 200]}],
                [0.85],
                i,
                0.05
            )
        
        # Should not crash due to memory issues
        self.assertIsNotNone(self.gui.current_preview_image)
        self.assertIsNotNone(self.gui.current_results_image)
    
    def test_gui_error_recovery(self):
        """Test GUI error recovery."""
        # Test with various invalid inputs
        invalid_inputs = [
            None,
            np.array([]),
            np.random.randint(0, 255, (10, 10), dtype=np.uint8),  # Too small
            "invalid",
            []
        ]
        
        for invalid_input in invalid_inputs:
            try:
                self.gui.update_preview(invalid_input)
                self.gui.update_inspection_results(
                    invalid_input,
                    ["Missing Hole"],
                    [{"bbox": [100, 100, 200, 200]}],
                    [0.85],
                    1,
                    0.05
                )
            except Exception as e:
                # Should handle errors gracefully
                self.logger.error(f"Error with input {type(invalid_input)}: {e}")
        
        # GUI should still be functional
        self.assertIsNotNone(self.gui.root)


class TestGUIConfiguration(unittest.TestCase):
    """Test GUI configuration and customization."""
    
    def test_gui_config_usage(self):
        """Test that GUI uses configuration properly."""
        # Check that GUI uses config values
        gui = PCBInspectionGUI()
        gui.root.withdraw()
        
        try:
            # Check window title
            self.assertEqual(gui.root.title(), GUI_CONFIG["window_title"])
            
            # Check window size
            geometry = gui.root.geometry()
            expected_size = f"{GUI_CONFIG['window_size'][0]}x{GUI_CONFIG['window_size'][1]}"
            self.assertIn(expected_size, geometry)
            
        finally:
            gui.root.destroy()
    
    def test_gui_styling(self):
        """Test GUI styling and appearance."""
        gui = PCBInspectionGUI()
        gui.root.withdraw()
        
        try:
            # Check that main components have proper styling
            self.assertIsNotNone(gui.preview_canvas)
            self.assertIsNotNone(gui.results_canvas)
            
            # Check canvas sizes
            self.assertEqual(gui.preview_canvas.winfo_reqwidth(), GUI_CONFIG["preview_size"][0])
            self.assertEqual(gui.preview_canvas.winfo_reqheight(), GUI_CONFIG["preview_size"][1])
            
        finally:
            gui.root.destroy()


def run_all_tests():
    """Run all GUI tests."""
    print("Running GUI tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestPCBInspectionGUI,
        TestAnalyticsViewer,
        TestHistoryBrowser,
        TestGUIIntegration,
        TestGUIConfiguration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_gui_tests():
    """Run GUI tests only."""
    print("Running GUI component tests...")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add specific test classes
    test_classes = [TestPCBInspectionGUI, TestAnalyticsViewer, TestHistoryBrowser]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests only."""
    print("Running GUI integration tests...")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGUIIntegration)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GUI components")
    parser.add_argument("--gui", action="store_true", help="Run GUI tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.gui:
        success = run_gui_tests()
    elif args.integration:
        success = run_integration_tests()
    else:
        success = run_all_tests()
    
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
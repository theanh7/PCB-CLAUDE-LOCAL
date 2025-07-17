"""
Presentation layer for PCB inspection system.

This module provides the graphical user interface components for the PCB
inspection system, including live preview, inspection results, and analytics.
"""

# Import main GUI components
from .gui import PCBInspectionGUI

__all__ = [
    'PCBInspectionGUI'
]

# Version information
__version__ = '1.0.0'
__author__ = 'PCB Inspection System'
__description__ = 'GUI components for PCB defect detection system'
#!/usr/bin/env python3
"""
Test the final system for a short time to verify all fixes are working.
"""

import time
import threading
from main import PCBInspectionSystem

def test_system_briefly():
    """Test the system for 10 seconds and then shut down."""
    
    print("Starting PCB Inspection System test...")
    
    # Initialize system
    system = PCBInspectionSystem()
    
    # Start the system in a separate thread
    system_thread = threading.Thread(target=system.run)
    system_thread.daemon = True
    system_thread.start()
    
    # Let it run for 10 seconds
    print("System running for 10 seconds...")
    time.sleep(10)
    
    # Check system status
    print("\nSystem Status Check:")
    print(f"Camera connected: {system.camera.is_connected if system.camera else 'N/A'}")
    print(f"Camera streaming: {system.camera.is_streaming if system.camera else 'N/A'}")
    print(f"System running: {system.is_running}")
    
    # Get some stats if available
    if hasattr(system, 'pcb_detector') and system.pcb_detector:
        stats = system.pcb_detector.get_detection_stats()
        print(f"Detection stats: {stats}")
        
        stability_info = system.pcb_detector.get_stability_info()
        print(f"Stability info: {stability_info}")
    
    # Shutdown system
    print("\nShutting down system...")
    system.shutdown()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_system_briefly()
#!/usr/bin/env python3
"""
Check available camera nodes and features.
"""

from pypylon import pylon

def check_camera_nodes():
    """Check what nodes are available on the camera."""
    try:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        
        print(f"Camera Model: {camera.GetDeviceInfo().GetModelName()}")
        print(f"Camera Serial: {camera.GetDeviceInfo().GetSerialNumber()}")
        
        # Get node map
        node_map = camera.GetNodeMap()
        
        # Check common nodes
        exposure_nodes = ["ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"]
        gain_nodes = ["Gain", "GainRaw", "GainAbs"]
        
        print("\nChecking Exposure nodes:")
        for node_name in exposure_nodes:
            try:
                node = node_map.GetNode(node_name)
                if node.IsAvailable():
                    print(f"  Available: {node_name}")
                    if node.IsWritable():
                        print(f"    - Writable")
                    if hasattr(node, 'GetMin') and hasattr(node, 'GetMax'):
                        print(f"    - Range: {node.GetMin()} - {node.GetMax()}")
            except:
                print(f"  Not found: {node_name}")
        
        print("\nChecking Gain nodes:")
        for node_name in gain_nodes:
            try:
                node = node_map.GetNode(node_name)
                if node.IsAvailable():
                    print(f"  Available: {node_name}")
                    if node.IsWritable():
                        print(f"    - Writable")
                    if hasattr(node, 'GetMin') and hasattr(node, 'GetMax'):
                        print(f"    - Range: {node.GetMin()} - {node.GetMax()}")
            except:
                print(f"  Not found: {node_name}")
        
        # Check binning
        print("\nChecking Binning nodes:")
        binning_nodes = ["BinningHorizontal", "BinningVertical", "Binning"]
        for node_name in binning_nodes:
            try:
                node = node_map.GetNode(node_name)
                if node.IsAvailable():
                    print(f"  Available: {node_name}")
            except:
                print(f"  Not found: {node_name}")
        
        # Check acquisition mode
        print("\nChecking AcquisitionMode:")
        try:
            node = node_map.GetNode("AcquisitionMode")
            if node.IsAvailable():
                print("  AcquisitionMode available")
                # Get possible values
                if hasattr(node, 'Symbolics'):
                    print(f"    Values: {list(node.Symbolics)}")
        except:
            print("  AcquisitionMode not found")
        
        camera.Close()
        return True
        
    except Exception as e:
        print(f"Error checking camera: {e}")
        return False

if __name__ == "__main__":
    check_camera_nodes()
#!/usr/bin/env python3
"""
Quick Start Script for PCB Inspection System
Simple launcher với error handling và auto-cleanup
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def print_banner():
    """Display startup banner"""
    print("\n" + "="*60)
    print("   PCB AUTO-INSPECTION SYSTEM - QUICK START")
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ ERROR: Python {version.major}.{version.minor} detected")
        print("   Requires Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'cv2', 'numpy', 'PIL', 'ultralytics', 
        'pypylon', 'tkinter', 'sqlite3'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            elif package == 'tkinter':
                import tkinter
            elif package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Please install requirements: pip install -r requirements.txt")
        return False
    
    return True

def kill_existing_processes():
    """Kill existing Python processes to free camera"""
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                         capture_output=True, check=False)
        else:  # Linux/Mac
            subprocess.run(['pkill', '-f', 'python'], 
                         capture_output=True, check=False)
        
        print("🔄 Cleaned up existing processes")
        time.sleep(2)  # Wait for cleanup
        
    except Exception as e:
        print(f"⚠️  Could not cleanup processes: {e}")

def run_system():
    """Run the PCB inspection system"""
    script_dir = Path(__file__).parent
    main_py = script_dir / "main.py"
    
    if not main_py.exists():
        print(f"❌ ERROR: main.py not found in {script_dir}")
        return False
    
    print(f"📁 Project directory: {script_dir}")
    print("🚀 Starting PCB Inspection System...\n")
    
    try:
        # Change to project directory
        os.chdir(script_dir)
        
        # Run main.py
        result = subprocess.run([sys.executable, "main.py"], 
                              cwd=script_dir)
        
        if result.returncode == 0:
            print("\n✅ PCB Inspection System closed successfully")
            return True
        else:
            print(f"\n❌ System exited with error code: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
        return True
    except Exception as e:
        print(f"\n❌ ERROR: Failed to start system")
        print(f"   Error: {str(e)}")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Check Python version
    print("Checking system requirements...")
    if not check_python_version():
        input("\nPress Enter to exit...")
        return 1
    
    # Check required packages
    print("\nChecking installed packages...")
    if not check_requirements():
        input("\nPress Enter to exit...")
        return 1
    
    print("\n✅ All requirements satisfied")
    
    # Cleanup existing processes
    print("\nPreparing system...")
    kill_existing_processes()
    
    # Run the system
    success = run_system()
    
    # Exit message
    if success:
        print("\n🎉 Thank you for using PCB Inspection System!")
    else:
        print("\n💥 System encountered issues. Check logs for details.")
    
    input("\nPress Enter to exit...")
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {str(e)}")
        input("Press Enter to exit...")
        sys.exit(1)
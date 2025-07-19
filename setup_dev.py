#!/usr/bin/env python3
"""
Development environment setup script for PCB inspection system.

This script automates the setup of development environment including:
- Python virtual environment creation
- Dependency installation with fallbacks
- Directory structure creation
- Configuration file setup
- Hardware check and validation
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup.log')
    ]
)
logger = logging.getLogger(__name__)


class DevEnvironmentSetup:
    """Development environment setup automation."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.system_info = self._get_system_info()
        self.python_executable = sys.executable
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for environment setup."""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "os_release": platform.release()
        }
    
    def print_banner(self):
        """Print setup banner."""
        print("=" * 80)
        print("🔧 PCB INSPECTION SYSTEM - DEVELOPMENT ENVIRONMENT SETUP")
        print("=" * 80)
        print(f"Python Version: {self.system_info['python_version']}")
        print(f"Platform: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"Project Root: {self.project_root}")
        print("=" * 80)
        print()
    
    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment."""
        logger.info("Creating virtual environment...")
        
        if self.venv_path.exists():
            logger.info("Virtual environment already exists")
            return True
        
        try:
            subprocess.run([
                self.python_executable, "-m", "venv", str(self.venv_path)
            ], check=True)
            logger.info(f"✅ Virtual environment created at {self.venv_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_venv_python(self) -> str:
        """Get virtual environment Python executable path."""
        if self.system_info["platform"] == "Windows":
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")
    
    def get_venv_pip(self) -> str:
        """Get virtual environment pip executable path."""
        if self.system_info["platform"] == "Windows":
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip in virtual environment."""
        logger.info("Upgrading pip...")
        
        try:
            subprocess.run([
                self.get_venv_python(), "-m", "pip", "install", "--upgrade", "pip"
            ], check=True)
            logger.info("✅ Pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to upgrade pip: {e}")
            return False
    
    def install_dependencies(self, mode: str = "dev") -> bool:
        """
        Install dependencies based on mode.
        
        Args:
            mode: "minimal", "test", "dev", or "full"
        """
        logger.info(f"Installing dependencies for {mode} mode...")
        
        requirements_files = {
            "minimal": ["requirements-test.txt"],
            "test": ["requirements-test.txt"],
            "dev": ["requirements-dev.txt"],
            "full": ["requirements.txt", "requirements-dev.txt"]
        }
        
        files_to_install = requirements_files.get(mode, ["requirements-test.txt"])
        
        for req_file in files_to_install:
            req_path = self.project_root / req_file
            if not req_path.exists():
                logger.warning(f"Requirements file not found: {req_file}")
                continue
                
            logger.info(f"Installing from {req_file}...")
            
            try:
                subprocess.run([
                    self.get_venv_pip(), "install", "-r", str(req_path)
                ], check=True)
                logger.info(f"✅ Installed dependencies from {req_file}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install from {req_file}: {e}")
                
                # Try essential packages individually
                if req_file == "requirements-test.txt":
                    logger.info("Attempting to install essential packages individually...")
                    essential_packages = [
                        "numpy", "opencv-python", "pillow", "pandas",
                        "pytest", "pytest-mock", "psutil"
                    ]
                    self._install_essential_packages(essential_packages)
                
                return False
        
        return True
    
    def _install_essential_packages(self, packages: List[str]) -> None:
        """Install essential packages individually with error handling."""
        for package in packages:
            try:
                subprocess.run([
                    self.get_venv_pip(), "install", package
                ], check=True)
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️ Failed to install {package}, skipping...")
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("Creating directory structure...")
        
        directories = [
            "data/images",
            "data/defects", 
            "logs",
            "weights",
            "tests/temp",
            "docs",
            ".vscode"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            
            logger.info("✅ Directory structure created")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def create_config_files(self) -> bool:
        """Create development configuration files."""
        logger.info("Creating configuration files...")
        
        try:
            # Create .env file for development
            env_content = """# Development Environment Configuration
DEBUG=true
LOG_LEVEL=DEBUG
DB_PATH=data/pcb_dev.db
CAMERA_MOCK=true
GPU_ENABLED=true
"""
            with open(self.project_root / ".env", "w") as f:
                f.write(env_content)
            
            # Create VSCode settings
            vscode_settings = {
                "python.defaultInterpreterPath": f"./venv/bin/python",
                "python.testing.pytestEnabled": True,
                "python.testing.pytestArgs": ["tests"],
                "python.linting.enabled": True,
                "python.linting.flake8Enabled": True,
                "editor.formatOnSave": True,
                "python.formatting.provider": "black"
            }
            
            vscode_dir = self.project_root / ".vscode"
            with open(vscode_dir / "settings.json", "w") as f:
                json.dump(vscode_settings, f, indent=2)
            
            logger.info("✅ Configuration files created")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create config files: {e}")
            return False
    
    def check_hardware_dependencies(self) -> Dict[str, bool]:
        """Check hardware-specific dependencies."""
        logger.info("Checking hardware dependencies...")
        
        checks = {
            "opencv": False,
            "camera_sdk": False,
            "cuda": False,
            "display": False
        }
        
        # Check OpenCV
        try:
            subprocess.run([
                self.get_venv_python(), "-c", "import cv2; print(cv2.__version__)"
            ], check=True, capture_output=True)
            checks["opencv"] = True
            logger.info("✅ OpenCV available")
        except:
            logger.warning("⚠️ OpenCV not available")
        
        # Check display (for GUI)
        if "DISPLAY" in os.environ or self.system_info["platform"] == "Windows":
            checks["display"] = True
            logger.info("✅ Display available")
        else:
            logger.warning("⚠️ No display detected (headless mode)")
        
        # Check CUDA (optional)
        try:
            subprocess.run([
                self.get_venv_python(), "-c", "import torch; print(torch.cuda.is_available())"
            ], check=True, capture_output=True)
            checks["cuda"] = True
            logger.info("✅ CUDA available")
        except:
            logger.info("ℹ️ CUDA not available (CPU mode)")
        
        return checks
    
    def run_tests(self) -> bool:
        """Run basic tests to verify setup."""
        logger.info("Running setup verification tests...")
        
        try:
            # Test basic imports
            test_script = '''
import sys
sys.path.append(".")
from tests.test_mocks import setup_test_environment, teardown_test_environment

# Setup test environment
setup_test_environment()

# Test core imports
try:
    from core.config import CAMERA_CONFIG, AI_CONFIG
    print("✅ Core imports successful")
except Exception as e:
    print(f"❌ Core imports failed: {e}")
    sys.exit(1)

# Test mock infrastructure
try:
    import cv2
    import numpy as np
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = cv2.resize(test_image, (50, 50))
    print("✅ Mock infrastructure working")
except Exception as e:
    print(f"❌ Mock infrastructure failed: {e}")
    sys.exit(1)

teardown_test_environment()
print("✅ All verification tests passed")
'''
            
            result = subprocess.run([
                self.get_venv_python(), "-c", test_script
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Setup verification tests passed")
                return True
            else:
                logger.error(f"❌ Setup verification failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to run verification tests: {e}")
            return False
    
    def setup(self, mode: str = "dev") -> bool:
        """Run complete setup process."""
        self.print_banner()
        
        steps = [
            ("Creating virtual environment", self.create_virtual_environment),
            ("Upgrading pip", self.upgrade_pip),
            ("Installing dependencies", lambda: self.install_dependencies(mode)),
            ("Creating directories", self.create_directories),
            ("Creating config files", self.create_config_files),
            ("Running verification tests", self.run_tests)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"🔄 {step_name}...")
            if not step_func():
                logger.error(f"❌ Setup failed at: {step_name}")
                return False
        
        # Check hardware dependencies (non-blocking)
        hardware_status = self.check_hardware_dependencies()
        
        # Print setup summary
        self.print_setup_summary(hardware_status)
        
        return True
    
    def print_setup_summary(self, hardware_status: Dict[str, bool]):
        """Print setup completion summary."""
        print()
        print("=" * 80)
        print("🎉 DEVELOPMENT ENVIRONMENT SETUP COMPLETED")
        print("=" * 80)
        print()
        print("📋 Setup Summary:")
        print(f"  ✅ Virtual environment: {self.venv_path}")
        print(f"  ✅ Python executable: {self.get_venv_python()}")
        print(f"  ✅ Project structure created")
        print(f"  ✅ Dependencies installed")
        print()
        
        print("🔧 Hardware Status:")
        for component, available in hardware_status.items():
            status = "✅" if available else "⚠️"
            print(f"  {status} {component.title()}: {'Available' if available else 'Not available'}")
        
        print()
        print("🚀 Next Steps:")
        print("  1. Activate virtual environment:")
        if self.system_info["platform"] == "Windows":
            print(f"     .\\venv\\Scripts\\activate")
        else:
            print(f"     source venv/bin/activate")
        print("  2. Run tests:")
        print("     python -m pytest tests/")
        print("  3. Start development:")
        print("     python main.py")
        print()
        print("📚 Documentation:")
        print("  - USER_MANUAL.md: User guide")
        print("  - API_DOCUMENTATION.md: Technical reference")
        print("  - DEPLOYMENT.md: Production deployment")
        print("=" * 80)


def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup PCB Inspection development environment")
    parser.add_argument(
        "--mode", 
        choices=["minimal", "test", "dev", "full"],
        default="dev",
        help="Installation mode (default: dev)"
    )
    
    args = parser.parse_args()
    
    setup = DevEnvironmentSetup()
    success = setup.setup(args.mode)
    
    if success:
        logger.info("✅ Setup completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
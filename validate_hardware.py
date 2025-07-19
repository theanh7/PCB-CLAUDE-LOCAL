#!/usr/bin/env python3
"""
Hardware validation script for PCB inspection system.

This script validates all hardware dependencies and system readiness
for production deployment.
"""

import sys
import os
import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HardwareValidator:
    """Hardware validation and system readiness checker."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "validations": {},
            "overall_status": "unknown"
        }
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get comprehensive system information."""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "os_release": platform.release(),
            "hostname": platform.node(),
            "processor": platform.processor()
        }
    
    def validate_python_environment(self) -> bool:
        """Validate Python environment and version."""
        logger.info("🐍 Validating Python environment...")
        
        try:
            version_info = sys.version_info
            python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
            
            # Check Python version compatibility
            min_version = (3, 8)
            max_version = (3, 11)
            
            if version_info[:2] < min_version:
                self.results["validations"]["python"] = {
                    "status": "fail",
                    "message": f"Python {python_version} too old, need >= 3.8",
                    "version": python_version
                }
                return False
            elif version_info[:2] > max_version:
                self.results["validations"]["python"] = {
                    "status": "warning",
                    "message": f"Python {python_version} newer than tested versions",
                    "version": python_version
                }
                logger.warning(f"⚠️ Python {python_version} newer than tested versions")
            else:
                self.results["validations"]["python"] = {
                    "status": "pass",
                    "message": f"Python {python_version} compatible",
                    "version": python_version
                }
                logger.info(f"✅ Python {python_version} compatible")
            
            return True
            
        except Exception as e:
            self.results["validations"]["python"] = {
                "status": "error",
                "message": f"Failed to validate Python: {str(e)}"
            }
            logger.error(f"❌ Python validation failed: {e}")
            return False
    
    def validate_dependencies(self) -> bool:
        """Validate critical dependencies."""
        logger.info("📦 Validating dependencies...")
        
        critical_deps = {
            "numpy": ">=1.24.0",
            "opencv-python": ">=4.8.0", 
            "pillow": ">=10.0.0",
            "pandas": ">=2.0.0"
        }
        
        optional_deps = {
            "torch": ">=2.0.0",
            "ultralytics": ">=8.0.0",
            "pypylon": ">=3.0.0"
        }
        
        dep_results = {}
        all_critical_ok = True
        
        # Check critical dependencies
        for dep_name, min_version in critical_deps.items():
            try:
                __import__(dep_name.replace('-', '_'))
                dep_results[dep_name] = {"status": "pass", "type": "critical"}
                logger.info(f"✅ {dep_name} available")
            except ImportError:
                dep_results[dep_name] = {
                    "status": "fail", 
                    "type": "critical",
                    "message": f"Missing critical dependency: {dep_name}"
                }
                logger.error(f"❌ Missing critical dependency: {dep_name}")
                all_critical_ok = False
        
        # Check optional dependencies
        for dep_name, min_version in optional_deps.items():
            try:
                if dep_name == "opencv-python":
                    import cv2
                elif dep_name == "pypylon":
                    from pypylon import pylon
                else:
                    __import__(dep_name)
                dep_results[dep_name] = {"status": "pass", "type": "optional"}
                logger.info(f"✅ {dep_name} available")
            except ImportError:
                dep_results[dep_name] = {
                    "status": "warning",
                    "type": "optional", 
                    "message": f"Optional dependency not available: {dep_name}"
                }
                logger.warning(f"⚠️ Optional dependency missing: {dep_name}")
        
        self.results["validations"]["dependencies"] = dep_results
        return all_critical_ok
    
    def validate_gpu(self) -> bool:
        """Validate GPU and CUDA availability."""
        logger.info("🎮 Validating GPU...")
        
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                self.results["validations"]["gpu"] = {
                    "status": "pass",
                    "cuda_available": True,
                    "device_count": device_count,
                    "gpu_name": gpu_name,
                    "memory_gb": round(gpu_memory_gb, 2)
                }
                
                logger.info(f"✅ GPU detected: {gpu_name}")
                logger.info(f"✅ GPU memory: {gpu_memory_gb:.1f} GB")
                
                # Check minimum memory requirement (4GB for Tesla P4)
                if gpu_memory_gb < 4.0:
                    logger.warning(f"⚠️ GPU memory ({gpu_memory_gb:.1f}GB) below recommended 4GB")
                
            else:
                self.results["validations"]["gpu"] = {
                    "status": "warning",
                    "cuda_available": False,
                    "message": "CUDA not available, will run on CPU"
                }
                logger.warning("⚠️ CUDA not available, will run on CPU")
            
            return True
            
        except ImportError:
            self.results["validations"]["gpu"] = {
                "status": "error",
                "message": "PyTorch not available for GPU validation"
            }
            logger.error("❌ PyTorch not available for GPU validation")
            return False
        except Exception as e:
            self.results["validations"]["gpu"] = {
                "status": "error",
                "message": f"GPU validation failed: {str(e)}"
            }
            logger.error(f"❌ GPU validation failed: {e}")
            return False
    
    def validate_camera(self) -> bool:
        """Validate camera hardware."""
        logger.info("📷 Validating camera...")
        
        try:
            # Try to import pypylon
            from pypylon import pylon
            
            # Try to get camera devices
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if len(devices) == 0:
                self.results["validations"]["camera"] = {
                    "status": "warning",
                    "message": "No cameras detected (may be mock mode)"
                }
                logger.warning("⚠️ No cameras detected")
                return True  # Allow mock mode
            
            # Try to connect to first camera
            camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            camera.Open()
            
            # Get camera info
            camera_info = {
                "model": camera.GetDeviceInfo().GetModelName(),
                "serial": camera.GetDeviceInfo().GetSerialNumber(),
                "vendor": camera.GetDeviceInfo().GetVendorName()
            }
            
            camera.Close()
            
            self.results["validations"]["camera"] = {
                "status": "pass",
                "camera_count": len(devices),
                "camera_info": camera_info
            }
            
            logger.info(f"✅ Camera detected: {camera_info['model']}")
            return True
            
        except ImportError:
            self.results["validations"]["camera"] = {
                "status": "warning",
                "message": "pypylon not available, camera validation skipped"
            }
            logger.warning("⚠️ pypylon not available, camera validation skipped")
            return True  # Allow development without camera
            
        except Exception as e:
            self.results["validations"]["camera"] = {
                "status": "error",
                "message": f"Camera validation failed: {str(e)}"
            }
            logger.error(f"❌ Camera validation failed: {e}")
            return False
    
    def validate_filesystem(self) -> bool:
        """Validate filesystem and permissions."""
        logger.info("💾 Validating filesystem...")
        
        required_dirs = [
            "data",
            "weights", 
            "logs",
            "tests"
        ]
        
        required_files = [
            "main.py",
            "requirements.txt",
            "CLAUDE.md"
        ]
        
        fs_results = {
            "directories": {},
            "files": {},
            "permissions": {}
        }
        
        # Check directories
        all_dirs_ok = True
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and dir_path.is_dir():
                fs_results["directories"][dir_name] = "exists"
                logger.info(f"✅ Directory exists: {dir_name}")
            else:
                fs_results["directories"][dir_name] = "missing"
                logger.warning(f"⚠️ Directory missing: {dir_name}")
                # Try to create
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"✅ Created directory: {dir_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to create directory {dir_name}: {e}")
                    all_dirs_ok = False
        
        # Check files
        all_files_ok = True
        for file_name in required_files:
            file_path = Path(file_name)
            if file_path.exists() and file_path.is_file():
                fs_results["files"][file_name] = "exists"
                logger.info(f"✅ File exists: {file_name}")
            else:
                fs_results["files"][file_name] = "missing"
                logger.error(f"❌ Required file missing: {file_name}")
                all_files_ok = False
        
        # Check write permissions
        try:
            test_file = Path("test_write_permission.tmp")
            test_file.write_text("test")
            test_file.unlink()
            fs_results["permissions"]["write"] = "ok"
            logger.info("✅ Write permissions OK")
        except Exception as e:
            fs_results["permissions"]["write"] = f"failed: {e}"
            logger.error(f"❌ Write permission test failed: {e}")
            all_files_ok = False
        
        self.results["validations"]["filesystem"] = fs_results
        return all_dirs_ok and all_files_ok
    
    def validate_system_resources(self) -> bool:
        """Validate system resources (CPU, Memory, Disk)."""
        logger.info("⚡ Validating system resources...")
        
        try:
            import psutil
            
            # CPU info
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory info
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # Disk info
            disk = psutil.disk_usage('.')
            disk_free_gb = disk.free / (1024**3)
            
            resources = {
                "cpu": {
                    "cores": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "usage_percent": cpu_percent
                },
                "memory": {
                    "total_gb": round(memory_gb, 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "usage_percent": memory.percent
                },
                "disk": {
                    "free_gb": round(disk_free_gb, 2),
                    "usage_percent": round((disk.used / disk.total) * 100, 1)
                }
            }
            
            # Check minimum requirements
            status = "pass"
            warnings = []
            
            if cpu_count < 4:
                warnings.append(f"CPU cores ({cpu_count}) below recommended 4+")
                status = "warning"
            
            if memory_gb < 16:
                warnings.append(f"Memory ({memory_gb:.1f}GB) below recommended 16GB")
                status = "warning"
            
            if disk_free_gb < 10:
                warnings.append(f"Disk space ({disk_free_gb:.1f}GB) below required 10GB")
                status = "warning"
            
            self.results["validations"]["system_resources"] = {
                "status": status,
                "resources": resources,
                "warnings": warnings
            }
            
            logger.info(f"✅ System resources: {cpu_count} cores, {memory_gb:.1f}GB RAM, {disk_free_gb:.1f}GB free")
            
            for warning in warnings:
                logger.warning(f"⚠️ {warning}")
            
            return status != "fail"
            
        except Exception as e:
            self.results["validations"]["system_resources"] = {
                "status": "error",
                "message": f"Resource validation failed: {str(e)}"
            }
            logger.error(f"❌ System resource validation failed: {e}")
            return False
    
    def run_validation(self) -> bool:
        """Run complete hardware validation."""
        logger.info("🔍 Starting hardware validation...")
        print("=" * 80)
        print("🔧 PCB INSPECTION SYSTEM - HARDWARE VALIDATION")
        print("=" * 80)
        
        validations = [
            ("Python Environment", self.validate_python_environment),
            ("Dependencies", self.validate_dependencies),
            ("GPU/CUDA", self.validate_gpu),
            ("Camera Hardware", self.validate_camera),
            ("Filesystem", self.validate_filesystem),
            ("System Resources", self.validate_system_resources)
        ]
        
        all_passed = True
        
        for validation_name, validation_func in validations:
            logger.info(f"Running {validation_name} validation...")
            try:
                result = validation_func()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {validation_name} validation crashed: {e}")
                all_passed = False
        
        # Determine overall status
        if all_passed:
            self.results["overall_status"] = "pass"
            logger.info("✅ All validations passed")
        else:
            self.results["overall_status"] = "fail"
            logger.error("❌ Some validations failed")
        
        return all_passed
    
    def generate_report(self) -> str:
        """Generate validation report."""
        report = f"""
# Hardware Validation Report

**Generated**: {self.results['timestamp']}  
**Status**: {self.results['overall_status'].upper()}

## System Information
- **Platform**: {self.results['system_info']['platform']} {self.results['system_info']['architecture']}
- **Python**: {self.results['system_info']['python_version']}
- **Hostname**: {self.results['system_info']['hostname']}

## Validation Results
"""
        
        for component, result in self.results["validations"].items():
            status_emoji = {
                "pass": "✅",
                "warning": "⚠️", 
                "fail": "❌",
                "error": "💥"
            }.get(result.get("status", "unknown"), "❓")
            
            report += f"\n### {component.title()}\n"
            report += f"**Status**: {status_emoji} {result.get('status', 'unknown').upper()}\n"
            
            if "message" in result:
                report += f"**Message**: {result['message']}\n"
            
            # Add specific details for each component
            if component == "gpu" and result.get("cuda_available"):
                report += f"**GPU**: {result.get('gpu_name', 'Unknown')}\n"
                report += f"**Memory**: {result.get('memory_gb', 0)} GB\n"
            
            if component == "camera" and result.get("camera_info"):
                info = result["camera_info"]
                report += f"**Model**: {info.get('model', 'Unknown')}\n"
                report += f"**Serial**: {info.get('serial', 'Unknown')}\n"
        
        return report
    
    def save_report(self, filename: str = "validation_report.md"):
        """Save validation report to file."""
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        # Also save JSON results
        json_filename = filename.replace('.md', '.json')
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"✅ Reports saved: {filename}, {json_filename}")


def main():
    """Main validation function."""
    validator = HardwareValidator()
    
    try:
        success = validator.run_validation()
        
        # Generate and save report
        validator.save_report()
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 VALIDATION SUMMARY")
        print("=" * 80)
        
        total_validations = len(validator.results["validations"])
        passed_validations = sum(
            1 for v in validator.results["validations"].values() 
            if v.get("status") == "pass"
        )
        
        print(f"Total validations: {total_validations}")
        print(f"Passed: {passed_validations}")
        print(f"Overall status: {validator.results['overall_status'].upper()}")
        
        if success:
            print("\n🎉 System ready for deployment!")
            return 0
        else:
            print("\n⚠️ System has issues that need attention.")
            print("Please check the validation report for details.")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
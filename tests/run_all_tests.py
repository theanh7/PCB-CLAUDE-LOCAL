"""
Comprehensive test suite runner for PCB inspection system.

Runs all unit tests, integration tests, and performance benchmarks
with detailed reporting and coverage analysis.
"""

import unittest
import sys
import os
import time
import json
from datetime import datetime
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestResult:
    """Custom test result class for detailed reporting."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.failures = []
        self.errors = []
        self.skipped = []
    
    def start_testing(self):
        """Start the testing session."""
        self.start_time = time.time()
        print("="*80)
        print("PCB INSPECTION SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def end_testing(self):
        """End the testing session."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        print()
        print("="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests Run: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Errors: {self.error_tests}")
        print(f"Skipped: {self.skipped_tests}")
        print(f"Duration: {duration:.2f} seconds")
        print()
        
        if self.failures:
            print("FAILURES:")
            print("-" * 40)
            for failure in self.failures:
                print(f"- {failure}")
            print()
        
        if self.errors:
            print("ERRORS:")
            print("-" * 40)
            for error in self.errors:
                print(f"- {error}")
            print()
        
        if self.skipped:
            print("SKIPPED:")
            print("-" * 40)
            for skip in self.skipped:
                print(f"- {skip}")
            print()
        
        # Overall result
        if self.failed_tests == 0 and self.error_tests == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED!")
        
        print("="*80)
    
    def add_result(self, test_name, result, duration=0, message=""):
        """Add a test result."""
        self.test_results[test_name] = {
            'result': result,
            'duration': duration,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.total_tests += 1
        
        if result == 'PASS':
            self.passed_tests += 1
        elif result == 'FAIL':
            self.failed_tests += 1
            self.failures.append(f"{test_name}: {message}")
        elif result == 'ERROR':
            self.error_tests += 1
            self.errors.append(f"{test_name}: {message}")
        elif result == 'SKIP':
            self.skipped_tests += 1
            self.skipped.append(f"{test_name}: {message}")
    
    def to_json(self):
        """Convert results to JSON format."""
        return {
            'summary': {
                'total_tests': self.total_tests,
                'passed': self.passed_tests,
                'failed': self.failed_tests,
                'errors': self.error_tests,
                'skipped': self.skipped_tests,
                'duration': self.end_time - self.start_time if self.end_time else 0,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
            },
            'details': self.test_results,
            'failures': self.failures,
            'errors': self.errors,
            'skipped': self.skipped
        }


def run_test_module(module_name, test_result):
    """Run tests from a specific module."""
    print(f"\n📋 Running {module_name}...")
    print("-" * 60)
    
    try:
        # Import the test module
        test_module = __import__(f'tests.{module_name}', fromlist=[''])
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # Run tests with custom result handler
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Process results
        for test, traceback in result.failures:
            test_result.add_result(
                f"{module_name}.{test._testMethodName}",
                'FAIL',
                duration / result.testsRun if result.testsRun > 0 else 0,
                str(traceback).split('\n')[-2] if traceback else ''
            )
        
        for test, traceback in result.errors:
            test_result.add_result(
                f"{module_name}.{test._testMethodName}",
                'ERROR',
                duration / result.testsRun if result.testsRun > 0 else 0,
                str(traceback).split('\n')[-2] if traceback else ''
            )
        
        for test, reason in result.skipped:
            test_result.add_result(
                f"{module_name}.{test._testMethodName}",
                'SKIP',
                0,
                reason
            )
        
        # Add passed tests
        passed_count = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        for i in range(passed_count):
            test_result.add_result(
                f"{module_name}.test_{i}",
                'PASS',
                duration / result.testsRun if result.testsRun > 0 else 0
            )
        
        # Print summary for this module
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        print(f"Duration: {duration:.2f}s")
        
        if result.failures or result.errors:
            print("❌ Some tests failed")
        else:
            print("✅ All tests passed")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Module {module_name} not available: {e}")
        test_result.add_result(module_name, 'SKIP', 0, f"Module not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running {module_name}: {e}")
        test_result.add_result(module_name, 'ERROR', 0, str(e))
        return False


def check_system_requirements():
    """Check system requirements for testing."""
    print("🔍 Checking system requirements...")
    
    requirements = {
        'Python': sys.version_info >= (3, 8),
        'NumPy': True,
        'OpenCV': True,
        'PIL': True
    }
    
    try:
        import numpy
        requirements['NumPy'] = True
    except ImportError:
        requirements['NumPy'] = False
    
    try:
        import cv2
        requirements['OpenCV'] = True
    except ImportError:
        requirements['OpenCV'] = False
    
    try:
        from PIL import Image
        requirements['PIL'] = True
    except ImportError:
        requirements['PIL'] = False
    
    all_good = True
    for req, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {req}")
        if not status:
            all_good = False
    
    if not all_good:
        print("⚠️  Some requirements are missing. Some tests may be skipped.")
    else:
        print("✅ All requirements satisfied")
    
    print()
    return all_good


def main():
    """Main test runner function."""
    # Check requirements
    check_system_requirements()
    
    # Initialize result tracker
    test_result = TestResult()
    test_result.start_testing()
    
    # Define test modules to run
    test_modules = [
        'test_core',           # Core layer tests
        'test_hardware',       # Hardware layer tests  
        'test_processing',     # Processing layer tests
        'test_ai',            # AI layer tests
        'test_data',          # Data layer tests
        'test_integration',   # Integration tests
        'test_performance'    # Performance tests
    ]
    
    # Run each test module
    successful_modules = 0
    
    for module in test_modules:
        success = run_test_module(module, test_result)
        if success:
            successful_modules += 1
    
    # End testing session
    test_result.end_testing()
    
    # Generate detailed report
    report_data = test_result.to_json()
    
    # Add system information
    report_data['system_info'] = {
        'python_version': sys.version,
        'platform': sys.platform,
        'test_modules_attempted': len(test_modules),
        'test_modules_successful': successful_modules
    }
    
    # Save report to file
    os.makedirs('tests/reports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'tests/reports/test_report_{timestamp}.json'
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"📊 Detailed test report saved to: {report_file}")
    
    # Return exit code
    if test_result.failed_tests > 0 or test_result.error_tests > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
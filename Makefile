# Makefile for PCB Inspection System Development
# Provides convenient commands for common development tasks

.PHONY: help setup test clean lint format run docs

# Default target
help:
	@echo "PCB Inspection System - Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup          - Setup development environment"
	@echo "  make setup-minimal  - Setup minimal environment (testing only)"
	@echo "  make setup-full     - Setup full environment with all dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint           - Run code linting"
	@echo "  make format         - Format code with black"
	@echo "  make run            - Run the main application"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  make clean          - Clean temporary files"
	@echo "  make docs           - Generate documentation"
	@echo "  make requirements   - Update requirements files"
	@echo ""

# Setup commands
setup:
	python setup_dev.py --mode dev

setup-minimal:
	python setup_dev.py --mode minimal

setup-full:
	python setup_dev.py --mode full

# Testing commands
test:
	@echo "Running comprehensive test suite..."
	./venv/bin/python -m pytest tests/ -v --cov=. --cov-report=html

test-unit:
	@echo "Running unit tests..."
	./venv/bin/python -m pytest tests/test_*.py -v

test-integration:
	@echo "Running integration tests..."
	./venv/bin/python -m pytest tests/test_integration.py -v

test-system:
	@echo "Running system integration test..."
	./venv/bin/python test_system_integration.py

# Code quality commands
lint:
	@echo "Running code linting..."
	./venv/bin/python -m flake8 . --max-line-length=100 --exclude=venv,__pycache__
	./venv/bin/python -m black --check .

format:
	@echo "Formatting code..."
	./venv/bin/python -m black .
	./venv/bin/python -m isort .

# Run commands
run:
	@echo "Starting PCB Inspection System..."
	./venv/bin/python main.py

run-mock:
	@echo "Starting with mock hardware..."
	CAMERA_MOCK=true ./venv/bin/python main.py

# Maintenance commands
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

docs:
	@echo "Generating documentation..."
	./venv/bin/python -c "
import os
print('📚 Available Documentation:')
docs = ['USER_MANUAL.md', 'API_DOCUMENTATION.md', 'SYSTEM_ARCHITECTURE.md', 'DEPLOYMENT.md']
for doc in docs:
    if os.path.exists(doc):
        print(f'  ✅ {doc}')
    else:
        print(f'  ❌ {doc} (missing)')
"

requirements:
	@echo "Updating requirements..."
	./venv/bin/python -m pip freeze > requirements-freeze.txt
	@echo "Requirements frozen to requirements-freeze.txt"

# Development helpers
install-dev-tools:
	@echo "Installing additional development tools..."
	./venv/bin/pip install pre-commit jupyter ipython

setup-pre-commit:
	@echo "Setting up pre-commit hooks..."
	./venv/bin/pre-commit install

# Hardware testing
test-camera:
	@echo "Testing camera connection..."
	./venv/bin/python -c "
from tests.test_mocks import setup_test_environment
setup_test_environment()
from hardware.test_camera import main
main()
"

# Monitoring and profiling
monitor:
	@echo "System monitoring..."
	./venv/bin/python -c "
import psutil
print(f'CPU Usage: {psutil.cpu_percent()}%')
print(f'Memory Usage: {psutil.virtual_memory().percent}%')
print(f'Disk Usage: {psutil.disk_usage(\".\").percent}%')
"

# Deployment preparation
build-requirements:
	@echo "Building requirements for deployment..."
	echo "# Production Requirements" > requirements-prod.txt
	echo "# Generated on $$(date)" >> requirements-prod.txt
	echo "" >> requirements-prod.txt
	./venv/bin/python -c "
import pkg_resources
essential = [
    'ultralytics', 'torch', 'torchvision', 'opencv-python', 
    'pillow', 'numpy', 'pandas', 'psutil', 'python-dateutil'
]
installed = [pkg.project_name for pkg in pkg_resources.working_set]
for pkg_name in essential:
    if pkg_name in installed:
        pkg = pkg_resources.get_distribution(pkg_name)
        print(f'{pkg.project_name}=={pkg.version}')
" >> requirements-prod.txt
	@echo "Production requirements saved to requirements-prod.txt"
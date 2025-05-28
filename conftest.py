"""Pytest configuration file."""
import os
import sys

from browser_use.logging_config import setup_logging

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Add test_framework to Python path
test_framework_path = os.path.join(project_root, "test_framework")
sys.path.insert(0, test_framework_path)

setup_logging()

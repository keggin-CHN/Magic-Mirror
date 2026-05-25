"""Shared test fixtures for Magic-Mirror test suite."""

import os
import sys

# Add src-python to path so we can import the magic package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src-python'))

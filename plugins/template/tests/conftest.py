"""Shared test fixtures and configuration for template plugin tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import fiftyone as fo

# Ensure plugin parent directory is in Python path for imports
PLUGIN_PARENT = Path(__file__).resolve().parents[2]
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))


@pytest.fixture
def fo_version():
    """Get the current FiftyOne version."""
    return fo.__version__


@pytest.fixture
def sample_dataset():
    """Create a minimal test dataset."""
    dataset = fo.Dataset("test_dataset")
    
    # Add a few samples with metadata (no actual files needed for basic tests)
    for i in range(3):
        sample = fo.Sample(
            filepath=f"test_image_{i}.jpg",  # Relative path, file doesn't need to exist
            tags=[f"test_{i}"],
        )
        dataset.add_sample(sample)
    
    yield dataset
    
    # Cleanup
    try:
        dataset.delete()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def empty_dataset():
    """Create an empty test dataset."""
    dataset = fo.Dataset("test_empty_dataset")
    yield dataset
    dataset.delete()


@pytest.fixture
def plugin_module():
    """Import and return the template plugin module."""
    from template import __init__ as plugin_module
    return plugin_module


# Cloud credentials fixtures - table for later
# These will be implemented when cloud media testing is added
# @pytest.fixture
# def cloud_credentials():
#     """Get cloud credentials from environment variables."""
#     import os
#     return {
#         "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
#         "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
#     }

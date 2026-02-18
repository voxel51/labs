"""FO Version Compatibility Tests for Template Plugin.

These tests ensure the plugin is compatible with the latest (or pinned) FO version
and that all plugin functionality works correctly.
"""

from __future__ import annotations

import pytest
import fiftyone as fo
import fiftyone.operators as foo


def test_plugin_imports():
    """Test that the plugin module can be imported successfully."""
    from template import TemplateOperator
    operator = TemplateOperator()

    assert operator is not None
    assert operator.config.name == "template_operator"
    assert operator.config.label == "Dummy Template Operator"

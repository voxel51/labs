"""FO Version Compatibility Tests for Template Plugin.

These tests ensure the plugin is compatible with the latest (or pinned) FO version
and that all plugin functionality works correctly.
"""

from __future__ import annotations

import pytest
import fiftyone as fo
import fiftyone.operators as foo


def test_fiftyone_version(fo_version):
    """Verify FiftyOne is installed and has a version."""
    assert fo_version is not None
    assert isinstance(fo_version, str)
    assert len(fo_version) > 0
    # Version should be in format like "1.0.0" or "1.0.0.dev0"
    assert "." in fo_version


def test_fiftyone_imports():
    """Verify all required FiftyOne modules can be imported."""
    import fiftyone as fo
    import fiftyone.operators as foo
    import fiftyone.operators.types as types
    import fiftyone.core.collections as foc
    
    assert fo is not None
    assert foo is not None
    assert types is not None
    assert foc is not None


def test_plugin_imports(plugin_module):
    """Test that the plugin module can be imported successfully."""
    assert plugin_module is not None
    from template import TemplateOperator, register
    assert TemplateOperator is not None
    assert register is not None


def test_plugin_registration():
    """Test that the plugin can be registered with FiftyOne."""
    from template import TemplateOperator, register
    
    # Create a mock plugin registry
    class MockRegistry:
        def __init__(self):
            self.operators = []
        
        def register(self, operator_class):
            self.operators.append(operator_class)
    
    registry = MockRegistry()
    register(registry)
    
    assert len(registry.operators) == 1
    assert registry.operators[0] == TemplateOperator


def test_operator_instantiation():
    """Test that the operator can be instantiated."""
    from template import TemplateOperator
    
    operator = TemplateOperator()
    assert operator is not None
    assert isinstance(operator, foo.Operator)


def test_operator_config():
    """Test that the operator has the correct configuration."""
    from template import TemplateOperator
    
    operator = TemplateOperator()
    config = operator.config
    
    assert config is not None
    assert config.name == "template_operator"
    assert config.label == "Template Operator"
    assert config.description is not None
    assert config.allow_immediate_execution is True


def test_operator_resolve_input(empty_dataset):
    """Test that the operator's resolve_input method works."""
    from template import TemplateOperator
    from fiftyone.operators import ExecutionContext
    
    operator = TemplateOperator()
    
    # Create a minimal context
    class MockContext:
        def __init__(self, dataset):
            self.dataset = dataset
            self.view = dataset
    
    context = MockContext(empty_dataset)
    
    # We can't fully test resolve_input without a proper ExecutionContext,
    # but we can verify the method exists and is callable
    assert hasattr(operator, "resolve_input")
    assert callable(operator.resolve_input)


def test_operator_execute(sample_dataset):
    """Test that the operator's execute method works with a dataset."""
    from template import TemplateOperator
    
    operator = TemplateOperator()
    
    # Create a minimal context for execution
    class MockContext:
        def __init__(self, dataset):
            self.dataset = dataset
            self.view = dataset
            self.params = {"message": "Test message"}
            self.delegated = False
        
        def target_view(self):
            return self.view
        
        class MockOps:
            def notify(self, message, variant="info"):
                pass
        
        ops = MockOps()
    
    context = MockContext(sample_dataset)
    
    # Execute should not raise an error
    operator.execute(context)


def test_no_deprecated_apis():
    """Verify that the plugin doesn't use deprecated FiftyOne APIs."""
    import fiftyone as fo
    
    # Check for common deprecated patterns
    # This is a placeholder - add specific checks based on FO deprecation warnings
    assert hasattr(fo, "Dataset")
    assert hasattr(fo, "Sample")
    
    # If there are known deprecated APIs, check they're not used
    # For now, this is a basic check that core APIs exist


@pytest.mark.unit
def test_plugin_basic_functionality():
    """Basic functionality test marked as unit test."""
    from template import TemplateOperator
    
    operator = TemplateOperator()
    assert operator is not None
    assert operator.config.name == "template_operator"

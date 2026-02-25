"""Template Plugin for FiftyOne Labs - Testing Framework Example.

This is a minimal plugin implementation used as a template for testing
the CI infrastructure. It exercises basic FiftyOne APIs to validate
plugin compatibility with the latest FO version.
"""

import fiftyone.operators as foo
from fiftyone.operators import types

# test: a third-party package dependency
import sam2


class TemplateOperator(foo.Operator):
    """A simple operator that exercises basic FiftyOne functionality."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="template_operator",
            label="Dummy Template Operator",
            description="A template operator for demonstrating plugin testing",
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.view_target(ctx)
        inputs.str(
            "input_message",
            label="Input Message",
            required=False,
            default="Hello from template plugin",
        )

        return types.Property(
            inputs, view=types.View(label="Template Operator")
        )

    def execute(self, ctx):
        """Execute the template operator."""
        target_view = ctx.target_view()
        input_message = ctx.params.get("input_message", "Hello from template plugin")
        sample_count = len(target_view)

        if not ctx.delegated:
            ctx.ops.notify(
                f"Template operator executed on {sample_count} samples. Message: {input_message}",
                variant="info",
            )
        
        output_message = f"{input_message}\nSuccessfully imported the {sam2.__name__} package"
        return {
            "message": output_message,
            "status": "success",
            "sample_count": sample_count,
        }


def register(p):
    """Register the template plugin with FiftyOne."""
    p.register(TemplateOperator)

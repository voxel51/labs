"""Template Plugin for FiftyOne Labs - Testing Framework Example.

This is a minimal plugin implementation used as a template for testing
the CI infrastructure. It exercises basic FiftyOne APIs to validate
plugin compatibility with the latest FO version.
"""

import fiftyone.operators as foo
from fiftyone.operators import types


class TemplateOperator(foo.Operator):
    """A simple operator that exercises basic FiftyOne functionality."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="template_operator",
            label="Template Operator",
            description="A template operator for testing plugin compatibility",
            allow_delegated_execution=False,
            allow_immediate_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.view_target(ctx)
        inputs.str(
            "message",
            label="Message",
            description="A test message parameter",
            required=False,
            default="Hello from template plugin",
        )

        return types.Property(
            inputs, view=types.View(label="Template Operator")
        )

    def execute(self, ctx):
        """Execute the template operator."""
        target_view = ctx.target_view()
        message = ctx.params.get("message", "Hello from template plugin")

        # Exercise basic FiftyOne API
        sample_count = len(target_view)

        if not ctx.delegated:
            ctx.ops.notify(
                f"Template operator executed on {sample_count} samples. Message: {message}",
                variant="info",
            )


def register(p):
    """Register the template plugin with FiftyOne."""
    p.register(TemplateOperator)

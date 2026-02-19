from itertools import chain
from packaging.version import Version

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone.zoo as foz


class ComputeMetadata(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_metadata",
            label="Compute Metadata",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.bool(
            "overwrite", default=False, label="Overwrite", required=False
        )
        return types.Property(inputs)

    def execute(self, ctx):
        if not ctx.current_sample:
            raise Exception("Operator expects an active Sample in the App.")
        sample = ctx.dataset[ctx.current_sample]
        overwrite = ctx.params.get("overwrite", False)
        sample.compute_metadata(overwrite=overwrite)
        sample.save()

        ctx.ops.notify(
            "Sample metadata computed. You may need to reload the Sample modal.",
            variant="success",
        )


class SaveKeypoints(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_keypoints",
            label="Save Keypoints",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.list(
            "keypoints",
            types.List(types.Number()),
            required=True,
            label="Keypoint Coordinates",
        )
        inputs.list(
            "keypoint_labels",
            types.List(types.Number()),
            required=False,
            label="Keypoint Labels (pos/neg)",
        )
        inputs.str(
            "kpts_field_name", default="user_clicks", label="Field Name"
        )
        inputs.str("label_name", default="label", label="Label Name")
        return types.Property(inputs)

    def execute(self, ctx):
        if not ctx.current_sample:
            raise Exception("No sample is active in the App's Sample modal.")
        sample = ctx.dataset[ctx.current_sample]
        keypoints = ctx.params["keypoints"]
        keypoint_labels = ctx.params.get("keypoint_labels", [])
        keypoint_labels = keypoint_labels if len(keypoint_labels) else None
        if keypoint_labels is not None:
            keypoint_labels = list(chain.from_iterable(keypoint_labels))
        field_name = ctx.params["kpts_field_name"]
        label_name = ctx.params["label_name"]

        # NOTE: Negative prompting requires https://github.com/voxel51/fiftyone/pull/6941
        # If fix is not available, raise a warning and remove negative prompts
        remove_neg_pts = False
        if Version(fo.constants.VERSION) < Version("1.14.0"):
            remove_neg_pts = True
        elif hasattr(fo.constants, "TEAM_VERSION"):
            if Version(fo.constants.TEAM_VERSION) < Version("2.17.0"):
                remove_neg_pts = True
        if remove_neg_pts and keypoint_labels:
            ctx.ops.notify(
                "Negative prompting not available with the installed fiftyone version. Removing negative points.",
                variant="warning",
            )
            keypoints = [
                kpt for kpt, lbl in zip(keypoints, keypoint_labels) if lbl != 0
            ]
            keypoint_labels = None

        keypoint = fo.Keypoint(
            points=keypoints,
            label=label_name,
        )
        if keypoint_labels:
            keypoint.sam_labels = keypoint_labels

        if sample.has_field(field_name) and sample[field_name] is not None:
            num_kpts = len(sample[field_name].keypoints)
            ctx.ops.notify(
                f"Appending keypoints to {field_name} with {num_kpts} existing keypoint(s).",
                variant="warning",
            )
            sample[field_name].keypoints.append(keypoint)
        else:
            sample[field_name] = fo.Keypoints(keypoints=[keypoint])
        sample.save()


class SegmentWithPrompts(foo.Operator):
    _model_cache = None

    @property
    def config(self):
        return foo.OperatorConfig(
            name="segment_with_prompts",
            label="Segment With Prompts",
            description="Apply a promptable segmentation model to images",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
            allow_delegated_execution=True,
            allow_immediate_execution=True,
            default_choice_to_delegated=True,
            allow_distributed_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.view_target(ctx)

        inputs.str(
            "prompt_field",
            required=True,
            label="Prompt Field",
        )
        inputs.str(
            "label_field",
            required=False,
            label="Segmentation Label Field",
        )

        inputs.str(
            "model_name",
            required=True,
            label="Promptable Segmentation Model Name",
            default="segment-anything-2-hiera-small-image-torch",
        )

        inputs.int(
            "batch_size",
            label="Batch Size",
            description=("A batch size to use for segmentation"),
            required=False,
            default=1,
        )

        inputs.int(
            "num_workers",
            label="Num Workers",
            description=(
                "The number of workers to use for torch dataloaders "
                "(if applicable)"
            ),
            required=False,
        )

        inputs.bool(
            "skip_failures",
            label="Skip Failures",
            description=(
                "Whether to gracefully continue without raising an error if "
                "prediction cannot be generated for a sample"
            ),
            required=False,
        )

        return types.Property(
            inputs, view=types.View(label="Segment with prompts operator")
        )

    def _get_or_load_model(self, model_name):
        if self._model_cache is None or self._model_cache[0] != model_name:
            self._model_cache = [model_name, foz.load_zoo_model(model_name)]

        return self._model_cache[1]

    def execute(self, ctx):
        if ctx.current_sample:
            target_view = ctx.dataset.select(ctx.current_sample)
        else:
            target_view = ctx.target_view()

        if not target_view.has_sample_field("metadata") or len(
            target_view
        ) != target_view.count("metadata"):
            target_view.compute_metadata()

        prompt_field = ctx.params["prompt_field"]
        model_name = ctx.params["model_name"]
        label_field = (
            ctx.params.get("label_field", None) or prompt_field + "_seg"
        )
        batch_size = ctx.params.get("batch_size", None)
        num_workers = ctx.params.get("num_workers", None)
        skip_failures = ctx.params.get("skip_failures", False)

        if not target_view.has_sample_field(prompt_field):
            ctx.ops.notify(
                f"Prompt field {prompt_field} doesn't exist.", variant="error"
            )
            raise IOError(f"Prompt field {prompt_field} doesn't exist.")

        model = self._get_or_load_model(model_name)

        target_view.apply_model(
            model,
            label_field=label_field,
            prompt_field=prompt_field,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
        )
        target_view.save()

        ctx.ops.notify(
            f"Segmentation for prompts in {prompt_field} saved to {label_field}",
            variant="success",
        )

        if not ctx.delegated:
            ctx.ops.reload_dataset()


def register(p):
    p.register(ComputeMetadata)
    p.register(SaveKeypoints)
    p.register(SegmentWithPrompts)

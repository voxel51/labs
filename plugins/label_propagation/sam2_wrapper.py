import os
import tempfile
from pathlib import Path
import logging
from typing import Tuple, Union, Optional, Any, List
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import cv2

import fiftyone as fo
from .utils import bbox_corners_in_pixel_coords, fit_mask_to_bbox


logger = logging.getLogger(__name__)


SAM2_MODEL_CONFIG_PATH = "configs/sam2/sam2_hiera_t.yaml"
SAM2_MODEL_CHECKPOINT_PATH_CACHE = None


@dataclass(frozen=True)
class SAM2ObjID:
    label: str
    detection_id: str


class PropagatorSAM2:
    def __init__(self):
        """
        Initialize SAM2 propagator
        """
        self.sam2_predictor: Any = None  # type: ignore[assignment]
        self.inference_state: Any = None  # type: ignore[assignment]
        self.preds_dict = OrderedDict()
        self.label_type = "bounding_box"
        self.setup()

    def setup(self):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        import fiftyone.zoo as foz

        device = torch.device(
            # "mps" if torch.backends.mps.is_available() else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            # )  # avoid MPS to prevent Metal SIGABRTs
        )

        # Load model from zoo to get checkpoint path
        global SAM2_MODEL_CHECKPOINT_PATH_CACHE
        if SAM2_MODEL_CHECKPOINT_PATH_CACHE is None:
            zoo_model = foz.load_zoo_model(
                "segment-anything-2-hiera-tiny-image-torch"
            )
            SAM2_MODEL_CHECKPOINT_PATH_CACHE = zoo_model.config.model_path
            # Delete zoo model to free memory
            del zoo_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # build_sam2_video_predictor expects a relative config path that Hydra can resolve
        # The config path should be relative to the sam2 package's config directory
        self.sam2_predictor = build_sam2_video_predictor(
            SAM2_MODEL_CONFIG_PATH,
            SAM2_MODEL_CHECKPOINT_PATH_CACHE,
            device=str(device),
        )
        logger.info("SAM2 predictor initialized successfully")

    def _path_list_to_dir(self, image_path_list, temp_dir):
        """
        Convert a list of image paths to a temporary directory
        using simlinks, maintaining the order of the images.

        Args:
            image_path_list: List of image file paths
            temp_dir: Temporary directory to create the symlinks in
        Returns:
            Temporary directory path
        """
        temp_dir_path = Path(temp_dir)
        for ii, pp in enumerate(image_path_list):
            temp_path = temp_dir_path / f"{ii:06d}{Path(pp).suffix}"
            temp_path.symlink_to(Path(pp).resolve())
        return temp_dir_path

    def initialize(self, frame_path_list):
        """
        Initialize the inference state with the frames list.
        Args:
            frame_path_list: List of frame file paths ordered by frame number
            [video file support coming soon]
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_populated = self._path_list_to_dir(frame_path_list, temp_dir)
            self.inference_state = self.sam2_predictor.init_state(
                str(temp_dir_populated)
            )

        self.preds_dict.clear()
        for idx, frame_path in enumerate(frame_path_list):
            self.preds_dict[os.path.abspath(frame_path)] = None
        logger.info(
            f"Inference state initialized with {len(frame_path_list)} frames"
        )

    def _sam2_detections_to_fiftyone_detections(
        self, sam2_detections: Tuple
    ) -> fo.Detections:
        cv2.setNumThreads(1)

        obj_ids, mask_logits = sam2_detections
        fiftyone_detections = []

        for oi, obj_id in enumerate(obj_ids):
            logits = mask_logits[oi].squeeze(0)
            pred = (logits > 0).cpu().numpy().astype(
                np.uint8
            ) * 255  # threshold at 0

            # Find new bbox from pred
            contours, _ = cv2.findContours(
                pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                new_bbox = [
                    x / self.inference_state["video_width"],
                    y / self.inference_state["video_height"],
                    w / self.inference_state["video_width"],
                    h / self.inference_state["video_height"],
                ]

                if self.label_type == "segmentation_mask":
                    (x1, y1, x2, y2) = bbox_corners_in_pixel_coords(
                        new_bbox,
                        self.inference_state["video_width"],
                        self.inference_state["video_height"],
                    )
                    mask_fitted = pred[y1:y2, x1:x2]
                    # since contours exist, np.max(mask_fitted) must be > 0
                    mask_fitted = (
                        mask_fitted.astype(np.float32) / np.max(mask_fitted)
                    ).astype(np.uint8)
                else:
                    mask_fitted = None

                new_detection = fo.Detection(
                    bounding_box=new_bbox,
                    mask=mask_fitted,
                    label=obj_id.label,
                )
                fiftyone_detections.append(new_detection)
            else:
                logger.warning("Warning: No contour found for detection")

        return fo.Detections(detections=fiftyone_detections)

    def _fiftyone_detections_to_sam2_detections(
        self, fiftyone_detections: fo.Detections
    ) -> List[Tuple]:
        """
        fiftyone_detections is a fo.Detections object
        Returns: List of tuples of (detection_type, detection_array, Sam2ObjID)
        """
        detection_tuples = []
        if not hasattr(fiftyone_detections, "detections"):
            logger.warning(
                f"Source detections is either empty, or not a fo.Detections object: {fiftyone_detections}"
            )
            return detection_tuples

        for detection in fiftyone_detections.detections:  # type: ignore[attr-defined]
            # Get source bbox and convert to pixel coordinates
            source_bbox = detection.bounding_box
            x1, y1, x2, y2 = bbox_corners_in_pixel_coords(
                source_bbox,
                self.inference_state["video_width"],
                self.inference_state["video_height"],
            )

            source_mask = detection.mask
            if source_mask is not None:
                source_mask_fitted = fit_mask_to_bbox(
                    source_mask, (y2 - y1, x2 - x1)
                )
                # make it relative to the target frame
                source_mask_framed = np.zeros(
                    (
                        self.inference_state["video_height"],
                        self.inference_state["video_width"],
                    ),
                    bool,
                )
                source_mask_framed[y1:y2, x1:x2] = source_mask_fitted
                source_mask_framed = source_mask_framed.astype(np.uint8)
                detection_tuples.append(
                    (
                        "segmentation_mask",
                        source_mask_framed,
                        SAM2ObjID(
                            label=detection.label, detection_id=detection.id
                        ),
                    )
                )
            else:
                detection_tuples.append(
                    (
                        "bounding_box",
                        [x1, y1, x2, y2],
                        SAM2ObjID(
                            label=detection.label, detection_id=detection.id
                        ),
                    )
                )

        return detection_tuples

    def register_source_frame(self, source_filepath, source_detections):
        """
        Register the source frame and detections with SAM2.

        Args:
            source_filepath: The source frame file path
            source_detections: The detections from source_frame (fo.Detections)
        """

        source_frame_idx = list(self.preds_dict.keys()).index(
            os.path.abspath(source_filepath)
        )
        logger.debug(
            f"Registering source frame {source_filepath} at index {source_frame_idx}"
        )

        for detection_tuple in self._fiftyone_detections_to_sam2_detections(
            source_detections
        ):
            detection_type, detection_array, obj_id = detection_tuple
            if detection_type == "segmentation_mask":
                self.label_type = "segmentation_mask"
                self.sam2_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=source_frame_idx,
                    obj_id=obj_id,
                    mask=detection_array,
                )
                logger.debug(f"Added new segmentation mask: {obj_id.label}")
            else:
                self.sam2_predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=source_frame_idx,
                    obj_id=obj_id,
                    box=detection_array,
                )
                logger.debug(f"Added new bounding box: {obj_id.label}")

    def propagate_to_all_frames(self):
        """
        Propagate detections to all frames in the inference state.
        Uses SAM2's propagate_in_video API.
        Returns:
            None
            Populates the preds_dict with predictions for each frame
        """
        propagated_to_count = 0
        for frame_idx, obj_ids, mask_logits in self.sam2_predictor.propagate_in_video(self.inference_state):  # type: ignore[union-attr]
            target_filepath = list(self.preds_dict.keys())[frame_idx]

            obj_ids = list(obj_ids)
            logger.debug(
                f"Found {len(obj_ids)} detections for frame {target_filepath}"
            )
            propagated_detections = (
                self._sam2_detections_to_fiftyone_detections(
                    (obj_ids, mask_logits)
                )
            )

            logger.debug(
                f"Propagated {len(propagated_detections)} detections for frame {target_filepath}"
            )
            self.preds_dict[target_filepath] = propagated_detections
            propagated_to_count += 1

        logger.info(
            f"Propagated detections to {propagated_to_count} frames out of {len(self.preds_dict)}"
        )

    def get_propagated_detections(self, target_filepath) -> fo.Detections:
        """
        Get propagated detections for target frame.
        This does not modify the inference state.

        Args:
            target_filepath: Target frame file path
        Returns:
            fo.Detections: Propagated detections with or without masks
        """
        if self.inference_state is None:
            raise RuntimeError(
                "Must call register_source_frame() before get_propagated_detections()"
            )

        if os.path.abspath(target_filepath) not in self.preds_dict:
            logger.warning(
                f"Target frame {target_filepath} not found in predictions"
            )
            return fo.Detections(detections=[])

        result = self.preds_dict.get(
            os.path.abspath(target_filepath), fo.Detections(detections=[])
        )
        return result


def propagate_annotations_sam2_neeraja(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
    progress: Optional[bool] = True,
) -> dict[str, float]:
    """
    Propagate annotations from exemplar frames (containing labels in input_annotation_field) to all the frames.
    Args:
        view: The view to propagate annotations from
        input_annotation_field: The field name of the annotation to copy from the exemplar frame field
        output_annotation_field: The field name of the annotation to save to the target frame
        sort_field: Field to sort samples by
        progress: Whether to show progress bars (True/False) or use default (None)
    """

    # Set up the propagator
    propagator = PropagatorSAM2()

    if sort_field and view.has_field(sort_field):
        image_path_list = view.sort_by(sort_field).values("filepath")
    else:
        image_path_list = view.values("filepath")
    propagator.initialize(image_path_list)

    # Register all frames
    def register_sample(sample):
        if sample[input_annotation_field]:
            propagator.register_source_frame(
                sample.filepath, sample[input_annotation_field]
            )

    _ = list(
        view.map_samples(register_sample, num_workers=1, progress=progress)
    )

    # Propagate
    try:
        propagator.propagate_to_all_frames()
    except Exception as e:
        """
        ------------------------------------------------------------
        This error typically occurs due to a mixed-precision dtype mismatch in SAM2's
        internal transformer layers. SAM2 uses bfloat16 for some tensors while keeping
        others (or layer weights) in float32, causing "mat1 and mat2 must have the same
        dtype, but got BFloat16 and Float" errors in Linear layers during propagation.

        This is a known issue in SAM2's implementation that is more likely to occur
        when propagating across multiple sequences with different label sets,
        as this triggers more complex memory attention paths in SAM2's tracking stack.
        ------------------------------------------------------------
        """
        raise RuntimeError(
            f"Error propagating to all frames; \
        please try with a shorter sequence with continuous frames and consistent labels."
        )

    # Populate propagations
    def populate_propagations(sample):
        propagated_detections = propagator.get_propagated_detections(
            sample.filepath
        )
        sample[output_annotation_field] = propagated_detections
        return

    if sort_field and view.has_field(sort_field):
        _ = list(
            view.sort_by(sort_field).map_samples(
                populate_propagations,
                num_workers=1,
                save=True,
                progress=progress,
            )
        )
    else:
        _ = list(
            view.map_samples(
                populate_propagations,
                num_workers=1,
                save=True,
                progress=progress,
            )
        )

    return {}



def propagate_annotations_sam2(
    view: Union[fo.Dataset, fo.DatasetView],
    input_annotation_field: str,
    output_annotation_field: str,
    sort_field: Optional[str] = None,
    progress: Optional[bool] = True,
) -> dict[str, float]:
    """
    Propagate annotations from exemplar frames (containing labels in input_annotation_field) to all the frames.
    Args:
        view: The view to propagate annotations from
        input_annotation_field: The field name of the annotation to copy from the exemplar frame field
        output_annotation_field: The field name of the annotation to save to the target frame
        sort_field: Field to sort samples by
        progress: Whether to show progress bars (True/False) or use default (None)
    """
    import fiftyone.core.labels as fol
    import fiftyone.zoo as foz
    # Fo SAM2 video model (fiftyone.utils.sam2.SegmentAnything2VideoModel)
    # is loaded via zoo; its predict() does init_state + propagation internally.

    # Ordered frame paths and per-frame prompts from view
    if sort_field and view.has_field(sort_field):
        ordered = list(
            view.sort_by(sort_field).iter_samples(progress=progress)
        )
    else:
        ordered = list(view.iter_samples(progress=progress))
    paths = [s.filepath for s in ordered]
    prompts_per_frame = [
        s.get_field(input_annotation_field) if s.get_field(input_annotation_field) else fol.Detections(detections=[])
        for s in ordered
    ]

    # Mock frame: frame field "detections" for Fo model's _get_prompts
    class MockFrame:
        def __init__(self, detections):
            self._detections = detections

        def get_field(self, name):
            if name == "detections":
                return self._detections
            return None

    # Mock sample: .frames = {1: ..., 2: ..., ...} for Fo's load_fiftyone_video_frames and _get_prompts
    class MockSample:
        def __init__(self, frames_dict):
            self.frames = frames_dict

    mock_frames = {
        i + 1: MockFrame(prompts_per_frame[i]) for i in range(len(paths))
    }
    mock_sample = MockSample(mock_frames)

    # Video reader: .frame_size and .read() for Fo's load_fiftyone_video_frames
    first_img = cv2.imread(paths[0])
    if first_img is None:
        raise RuntimeError(f"Could not read first frame: {paths[0]}")
    first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    w, h = first_img.shape[1], first_img.shape[0]

    class ImageSequenceReader:
        def __init__(self, path_list):
            self.path_list = path_list
            self._idx = 0
            self._first = first_img

        @property
        def frame_size(self):
            return (w, h)

        def read(self):
            if self._first is not None:
                out = self._first
                self._first = None
                return out
            img = cv2.imread(self.path_list[self._idx])
            self._idx += 1
            if img is None:
                return np.zeros((h, w, 3), dtype=np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    video_reader = ImageSequenceReader(paths)

    # Load Fo SAM2 video model and run predict (no propagate_in_video in our code)
    model = foz.load_zoo_model("segment-anything-2-hiera-tiny-video-torch")
    model.needs_fields = {"prompt_field": "frames.detections"}
    sample_detections = model.predict(video_reader, mock_sample)

    # sample_detections is {frame_number: fol.Detections} (1-based)
    for i, sample in enumerate(ordered):
        fn = i + 1
        sample[output_annotation_field] = sample_detections.get(
            fn, fol.Detections(detections=[])
        )
        sample.save()
    return {}
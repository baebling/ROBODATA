"""Tool pose from calibrated hand geometry and an explicitly verified fixed grasp.

The calibration supplies metric hand points and T_hand_tool. Model-local world
landmarks are not a camera pose or a substitute for this calibration. Synthetic
fixed-grasp calibrations remain virtual fixtures, not measured Graco geometry.
"""

import numpy as np

from .config import HandAssistanceSpec, RunConfig
from .geometry import estimate_pnp, validate_transform
from .observations import Camera, Detection
from .schema import Estimate

PALM_KEYS = frozenset({"0", "5", "9", "13", "17"})


def hand_tool_observation(
    detection: Detection, spec: HandAssistanceSpec, camera: Camera, cfg: RunConfig
) -> Estimate:
    """Estimate C_tool = C_hand @ T_hand_tool, or raise for unsupported evidence.

    Keypoint confidence and visibility use the existing observation gates. The
    returned confidence is a geometry/accepted-keypoint heuristic, not an error
    probability. For MediaPipe predictions, visible only means coordinate
    availability and confidence=1 only records an accepted model prediction.
    """
    if spec is None or getattr(spec, "grasp_verified", False) is not True:
        raise ValueError("Hand assistance requires an explicitly verified fixed grasp")
    source = getattr(spec, "calibration_source", None)
    if not isinstance(source, str) or source.strip().lower() in {
        "",
        "unknown",
        "unverified",
        "none",
        "n/a",
        "unspecified",
    }:
        raise ValueError("Hand assistance requires a known calibration source")
    hand_tool = validate_transform(spec.T_hand_tool)
    if detection is None or not detection.keypoints:
        raise ValueError("Missing hand predictions")
    if detection.handedness is None or detection.handedness != spec.handedness:
        raise ValueError("Hand handedness differs from calibrated grasp or is unknown")
    if detection.occluded:
        raise ValueError("Annotated hand occlusion")
    names = [
        name
        for name, point in detection.keypoints.items()
        if name in spec.points_hand_m
        and point.visible
        and point.confidence >= cfg.min_keypoint_confidence
    ]
    if not PALM_KEYS.issubset(names):
        raise ValueError("Missing visible confident calibrated wrist/MCP hand predictions")
    result = estimate_pnp(
        [spec.points_hand_m[name] for name in names],
        [detection.keypoints[name].uv for name in names],
        camera.matrix,
        camera.distortion,
        cfg.max_reprojection_error_px,
    )
    camera_tool = validate_transform(result.transform @ hand_tool)
    confidence = result.confidence * min(detection.keypoints[n].confidence for n in names)
    return Estimate(
        T_camera_frame=camera_tool.tolist(),
        method="calibrated_hand_grasp_pnp",
        residual=result.residual,
        residual_unit="pixel",
        confidence=float(np.clip(confidence, 0, 1)),
        origin_camera_m=camera_tool[:3, 3].tolist(),
        axis_camera=camera_tool[:3, 2].tolist(),
        frame_method="calibrated_fixed_hand_tool_grasp",
        metric_scale_source="calibrated_hand_template",
        calibration_source=spec.calibration_source,
    )

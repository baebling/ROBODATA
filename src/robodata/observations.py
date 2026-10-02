"""Timestamped video input, semantic observations and checked optical flow."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import Field, model_validator

from .config import RunConfig, resolve
from .geometry import backproject, estimate_pnp, frame_from_axis, unit
from .schema import Contract, Estimate


class Camera(Contract):
    matrix: list[list[float]]
    distortion: list[float] = Field(default_factory=lambda: [0.0] * 5)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @model_validator(mode="after")
    def calibrated(self):
        K = np.asarray(self.matrix, float)
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or K[0, 0] <= 0
            or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])
        ):
            raise ValueError("Invalid calibrated camera matrix")
        return self


class Keypoint(Contract):
    uv: list[float] = Field(min_length=2, max_length=2)
    confidence: float = Field(default=1.0, ge=0, le=1)
    visible: bool = True


class Detection(Contract):
    keypoints: dict[str, Keypoint] = Field(default_factory=dict)
    mask_path: str | None = None
    occluded: bool = False


class FrameInput(Contract):
    frame_index: int = Field(ge=0)
    tool: Detection = Field(default_factory=Detection)
    surface: Detection = Field(default_factory=Detection)
    depth_path: str | None = None
    work_on: bool | None = None
    primitive_label: str | None = None
    process_state_source: str | None = None


class Manifest(Contract):
    episode_id: str
    task_name: str
    video_path: str
    camera: Camera
    data_kind: Literal["real", "synthetic"] = "real"
    timestamps_sec: list[float] | None = None
    observations: list[FrameInput] = Field(default_factory=list)
    landmark_source: str = "manual_annotation"
    depth_aligned_to_rgb: bool = False
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered(self):
        ids = [o.frame_index for o in self.observations]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate observation frame indices")
        if self.timestamps_sec is not None:
            t = np.asarray(self.timestamps_sec)
            if not len(t) or not np.isfinite(t).all() or t[0] < 0 or np.any(np.diff(t) <= 0):
                raise ValueError("Explicit timestamps must be finite and strictly increasing")
        return self


def read_mask(root: Path, name: str | None, shape) -> np.ndarray | None:
    if name is None:
        return None
    mask = cv2.imread(str(resolve(root, name)), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(name)
    if mask.shape != shape:
        raise ValueError("Mask must match RGB resolution")
    return mask > 0


def read_depth(root: Path, name: str | None, cfg: RunConfig, shape) -> np.ndarray:
    if name is None:
        raise ValueError("No aligned depth for this frame")
    path = resolve(root, name)
    raw = (
        np.load(path, allow_pickle=False)
        if path.suffix == ".npy"
        else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    )
    if raw is None:
        raise FileNotFoundError(path)
    if raw.shape != shape:
        raise ValueError("Depth must match RGB resolution")
    depth = raw.astype(float) * cfg.depth_unit_m
    good = np.isfinite(depth) & (depth >= cfg.depth_range_m[0]) & (depth <= cfg.depth_range_m[1])
    depth[~good] = np.nan
    return depth


def video_frames(path: Path, camera: Camera, timestamps):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot decode video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    try:
        if timestamps is None and (not np.isfinite(fps) or fps <= 0):
            raise ValueError("Video requires explicit timestamps or valid constant FPS metadata")
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if bgr.shape[:2] != (camera.height, camera.width):
                raise ValueError("Video resolution does not match calibrated camera")
            if timestamps is not None and i >= len(timestamps):
                raise ValueError("Explicit timestamp count does not match video")
            timestamp = timestamps[i] if timestamps is not None else i / fps
            yield i, float(timestamp), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
        if i == 0:
            raise ValueError("Empty video")
        if timestamps is not None and i != len(timestamps):
            raise ValueError("Explicit timestamp count does not match video")
    finally:
        cap.release()


def pnp_observation(detection: Detection, spec, camera: Camera, cfg: RunConfig) -> Estimate:
    if detection.occluded:
        raise ValueError("Annotated occlusion")
    names = [
        name
        for name, p in detection.keypoints.items()
        if name in spec.points_m and p.visible and p.confidence >= cfg.min_keypoint_confidence
    ]
    result = estimate_pnp(
        [spec.points_m[n] for n in names],
        [detection.keypoints[n].uv for n in names],
        camera.matrix,
        camera.distortion,
        cfg.max_reprojection_error_px,
    )
    return Estimate(
        T_camera_frame=result.transform.tolist(),
        method=result.method,
        confidence=result.confidence,
        residual=result.residual,
        residual_unit="pixel",
        origin_camera_m=result.transform[:3, 3].tolist(),
        axis_camera=result.transform[:3, 2].tolist(),
        frame_method="semantic_geometry",
    )


def depth_tool(detection: Detection, depth, camera: Camera, cfg: RunConfig) -> Estimate:
    if detection.occluded:
        raise ValueError("Annotated occlusion")
    pts = {}
    scores = []
    for name, p in detection.keypoints.items():
        if not p.visible or p.confidence < cfg.min_keypoint_confidence:
            continue
        u, v = np.rint(p.uv).astype(int)
        if not (0 <= u < camera.width and 0 <= v < camera.height):
            continue
        patch = depth[max(v - 1, 0) : v + 2, max(u - 1, 0) : u + 2]
        values = patch[np.isfinite(patch)]
        if not len(values):
            continue
        pts[name] = backproject([p.uv], [np.median(values)], camera.matrix)[0]
        scores.append(p.confidence)
    spec = cfg.tool
    if any(k not in pts for k in [spec.origin_key, spec.axis_from, spec.axis_to]):
        raise ValueError("Missing functional-point or axis depth")
    origin = pts[spec.origin_key]
    z = unit(pts[spec.axis_to] - pts[spec.axis_from])
    T = None
    if spec.x_key is not None and spec.x_key in pts:
        try:
            T = frame_from_axis(origin, z, pts[spec.x_key] - origin).tolist()
        except ValueError:
            pass
    return Estimate(
        T_camera_frame=T,
        method="semantic_keypoints_aligned_depth",
        origin_camera_m=origin.tolist(),
        axis_camera=z.tolist(),
        confidence=min(scores),
        frame_method="functional_point_and_axis"
        if T is None
        else "functional_point_axis_and_secondary_landmark",
    )


class FlowTracker:
    """Forward/backward checked KLT, reset by every supplied annotation."""

    def __init__(self, max_error_px: float):
        self.previous = None
        self.points: dict[str, Keypoint] = {}
        self.max_error_px = max_error_px

    def update(self, rgb, supplied: Detection) -> Detection:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        result = supplied.model_copy(deep=True)
        if (
            not supplied.occluded
            and not supplied.keypoints
            and self.previous is not None
            and self.points
        ):
            names = list(self.points)
            a = np.float32([self.points[n].uv for n in names]).reshape(-1, 1, 2)
            b, ok1, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, a, None)
            if b is not None:
                back, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, b, None)
                if back is not None:
                    errors = np.linalg.norm(back - a, axis=2).ravel()
                    h, w = gray.shape
                    for j, name in enumerate(names):
                        uv = b[j, 0]
                        if (
                            ok1[j, 0]
                            and ok2[j, 0]
                            and errors[j] < self.max_error_px
                            and 0 <= uv[0] < w
                            and 0 <= uv[1] < h
                        ):
                            result.keypoints[name] = Keypoint(
                                uv=uv.tolist(),
                                confidence=float(
                                    min(self.points[name].confidence, np.exp(-errors[j]))
                                ),
                            )
        self.previous = gray
        self.points = {n: p for n, p in result.keypoints.items() if p.visible}
        return result

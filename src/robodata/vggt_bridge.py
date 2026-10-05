"""Measured RGB anchors connect VGGT's normalized scene to a physical surface."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .geometry import inverse, validate_transform
from .observations import Camera
from .schema import Contract, Estimate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_affine(width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Original pixel centers to official VGGT crop-mode pixel centers."""
    if width <= 0 or height <= 0:
        raise ValueError("Positive image dimensions required")
    new_width = 518
    new_height = round(height * (new_width / width) / 14) * 14
    if new_height < 14:
        raise ValueError("Unsupported extreme image aspect ratio")
    sx, sy = new_width / width, new_height / height
    offset_y = max(0, (new_height - 518) // 2)
    A = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2 - offset_y], [0, 0, 1]], float)
    return A, (min(new_height, 518), 518)


@dataclass
class MetricAlignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    residual_m: float
    max_error_m: float


def fit_metric_alignment(model_points, measured_points, max_error_m: float) -> MetricAlignment:
    """Fit X_surface = s R X_model + t; no GT trajectory enters this fit."""
    src, dst = np.asarray(model_points, float), np.asarray(measured_points, float)
    if (
        src.ndim != 2
        or src.shape[1] != 3
        or dst.shape != src.shape
        or len(src) < 4
        or not np.isfinite(src).all()
        or not np.isfinite(dst).all()
        or not np.isfinite(max_error_m)
        or max_error_m <= 0
    ):
        raise ValueError("Alignment needs four finite corresponding anchors and a positive gate")
    a, b = src - src.mean(axis=0), dst - dst.mean(axis=0)
    if np.linalg.matrix_rank(a, tol=1e-7) < 2 or np.linalg.matrix_rank(b, tol=1e-7) < 2:
        raise ValueError("Collinear anchors cannot define a surface frame")
    U, singular, Vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag(sign) @ Vt
    scale = float(np.dot(singular, sign) / np.mean(np.sum(a * a, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid metric alignment scale")
    t = dst.mean(axis=0) - scale * R @ src.mean(axis=0)
    errors = np.linalg.norm(scale * src @ R.T + t - dst, axis=1)
    if errors.max() > max_error_m:
        raise ValueError("Measured anchor alignment exceeds metric residual gate")
    return MetricAlignment(scale, R, t, float(np.sqrt(np.mean(errors**2))), float(errors.max()))


def metric_camera_surface(extrinsic, alignment: MetricAlignment) -> np.ndarray:
    E = np.asarray(extrinsic, float)
    if E.shape != (3, 4):
        raise ValueError("VGGT extrinsics must be camera-from-model 3x4")
    camera_model = validate_transform(np.vstack([E, [0, 0, 0, 1]]))
    T = np.eye(4)
    T[:3, :3] = camera_model[:3, :3] @ alignment.rotation.T
    T[:3, 3] = alignment.scale * E[:, 3] - T[:3, :3] @ alignment.translation
    return validate_transform(T)


def sample_world_point(uv, depth, confidence, intrinsics, extrinsic, min_confidence):
    depth, confidence = np.asarray(depth), np.asarray(confidence)
    uv = np.asarray(uv, float)
    if (
        uv.shape != (2,)
        or not np.isfinite(uv).all()
        or depth.ndim != 2
        or confidence.shape != depth.shape
    ):
        raise ValueError("Invalid RGB/depth anchor shape")
    h, w = depth.shape
    if not (0 <= uv[0] <= w - 1 and 0 <= uv[1] <= h - 1):
        raise ValueError("Anchor outside processed RGB image")
    u, v = np.rint(uv).astype(int)
    z, score = float(depth[v, u]), float(confidence[v, u])
    if not np.isfinite(z) or z <= 0 or not np.isfinite(score) or score < min_confidence:
        raise ValueError("Anchor has invalid or low-confidence predicted depth")
    K = Camera(matrix=np.asarray(intrinsics).tolist(), width=w, height=h).matrix
    camera_point = np.linalg.solve(K, [uv[0], uv[1], 1]) * z
    E = np.asarray(extrinsic, float)
    if E.shape != (3, 4):
        raise ValueError("Invalid extrinsics shape")
    model_camera = inverse(np.vstack([E, [0, 0, 0, 1]]))
    return model_camera[:3, :3] @ camera_point + model_camera[:3, 3]


def surface_estimate(T, residual_m, gate_m, calibration_source) -> Estimate:
    T = validate_transform(T)
    return Estimate(
        T_camera_frame=T.tolist(),
        method="vggt_measured_surface_alignment",
        origin_camera_m=T[:3, 3].tolist(),
        axis_camera=T[:3, 2].tolist(),
        confidence=float(np.exp(-residual_m / gate_m)),
        residual=residual_m,
        residual_unit="meter",
        frame_method="persistent_measured_surface_anchors",
        metric_scale_source="measured_geometry",
        calibration_source=calibration_source,
    )


class SurfaceSample(Contract):
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    estimate: Estimate | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def complete(self):
        if (self.estimate is None) == (self.failure_reason is None):
            raise ValueError("A surface sample needs either an estimate or a failure reason")
        if self.estimate is not None and (
            self.estimate.T_camera_frame is None
            or self.estimate.confidence is None
            or self.estimate.metric_scale_source != "measured_geometry"
            or self.estimate.calibration_source is None
        ):
            raise ValueError("Surface estimate requires a metric sourced full pose")
        return self


class SurfaceCache(Contract):
    schema_version: Literal[1] = 1
    video_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    camera: Camera
    data_kind: Literal["real", "synthetic"]
    calibration_source: str = Field(min_length=1)
    surface_frame_id: str = Field(min_length=1)
    frames: list[SurfaceSample] = Field(min_length=1)
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered(self):
        if any(
            b.frame_index <= a.frame_index or b.timestamp_sec <= a.timestamp_sec
            for a, b in zip(self.frames, self.frames[1:])
        ):
            raise ValueError("Cache indices/timestamps must be strictly increasing")
        return self


def load_surface_cache(
    path: Path, video: Path, camera: Camera, data_kind: str, surface_frame_id: str
) -> SurfaceCache:
    cache = SurfaceCache.model_validate_json(path.read_text(encoding="utf-8"))
    if cache.video_sha256 != file_sha256(video):
        raise ValueError("VGGT surface cache belongs to a different video")
    if cache.camera != camera or cache.data_kind != data_kind:
        raise ValueError("VGGT surface cache camera/data-kind mismatch")
    if cache.surface_frame_id != surface_frame_id:
        raise ValueError("VGGT surface cache physical frame identity mismatch")
    return cache


def cached_surface(cache: SurfaceCache, index: int, timestamp: float) -> Estimate:
    for sample in cache.frames:
        if sample.frame_index == index:
            if not np.isclose(sample.timestamp_sec, timestamp, rtol=0, atol=1e-7):
                raise ValueError("VGGT surface sample timestamp mismatch")
            if sample.estimate is None:
                raise ValueError(f"VGGT surface sample rejected: {sample.failure_reason}")
            return sample.estimate.model_copy(deep=True)
    raise ValueError("No VGGT surface sample for this frame; no interpolation applied")

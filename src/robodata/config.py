"""Versioned experiment configuration and path resolution."""

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import Field, model_validator

from .schema import Contract


class ObjectSpec(Contract):
    points_m: dict[str, list[float]] = Field(default_factory=dict)
    origin_key: str = "tip"
    axis_from: str = "axis_base"
    axis_to: str = "tip"
    x_key: str | None = "side"

    @model_validator(mode="after")
    def geometry(self):
        if any(len(p) != 3 or not np.isfinite(p).all() for p in self.points_m.values()):
            raise ValueError("Geometry points must be finite 3D meter values")
        return self


class RunConfig(Contract):
    schema_version: Literal[1] = 1
    manifest_path: str
    mode: Literal["rgb", "rgbd"] = "rgb"
    metric_scale_source: Literal["measured_geometry", "depth_sensor"] = "measured_geometry"
    camera_motion: Literal["static", "moving"] = "moving"
    surface_frame_id: str = "reference_patch"
    surface_method: Literal["pnp", "depth_plane"] = "pnp"
    tool: ObjectSpec
    surface: ObjectSpec
    horizon_sec: float = Field(default=0.2, gt=0)
    timestamp_tolerance_sec: float = Field(default=0.04, ge=0)
    max_reprojection_error_px: float = Field(default=4.0, gt=0)
    min_keypoint_confidence: float = Field(default=0.5, ge=0, le=1)
    plane_threshold_m: float = Field(default=0.01, gt=0)
    min_plane_inlier_ratio: float = Field(default=0.7, gt=0, le=1)
    depth_unit_m: float = Field(default=0.001, gt=0)
    depth_range_m: tuple[float, float] = (0.05, 10.0)
    tracking: bool = False
    max_flow_fb_error_px: float = Field(default=1.0, gt=0)
    seed: int = 42

    @model_validator(mode="after")
    def supported(self):
        if self.mode == "rgb" and self.metric_scale_source != "measured_geometry":
            raise ValueError("RGB PnP requires measured geometry for metric scale")
        if self.mode == "rgbd" and self.metric_scale_source != "depth_sensor":
            raise ValueError("RGB-D requires explicit depth_sensor scale")
        if self.mode == "rgb" and not self.tool.points_m:
            raise ValueError("RGB tool geometry is required")
        if self.surface_method == "pnp" and len(self.surface.points_m) < 4:
            raise ValueError("Surface PnP requires persistent measured landmarks")
        if self.surface_method == "depth_plane" and (
            self.mode != "rgbd" or self.camera_motion != "static"
        ):
            raise ValueError("Plane-only surface reference requires static RGB-D camera")
        if not 0 < self.depth_range_m[0] < self.depth_range_m[1]:
            raise ValueError("Invalid depth range")
        return self


def load_config(path: Path | str) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def resolve(root: Path, name: str) -> Path:
    return (root / name).resolve()

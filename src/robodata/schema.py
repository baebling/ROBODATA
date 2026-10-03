"""Strict portable dataset contract, including provenance and missingness."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .geometry import validate_transform


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Source(Contract):
    mode: Literal["rgb", "rgbd", "external_pose"]
    metric_scale_source: Literal[
        "measured_geometry",
        "depth_sensor",
        "calibrated_external",
        "mixed_depth_and_hand_calibration",
    ]
    data_kind: Literal["real", "synthetic"]
    manifest_path: str | None = None
    rgb_video_path: str | None = None
    timestamp_source: str = "explicit"


class Estimate(Contract):
    T_camera_frame: list[list[float]] | None = None
    method: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    residual: float | None = Field(default=None, ge=0)
    residual_unit: str | None = None
    origin_camera_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    axis_camera: list[float] | None = Field(default=None, min_length=3, max_length=3)
    frame_method: str | None = None
    metric_scale_source: str | None = None
    calibration_source: str | None = None

    @field_validator("T_camera_frame")
    @classmethod
    def check_transform(cls, v):
        return None if v is None else validate_transform(v).tolist()

    @field_validator("axis_camera")
    @classmethod
    def check_axis(cls, v):
        if v is not None and not np.isclose(np.linalg.norm(v), 1.0, atol=1e-6):
            raise ValueError("Tool/surface direction must be a unit vector")
        return v


class Observation(Contract):
    rgb_path: str | None = None
    depth_path: str | None = None
    tool_mask_path: str | None = None
    surface_mask_path: str | None = None
    tool_centroid_pixel: list[float] | None = Field(default=None, min_length=2, max_length=2)
    landmark_source: str | None = None


class Canonical(Contract):
    T_surface_tool: list[list[float]] | None = None
    tip_position_surface_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    tool_axis_surface: list[float] | None = Field(default=None, min_length=3, max_length=3)
    standoff_distance_m: float | None = Field(default=None, ge=0)
    signed_surface_distance_m: float | None = None
    incidence_angle_rad: float | None = Field(default=None, ge=0, le=np.pi)
    surface_projected_path_m: list[float] | None = Field(default=None, min_length=2, max_length=2)
    linear_velocity_mps: list[float] | None = Field(default=None, min_length=3, max_length=3)
    angular_velocity_radps: list[float] | None = Field(default=None, min_length=3, max_length=3)

    @field_validator("T_surface_tool")
    @classmethod
    def check_transform(cls, v):
        return None if v is None else validate_transform(v).tolist()

    @field_validator("tool_axis_surface")
    @classmethod
    def check_axis(cls, v):
        if v is not None and not np.isclose(np.linalg.norm(v), 1.0, atol=1e-6):
            raise ValueError("Functional direction must be a unit vector")
        return v

    @model_validator(mode="after")
    def consistent_pose(self):
        if self.T_surface_tool is not None:
            T = np.asarray(self.T_surface_tool)
            if self.tip_position_surface_m is not None and not np.allclose(
                T[:3, 3], self.tip_position_surface_m, atol=1e-6
            ):
                raise ValueError("Functional origin must match the pose translation")
            if self.tool_axis_surface is not None and not np.allclose(
                T[:3, 2], self.tool_axis_surface, atol=1e-6
            ):
                raise ValueError("Functional axis must match the pose +z direction")
        return self


class Quality(Contract):
    valid: bool = False
    full_pose_valid: bool = False
    occluded: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_kind: str = "uncalibrated_quality_score"
    failure_reason: str | None = None
    estimated_fields: list[str] = Field(default_factory=list)
    interpolated: bool = False

    @model_validator(mode="after")
    def full_pose_needs_valid_state(self):
        if self.full_pose_valid and not self.valid:
            raise ValueError("Full pose requires a valid state")
        return self


class ActionLabel(Contract):
    valid: bool = False
    target_timestamp_sec: float | None = Field(default=None, ge=0)
    horizon_actual_sec: float | None = Field(default=None, gt=0)
    target_T_surface_tool: list[list[float]] | None = None
    delta_position_surface_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    delta_T_tool_current_tool_future: list[list[float]] | None = None
    delta_rotation_tool_current_rad: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )
    failure_reason: str | None = "no_future_sample"

    @field_validator("target_T_surface_tool", "delta_T_tool_current_tool_future")
    @classmethod
    def check_transform(cls, v):
        return None if v is None else validate_transform(v).tolist()

    @model_validator(mode="after")
    def complete_target(self):
        if self.valid and any(
            value is None
            for value in [
                self.target_timestamp_sec,
                self.horizon_actual_sec,
                self.target_T_surface_tool,
                self.delta_position_surface_m,
                self.delta_T_tool_current_tool_future,
                self.delta_rotation_tool_current_rad,
            ]
        ):
            raise ValueError("Valid action needs a complete future-pose target")
        if self.valid and self.failure_reason is not None:
            raise ValueError("Valid action cannot have a failure reason")
        return self


class FrameRecord(Contract):
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    observation: Observation = Field(default_factory=Observation)
    surface: Estimate | None = None
    tool: Estimate | None = None
    canonical: Canonical = Field(default_factory=Canonical)
    quality: Quality = Field(default_factory=Quality)
    action: ActionLabel = Field(default_factory=ActionLabel)
    work_on: bool | None = None
    primitive_label: str | None = None
    process_state_source: str | None = None

    @model_validator(mode="after")
    def check_validity(self):
        if self.quality.valid and (
            self.canonical.tip_position_surface_m is None
            or self.canonical.tool_axis_surface is None
        ):
            raise ValueError("Valid state needs a functional point and axis")
        if self.quality.full_pose_valid and self.canonical.T_surface_tool is None:
            raise ValueError("Full pose validity requires SE(3)")
        if self.action.valid and not self.quality.full_pose_valid:
            raise ValueError("Valid action requires a valid current full pose")
        return self


class Episode(Contract):
    schema_version: Literal["0.1.0"] = "0.1.0"
    episode_id: str
    task_name: str
    surface_frame_id: str
    source: Source
    frames: list[FrameRecord] = Field(min_length=1)
    provenance: dict = Field(default_factory=dict)
    units: dict[str, str] = Field(
        default_factory=lambda: {"position": "meter", "angle": "radian", "timestamp": "second"}
    )

    @model_validator(mode="after")
    def ordered(self):
        times = [f.timestamp_sec for f in self.frames]
        indices = [f.frame_index for f in self.frames]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Frame timestamps must be strictly increasing")
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError("Frame indices must be strictly increasing")
        if self.units != {"position": "meter", "angle": "radian", "timestamp": "second"}:
            raise ValueError("Unsupported units")
        return self

"""Time-aware labels with explicit missing-data boundaries."""

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import inverse
from .schema import ActionLabel, Episode


def add_labels(episode: Episode, horizon_sec: float, tolerance_sec: float) -> None:
    if not np.isfinite([horizon_sec, tolerance_sec]).all() or horizon_sec <= 0 or tolerance_sec < 0:
        raise ValueError("Invalid label horizon/tolerance")
    frames = episode.frames
    times = np.array([f.timestamp_sec for f in frames])
    for f in frames:
        f.action = ActionLabel()
        f.canonical.linear_velocity_mps = None
        f.canonical.angular_velocity_radps = None
    for i, f in enumerate(frames):
        if not f.quality.valid:
            f.action.failure_reason = "invalid_current_state"
            continue
        if (
            i + 1 < len(frames)
            and frames[i + 1].quality.valid
            and frames[i + 1].frame_index == f.frame_index + 1
        ):
            nxt = frames[i + 1]
            dt = nxt.timestamp_sec - f.timestamp_sec
            f.canonical.linear_velocity_mps = (
                (
                    np.array(nxt.canonical.tip_position_surface_m)
                    - np.array(f.canonical.tip_position_surface_m)
                )
                / dt
            ).tolist()
            if f.quality.full_pose_valid and nxt.quality.full_pose_valid:
                a = np.array(f.canonical.T_surface_tool)[:3, :3]
                b = np.array(nxt.canonical.T_surface_tool)[:3, :3]
                f.canonical.angular_velocity_radps = (
                    Rotation.from_matrix(b @ a.T).as_rotvec() / dt
                ).tolist()
        candidates = np.arange(i + 1, len(frames))
        if not len(candidates):
            continue
        j = int(candidates[np.argmin(abs(times[candidates] - (times[i] + horizon_sec)))])
        if abs(times[j] - times[i] - horizon_sec) > tolerance_sec + 1e-9:
            continue
        if not all(x.quality.full_pose_valid for x in frames[i : j + 1]):
            f.action.failure_reason = "invalid_pose_in_horizon"
            continue
        if any(
            b.frame_index != a.frame_index + 1 for a, b in zip(frames[i:j], frames[i + 1 : j + 1])
        ):
            f.action.failure_reason = "missing_frame_in_horizon"
            continue
        a = np.asarray(f.canonical.T_surface_tool)
        b = np.asarray(frames[j].canonical.T_surface_tool)
        delta = inverse(a) @ b
        f.action = ActionLabel(
            valid=True,
            target_timestamp_sec=float(times[j]),
            horizon_actual_sec=float(times[j] - times[i]),
            target_T_surface_tool=b.tolist(),
            delta_position_surface_m=(b[:3, 3] - a[:3, 3]).tolist(),
            delta_T_tool_current_tool_future=delta.tolist(),
            delta_rotation_tool_current_rad=Rotation.from_matrix(delta[:3, :3])
            .as_rotvec()
            .tolist(),
            failure_reason=None,
        )

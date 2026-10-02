"""Independent-reference errors without fitting away prediction error."""

import numpy as np
from scipy.spatial.transform import Rotation

from .schema import Episode


def _stats(values, rmse=False):
    if not values:
        return None
    x = np.asarray(values)
    return float(np.sqrt(np.mean(x * x)) if rmse else np.mean(x))


def evaluate(pred: Episode, gt: Episode, tolerance_sec=0.02) -> dict:
    if not np.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("Timestamp tolerance must be finite and nonnegative")
    if pred.surface_frame_id != gt.surface_frame_id:
        raise ValueError("Evaluation requires the same calibrated physical surface frame")
    if pred.task_name != gt.task_name:
        raise ValueError("Evaluation task mismatch")
    times = np.array([f.timestamp_sec for f in gt.frames])
    used = set()
    position: list[float] = []
    axis: list[float] = []
    rotation: list[float] = []
    standoff: list[float] = []
    incidence: list[float] = []
    action_position: list[float] = []
    matched, valid, action_matched = 0, 0, 0
    for p in pred.frames:
        j = int(np.argmin(abs(times - p.timestamp_sec)))
        if abs(times[j] - p.timestamp_sec) > tolerance_sec or j in used:
            continue
        used.add(j)
        matched += 1
        g = gt.frames[j]
        if not p.quality.valid or not g.quality.valid:
            continue
        valid += 1
        position.append(
            float(
                np.linalg.norm(
                    np.array(p.canonical.tip_position_surface_m)
                    - np.array(g.canonical.tip_position_surface_m)
                )
            )
        )
        dot = np.dot(
            np.asarray(p.canonical.tool_axis_surface), np.asarray(g.canonical.tool_axis_surface)
        )
        axis.append(float(np.arccos(np.clip(dot, -1, 1))))
        if p.quality.full_pose_valid and g.quality.full_pose_valid:
            a = np.array(p.canonical.T_surface_tool)[:3, :3]
            b = np.array(g.canonical.T_surface_tool)[:3, :3]
            rotation.append(float(Rotation.from_matrix(a @ b.T).magnitude()))
        for key, container in [
            ("standoff_distance_m", standoff),
            ("incidence_angle_rad", incidence),
        ]:
            pv, gv = getattr(p.canonical, key), getattr(g.canonical, key)
            if pv is not None and gv is not None:
                container.append(abs(pv - gv))
        if (
            p.action.valid
            and g.action.valid
            and p.action.target_timestamp_sec is not None
            and g.action.target_timestamp_sec is not None
            and abs(p.action.target_timestamp_sec - g.action.target_timestamp_sec) <= tolerance_sec
        ):
            action_matched += 1
            action_position.append(
                float(
                    np.linalg.norm(
                        np.array(p.action.delta_position_surface_m)
                        - np.array(g.action.delta_position_surface_m)
                    )
                )
            )
    return {
        "reference_data_kind": gt.source.data_kind,
        "reference_provenance": gt.provenance,
        "predicted_frames": len(pred.frames),
        "reference_frames": len(gt.frames),
        "matched_frames": matched,
        "matched_valid_frames": valid,
        "valid_coverage": valid / len(gt.frames),
        "prediction_valid_ratio": sum(f.quality.valid for f in pred.frames) / len(pred.frames),
        "position_rmse_m": _stats(position, rmse=True),
        "position_mae_m": _stats(position),
        "axis_mae_rad": _stats(axis),
        "rotation_mae_rad": _stats(rotation),
        "rotation_evaluated_frames": len(rotation),
        "standoff_mae_m": _stats(standoff),
        "incidence_mae_rad": _stats(incidence),
        "matched_actions": action_matched,
        "action_delta_position_rmse_m": _stats(action_position, True),
        "timestamp_tolerance_sec": tolerance_sec,
        "alignment": "explicit_same_surface_frame_no_trajectory_fitting",
    }

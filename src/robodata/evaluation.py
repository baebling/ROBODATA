"""Independent-reference errors without fitting away prediction error."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial.transform import Rotation

from .schema import Episode


def _stats(values, rmse=False):
    if not values:
        return None
    x = np.asarray(values)
    return float(np.sqrt(np.mean(x * x)) if rmse else np.mean(x))


def _match_timestamps(pred_times, gt_times, tolerance_sec) -> list[tuple[int, int]]:
    n, m = len(pred_times), len(gt_times)
    rows, columns, costs = [], [], []
    for i, timestamp in enumerate(pred_times):
        lo = int(np.searchsorted(gt_times, timestamp - tolerance_sec, side="left"))
        hi = int(np.searchsorted(gt_times, timestamp + tolerance_sec, side="right"))
        for j in range(lo, hi):
            distance = abs(timestamp - gt_times[j])
            if distance <= tolerance_sec:
                rows.append(i)
                columns.append(j)
                costs.append((distance / tolerance_sec if tolerance_sec else 0) + 1e-12)
        rows.append(i)
        columns.append(m + i)
        costs.append(float(min(n, m) + 1))
    # Private dummy columns make every row assignable. Their penalty prioritizes
    # match count, then timestamp distance, without a dense video-by-tracker matrix.
    graph = csr_matrix((costs, (rows, columns)), shape=(n, m + n))
    assigned_rows, assigned_columns = min_weight_full_bipartite_matching(graph)
    pairs = [(int(i), int(j)) for i, j in zip(assigned_rows, assigned_columns) if j < m]
    # Ordered times with a common tolerance allow uncrossing equal-cost ties.
    return list(zip(sorted(i for i, _ in pairs), sorted(j for _, j in pairs)))


def evaluate(pred: Episode, gt: Episode, tolerance_sec=0.02) -> dict:
    if not np.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("Timestamp tolerance must be finite and nonnegative")
    if pred.surface_frame_id != gt.surface_frame_id:
        raise ValueError("Evaluation requires the same calibrated physical surface frame")
    if pred.task_name != gt.task_name:
        raise ValueError("Evaluation task mismatch")
    times = np.array([f.timestamp_sec for f in gt.frames])
    pred_times = np.array([f.timestamp_sec for f in pred.frames])
    position: list[float] = []
    axis: list[float] = []
    rotation: list[float] = []
    standoff: list[float] = []
    incidence: list[float] = []
    action_position: list[float] = []
    matched, valid, action_matched = 0, 0, 0
    for i, j in _match_timestamps(pred_times, times, tolerance_sec):
        matched += 1
        p = pred.frames[i]
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
        "timestamp_matching": "max_cardinality_min_time_error_ordered_one_to_one",
        "alignment": "explicit_same_surface_frame_no_trajectory_fitting",
    }

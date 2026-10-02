"""Portable validated JSON and masked training tensors."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .schema import Episode


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def read_episode(path: Path | str) -> Episode:
    return Episode.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_episode(episode: Episode, path: Path) -> None:
    # Revalidate mutations made by the temporal stage before serialization.
    checked = Episode.model_validate(episode.model_dump())
    write_json(path, checked.model_dump(mode="json"))


def export_tensors(episode: Episode, path: Path) -> None:
    n = len(episode.frames)
    state, target, delta = np.zeros((n, 12)), np.zeros((n, 12)), np.zeros((n, 6))
    state_valid: NDArray[np.bool_] = np.zeros(n, bool)
    action_valid: NDArray[np.bool_] = np.zeros(n, bool)
    axis_state = np.zeros((n, 6))
    axis_valid: NDArray[np.bool_] = np.zeros(n, bool)
    for i, f in enumerate(episode.frames):
        if f.quality.valid:
            axis_state[i] = np.r_[f.canonical.tip_position_surface_m, f.canonical.tool_axis_surface]
            axis_valid[i] = True
        if f.quality.full_pose_valid:
            T = np.asarray(f.canonical.T_surface_tool)
            state[i] = np.r_[T[:3, 3], T[:3, :3].ravel()]
            state_valid[i] = True
        if f.action.valid:
            T = np.asarray(f.action.target_T_surface_tool)
            target[i] = np.r_[T[:3, 3], T[:3, :3].ravel()]
            D = np.asarray(f.action.delta_T_tool_current_tool_future)
            delta[i] = np.r_[D[:3, 3], Rotation.from_matrix(D[:3, :3]).as_rotvec()]
            action_valid[i] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps_sec=[f.timestamp_sec for f in episode.frames],
        state=state,
        target=target,
        delta_tool=delta,
        axis_state=axis_state,
        state_valid=state_valid,
        action_valid=action_valid,
        axis_valid=axis_valid,
        surface_frame_id=episode.surface_frame_id,
        schema_version=episode.schema_version,
    )

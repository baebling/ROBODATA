import numpy as np
import pytest

from robodata.schema import ActionLabel, Canonical, FrameRecord, Quality


def test_direction_vectors_must_have_unit_length():
    with pytest.raises(ValueError, match="unit"):
        Canonical(tool_axis_surface=[0, 0, 2])


def test_full_pose_flag_requires_valid_state():
    with pytest.raises(ValueError, match="valid"):
        Quality(valid=False, full_pose_valid=True)


def test_pose_and_functional_axis_cannot_disagree():
    with pytest.raises(ValueError, match="axis"):
        Canonical(T_surface_tool=np.eye(4).tolist(), tool_axis_surface=[0, 0, -1])


def test_pose_and_functional_origin_cannot_disagree():
    with pytest.raises(ValueError, match="origin"):
        Canonical(T_surface_tool=np.eye(4).tolist(), tip_position_surface_m=[0, 0, 1])


def test_valid_action_requires_complete_target_and_positive_horizon():
    with pytest.raises(ValueError):
        ActionLabel(valid=True)
    with pytest.raises(ValueError):
        ActionLabel(horizon_actual_sec=0)


def test_action_validity_requires_valid_current_full_pose():
    action = ActionLabel(
        valid=True,
        target_timestamp_sec=0.2,
        horizon_actual_sec=0.2,
        target_T_surface_tool=np.eye(4).tolist(),
        delta_position_surface_m=[0, 0, 0],
        delta_T_tool_current_tool_future=np.eye(4).tolist(),
        delta_rotation_tool_current_rad=[0, 0, 0],
        failure_reason=None,
    )
    with pytest.raises(ValueError, match="current"):
        FrameRecord(frame_index=0, timestamp_sec=0, action=action)

import numpy as np
import pytest

from robodata.evaluation import evaluate
from robodata.schema import Canonical, Episode, FrameRecord, Quality, Source
from robodata.temporal import add_labels


def episode(points, times=None, invalid=()):
    times = times or [i * 0.1 for i in range(len(points))]
    frames = []
    for i, (t, x) in enumerate(zip(times, points)):
        T = np.diag([1.0, -1, -1, 1])
        T[:3, 3] = [x, 0, 0.3]
        valid = i not in invalid
        frames.append(
            FrameRecord(
                frame_index=i,
                timestamp_sec=t,
                canonical=Canonical(
                    T_surface_tool=T.tolist() if valid else None,
                    tip_position_surface_m=[x, 0, 0.3] if valid else None,
                    tool_axis_surface=[0, 0, -1] if valid else None,
                ),
                quality=Quality(
                    valid=valid, full_pose_valid=valid, failure_reason=None if valid else "occluded"
                ),
            )
        )
    return Episode(
        episode_id="a",
        task_name="spray_proxy",
        surface_frame_id="patch_A",
        source=Source(mode="rgb", metric_scale_source="measured_geometry", data_kind="synthetic"),
        frames=frames,
    )


def test_future_target_and_surface_displacement_use_seconds():
    data = episode([0, 0.05, 0.1, 0.15])
    add_labels(data, horizon_sec=0.2, tolerance_sec=0.001)
    np.testing.assert_allclose(data.frames[0].action.delta_position_surface_m, [0.1, 0, 0])
    assert data.frames[0].action.target_timestamp_sec == pytest.approx(0.2)
    np.testing.assert_allclose(data.frames[1].canonical.linear_velocity_mps, [0.5, 0, 0])
    assert not data.frames[-1].action.valid


def test_invalid_intermediate_frame_blocks_future_label():
    data = episode([0, 0.05, 0.1], invalid=[1])
    add_labels(data, horizon_sec=0.2, tolerance_sec=0.001)
    assert not data.frames[0].action.valid
    assert data.frames[0].canonical.linear_velocity_mps is None


def test_timestamp_tolerance_does_not_invent_future_samples():
    data = episode([0, 0.1, 0.2], times=[0, 0.13, 0.4])
    add_labels(data, horizon_sec=0.2, tolerance_sec=0.01)
    assert not data.frames[0].action.valid


def test_repeated_timestamps_are_rejected():
    with pytest.raises(ValueError):
        episode([0, 1], times=[0, 0])


def test_eval_includes_missing_frames_and_preserves_real_error():
    gt = episode([0, 0.1, 0.2])
    pred = episode([0.01, 0.11, 0.21], invalid=[1])
    result = evaluate(pred, gt, tolerance_sec=0.01)
    assert result["matched_valid_frames"] == 2
    assert result["valid_coverage"] == pytest.approx(2 / 3)
    assert result["position_rmse_m"] == pytest.approx(0.01)
    assert result["axis_mae_rad"] == pytest.approx(0)


def test_eval_rejects_different_surface_reference():
    pred = episode([0, 0.1])
    gt = episode([0, 0.1])
    gt.surface_frame_id = "patch_B"
    with pytest.raises(ValueError):
        evaluate(pred, gt)


def test_skipped_frame_indices_block_motion_labels():
    data = episode([0, 0.1, 0.2])
    data.frames[1].frame_index = 2
    data.frames[2].frame_index = 3
    add_labels(data, horizon_sec=0.2, tolerance_sec=0.001)
    assert data.frames[0].canonical.linear_velocity_mps is None
    assert not data.frames[0].action.valid


def test_body_delta_translation_is_not_surface_displacement():
    data = episode([0, 0.1])
    rotation = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])
    for frame in data.frames:
        T = np.array(frame.canonical.T_surface_tool)
        T[:3, :3] = rotation
        frame.canonical.T_surface_tool = T.tolist()
    add_labels(data, horizon_sec=0.1, tolerance_sec=0.001)
    action = data.frames[0].action
    np.testing.assert_allclose(action.delta_position_surface_m, [0.1, 0, 0])
    np.testing.assert_allclose(
        np.array(action.delta_T_tool_current_tool_future)[:3, 3], [0, 0.1, 0]
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_temporal_parameters_are_rejected(value):
    data = episode([0, 0.1])
    with pytest.raises(ValueError):
        add_labels(data, horizon_sec=value, tolerance_sec=0.01)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_evaluation_tolerance_is_rejected(value):
    data = episode([0, 0.1])
    with pytest.raises(ValueError):
        evaluate(data, data, tolerance_sec=value)

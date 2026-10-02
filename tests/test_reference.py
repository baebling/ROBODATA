import numpy as np
import pytest
import yaml

from robodata.reference import import_reference


def test_marker_calibration_and_clock_offset_are_applied(tmp_path):
    csv = tmp_path / "mocap.csv"
    csv.write_text(
        "frame_index,timestamp_sec,x_m,y_m,z_m,qx,qy,qz,qw\n0,0,1,0,0,0,0,0,1\n1,.1,1.1,0,0,0,0,0,1\n"
    )

    tracking = np.eye(4)
    tracking[0, 3] = -1
    marker_tool = np.eye(4)
    marker_tool[2, 3] = 0.3
    config = tmp_path / "calibration.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "episode_id": "gold",
                "task_name": "spray",
                "surface_frame_id": "patch",
                "instrument": "mocap",
                "time_offset_sec": 0.05,
                "T_surface_tracking": tracking.tolist(),
                "T_marker_tool": marker_tool.tolist(),
            }
        )
    )
    result = import_reference(csv, config, tmp_path / "gold.json")
    np.testing.assert_allclose(result.frames[0].canonical.tip_position_surface_m, [0, 0, 0.3])
    assert result.frames[0].timestamp_sec == 0.05
    assert result.source.data_kind == "real"
    assert result.provenance["instrument"] == "mocap"


@pytest.mark.parametrize("with_indices", [True, False])
def test_tracker_gaps_are_not_renumbered_as_contiguous(tmp_path, with_indices):
    csv = tmp_path / "tracker.csv"
    header = "timestamp_sec,x_m,y_m,z_m,qx,qy,qz,qw"
    rows = ["0,0,0,.3,0,0,0,1", ".01,.01,0,.3,0,0,0,1", ".2,.2,0,.3,0,0,0,1"]
    if with_indices:
        header = "frame_index," + header
        rows = [f"{i},{row}" for i, row in zip([0, 1, 20], rows)]
    csv.write_text(header + "\n" + "\n".join(rows) + "\n")
    config = tmp_path / "calibration.yaml"
    cfg = {
        "episode_id": "gold",
        "task_name": "spray",
        "surface_frame_id": "patch",
        "instrument": "tracker",
        "T_surface_tracking": np.eye(4).tolist(),
        "T_marker_tool": np.eye(4).tolist(),
    }
    if not with_indices:
        cfg["sample_period_sec"] = 0.01
    config.write_text(yaml.safe_dump(cfg))
    result = import_reference(csv, config, tmp_path / "gold.json")
    assert [f.frame_index for f in result.frames] == [0, 1, 20]
    assert not result.frames[0].action.valid
    assert result.frames[1].canonical.linear_velocity_mps is None


def test_gt_without_source_indices_or_cadence_is_rejected(tmp_path):
    csv = tmp_path / "tracker.csv"
    csv.write_text("timestamp_sec,x_m,y_m,z_m,qx,qy,qz,qw\n0,0,0,.3,0,0,0,1\n")
    config = tmp_path / "calibration.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "episode_id": "gold",
                "task_name": "spray",
                "surface_frame_id": "patch",
                "instrument": "tracker",
                "T_surface_tracking": np.eye(4).tolist(),
                "T_marker_tool": np.eye(4).tolist(),
            }
        )
    )
    with pytest.raises(ValueError, match="frame_index.*sample_period"):
        import_reference(csv, config, tmp_path / "gold.json")

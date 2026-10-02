import numpy as np

from robodata.reference import import_reference


def test_marker_calibration_and_clock_offset_are_applied(tmp_path):
    csv = tmp_path / "mocap.csv"
    csv.write_text("timestamp_sec,x_m,y_m,z_m,qx,qy,qz,qw\n0,1,0,0,0,0,0,1\n.1,1.1,0,0,0,0,0,1\n")
    import yaml

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

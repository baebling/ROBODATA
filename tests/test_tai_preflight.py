from pathlib import Path

import numpy as np
import pytest

from reproductions.tool_as_interface import preflight


@pytest.mark.parametrize(
    "matrix",
    [np.ones((4, 4)), np.diag([-1, 1, 1, 1]), np.eye(3), np.full((4, 4), np.nan)],
)
def test_calibration_rejects_non_se3_without_repair(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"SE\(3\)"):
        preflight.check_transform(matrix)


def test_calibration_accepts_proper_transform() -> None:
    matrix = np.eye(4)
    matrix[:3, 3] = [0.1, -0.2, 0.8]
    preflight.check_transform(matrix)


def test_missing_files_prevent_preprocessing_readiness(tmp_path: Path) -> None:
    report = preflight.check_assets(tmp_path, "camera_3")
    assert report["ready"] is False
    assert len(report["missing"]) >= 5
    assert "assets/mesh/hammer/textured_mesh.obj" in report["missing"]
    assert not list(tmp_path.iterdir())


def test_asset_check_flags_invalid_calibration(tmp_path: Path) -> None:
    path = tmp_path / "ti/real_world/robot_extrinsics/right_base_pose_in_world.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.ones((4, 4)))
    report = preflight.check_assets(tmp_path, "camera_3")
    assert "right_base_pose_in_world.npy" in report["invalid_calibrations"][0]
    assert not report["ready"]


def test_asset_check_rejects_unknown_camera(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="camera"):
        preflight.check_assets(tmp_path, "helmet_rgb")

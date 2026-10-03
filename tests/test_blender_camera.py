import importlib

import numpy as np
import pytest


def camera_module():
    return importlib.import_module("robodata.blender_camera")


def test_nominal_sensor_axes_produce_distinct_focal_lengths():
    cfg = {
        "width": 960,
        "height": 540,
        "sensor_profile": {"horizontal_fov_deg": 110, "vertical_fov_deg": 70},
    }
    K = camera_module().camera_intrinsics(cfg)
    assert K[0, 0] == pytest.approx(960 / (2 * np.tan(np.radians(55))))
    assert K[1, 1] == pytest.approx(540 / (2 * np.tan(np.radians(35))))


def test_helmet_forward_mount_is_tied_to_head_and_attitude_not_target():
    cfg = {
        "frames": 8,
        "helmet_mount": {
            "forward_offset_m": 0.06,
            "up_offset_m": 0.04,
            "bob_m": 0.01,
            "yaw_amplitude_deg": 2,
            "pitch_amplitude_deg": 1.5,
        },
    }
    scenario = {
        "head_position_surface_m": [0, 0.42, 0.85],
        "helmet_pitch_deg": 30,
        "camera_motion": "moving",
    }
    position, direction = camera_module().helmet_camera_pose(cfg, scenario, 0)
    np.testing.assert_allclose(direction, [0, -0.5, -np.sqrt(0.75)])
    up = np.array([0, np.sqrt(0.75), -0.5])
    np.testing.assert_allclose(position, [0, 0.42, 0.85] + direction * 0.06 + up * 0.04)
    p2, d2 = camera_module().helmet_camera_pose(cfg, scenario, 2)
    assert not np.allclose(position, p2)
    assert not np.allclose(direction, d2)


@pytest.mark.parametrize("pitch", [0, 30, 50])
def test_helmet_mount_length_is_invariant_under_head_pitch(pitch):
    cfg = {"frames": 8, "helmet_mount": {"forward_offset_m": 0.06, "up_offset_m": 0.04}}
    scenario = {
        "head_position_surface_m": [0, 0.42, 0.85],
        "helmet_pitch_deg": pitch,
        "camera_motion": "static",
    }
    position, _ = camera_module().helmet_camera_pose(cfg, scenario, 0)
    assert np.linalg.norm(position - scenario["head_position_surface_m"]) == pytest.approx(
        np.hypot(0.06, 0.04)
    )


def test_sensor_profile_keeps_base_config_immutable():
    cfg = {
        "width": 1280,
        "height": 720,
        "sensor_profiles": {
            "helios": {
                "width": 640,
                "height": 480,
                "horizontal_fov_deg": 67,
                "vertical_fov_deg": 51,
                "depth_available": True,
            }
        },
    }
    result = camera_module().scenario_camera_config(cfg, {"sensor": "helios"})
    assert cfg["width"] == 1280
    assert result["width"] == 640 and result["height"] == 480
    assert result["sensor_profile"]["depth_available"] is True


def test_behind_or_invalid_fov_is_rejected():
    with pytest.raises(ValueError):
        camera_module().camera_intrinsics(
            {
                "width": 640,
                "height": 480,
                "sensor_profile": {"horizontal_fov_deg": 180, "vertical_fov_deg": 51},
            }
        )


@pytest.mark.parametrize("fx,fy", [(336.1, 385.6), (500, 400), (500, 500)])
def test_blender_pixel_aspect_preserves_ratio_without_clamping(fx, fy):
    x, y = camera_module().blender_pixel_aspect(np.diag([fx, fy, 1]))
    assert min(x, y) == pytest.approx(1)
    assert x >= 1 and y >= 1
    assert y / x == pytest.approx(fx / fy)

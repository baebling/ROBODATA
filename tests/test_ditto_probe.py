"""Quality gates for the optional official-DITTO experiment."""

import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def assess(*args, **kwargs):
    script = Path(__file__).resolve().parents[1] / "scripts/probe_ditto_blender.py"
    assert script.is_file(), "Optional DITTO experiment adapter has not been implemented"
    return runpy.run_path(str(script))["assess_fit"](*args, **kwargs)


def points():
    return np.random.default_rng(42).uniform(-0.1, 0.1, (80, 3)) + [0, 0, 1]


def test_transform_direction_maps_source_points_into_target():
    source = points()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix()
    transform[:3, 3] = [0.05, -0.02, 0.03]
    target = source @ transform[:3, :3].T + transform[:3, 3]
    result = assess(source, target, transform)
    assert result["valid"]
    assert result["inlier_fraction"] == 1
    assert result["inlier_rmse_m"] < 1e-12
    assert not assess(source, target, np.linalg.inv(transform))["valid"]


def test_identity_is_valid_for_observed_stationary_motion():
    source = points()
    assert assess(source, source, np.eye(4))["valid"]


def test_upstream_failure_identity_is_not_a_stationary_estimate():
    source = points()
    result = assess(source, source, np.eye(4), upstream_failed=True)
    assert not result["valid"]
    assert result["failure_reason"] == "upstream_fit_failed"


@pytest.mark.parametrize("count", [0, 2, 10])
def test_insufficient_correspondences_are_invalid(count):
    source = points()[:count]
    result = assess(source, source, np.eye(4))
    assert not result["valid"]
    assert result["failure_reason"] == "insufficient_correspondences"


def test_collinear_points_do_not_constrain_full_rotation():
    source = np.column_stack([np.linspace(0, 0.1, 40), np.zeros(40), np.ones(40)])
    result = assess(source, source, np.eye(4))
    assert not result["valid"]
    assert result["failure_reason"] == "degenerate_inliers"


def test_nonfinite_fit_and_nonrigid_transform_are_invalid():
    source = points()
    transform = np.eye(4)
    transform[0, 0] = 2
    assert not assess(source, source, transform)["valid"]
    source[0, 0] = np.nan
    assert not assess(source, source, np.eye(4))["valid"]


def test_prepared_metric_axial_depth_keeps_global_camera_rays(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/probe_ditto_blender.py"
    rgb = np.zeros((5, 7, 3), dtype=np.uint8)
    labels = np.full((5, 7), 2, dtype=np.uint8)
    depth = np.full((5, 7), 2, dtype=np.float32)
    depth[0, 0] = 0
    cv2.imwrite(str(tmp_path / "rgb.png"), rgb)
    cv2.imwrite(str(tmp_path / "labels.png"), labels)
    np.save(tmp_path / "depth.npy", depth)
    matrix = np.array([[10, 0, 3], [0, 10, 2], [0, 0, 1]])
    _, xyz, decoded_labels = runpy.run_path(str(script))["read_observation"](
        tmp_path,
        {"rgb_path": "rgb.png", "labels_path": "labels.png"},
        matrix,
        tmp_path / "depth.npy",
    )
    np.testing.assert_allclose(xyz[3, 4], [0.2, 0.2, 2])
    assert np.isnan(xyz[0, 0]).all()
    np.testing.assert_array_equal(decoded_labels, labels)


def test_disappearing_target_tool_is_invalid_before_registration():
    script = Path(__file__).resolve().parents[1] / "scripts/probe_ditto_blender.py"
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    xyz = np.ones((30, 30, 3))
    source = np.full((30, 30), 2, dtype=np.uint8)
    target = np.zeros((30, 30), dtype=np.uint8)
    result, transform = runpy.run_path(str(script))["probe_pair"](
        (image, xyz, source),
        (image, xyz, target),
        {},
        None,
        None,
        None,
    )
    assert not result["valid"]
    assert result["failure_reason"] == "empty_target_tool_mask"
    assert transform is None


def test_target_mask_filter_rejects_background_and_out_of_bounds():
    script = Path(__file__).resolve().parents[1] / "scripts/probe_ditto_blender.py"
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 3] = True
    source = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    target = np.array([[2, 3], [1, 1], [-1, 2], [2, 5]])
    a, b, scores = runpy.run_path(str(script))["filter_target_tool_matches"](
        source,
        target,
        np.array([0.9, 0.8, 0.7, 0.6]),
        mask,
    )
    np.testing.assert_array_equal(a, source[:1])
    np.testing.assert_array_equal(b, target[:1])
    np.testing.assert_array_equal(scores, [0.9])


@pytest.mark.parametrize("changed_field", [None, "hash", "units", "episode"])
def test_prepared_depth_is_bound_to_source_and_declared_meters(changed_field):
    script = Path(__file__).resolve().parents[1] / "scripts/probe_ditto_blender.py"
    matrix = [[10, 0, 3], [0, 10, 2], [0, 0, 1]]
    metadata = {"episode_id": "a", "width": 7, "height": 5, "camera_matrix": matrix}
    manifest = {
        "episode_id": "a",
        "data_kind": "synthetic",
        "depth_aligned_to_rgb": True,
        "camera": {"matrix": matrix, "width": 7, "height": 5},
        "provenance": {
            "render_metadata_sha256": "correct",
            "depth_convention_output": "axial_z_meters",
        },
    }
    if changed_field == "hash":
        manifest["provenance"]["render_metadata_sha256"] = "another_episode"
    elif changed_field == "units":
        manifest["provenance"]["depth_convention_output"] = "millimeters"
    elif changed_field == "episode":
        manifest["episode_id"] = "b"
    validate = runpy.run_path(str(script))["validate_prepared_depth"]
    if changed_field is None:
        validate(manifest, metadata, "correct")
    else:
        with pytest.raises(ValueError):
            validate(manifest, metadata, "correct")

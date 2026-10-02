import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robodata.geometry import estimate_pnp, fit_plane, inverse, validate_transform


def test_pnp_recovers_metric_pose_from_known_geometry():
    # Wrong transform direction or scale makes the 0.7m literal fail.
    xyz = np.array(
        [
            [0, 0, 0],
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0.1, 0.1, 0],
            [0, 0, 0.1],
            [0.1, 0, 0.1],
            [0, 0.1, 0.1],
            [0.1, 0.1, 0.1],
        ]
    )
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    uv, _ = cv2.projectPoints(
        xyz, np.array([0.1, -0.2, 0.05]), np.array([0.03, 0.02, 0.7]), K, np.zeros(5)
    )
    result = estimate_pnp(xyz, uv.reshape(-1, 2), K, max_error_px=2)
    np.testing.assert_allclose(result.transform[:3, 3], [0.03, 0.02, 0.7], atol=1e-5)
    assert result.residual < 1e-4


def test_plane_outliers_do_not_change_standoff_reference():
    rng = np.random.default_rng(4)
    points = np.c_[rng.uniform(-0.4, 0.4, (200, 2)), np.ones(200)]
    points = np.vstack([points, rng.uniform(-2, 2, (30, 3))])
    result = fit_plane(points, threshold_m=0.005, min_inlier_ratio=0.7, seed=5)
    np.testing.assert_allclose(result.normal, [0, 0, -1], atol=1e-4)
    assert abs(result.origin[2] - 1) < 1e-4
    assert result.inlier_ratio > 0.8


def test_reflection_and_bad_homogeneous_row_are_rejected():
    reflection = np.diag([1.0, 1, -1, 1])
    with pytest.raises(ValueError):
        validate_transform(reflection)
    bad = np.eye(4)
    bad[3, 0] = 0.1
    with pytest.raises(ValueError):
        inverse(bad)


def test_sparse_or_collinear_plane_is_not_valid():
    with pytest.raises(ValueError):
        fit_plane(np.zeros((3, 3)), threshold_m=0.01)
    with pytest.raises(ValueError):
        fit_plane(np.c_[np.arange(30), np.zeros((30, 2))], threshold_m=0.01)


def test_frontal_surface_with_working_normal_toward_camera():
    xyz = np.array([[-0.35, -0.25, 0], [0.35, -0.25, 0], [0.35, 0.25, 0], [-0.35, 0.25, 0]])
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    uv = np.array([[55.0, 195], [265, 195], [265, 45], [55, 45]])
    result = estimate_pnp(xyz, uv, K)
    np.testing.assert_allclose(result.transform[:3, 2], [0, 0, -1], atol=1e-6)


def test_pnp_surface_relative_pose_is_invariant_to_camera_motion():
    tool_points = np.array(
        [[x, y, z] for x in [-0.03, 0.03] for y in [-0.02, 0.02] for z in [-0.06, 0]]
    )
    surface_points = np.array([[-0.3, -0.2, 0], [0.3, -0.2, 0], [0.3, 0.2, 0], [-0.3, 0.2, 0]])
    K = np.array([[400.0, 0, 160], [0, 400.0, 120], [0, 0, 1]])
    surface = np.diag([1.0, -1, -1, 1])
    surface[2, 3] = 1
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_euler("xyz", [0.1, 0.4, -0.1]).as_matrix()
    tool[:3, 3] = [0.05, 0.03, 0.7]
    moved_camera = np.eye(4)
    moved_camera[:3, :3] = Rotation.from_euler("xyz", [0.08, -0.1, 0.05]).as_matrix()
    moved_camera[:3, 3] = [0.02, -0.03, 0.1]
    expected = inverse(surface) @ tool
    for change in [np.eye(4), moved_camera]:
        estimates = []
        for points, pose in [(surface_points, surface), (tool_points, tool)]:
            camera_pose = change @ pose
            pixels, _ = cv2.projectPoints(
                points,
                Rotation.from_matrix(camera_pose[:3, :3]).as_rotvec(),
                camera_pose[:3, 3],
                K,
                np.zeros(5),
            )
            estimates.append(estimate_pnp(points, pixels.reshape(-1, 2), K).transform)
        np.testing.assert_allclose(inverse(estimates[0]) @ estimates[1], expected, atol=1e-5)

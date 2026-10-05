import cv2
import numpy as np
import pytest


def test_rgb_marker_detection_uses_pixels_and_requires_all_four():
    from robodata.surface_tracking import detect_surface_markers

    image = np.full((120, 160, 3), 100, np.uint8)
    for xy, color in zip(
        [(30, 90), (130, 90), (130, 30), (30, 30)],
        [(0, 0, 240), (0, 240, 0), (240, 0, 0), (0, 210, 240)],
        strict=True,
    ):
        cv2.circle(image, xy, 3, color, -1)
    points = detect_surface_markers(image)
    np.testing.assert_allclose(points, [[30, 90], [130, 90], [130, 30], [30, 30]], atol=0.5)
    image[25:36, 25:36] = 100
    with pytest.raises(ValueError):
        detect_surface_markers(image)


def test_pixel_rays_recover_known_plane_without_depth():
    from robodata.surface_tracking import pixels_to_surface

    K = np.array([[100, 0, 80], [0, 100, 60], [0, 0, 1]], float)
    T = np.eye(4)
    T[:3, 3] = [0, 0, 2]
    xyz = pixels_to_surface([[80, 60], [90, 70]], K, T)
    np.testing.assert_allclose(xyz, [[0, 0, 0], [0.2, 0.2, 0]], atol=1e-8)


def test_empty_mask_does_not_count_as_perfect_detection():
    from robodata.surface_tracking import mask_scores

    blank = np.zeros((20, 30), bool)
    assert mask_scores(blank, blank)["iou"] is None
    target = blank.copy()
    target[3:8, 4:9] = True
    assert mask_scores(blank, target)["iou"] == 0.0
    assert mask_scores(target, blank)["false_positive_pixels"] == 25


def _textured_plane():
    image = np.zeros((200, 240, 3), np.uint8)
    rng = np.random.default_rng(5)
    for xy in rng.integers([20, 20], [220, 180], (120, 2)):
        cv2.circle(image, tuple(map(int, xy)), 2, (200, 200, 200), -1)
    K = np.array([[200, 0, 120], [0, 200, 100], [0, 0, 1]], float)
    T = np.eye(4)
    T[:3, 3] = [0, 0, 2]
    return image, K, T


def test_persistent_map_tracks_camera_instead_of_freezing_pose():
    from robodata.surface_tracking import PlanarSurfaceMap

    image, K, T = _textured_plane()
    mask = np.ones(image.shape[:2], bool)
    tracker = PlanarSurfaceMap(K, max_fb_error_px=1.0, max_reprojection_px=2.0)
    tracker.initialize(image, mask, T)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, 0]]), (240, 200))
    pose, diagnostics = tracker.update(shifted, mask)
    assert pose is not None
    assert diagnostics["inliers"] >= 6
    assert pose[0, 3] == pytest.approx(0.04, abs=0.008)
    assert pose[2, 3] == pytest.approx(2, abs=0.03)


def test_full_occlusion_is_invalid_not_a_copied_previous_pose():
    from robodata.surface_tracking import PlanarSurfaceMap

    image, K, T = _textured_plane()
    tracker = PlanarSurfaceMap(K)
    tracker.initialize(image, np.ones(image.shape[:2], bool), T)
    pose, diagnostics = tracker.update(image, np.zeros(image.shape[:2], bool))
    assert pose is None
    assert diagnostics["failure_reason"]
    recovered, _ = tracker.update(image, np.ones(image.shape[:2], bool))
    assert recovered is not None


def test_uninitialized_map_and_bad_calibration_fail_closed():
    from robodata.surface_tracking import PlanarSurfaceMap

    image, K, T = _textured_plane()
    tracker = PlanarSurfaceMap(K)
    pose, _ = tracker.update(image, np.ones(image.shape[:2], bool))
    assert pose is None
    T[0, 0] = 2
    with pytest.raises(ValueError):
        tracker.initialize(image, np.ones(image.shape[:2], bool), T)

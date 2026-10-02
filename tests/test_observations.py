import cv2
import numpy as np

from robodata.config import ObjectSpec, RunConfig
from robodata.observations import Camera, Detection, FlowTracker, Keypoint, depth_tool


def test_tracking_keeps_supplied_masks_without_keypoints():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = FlowTracker(1.0).update(image, Detection(mask_path="surface.png"))
    assert result.mask_path == "surface.png"


def test_forward_backward_tracking_and_occlusion_reset():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    translated = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (80, 80))
    tracker = FlowTracker(0.5)
    tracker.update(image, Detection(keypoints={"tip": Keypoint(uv=[30, 30])}))
    tracked = tracker.update(translated, Detection(mask_path="tool.png"))
    np.testing.assert_allclose(tracked.keypoints["tip"].uv, [32, 31], atol=0.2)
    assert tracked.mask_path == "tool.png"
    assert tracker.update(translated, Detection(occluded=True)).occluded
    assert not tracker.update(translated, Detection()).keypoints


def test_axis_only_depth_does_not_invent_roll():
    camera = Camera(matrix=[[100, 0, 50], [0, 100, 50], [0, 0, 1]], width=100, height=100)
    config = RunConfig(
        manifest_path="manifest.json",
        mode="rgbd",
        metric_scale_source="depth_sensor",
        surface_method="depth_plane",
        camera_motion="static",
        tool=ObjectSpec(),
        surface=ObjectSpec(),
    )
    detection = Detection(
        keypoints={"tip": Keypoint(uv=[50, 50]), "axis_base": Keypoint(uv=[40, 50])}
    )
    estimate = depth_tool(detection, np.full((100, 100), 0.7), camera, config)
    assert estimate.T_camera_frame is None
    np.testing.assert_allclose(estimate.origin_camera_m, [0, 0, 0.7])
    np.testing.assert_allclose(estimate.axis_camera, [1, 0, 0])

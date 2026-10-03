from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from robodata import hand_assistance
from robodata.observations import Camera, Detection, Keypoint


@pytest.fixture
def calibrated_grasp():
    # Independent, nonplanar virtual calibration, not measured Graco geometry.
    points = {
        "0": [0.0, 0.0, 0.0],
        "1": [-0.032, 0.025, 0.018],
        "5": [-0.027, 0.073, 0.005],
        "9": [0.0, 0.084, 0.011],
        "13": [0.026, 0.076, 0.002],
        "17": [0.043, 0.063, -0.008],
    }
    camera = Camera(
        matrix=[[820, 0, 320], [0, 810, 240], [0, 0, 1]],
        width=640,
        height=480,
        distortion=[0.03, -0.01, 0.001, -0.002, 0],
    )
    rvec = np.array([0.23, -0.31, 0.17])
    translation = np.array([0.025, -0.055, 0.72])
    pixels, _ = cv2.projectPoints(
        np.array(list(points.values())),
        rvec,
        translation,
        np.array(camera.matrix, float),
        np.array(camera.distortion),
    )
    detection = Detection(
        handedness="Right",
        keypoints={
            name: Keypoint(uv=uv.tolist(), confidence=0.9)
            for name, uv in zip(points, pixels.reshape(-1, 2))
        },
    )
    grasp = np.eye(4)
    grasp[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    grasp[:3, 3] = [0.04, 0.02, -0.09]
    spec = SimpleNamespace(
        points_hand_m=points,
        T_hand_tool=grasp.tolist(),
        calibration_source="synthetic_fixed_grasp_virtual_calibration",
        grasp_verified=True,
        handedness="Right",
    )
    cfg = SimpleNamespace(min_keypoint_confidence=0.5, max_reprojection_error_px=2.0)
    rotation, _ = cv2.Rodrigues(rvec)
    # Derive expected functional point/axis independently of transform composition.
    expected_origin = rotation @ np.array([0.04, 0.02, -0.09]) + translation
    expected_axis = rotation @ np.array([1.0, 0.0, 0.0])
    return detection, spec, camera, cfg, expected_origin, expected_axis


def test_recovers_calibrated_functional_point_and_axis(calibrated_grasp):
    detection, spec, camera, cfg, origin, axis = calibrated_grasp
    detection.keypoints["unconfigured"] = Keypoint(uv=[1, 2])
    result = hand_assistance.hand_tool_observation(detection, spec, camera, cfg)
    np.testing.assert_allclose(result.origin_camera_m, origin, atol=1e-6)
    np.testing.assert_allclose(result.axis_camera, axis, atol=1e-6)
    np.testing.assert_allclose(np.array(result.T_camera_frame)[:3, 3], origin, atol=1e-6)
    np.testing.assert_allclose(np.array(result.T_camera_frame)[:3, 2], axis, atol=1e-6)
    assert result.method == "calibrated_hand_grasp_pnp"
    assert result.residual_unit == "pixel"
    assert result.residual < 1e-5
    assert 0 < result.confidence <= 1
    assert result.metric_scale_source == "calibrated_hand_template"
    assert result.calibration_source == spec.calibration_source


@pytest.mark.parametrize("handedness", ["Left", None])
def test_opposite_or_unknown_hand_cannot_use_calibrated_template(calibrated_grasp, handedness):
    detection, spec, camera, cfg, *_ = calibrated_grasp
    detection.handedness = handedness
    with pytest.raises(ValueError, match="handedness"):
        hand_assistance.hand_tool_observation(detection, spec, camera, cfg)


@pytest.mark.parametrize("source", ["", "   ", "unknown", "UNKNOWN", "unverified", None])
def test_rejects_unknown_calibration(calibrated_grasp, source):
    detection, spec, camera, cfg, *_ = calibrated_grasp
    spec.calibration_source = source
    with pytest.raises(ValueError, match="calibration"):
        hand_assistance.hand_tool_observation(detection, spec, camera, cfg)


def test_rejects_unverified_grasp(calibrated_grasp):
    detection, spec, camera, cfg, *_ = calibrated_grasp
    spec.grasp_verified = False
    with pytest.raises(ValueError, match="grasp"):
        hand_assistance.hand_tool_observation(detection, spec, camera, cfg)


@pytest.mark.parametrize(
    "bad_transform",
    [
        np.diag([-1, 1, 1, 1]),
        np.diag([2, 1, 1, 1]),
        np.eye(3),
        np.full((4, 4), np.nan),
        np.diag([1, 1, 1, 2]),
    ],
)
def test_rejects_improper_grasp_transform(calibrated_grasp, bad_transform):
    detection, spec, camera, cfg, *_ = calibrated_grasp
    spec.T_hand_tool = bad_transform.tolist()
    with pytest.raises(ValueError):
        hand_assistance.hand_tool_observation(detection, spec, camera, cfg)


@pytest.mark.parametrize("gate", ["missing", "invisible", "low_confidence", "occluded"])
def test_rejects_missing_or_gated_palm_predictions(calibrated_grasp, gate):
    detection, spec, camera, cfg, *_ = calibrated_grasp
    if gate == "missing":
        del detection.keypoints["9"]
    elif gate == "invisible":
        detection.keypoints["9"].visible = False
    elif gate == "low_confidence":
        detection.keypoints["9"].confidence = 0.49
    else:
        detection.occluded = True
    with pytest.raises(ValueError):
        hand_assistance.hand_tool_observation(detection, spec, camera, cfg)


def test_rejects_nonplanar_geometry_with_only_five_predictions(calibrated_grasp):
    detection, spec, camera, cfg, *_ = calibrated_grasp
    del detection.keypoints["1"]
    with pytest.raises(ValueError):
        hand_assistance.hand_tool_observation(detection, spec, camera, cfg)


def test_rejects_bad_reprojection(calibrated_grasp):
    detection, spec, camera, cfg, *_ = calibrated_grasp
    detection.keypoints["9"].uv[0] += 60
    with pytest.raises(ValueError):
        hand_assistance.hand_tool_observation(detection, spec, camera, cfg)


def test_uses_main_agent_calibration_contract(calibrated_grasp):
    from robodata.config import HandAssistanceSpec

    detection, spec, camera, cfg, origin, axis = calibrated_grasp
    contract = HandAssistanceSpec.model_validate(vars(spec))
    result = hand_assistance.hand_tool_observation(detection, contract, camera, cfg)
    np.testing.assert_allclose(result.origin_camera_m, origin, atol=1e-6)
    np.testing.assert_allclose(result.axis_camera, axis, atol=1e-6)

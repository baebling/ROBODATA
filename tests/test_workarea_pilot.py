import hashlib
import json

import numpy as np
import pytest


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    source = root / "tracker.py"
    source.write_bytes(b"original algorithm\n")
    entries = {
        "tracker.py": {"bytes": 19, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    }
    (root / "MANIFEST_SHA256.json").write_text(json.dumps(entries), encoding="utf-8")
    return root, source, entries


def test_bundle_audit_validates_payload_and_rejects_changed_source(bundle):
    from robodata.workarea_pilot import verify_bundle

    root, source, _ = bundle
    assert verify_bundle(root)["verified_files"] == 1
    source.write_bytes(b"modified algorithm\n")
    with pytest.raises(ValueError, match="changed"):
        verify_bundle(root)


def test_bundle_audit_rejects_manifest_path_escape(bundle):
    from robodata.workarea_pilot import verify_bundle

    root, source, entries = bundle
    entries["../outside.py"] = entries.pop("tracker.py")
    (root / "MANIFEST_SHA256.json").write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="leaves"):
        verify_bundle(root)
    assert source.is_file()


def test_initial_depth_is_not_accepted_as_a_current_tracking_success():
    from robodata.workarea_pilot import summarize_camera_tracking

    reference = {
        "reference_capture": True,
        "accepted": False,
        "position_error_m": None,
        "rotation_error_deg": None,
        "failure_reason": "reference_only",
    }
    good = {
        "reference_capture": False,
        "accepted": True,
        "position_error_m": 0.02,
        "rotation_error_deg": 2.0,
        "failure_reason": None,
    }
    failed = {
        "reference_capture": False,
        "accepted": False,
        "position_error_m": None,
        "rotation_error_deg": None,
        "failure_reason": "pose_review",
    }
    result = summarize_camera_tracking([reference, good, failed])
    assert result["evaluated_frames"] == 2
    assert result["accepted_frames"] == 1
    assert result["mean_position_error_m"] == pytest.approx(0.02)


def test_all_failed_tracking_has_null_error_not_zero():
    from robodata.workarea_pilot import summarize_camera_tracking

    result = summarize_camera_tracking(
        [
            {
                "reference_capture": False,
                "accepted": False,
                "position_error_m": None,
                "rotation_error_deg": None,
                "failure_reason": "not_initialized",
            }
        ]
    )
    assert result["mean_position_error_m"] is None
    assert result["mean_rotation_error_deg"] is None


def test_reference_camera_transform_uses_current_from_reference_not_reverse():
    from robodata.workarea_pilot import camera_motion_reference

    initial = np.eye(4)
    initial[0, 3] = 10.0
    current = initial.copy()
    current[0, 3] = 10.2
    expected = np.eye(4)
    expected[0, 3] = -0.2
    np.testing.assert_allclose(camera_motion_reference(initial, current), expected, atol=1e-12)


def test_inference_input_rejects_bad_depth_shape_before_constructing_tracker():
    from robodata.workarea_pilot import track_sequence

    def forbidden_factory(_K):
        pytest.fail("Invalid data should fail before invoking the external algorithm")

    with pytest.raises(ValueError, match="shape"):
        track_sequence(
            [np.zeros((10, 12, 3), np.uint8)], np.zeros((5, 6)), np.eye(3), forbidden_factory
        )


def test_adapter_never_copies_pose_over_failed_rgb_update():
    from robodata.workarea_pilot import track_sequence

    class ExternalTracker:
        def __init__(self, _K):
            self.index = 0
            self.anchor_obj = np.zeros((20, 3))

        def initialize(self, image, depth):
            assert image.shape[:2] == depth.shape

        def update(self, image):
            self.index += 1
            if self.index == 2:
                return None, {"accepted": False, "reason": "pose_review"}
            T = np.eye(4)
            T[0, 3] = 0.1 if self.index == 1 else 0.2
            return T, {"accepted": True, "reason": "tracked"}

    image = np.zeros((10, 12, 3), np.uint8)
    records, init = track_sequence([image] * 4, np.ones((10, 12)), np.eye(3), ExternalTracker)
    assert init["accepted"]
    assert records[1]["T_camera_reference"][0][3] == pytest.approx(0.1)
    assert records[2]["T_camera_reference"] is None
    assert not records[2]["accepted"]
    assert records[3]["T_camera_reference"][0][3] == pytest.approx(0.2)


def test_failed_initialization_cannot_emit_identity_tracking_success():
    from robodata.workarea_pilot import track_sequence

    class ExternalTracker:
        def __init__(self, _K):
            pass

        def initialize(self, image, depth):
            raise ValueError("No usable reference points")

        def update(self, image):
            pytest.fail("Failed initialization must not enter tracking")

    image = np.zeros((10, 12, 3), np.uint8)
    records, init = track_sequence([image] * 3, np.ones((10, 12)), np.eye(3), ExternalTracker)
    assert not init["accepted"]
    assert all(r["T_camera_reference"] is None for r in records)
    assert records[1]["failure_reason"] == "not_initialized"


@pytest.mark.parametrize("reflection", [False, True])
def test_adapter_requires_right_handed_pose_and_explicit_acceptance(reflection):
    from robodata.workarea_pilot import track_sequence

    class ExternalTracker:
        def __init__(self, _K):
            self.anchor_obj = np.zeros((20, 3))

        def initialize(self, image, depth):
            pass

        def update(self, image):
            T = np.eye(4)
            if reflection:
                T[0, 0] = -1
            return T, {"accepted": reflection}

    image = np.zeros((10, 12, 3), np.uint8)
    records, _ = track_sequence([image] * 2, np.ones((10, 12)), np.eye(3), ExternalTracker)
    assert records[1]["T_camera_reference"] is None
    assert records[1]["failure_reason"] == "invalid_pose"


def test_camera_error_compares_camera_centers_not_map_origin_translations():
    from robodata.workarea_pilot import camera_motion_error

    reference = np.eye(4)
    reference[0, 3] = -1
    predicted = np.array([[0, -1, 0, 0], [1, 0, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    result = camera_motion_error(predicted, reference)
    assert result["position_error_m"] == pytest.approx(0.0)
    assert result["rotation_error_deg"] == pytest.approx(90.0)


@pytest.mark.parametrize(
    "K",
    [
        [[1, 0.2, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
    ],
)
def test_unsupported_intrinsics_fail_before_external_depth_lifting(K):
    from robodata.workarea_pilot import track_sequence

    def forbidden_factory(_K):
        pytest.fail("Unsupported calibration must not enter the tracker")

    with pytest.raises(ValueError, match="calibration"):
        track_sequence(
            [np.zeros((10, 12, 3), np.uint8)], np.ones((10, 12)), np.asarray(K), forbidden_factory
        )


def test_initial_simulator_depth_checks_reject_wrong_metric_scale():
    from robodata.workarea_pilot import validate_depth_seed

    checks = [{"u": 1, "v": 2, "expected_m": 2.0}]
    validate_depth_seed(np.full((5, 6), 2.0), checks, 0.002, 0.02)
    with pytest.raises(ValueError, match="depth"):
        validate_depth_seed(np.full((5, 6), 4.0), checks, 0.002, 0.02)
    with pytest.raises(ValueError, match="depth"):
        validate_depth_seed(np.full((5, 6), 2.0), [], 0.002, 0.02)

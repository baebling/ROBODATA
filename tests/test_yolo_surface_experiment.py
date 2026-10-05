import numpy as np
import pytest


@pytest.fixture
def bound_training(tmp_path):
    import hashlib
    import json

    import cv2

    training = tmp_path / "training"
    training.mkdir()
    checkpoint = training / "checkpoint.pt"
    checkpoint.write_bytes(b"validation-selected-weights")
    metadata = tmp_path / "test" / "metadata.json"
    metadata.parent.mkdir()
    metadata.write_text('{"frames": 1}', encoding="utf-8")
    image = metadata.parent / "rgb.png"
    pixels = np.full((10, 12, 3), 17, np.uint8)
    cv2.imwrite(str(image), pixels)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pixel_sha = hashlib.sha256(str(pixels.shape).encode("ascii") + pixels.tobytes()).hexdigest()
    record = {
        "split": "test",
        "source_rgb": "test/rgb.png",
        "source_rgb_sha256": sha(image),
        "source_rgb_pixels_sha256": pixel_sha,
        "source_metadata": "test/metadata.json",
        "source_metadata_sha256": sha(metadata),
        "frame_index": 0,
        "timestamp_sec": 0.0,
    }
    manifest = {
        "images": [
            record,
            {"split": "train", "source_rgb_sha256": "other", "source_rgb_pixels_sha256": "other"},
        ]
    }
    manifest_path = training / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    provenance = {
        "status": "completed",
        "checkpoint_sha256": sha(checkpoint),
        "manifest_sha256": sha(manifest_path),
    }
    (training / "training_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    inputs = [(metadata, {"frames": [{"frame_index": 0, "timestamp_sec": 0.0}]}, [image])]
    return tmp_path, training, checkpoint, inputs, manifest, provenance


def test_holdout_audit_binds_completed_training_and_exact_inference_inputs(bound_training):
    from robodata.yolo_surface_experiment import audit_training_inputs

    root, training, checkpoint, inputs, _, _ = bound_training
    assert audit_training_inputs(root, training, checkpoint, inputs)["evaluated_RGB_count"] == 1


@pytest.mark.parametrize("change", ["weights", "manifest", "incomplete"])
def test_holdout_audit_rejects_unbound_training(bound_training, change):
    import json

    from robodata.yolo_surface_experiment import audit_training_inputs

    root, training, checkpoint, inputs, _, provenance = bound_training
    if change == "weights":
        checkpoint.write_bytes(b"different-weights")
    elif change == "manifest":
        (training / "manifest.json").write_text("{}", encoding="utf-8")
    else:
        provenance["status"] = "running"
        (training / "training_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError):
        audit_training_inputs(root, training, checkpoint, inputs)


def test_holdout_audit_checks_metadata_listed_path_not_directory_glob(bound_training):
    from robodata.yolo_surface_experiment import audit_training_inputs

    root, training, checkpoint, inputs, _, _ = bound_training
    extra = inputs[0][2][0].with_name("different.png")
    extra.write_bytes(inputs[0][2][0].read_bytes())
    inputs[0][2][0] = extra
    with pytest.raises(ValueError, match="inventory"):
        audit_training_inputs(root, training, checkpoint, inputs)


@pytest.mark.parametrize("changed", ["rgb", "metadata"])
def test_holdout_audit_rejects_source_drift(bound_training, changed):
    from robodata.yolo_surface_experiment import audit_training_inputs

    root, training, checkpoint, inputs, _, _ = bound_training
    path = inputs[0][2][0] if changed == "rgb" else inputs[0][0]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        audit_training_inputs(root, training, checkpoint, inputs)


def test_holdout_audit_rejects_pixel_equivalent_training_image(bound_training):
    import hashlib
    import json

    from robodata.yolo_surface_experiment import audit_training_inputs

    root, training, checkpoint, inputs, manifest, provenance = bound_training
    manifest["images"][1]["source_rgb_pixels_sha256"] = manifest["images"][0][
        "source_rgb_pixels_sha256"
    ]
    path = training / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    provenance["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (training / "training_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        audit_training_inputs(root, training, checkpoint, inputs)


def test_segmentation_reduction_keeps_disconnected_patches_and_ids_separate():
    from robodata.yolo_surface_experiment import reduce_masks

    masks = np.zeros((3, 12, 16), bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 8:10, 10:12] = True
    masks[2, 4:7, 5:8] = True
    web, tool, ids = reduce_masks([0, 0, 2], masks, [10, 11, 12], (12, 16))
    assert web.sum() == 8
    assert tool.sum() == 9
    assert ids == [10, 11]


def test_no_detections_is_empty_evidence_not_a_previous_mask():
    from robodata.yolo_surface_experiment import reduce_masks

    web, tool, ids = reduce_masks([], np.zeros((0, 12, 16)), None, (12, 16))
    assert not web.any() and not tool.any() and ids == []


def test_masks_cannot_silently_use_letterboxed_coordinates():
    from robodata.yolo_surface_experiment import reduce_masks

    with pytest.raises(ValueError):
        reduce_masks([0], np.ones((1, 10, 10)), None, (12, 16))


def test_test_image_leakage_is_rejected():
    from robodata.yolo_surface_experiment import ensure_disjoint_hashes

    ensure_disjoint_hashes({"train": ["a"], "val": ["b"]}, ["c"])
    with pytest.raises(ValueError):
        ensure_disjoint_hashes({"train": ["a"], "val": ["b"]}, ["b"])


def test_pose_error_does_not_count_missing_pose_as_zero():
    from robodata.yolo_surface_experiment import pose_error

    assert pose_error(None, np.eye(4)) == {"position_error_m": None, "rotation_error_deg": None}
    shifted = np.eye(4)
    shifted[0, 3] = 0.1
    assert pose_error(shifted, np.eye(4))["position_error_m"] == pytest.approx(0.1)


def test_frame_alignment_rejects_renumbered_gaps_and_nonmonotonic_time():
    from robodata.yolo_surface_experiment import validate_frame_sequence

    validate_frame_sequence(
        [
            {"frame_index": 0, "timestamp_sec": 0.0},
            {"frame_index": 1, "timestamp_sec": 0.1},
        ],
        2,
    )
    with pytest.raises(ValueError):
        validate_frame_sequence(
            [
                {"frame_index": 0, "timestamp_sec": 0.0},
                {"frame_index": 2, "timestamp_sec": 0.1},
            ],
            2,
        )
    with pytest.raises(ValueError):
        validate_frame_sequence(
            [
                {"frame_index": 0, "timestamp_sec": 0.0},
                {"frame_index": 1, "timestamp_sec": 0.0},
            ],
            2,
        )


def test_raw_image_paths_cannot_escape_source(tmp_path):
    from robodata.yolo_surface_experiment import contained_file

    (tmp_path / "image.png").touch()
    assert contained_file(tmp_path, "image.png") == tmp_path / "image.png"
    with pytest.raises(ValueError):
        contained_file(tmp_path, "../elsewhere.png")


def test_tool_free_reference_calibration_is_not_clipped_by_learned_web_mask():
    import cv2

    from robodata.yolo_surface_experiment import _fixture_pose

    image = np.full((120, 160, 3), 100, np.uint8)
    for xy, color in zip(
        [(30, 90), (130, 90), (130, 30), (30, 30)],
        [(0, 0, 240), (0, 240, 0), (240, 0, 0), (0, 210, 240)],
        strict=True,
    ):
        cv2.circle(image, xy, 3, color, -1)
    K = np.array([[100, 0, 80], [0, 100, 60], [0, 0, 1]], float)
    points = [[-1, 0.6, 0], [1, 0.6, 0], [1, -0.6, 0], [-1, -0.6, 0]]
    T = _fixture_pose(image, None, K, points, 1.0)
    np.testing.assert_allclose(T[:3, 3], [0, 0, 2], atol=1e-5)
    with pytest.raises(ValueError):
        _fixture_pose(image, np.zeros(image.shape[:2], bool), K, points, 1.0)

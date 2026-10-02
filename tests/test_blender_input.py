"""Independent camera/depth expectations for the Blender import boundary."""

import importlib
import importlib.util
import json

import cv2
import numpy as np
import pytest


def adapter():
    assert importlib.util.find_spec("robodata.blender_input") is not None, "Missing Blender adapter"
    return importlib.import_module("robodata.blender_input")


def test_blender_camera_forward_is_opencv_positive_z():
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    result = adapter().camera_from_blender(pose)
    np.testing.assert_allclose(result @ [1, 2, 1, 1], [0, 0, 2, 1])
    np.testing.assert_allclose(result @ [2, 3, 3, 1], [1, -1, 0, 1])


def test_radial_depth_is_not_axial_depth_off_axis():
    depth = np.array([[2.0, np.sqrt(8.0)]])
    result = adapter().radial_to_axial(depth, np.eye(3))
    np.testing.assert_allclose(result, [[2.0, 2.0]])


def test_radial_depth_keeps_missing_and_background_invalid():
    result = adapter().radial_to_axial(np.array([[0.0, np.inf, 1e10]]), np.eye(3))
    assert np.isnan(result).all()


def test_camera_transform_rejects_scaled_frame():
    pose = np.diag([2.0, 1, 1, 1])
    with pytest.raises(ValueError):
        adapter().camera_from_blender(pose)


def test_existing_dataset_is_not_overwritten(tmp_path):
    out = tmp_path / "dataset"
    out.mkdir()
    sentinel = out / "raw.txt"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        adapter().assemble_rendered(tmp_path / "render", out)
    assert sentinel.read_text() == "original"


def test_missing_metadata_does_not_create_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        adapter().assemble_rendered(tmp_path / "missing", tmp_path / "new")
    assert not (tmp_path / "new").exists()


def rendered_fixture(root):
    root.mkdir()
    cv2.imwrite(str(root / "rgb.png"), np.full((24, 32, 3), 100, np.uint8))
    cv2.imwrite(str(root / "labels.png"), np.full((24, 32), 3, np.uint16))
    np.save(root / "depth.npy", np.ones((24, 32), np.float32))
    tool = np.diag([1.0, -1, -1, 1])
    tool[2, 3] = -0.7
    surface = np.eye(4)
    surface[2, 3] = -1.0
    metadata = {
        "episode_id": "fixture",
        "camera_matrix": [[32, 0, 15.5], [0, 32, 11.5], [0, 0, 1]],
        "width": 32,
        "height": 24,
        "fps": 10,
        "expected_frames": 4,
        "camera_motion": "static",
        "depth_convention": "axial_z",
        "tool_points_m": {"tip": [0, 0, 0]},
        "surface_points_m": {
            "a": [-0.2, -0.2, 0],
            "b": [0.2, -0.2, 0],
            "c": [0.2, 0.2, 0],
            "d": [-0.2, 0.2, 0],
        },
        "frames": [
            {
                "frame_index": i,
                "timestamp_sec": i / 10,
                "rgb_path": "rgb.png",
                "depth_path": "depth.npy",
                "labels_path": "labels.png",
                "T_world_camera": np.eye(4).tolist(),
                "T_world_tool": tool.tolist(),
                "T_world_surface": surface.tolist(),
                "tool_keypoints": {},
                "surface_keypoints": {},
            }
            for i in range(4)
        ],
    }
    (root / "render_metadata.json").write_text(json.dumps(metadata))
    return metadata


def test_assembly_keeps_truth_out_of_inference_manifest(tmp_path):
    raw = tmp_path / "render"
    meta = rendered_fixture(raw)
    meta["settings"] = {"config": {"frames": 4, "standoff_m": 0.3}, "config_sha256": "fixture"}
    (raw / "render_metadata.json").write_text(json.dumps(meta))
    out = adapter().assemble_rendered(raw, tmp_path / "dataset")
    manifest = json.loads((out / "manifest.json").read_text())
    truth = json.loads((out / "ground_truth.json").read_text())
    assert manifest["data_kind"] == "synthetic"
    assert "T_world_tool" not in manifest["observations"][0]
    assert "ground_truth" not in json.dumps(manifest)
    assert "standoff_m" not in json.dumps(manifest)
    assert manifest["provenance"]["render_settings"]["config_sha256"] == "fixture"
    np.testing.assert_allclose(
        truth["frames"][0]["canonical"]["tip_position_surface_m"], [0, 0, 0.3]
    )
    assert truth["frames"][0]["canonical"]["incidence_angle_rad"] == pytest.approx(0.0)
    assert len(truth["frames"]) == 4
    assert np.load(out / "depth/00000.npy").shape == (24, 32)


def test_assembly_rejects_timestamp_reversal(tmp_path):
    raw = tmp_path / "render"
    metadata = rendered_fixture(raw)
    metadata["frames"][1]["timestamp_sec"] = 0.0
    (raw / "render_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        adapter().assemble_rendered(raw, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_assembly_checks_render_depth_against_independent_ray(tmp_path):
    raw = tmp_path / "render"
    metadata = rendered_fixture(raw)
    metadata["frames"][0]["depth_checks"] = [{"u": 10, "v": 10, "expected_m": 2.0}]
    (raw / "render_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="depth"):
        adapter().assemble_rendered(raw, tmp_path / "dataset")


def test_blender_workflow_retains_invalid_frames_and_reports_no_accuracy(tmp_path):
    assert importlib.util.find_spec("robodata.blender_experiment") is not None
    workflow = importlib.import_module("robodata.blender_experiment")
    raw = tmp_path / "render"
    rendered_fixture(raw)
    out = workflow.run_blender_experiment(raw, tmp_path / "results")
    report = json.loads((out / "comparison.json").read_text())
    assert report["data_kind"] == "synthetic"
    assert report["landmarks"] == "oracle_not_detector"
    assert len(report["results"]) == 2
    for row in report["results"]:
        assert row["predicted_frames"] == 4
        assert row["matched_valid_frames"] == 0
        assert row["position_rmse_m"] is None
        assert sum(row["failure_counts"].values()) == 4
    assert (out / "report.md").is_file()


def test_blender_workflow_requires_all_declared_scenarios(tmp_path):
    assert importlib.util.find_spec("robodata.blender_experiment") is not None
    workflow = importlib.import_module("robodata.blender_experiment")
    raw = tmp_path / "render"
    raw.mkdir()
    (raw / "effective_config.json").write_text(json.dumps({"scenarios": [{"name": "missing"}]}))
    with pytest.raises(FileNotFoundError):
        workflow.run_blender_experiment(raw, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_assembly_rejects_truncated_declared_episode(tmp_path):
    raw = tmp_path / "render"
    meta = rendered_fixture(raw)
    meta["expected_frames"] = 30
    (raw / "render_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="frame"):
        adapter().assemble_rendered(raw, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_assembly_rejects_conflicting_frame_count_declarations(tmp_path):
    raw = tmp_path / "render"
    meta = rendered_fixture(raw)
    meta["settings"] = {"config": {"frames": 30}}
    (raw / "render_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="frame"):
        adapter().assemble_rendered(raw, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_workflow_rejects_truncation_against_parent_configuration(tmp_path):
    workflow = importlib.import_module("robodata.blender_experiment")
    raw = tmp_path / "render"
    raw.mkdir()
    rendered_fixture(raw / "case")
    (raw / "effective_config.json").write_text(
        json.dumps(
            {
                "frames": 30,
                "scenarios": [{"name": "case"}],
            }
        )
    )
    with pytest.raises(ValueError, match="frame"):
        workflow.run_blender_experiment(raw, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_assembly_rejects_reanchoring_the_static_workpiece(tmp_path):
    raw = tmp_path / "render"
    meta = rendered_fixture(raw)
    meta["frames"][1]["T_world_surface"][0][3] = 0.1
    (raw / "render_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="surface"):
        adapter().assemble_rendered(raw, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize(
    "check",
    [
        {"u": 10, "v": 10, "expected_m": float("nan")},
        {"u": 10, "v": 10, "expected_m": -1},
        {"u": -1, "v": 10, "expected_m": 1},
        {"u": 10.5, "v": 10, "expected_m": 1},
        {"u": 32, "v": 10, "expected_m": 1},
    ],
)
def test_assembly_rejects_invalid_ray_checks_before_writing(tmp_path, check):
    raw = tmp_path / "render"
    meta = rendered_fixture(raw)
    meta["frames"][0]["depth_checks"] = [check]
    (raw / "render_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="ray"):
        adapter().assemble_rendered(raw, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_assembly_does_not_write_inside_raw_render_tree(tmp_path):
    raw = tmp_path / "render"
    rendered_fixture(raw)
    with pytest.raises(ValueError, match="raw"):
        adapter().assemble_rendered(raw, raw / "dataset")
    assert not (raw / "dataset").exists()


def test_stationary_tool_stays_stationary_with_moving_camera(tmp_path):
    workflow = importlib.import_module("robodata.blender_experiment")
    raw = tmp_path / "render"
    meta = rendered_fixture(raw)
    meta["camera_motion"] = "moving"
    meta["tool_points_m"] = {
        "tip": [0, 0, 0],
        "axis_base": [0, 0, -0.085],
        "side": [0.055, 0, 0],
        "top": [0, 0.045, -0.025],
        "rear_left": [-0.05, -0.035, -0.085],
        "rear_right": [0.05, -0.035, -0.085],
        "back_top": [0, 0.04, -0.1],
        "front_left": [-0.05, -0.035, 0],
    }
    K = np.asarray(meta["camera_matrix"])
    for i, frame in enumerate(meta["frames"]):
        frame["T_world_camera"][0][3] = 0.01 * i
        C = adapter().camera_from_blender(frame["T_world_camera"])
        for obj in ["tool", "surface"]:
            pose = np.asarray(frame[f"T_world_{obj}"])
            keypoints = {}
            for key, point in meta[f"{obj}_points_m"].items():
                xyz = (C @ pose @ [*point, 1])[:3]
                pixel = K @ xyz
                keypoints[key] = {"uv": (pixel[:2] / pixel[2]).tolist(), "visible": True}
            frame[f"{obj}_keypoints"] = keypoints
    (raw / "render_metadata.json").write_text(json.dumps(meta))
    out = workflow.run_blender_experiment(raw, tmp_path / "results")
    rows = json.loads((out / "comparison.json").read_text())["results"]
    rgb = next(row for row in rows if row["mode"] == "rgb")
    assert rgb["matched_valid_frames"] == 4
    assert rgb["position_rmse_m"] < 1e-5
    assert rgb["stationary_prediction_span_m"] < 1e-5

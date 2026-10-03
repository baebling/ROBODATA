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


def test_tool_mask_includes_hopper_and_nozzle_part_ids(tmp_path):
    root = tmp_path / "render"
    rendered_fixture(root)
    path = root / "render_metadata.json"
    meta = json.loads(path.read_text())
    meta["tool_label_ids"] = [2, 4, 5]
    path.write_text(json.dumps(meta))
    labels = np.full((24, 32), 3, np.uint16)
    labels[0, :4] = [2, 4, 5, 6]
    cv2.imwrite(str(root / "labels.png"), labels)
    out = adapter().assemble_rendered(root, tmp_path / "dataset")
    mask = cv2.imread(str(out / "masks/tool_00000.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[0, :4].tolist() == [255, 255, 255, 0]


def test_rgb_only_sensor_never_supplies_ideal_depth_to_inference(tmp_path):
    root = tmp_path / "render"
    meta = rendered_fixture(root)
    meta["sensor_profile"] = {"depth_available": False}
    (root / "render_metadata.json").write_text(json.dumps(meta))
    out = adapter().assemble_rendered(root, tmp_path / "dataset")
    manifest = json.loads((out / "manifest.json").read_text())
    assert all(o["depth_path"] is None for o in manifest["observations"])
    assert not manifest["depth_aligned_to_rgb"]
    assert not (out / "rgbd.yaml").exists()
    assert (out / "depth/00000.npy").exists(), "Depth retained only for visibility evaluation"


def test_occlusion_reference_stays_out_of_inference_manifest(tmp_path):
    from robodata.blender_experiment import measure_occlusion

    root = tmp_path / "render"
    rendered_fixture(root)
    path = root / "render_metadata.json"
    meta = json.loads(path.read_text())
    meta["settings"] = {
        "config": {
            "surface_roi_bounds_m": [-0.2, -0.2, 0.2, 0.2],
            "occlusion_depth_tolerance_m": 0.002,
        }
    }
    path.write_text(json.dumps(meta))
    before = path.read_bytes()
    dataset = adapter().assemble_rendered(root, tmp_path / "dataset")
    result = measure_occlusion(root, dataset, meta)
    assert result["summary"]["mean_tool_roi_occlusion"] == 0
    assert len(result["frames"]) == 4
    assert "occlusion" not in (dataset / "manifest.json").read_text()
    assert path.read_bytes() == before


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


def test_web_assembly_preserves_frame_identity_without_truth_leakage(tmp_path):
    raw = tmp_path / "render"
    meta = rendered_fixture(raw)
    meta["work_face"] = "web_front"
    meta["surface_frame_id"] = "synthetic_hbeam_web_front"
    meta["frames"][0]["nozzle_reference"] = {"status": "flange_blocked"}
    (raw / "render_metadata.json").write_text(json.dumps(meta))
    out = adapter().assemble_rendered(raw, tmp_path / "dataset")
    from robodata.config import load_config

    assert load_config(out / "rgb.yaml").surface_frame_id == "synthetic_hbeam_web_front"
    truth = json.loads((out / "ground_truth.json").read_text())
    assert truth["surface_frame_id"] == "synthetic_hbeam_web_front"
    manifest = json.loads((out / "manifest.json").read_text())
    assert "nozzle_reference" not in json.dumps(manifest)
    assert "flange_blocked" not in json.dumps(manifest)


def test_web_assembly_rejects_top_flange_identity(tmp_path):
    raw = tmp_path / "render"
    meta = rendered_fixture(raw)
    meta.update(work_face="web_front", surface_frame_id="synthetic_hbeam_top_patch")
    (raw / "render_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="identity"):
        adapter().assemble_rendered(raw, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize(
    "point,axis", [([0.55, 0, 0.32], [0.8, 0, -0.6]), ([0, 0, 0.32], [0, 0, 1])]
)
def test_preview_keeps_estimated_no_hit_ray_visible(point, axis):
    from PIL import Image

    from robodata.schema import Canonical, Estimate, FrameRecord, Quality

    workflow = importlib.import_module("robodata.blender_experiment")
    S = np.eye(4)
    S[2, 3] = -1
    T = np.diag([1.0, -1, -1, 1])
    T[2, 3] = -0.7
    raw = {
        "T_world_camera": np.eye(4).tolist(),
        "T_world_surface": S.tolist(),
        "T_world_tool": T.tolist(),
        "nozzle_reference": {
            "status": "work_face_hit",
            "first_beam_part": "web",
            "first_hit_surface_m": [0, 0, 0],
        },
    }
    metadata = {
        "width": 320,
        "episode_id": "test",
        "work_face": "web_front",
        "camera_matrix": [[100, 0, 160], [0, 100, 120], [0, 0, 1]],
        "settings": {
            "config": {
                "work_face": "web_front",
                "surface_height_m": 0.8,
                "beam_length_m": 1.8,
                "beam_width_m": 0.4,
                "beam_height_m": 0.4,
                "flange_thickness_m": 0.02,
                "web_thickness_m": 0.012,
            }
        },
    }
    prediction = FrameRecord(
        frame_index=0,
        timestamp_sec=0,
        quality=Quality(valid=True),
        canonical=Canonical(tip_position_surface_m=point, tool_axis_surface=axis),
        surface=Estimate(
            method="test", T_camera_frame=adapter().camera_from_blender(np.eye(4)) @ S
        ),
    )
    image = Image.new("RGB", (320, 240), "black")
    preview = workflow._relation_preview(image, raw, metadata, prediction)
    assert preview is not None, "Preview must preserve scene framing with a separate label band"
    assert preview.width == image.width and preview.height > image.height
    pixels = np.asarray(preview)
    assert np.any(np.all(pixels == [255, 0, 255], axis=-1)), "Valid estimated ray vanished"


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


def test_relationship_diagnostics_keep_unestimated_frames_unknown(tmp_path):
    workflow = importlib.import_module("robodata.blender_experiment")
    raw = tmp_path / "render"
    meta = rendered_fixture(raw)
    meta["work_face"] = "web_front"
    meta["surface_frame_id"] = "synthetic_hbeam_web_front"
    meta["settings"] = {
        "config": {
            "frames": 4,
            "work_face": "web_front",
            "surface_height_m": 0.8,
            "beam_length_m": 1.8,
            "beam_width_m": 0.4,
            "beam_height_m": 0.4,
            "flange_thickness_m": 0.02,
            "web_thickness_m": 0.012,
        }
    }
    for frame in meta["frames"]:
        frame["nozzle_reference"] = {
            "status": "work_face_hit",
            "first_beam_part": "web",
            "ray_plane_intersection_surface_m": [0, 0, 0],
            "first_hit_surface_m": [0, 0, 0],
            "scene_ray_agrees": True,
        }
    (raw / "render_metadata.json").write_text(json.dumps(meta))
    out = workflow.run_blender_experiment(raw, tmp_path / "results")
    records = json.loads((out / "render" / "rgb" / "nozzle_relationship.json").read_text())
    assert records["summary"]["evaluated_frames"] == 0
    assert records["summary"]["status_accuracy"] is None
    assert len(records["frames"]) == 4
    assert all(row["estimated"] is None for row in records["frames"])
    assert all(row["reference"]["status"] == "work_face_hit" for row in records["frames"])


def test_relationship_diagnostics_use_prediction_not_reference_pose():
    workflow = importlib.import_module("robodata.blender_experiment")
    assert hasattr(workflow, "relationship_diagnostics"), "Missing relationship evaluation"
    from robodata.schema import Canonical, Episode, FrameRecord, Quality, Source

    episode = Episode(
        episode_id="test",
        task_name="test",
        surface_frame_id="web",
        source=Source(mode="rgb", metric_scale_source="measured_geometry", data_kind="synthetic"),
        frames=[
            FrameRecord(
                frame_index=0,
                timestamp_sec=0,
                quality=Quality(valid=True),
                canonical=Canonical(
                    tip_position_surface_m=[1.1, 0, 0.32], tool_axis_surface=[0, 0, -1]
                ),
            )
        ],
    )
    meta = {
        "surface_frame_id": "web",
        "settings": {
            "config": {
                "work_face": "web_front",
                "surface_height_m": 0.8,
                "beam_length_m": 1.8,
                "beam_width_m": 0.4,
                "beam_height_m": 0.4,
                "flange_thickness_m": 0.02,
                "web_thickness_m": 0.012,
            }
        },
        "frames": [
            {
                "frame_index": 0,
                "nozzle_reference": {
                    "status": "work_face_hit",
                    "first_beam_part": "web",
                    "ray_plane_intersection_surface_m": [0, 0, 0],
                    "first_hit_surface_m": [0, 0, 0],
                    "scene_ray_agrees": True,
                },
            }
        ],
    }
    result = workflow.relationship_diagnostics(episode, meta)
    assert result["frames"][0]["estimated"]["status"] == "miss"
    assert result["summary"]["status_accuracy"] == 0.0
    assert result["frames"][0]["plane_intersection_error_m"] == pytest.approx(1.1)


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

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from robodata.evaluation import evaluate
from robodata.export import read_episode, write_json
from robodata.pipeline import run_episode
from robodata.synthetic import create_demo


def workflow():
    return importlib.import_module("robodata.helmet_hand_experiment")


@pytest.fixture
def experiment(tmp_path):
    render = tmp_path / "renders"
    baseline = tmp_path / "baseline"
    render.mkdir()
    baseline.mkdir()
    name = "main_near_aimed"
    raw = render / name
    raw.mkdir()
    inputs = create_demo(baseline / name / "inputs", frames=4, occluded=[1])
    manifest = json.loads((inputs / "manifest.json").read_text())
    gt = read_episode(inputs / "ground_truth.json")
    for mode in ("rgb", "rgbd"):
        predicted = run_episode(inputs / f"{mode}.yaml", baseline / name / mode)
        write_json(baseline / name / mode / "evaluation.json", evaluate(predicted, gt, 0.01))
    scenario = {
        "name": name,
        "sensor": "galaxy_main",
        "view": "near_aimed",
        "mount": "helmet_front",
    }
    calibration = {
        "points_hand_m": {
            "0": [0, 0, 0],
            "1": [-0.03, 0.02, 0.015],
            "5": [-0.03, 0.07, 0.005],
            "9": [0, 0.08, 0.01],
            "13": [0.025, 0.07, 0],
            "17": [0.04, 0.06, -0.01],
        },
        "T_hand_tool": np.eye(4).tolist(),
        "handedness": "Right",
        "grasp_verified": True,
        "calibration_source": "synthetic_fixed_grasp_WebXR_template_NOT_real_measurement",
    }
    frames = []
    for index in range(4):
        image = raw / f"rgb_{index:05d}.png"
        Image.open(inputs / f"frames/{index:05d}.png").save(image)
        frames.append(
            {
                "frame_index": index,
                "timestamp_sec": index * 0.1,
                "rgb_path": image.name,
                "T_world_tool": "DO_NOT_USE_AS_MODEL_INPUT",
            }
        )
    metadata = {
        "data_kind": "synthetic",
        "width": 320,
        "height": 240,
        "camera_matrix": manifest["camera"]["matrix"],
        "frames": frames,
        "settings": {"scenario": scenario},
        "hand_calibration": calibration,
    }
    write_json(raw / "render_metadata.json", metadata)
    manifest.setdefault("provenance", {})["render_metadata_sha256"] = hashlib.sha256(
        (raw / "render_metadata.json").read_bytes()
    ).hexdigest()
    write_json(inputs / "manifest.json", manifest)
    write_json(
        render / "effective_config.json",
        {
            "frames": 4,
            "scenarios": [scenario],
            "sensor_profiles": {
                key: {}
                for key in (
                    "galaxy_main",
                    "galaxy_ultrawide",
                    "zed2i",
                    "helios_ray_chroma",
                )
            },
        },
    )
    model = tmp_path / "hand.task"
    model.write_bytes(b"test hand model")
    return render, baseline, model, tmp_path / "hand_results", name, metadata


def fake_inference(monkeypatch):
    def infer(model_path, manifest_path, output_dir, min_confidence=0.5):
        # Replace only the external model. Real extraction/evaluation/plotting run below.
        source = Path(manifest_path)
        data = json.loads(source.read_text())
        data["video_path"] = str(source.parent / data["video_path"])
        for frame in data["observations"]:
            for group in ("tool", "surface"):
                frame[group]["mask_path"] = str(source.parent / frame[group]["mask_path"])
            frame["depth_path"] = str(source.parent / frame["depth_path"])
            frame["hand"] = {"keypoints": {}, "handedness": "Right"}
        # This deliberately fails PnP for the occluded frame: failures must be retained.
        data["observations"][2]["hand"] = {
            "handedness": "Right",
            "keypoints": {
                str(index): {"uv": [60 + index * 6, 80 + index], "confidence": 1, "visible": True}
                for index in range(21)
            },
        }
        output = Path(output_dir)
        provenance = {
            "model_path": str(model_path),
            "model_sha256": "fake-model-hash",
            "mediapipe_version": "0.10.21",
            "min_confidence": min_confidence,
            "keypoint_semantics": "confidence=1 accepted prediction; not GT visibility",
            "raw_detections_path": str(output / "hand_detections.json"),
        }
        data.setdefault("provenance", {})["hand_source"] = provenance
        write_json(output / "manifest.json", data)
        raw = {
            "provenance": provenance,
            "frames": [
                {
                    "frame_index": index,
                    "timestamp_sec": index * 0.1,
                    "timestamp_ms": index * 100,
                    "selected_hand_index": 0 if index == 2 else None,
                    "association_failure": None if index == 2 else "no_association",
                    "hands": [
                        {
                            "normalized_landmarks": [
                                [(60 + joint * 6) / 320, (80 + joint) / 240, 0]
                                for joint in range(21)
                            ],
                            "local_world_landmarks_m": [[0, 0, 0]] * 21,
                            "handedness": [],
                        }
                    ],
                }
                for index in range(4)
            ],
        }
        write_json(output / "hand_detections.json", raw)
        return output / "manifest.json"

    monkeypatch.setattr(workflow(), "infer_hands", infer)
    return infer


def test_fixture_runs_real_extraction_evaluation_and_selected_only_contact_sheet(
    experiment,
    monkeypatch,
):
    render, baseline, model, output, name, metadata = experiment
    original_config = (baseline / name / "inputs/rgb.yaml").read_bytes()
    fake_inference(monkeypatch)
    result = workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert result == output
    config = yaml.safe_load((output / name / "rgb_hand.yaml").read_text())
    assert config["manifest_path"] == str(output / name / "hand_predictions/manifest.json")
    assert config["hand_assistance"] == metadata["hand_calibration"]
    assert config["mode"] == "rgb"
    assert (baseline / name / "inputs/rgb.yaml").read_bytes() == original_config
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["data_kind"] == "synthetic"
    assert len(comparison["sensor_columns"]) == 4
    row = comparison["results"][0]
    assert row["scenario"] == name
    assert row["total_frames"] == 4
    assert row["selected_hand_frames"] == 1
    assert row["hand_source"]["mediapipe_version"] == "0.10.21"
    assert row["rgb"]["matched_valid_frames"] == 3
    assert row["rgbd"]["predicted_frames"] == 4
    assert row["rgb_hand"]["matched_valid_frames"] == 3
    assert row["rgb_hand"]["position_rmse_m"] < 1e-5
    assert len(row["frames"]) == 4
    assert row["frames"][1]["failure_reason"]
    episode = read_episode(output / name / "rgb_hand/episode.json")
    assert len(episode.frames) == 4 and not episode.frames[1].quality.valid
    report = (output / "report.md").read_text()
    for term in ("SYNTHETIC", "MediaPipe", "not measured", "heuristic", "RGB-D", "3/4"):
        assert term in report
    assert "improved" not in report.lower()
    with Image.open(output / "contact_sheet.png") as sheet:
        assert sheet.width > 4 * 400 and sheet.height > 300
        colors = np.asarray(sheet.convert("RGB"))
        assert np.any(np.all(colors == [255, 80, 80], axis=2))


def test_existing_output_is_not_overwritten(experiment):
    render, baseline, model, output, *_ = experiment
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("root_index", [0, 1])
def test_output_cannot_be_inside_raw_or_baseline(experiment, root_index):
    render, baseline, model, _, *_ = experiment
    output = (render, baseline)[root_index] / "results"
    with pytest.raises(ValueError, match="outside"):
        workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert not output.exists()


def test_all_cases_are_preflighted_before_any_output_or_inference(experiment, monkeypatch):
    render, baseline, model, output, name, _ = experiment
    config = json.loads((render / "effective_config.json").read_text())
    config["scenarios"].append({"name": "../escape", "sensor": "zed2i", "view": "far"})
    write_json(render / "effective_config.json", config)
    monkeypatch.setattr(workflow(), "infer_hands", lambda *args, **kwargs: pytest.fail("inferred"))
    with pytest.raises(ValueError, match="Scenario|scenario"):
        workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert not output.exists()
    assert (render / name / "render_metadata.json").is_file()


def test_raw_midframe_escape_is_rejected_before_output(experiment):
    render, baseline, model, output, name, metadata = experiment
    metadata["frames"][2]["rgb_path"] = "../../outside.png"
    write_json(render / name / "render_metadata.json", metadata)
    with pytest.raises(ValueError, match="path|directory"):
        workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert not output.exists()


def test_baseline_manifest_escape_is_rejected_before_output(experiment):
    render, baseline, model, output, name, _ = experiment
    config_path = baseline / name / "inputs/rgb.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["manifest_path"] = "../../../outside.json"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="path|directory"):
        workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert not output.exists()


def test_missing_calibration_is_rejected_before_inference(experiment, monkeypatch):
    render, baseline, model, output, name, metadata = experiment
    del metadata["hand_calibration"]
    write_json(render / name / "render_metadata.json", metadata)
    monkeypatch.setattr(workflow(), "infer_hands", lambda *args, **kwargs: pytest.fail("inferred"))
    with pytest.raises(ValueError, match="calibration"):
        workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert not output.exists()


def test_unavailable_rgbd_is_explicitly_null(experiment, monkeypatch):
    render, baseline, model, output, name, _ = experiment
    (baseline / name / "rgbd/episode.json").unlink()
    (baseline / name / "rgbd/evaluation.json").unlink()
    fake_inference(monkeypatch)
    workflow().run_helmet_hand_experiment(render, baseline, model, output)
    row = json.loads((output / "comparison.json").read_text())["results"][0]
    assert row["rgbd"] is None
    assert "unavailable" in (output / "report.md").read_text()


def test_cli_accepts_requested_roots_and_model(experiment, monkeypatch, capsys):
    render, baseline, model, output, *_ = experiment
    fake_inference(monkeypatch)
    workflow().main(
        [
            "--render-root",
            str(render),
            "--baseline-root",
            str(baseline),
            "--model",
            str(model),
            "--output",
            str(output),
        ]
    )
    assert capsys.readouterr().out.strip() == str(output)
    assert (output / "comparison.json").is_file()


def test_different_baseline_episode_identity_is_rejected_before_output(experiment, monkeypatch):
    render, baseline, model, output, name, _ = experiment
    path = baseline / name / "rgb/episode.json"
    data = json.loads(path.read_text())
    data["episode_id"] = "another_case_with_the_same_timestamps"
    write_json(path, data)
    fake_inference(monkeypatch)
    with pytest.raises(ValueError, match="identity|episode"):
        workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert not output.exists()


def test_unselected_raw_hand_never_appears_in_overlay():
    image = Image.new("RGB", (100, 80), (20, 30, 40))
    raw = {
        "selected_hand_index": None,
        "hands": [{"normalized_landmarks": [[0.5, 0.5, 0]] * 21}],
    }
    predicted = workflow()._prediction_overlay(image, raw)
    np.testing.assert_array_equal(np.asarray(predicted), np.asarray(image))


def test_symlinked_scenario_cannot_escape_raw_root(experiment):
    render, baseline, model, output, name, _ = experiment
    effective = json.loads((render / "effective_config.json").read_text())
    effective["scenarios"][0]["name"] = "linked_case"
    write_json(render / "effective_config.json", effective)
    (render / "linked_case").symlink_to(baseline / name, target_is_directory=True)
    with pytest.raises(ValueError, match="path|directory"):
        workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert not output.exists()


def test_tile_labels_fit_actual_font_and_464_pixel_text_area(experiment, monkeypatch):
    render, baseline, _, _, name, metadata = experiment
    metrics = json.loads((baseline / name / "rgb/evaluation.json").read_text())
    row = {
        "rgb_hand": metrics,
        "rgb": metrics,
        "rgbd": None,
        "selected_hand_frames": 8,
        "total_frames": 8,
        "hand_pose_accepted_frames": 8,
    }
    case = SimpleNamespace(
        name="long_synthetic_case_name_" * 5,
        source=render / name,
        metadata=metadata,
    )
    original_text = ImageDraw.ImageDraw.text
    bounds = []

    def measured_text(draw, xy, text, **kwargs):
        bounds.append(draw.textbbox(xy, text, font=kwargs["font"]))
        return original_text(draw, xy, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", measured_text)
    tile = workflow()._tile(case, row, {"selected_hand_index": None})
    assert tile.width == 480
    assert all(right - left <= 464 for left, _, right, _ in bounds)
    assert all(bottom <= tile.height for _, _, _, bottom in bounds)


def test_selected_hand_pose_rejections_and_episode_stages_are_reported(experiment, monkeypatch):
    render, baseline, model, output, _, _ = experiment
    fake_inference(monkeypatch)
    workflow().run_helmet_hand_experiment(render, baseline, model, output)
    row = json.loads((output / "comparison.json").read_text())["results"][0]
    assert row["selected_hand_frames"] == 1
    assert row["hand_pose_accepted_frames"] == 0
    assert sum(row["hand_pose_rejection_counts"].values()) == 1
    assert row["frames"][2]["hand_pose_accepted"] is False
    assert row["frames"][2]["hand_pose_rejection"]
    assert row["frames"][0]["hand_pose_accepted"] is None
    assert row["stage_counts"] == {"surface_unavailable": 0, "tool_unavailable": 1}
    report = (output / "report.md").read_text()
    assert "Hand PnP" in report and "Surface unavailable" in report


def test_hand_pose_can_pass_while_canonical_surface_stage_is_unavailable(experiment, monkeypatch):
    render, baseline, model, output, name, metadata = experiment
    infer = fake_inference(monkeypatch)

    def projected_hand_only(model_path, manifest_path, output_dir, min_confidence=0.5):
        path = infer(model_path, manifest_path, output_dir, min_confidence)
        data = json.loads(path.read_text())
        points = metadata["hand_calibration"]["points_hand_m"]
        pixels, _ = cv2.projectPoints(
            np.array(list(points.values()), dtype=float),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.0, -0.01, 0.7]),
            np.array(data["camera"]["matrix"], dtype=float),
            np.zeros(5),
        )
        selected = data["observations"][2]
        selected["hand"]["keypoints"].update(
            {
                key: {"uv": uv.tolist(), "confidence": 1, "visible": True}
                for key, uv in zip(points, pixels.reshape(-1, 2))
            }
        )
        for point in selected["surface"]["keypoints"].values():
            point["visible"] = False
        write_json(path, data)
        return path

    monkeypatch.setattr(workflow(), "infer_hands", projected_hand_only)
    workflow().run_helmet_hand_experiment(render, baseline, model, output)
    row = json.loads((output / "comparison.json").read_text())["results"][0]
    assert row["hand_pose_accepted_frames"] == 1
    assert row["hand_pose_rejection_counts"] == {}
    assert row["frames"][2]["hand_pose_accepted"] is True
    assert row["stage_counts"] == {"surface_unavailable": 1, "tool_unavailable": 2}
    assert row["rgb_hand"]["matched_valid_frames"] == 2
    episode = read_episode(output / name / "rgb_hand/episode.json")
    assert len(episode.frames) == 4
    assert not episode.frames[2].quality.valid


def test_raw_metadata_change_rejects_stale_baseline_before_inference(experiment, monkeypatch):
    render, baseline, model, output, name, metadata = experiment
    pose = np.eye(4)
    pose[0, 3] = 1.0
    metadata["frames"][0]["T_world_tool"] = pose.tolist()
    write_json(render / name / "render_metadata.json", metadata)
    fake_inference(monkeypatch)
    with pytest.raises(ValueError, match="SHA|hash|metadata"):
        workflow().run_helmet_hand_experiment(render, baseline, model, output)
    assert not output.exists()

"""Synthetic visible-mask conversion, holdout isolation and training provenance."""

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml


def runner():
    assert importlib.util.find_spec("robodata.yolo_training") is not None, "Missing YOLO runner"
    return importlib.import_module("robodata.yolo_training")


def fixture(tmp_path):
    config = {
        "schema_version": 1,
        "data_kind": "synthetic",
        "source_root": "raw",
        "reserved_test_root": "reserved",
        "splits": {"train": ["frontal_base"], "val": ["stationary_base"]},
        "expected_frames_per_scenario": 2,
        "expected_counts": {"train": 2, "val": 2, "test": 2},
        "classes": [
            {"id": 0, "name": "web_patch", "label_ids": [3]},
            {"id": 1, "name": "flange", "label_ids": [1]},
            {"id": 2, "name": "tool", "label_ids": [2, 4, 5]},
        ],
        "model": "yolo26n-seg.pt",
        "train": {
            "seed": 42,
            "epochs": 40,
            "imgsz": 640,
            "batch": 8,
            "workers": 0,
            "patience": 0,
            "deterministic": True,
            "device": 0,
            "pretrained": True,
            "amp": False,
            "val": True,
            "save": True,
            "resume": False,
        },
        "inference": {"conf": 0.25, "iou": 0.7},
    }
    masks = np.zeros((24, 32), np.uint16)
    masks[1:12, 1:14] = 3
    masks[4:7, 4:7] = 0  # A hole must be recorded as an approximation.
    masks[15:20, 1:7] = 3  # Disconnected web patch stays a separate instance.
    masks[1:8, 18:28] = 1
    masks[13:18, 12:16] = 2
    masks[13:18, 16:20] = 4
    masks[13:18, 20:24] = 5
    masks[22, 30] = 5  # Tiny visible fragment must not disappear silently.
    for root_name, scenarios, offset in [
        ("raw", ["frontal_base", "stationary_base"], 1),
        ("reserved", ["helmet_external"], 10),
    ]:
        root = tmp_path / root_name
        root.mkdir()
        (root / "effective_config.json").write_text(
            json.dumps(
                {
                    "scenarios": [{"name": s} for s in scenarios],
                    "frames": 2,
                }
            )
        )
        for s_index, scenario in enumerate(scenarios):
            source = root / scenario
            source.mkdir()
            frames = []
            for index in range(2):
                rgb = np.full((24, 32, 3), offset + 2 * s_index + index, np.uint8)
                cv2.imwrite(str(source / f"rgb{index}.png"), rgb)
                # Reserved labels deliberately do not exist: audit is RGB-only.
                if root_name == "raw":
                    cv2.imwrite(str(source / f"labels{index}.png"), masks)
                frames.append(
                    {
                        "frame_index": index,
                        "timestamp_sec": index / 10,
                        "rgb_path": f"rgb{index}.png",
                        "labels_path": f"labels{index}.png",
                    }
                )
            (source / "render_metadata.json").write_text(
                json.dumps(
                    {
                        "episode_id": scenario,
                        "width": 32,
                        "height": 24,
                        "expected_frames": 2,
                        "frames": frames,
                    }
                )
            )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config


def prepare(tmp_path, path=None, output="prepared"):
    path = path or fixture(tmp_path)[0]
    return runner().prepare_dataset(path, tmp_path / output, repo_root=tmp_path)


def test_visible_components_are_normalized_and_topology_is_recorded(tmp_path):
    manifest_path = prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["counts"] == {"train": 2, "val": 2, "test": 2}
    item = next(r for r in manifest["images"] if r["split"] == "train")
    rows = (tmp_path / "prepared" / item["dataset_label"]).read_text().splitlines()
    assert [int(row.split()[0]) for row in rows] == [0, 0, 1, 2, 2]
    for row in rows:
        coordinates = np.array([float(v) for v in row.split()[1:]])
        assert len(coordinates) >= 6 and len(coordinates) % 2 == 0
        assert np.all((coordinates >= 0) & (coordinates <= 1))
    assert item["conversion"]["holes_filled"] == 1
    assert item["conversion"]["degenerate_components"] == 1
    assert item["conversion"]["visible_pixels"] == {"0": 164, "1": 70, "2": 61}
    assert "holes" in manifest["annotation_note"].lower()
    assert "disconnected" in manifest["annotation_note"].lower()
    assert "fixture" in manifest["annotation_note"].lower()


def test_manifest_hashes_sources_and_test_never_enters_yolo_yaml(tmp_path):
    module_was_loaded = "ultralytics" in sys.modules
    path = prepare(tmp_path)
    manifest = json.loads(path.read_text())
    data = yaml.safe_load((path.parent / "dataset/data.yaml").read_text())
    assert "test" not in data
    assert data["names"] == {0: "web_patch", 1: "flange", 2: "tool"}
    assert ("ultralytics" in sys.modules) == module_was_loaded
    for item in manifest["images"]:
        assert (
            item["source_rgb_sha256"]
            == hashlib.sha256((tmp_path / item["source_rgb"]).read_bytes()).hexdigest()
        )
        assert (
            item["source_metadata_sha256"]
            == hashlib.sha256((tmp_path / item["source_metadata"]).read_bytes()).hexdigest()
        )
        if item["split"] == "test":
            assert "source_labels" not in item and "dataset_image" not in item
        else:
            assert (
                item["source_labels_sha256"]
                == hashlib.sha256((tmp_path / item["source_labels"]).read_bytes()).hexdigest()
            )
            assert (path.parent / item["dataset_image"]).read_bytes() == (
                tmp_path / item["source_rgb"]
            ).read_bytes()


def test_preparation_is_deterministic_and_does_not_touch_raw_files(tmp_path):
    config, _ = fixture(tmp_path)
    before = {p: p.read_bytes() for p in (tmp_path / "raw").rglob("*") if p.is_file()}
    first = prepare(tmp_path, config, "first")
    second = prepare(tmp_path, config, "second")
    assert first.read_bytes() == second.read_bytes()
    for p in (first.parent / "dataset/labels").rglob("*.txt"):
        assert p.read_bytes() == (second.parent / p.relative_to(first.parent)).read_bytes()
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("reencode", [False, True])
def test_duplicate_rgb_across_splits_fails_before_creating_output(tmp_path, split, reencode):
    config, _ = fixture(tmp_path)
    source = tmp_path / "raw/frontal_base/rgb0.png"
    destination = tmp_path / (
        "raw/stationary_base/rgb0.png" if split == "val" else "reserved/helmet_external/rgb0.png"
    )
    if reencode:
        cv2.imwrite(str(destination), cv2.imread(str(source)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        shutil.copyfile(source, destination)
    with pytest.raises(ValueError, match="Duplicate RGB"):
        prepare(tmp_path, config)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("problem", ["overlap", "unassigned", "absolute", "escape", "count"])
def test_invalid_partition_or_paths_fail_before_writes(tmp_path, problem):
    path, config = fixture(tmp_path)
    if problem == "overlap":
        config["splits"]["val"] = ["frontal_base"]
    elif problem == "unassigned":
        config["splits"]["train"] = ["unknown"]
    elif problem == "absolute":
        config["source_root"] = str(tmp_path / "raw")
    elif problem == "escape":
        meta_path = tmp_path / "raw/frontal_base/render_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["frames"][0]["rgb_path"] = "../stationary_base/rgb0.png"
        meta_path.write_text(json.dumps(meta))
    else:
        config["expected_counts"]["train"] = 3
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        prepare(tmp_path, path)
    assert not (tmp_path / "prepared").exists()


def test_output_must_be_fresh_and_outside_raw_roots(tmp_path):
    path, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        runner().prepare_dataset(path, tmp_path / "raw/new", repo_root=tmp_path)
    (tmp_path / "prepared").mkdir()
    sentinel = tmp_path / "prepared/sentinel"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        prepare(tmp_path, path)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("problem", ["uint8", "unknown_id", "shape"])
def test_labels_reject_wrong_bit_depth_ids_or_alignment(tmp_path, problem):
    path, _ = fixture(tmp_path)
    label_path = tmp_path / "raw/frontal_base/labels0.png"
    mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if problem == "uint8":
        mask = mask.astype(np.uint8)
    elif problem == "unknown_id":
        mask[0, 0] = 42
    else:
        mask = mask[:10]
    cv2.imwrite(str(label_path), mask)
    with pytest.raises(ValueError):
        prepare(tmp_path, path)
    assert not (tmp_path / "prepared").exists()


def test_training_records_checkpoint_hash_and_forwards_only_training_data(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path)
    module = runner()

    class Model:
        def __init__(self, weights, task):
            assert Path(weights).name == "yolo26n-seg.pt" and task == "segment"
            self.ckpt_path = weights
            Path(weights).write_bytes(b"pretrained fixture")

        def train(self, **kwargs):
            assert kwargs["epochs"] == 40 and kwargs["imgsz"] == 640
            assert kwargs["batch"] == 8 and kwargs["workers"] == 0
            assert kwargs["patience"] == 0 and kwargs["deterministic"] is True
            assert kwargs["device"] == 0 and kwargs["pretrained"] is True
            assert "conf" not in kwargs and "test" not in kwargs
            data = yaml.safe_load(Path(kwargs["data"]).read_text())
            assert "test" not in data
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            weights = save_dir / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"trained best")
            (weights / "last.pt").write_bytes(b"trained last")
            (save_dir / "args.yaml").write_text(yaml.safe_dump(kwargs))
            self.trainer = SimpleNamespace(
                best=weights / "best.pt",
                last=weights / "last.pt",
                save_dir=save_dir,
                epoch=39,
                args=SimpleNamespace(**kwargs),
            )
            print("actual trainer log fixture")

    monkeypatch.setattr(module, "_load_ultralytics", lambda: (Model, "test-version"))
    provenance = module.run_training(path, tmp_path / "run", repo_root=tmp_path)
    assert provenance["status"] == "completed"
    assert provenance["epochs_completed"] == 40
    assert provenance["ultralytics_version"] == "test-version"
    assert provenance["checkpoint_sha256"] == hashlib.sha256(b"trained best").hexdigest()
    assert (tmp_path / provenance["checkpoint"]).is_file()
    assert provenance["source"]["git_dirty"] is None  # Fixture has no Git repo.
    assert provenance["inference"] == {"conf": 0.25, "iou": 0.7}
    assert "actual trainer log fixture" in (tmp_path / "run/training.log").read_text()
    assert json.loads((tmp_path / "run/training_provenance.json").read_text()) == provenance


def test_failed_training_preserves_manifest_log_and_failure_provenance(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path)
    module = runner()

    def unavailable():
        raise ImportError("optional ultralytics unavailable")

    monkeypatch.setattr(module, "_load_ultralytics", unavailable)
    with pytest.raises(ImportError):
        module.run_training(path, tmp_path / "run", repo_root=tmp_path)
    provenance = json.loads((tmp_path / "run/training_provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert "optional ultralytics unavailable" in provenance["error"]
    assert (tmp_path / "run/manifest.json").is_file()
    assert "ImportError" in (tmp_path / "run/training.log").read_text()


def test_ultralytics_settings_directory_exists_before_lazy_import(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path)
    module = runner()
    existing = os.environ.get("YOLO_CONFIG_DIR")

    def dependency_boundary():
        directory = Path(os.environ["YOLO_CONFIG_DIR"])
        assert directory.is_dir(), "Missing runtime settings directory before import"
        assert directory.is_relative_to(tmp_path / "run")
        assert (directory / "Ultralytics").is_dir()
        assert os.environ["YOLO_AUTOINSTALL"] == "false"
        raise ImportError("boundary checked")

    monkeypatch.setattr(module, "_load_ultralytics", dependency_boundary)
    with pytest.raises(ImportError, match="boundary checked"):
        module.run_training(path, tmp_path / "run", repo_root=tmp_path)
    assert os.environ.get("YOLO_CONFIG_DIR") == existing

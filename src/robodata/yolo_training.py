"""Bounded synthetic YOLO26 segmentation preparation and training.

Only visible ushort object-ID masks are converted. One connected component is
one YOLO instance, including disconnected patches of the same physical object.
External contours fill holes: these polygon labels approximate topology, not
amodal masks or physical surface boundaries. ID 3 includes fixture markers;
the web_patch class is a synthetic proxy, not a measured persistent web ROI.
Reserved test RGB is hashed for leakage checks only; test labels are never read.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import cv2
import numpy as np
import yaml

ANNOTATION_NOTE = (
    "Synthetic visible-label proxy only. ID 3 contains web and surface fixture markers; "
    "ID 1 maps the other-beam/flange proxy; IDs 2/4/5 include tool fixture markers. "
    "One 8-connected component per polygon instance; disconnected patches are separate "
    "instances, not separate physical objects. External contours fill holes and approximate "
    "pixel boundaries; zero-area fragments use their pixel bounding rectangle. No amodal "
    "completion, measured geometry, real accuracy or cross-tool generalization is claimed."
)
CLASS_SPEC = [
    {"id": 0, "name": "web_patch", "label_ids": [3]},
    {"id": 1, "name": "flange", "label_ids": [1]},
    {"id": 2, "name": "tool", "label_ids": [2, 4, 5]},
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    # Replace atomically so an interrupted run retains readable provenance.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _relative_path(root: Path, name: str, *, file: bool = True) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or PureWindowsPath(name).drive
        or "\\" in name
        or ".." in Path(name).parts
    ):
        raise ValueError(f"Expected a portable contained relative path: {name}")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Input path escapes source root: {name}")
    if file and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _repo_root(repo_root: Path | str | None) -> Path:
    return Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]


def _config(path: Path, repo: Path) -> dict:
    if not path.is_relative_to(repo):
        raise ValueError("Configuration must be inside the repository")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("data_kind") != "synthetic":
        raise ValueError("Requires version 1 synthetic experiment configuration")
    if config.get("classes") != CLASS_SPEC:
        raise ValueError("Requires the approved web_patch/flange/tool object-ID mapping")
    if config.get("model") != "yolo26n-seg.pt":
        raise ValueError("Requires pretrained YOLO26n-seg weights")
    splits = config["splits"]
    if set(splits) != {"train", "val"}:
        raise ValueError("Only complete train/val scenarios may enter training")
    names = [name for split in ("train", "val") for name in splits[split]]
    if (
        not all(isinstance(splits[s], list) and splits[s] for s in splits)
        or any(not isinstance(n, str) or not n or Path(n).name != n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Scenario partitions must be nonempty, unique and disjoint")
    for split in ("train", "val", "test"):
        if (
            type(config["expected_counts"][split]) is not int
            or config["expected_counts"][split] <= 0
        ):
            raise ValueError("Expected split counts must be positive integers")
    if (
        type(config["expected_frames_per_scenario"]) is not int
        or config["expected_frames_per_scenario"] <= 0
    ):
        raise ValueError("Frame bound must be a positive integer")
    parameters = config["train"]
    required = {
        "epochs": 40,
        "imgsz": 640,
        "batch": 8,
        "workers": 0,
        "patience": 0,
        "deterministic": True,
        "device": 0,
        "pretrained": True,
        "save": True,
        "val": True,
        "resume": False,
    }
    if any(
        type(parameters.get(k)) is not type(v) or parameters[k] != v for k, v in required.items()
    ):
        raise ValueError("Training parameters differ from the approved bounded pilot")
    if type(parameters.get("seed")) is not int or parameters["seed"] < 0:
        raise ValueError("Requires an explicit nonnegative training seed")
    if set(parameters) & {"data", "model", "project", "name", "cfg", "split", "conf", "iou"}:
        raise ValueError("Training paths and validation split are controlled by the runner")
    if config.get("inference") != {"conf": 0.25, "iou": 0.7}:
        raise ValueError("Requires the fixed, independently evaluated inference thresholds")
    return config


def _polygons(labels: np.ndarray) -> tuple[str, dict]:
    height, width = labels.shape
    rows = []
    stats: dict[str, Any] = {
        "visible_pixels": {},
        "components": {},
        "holes_filled": 0,
        "degenerate_components": 0,
    }
    for spec in CLASS_SPEC:
        mask = np.isin(labels, spec["label_ids"]).astype(np.uint8)
        count, components, bounds, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        key = str(spec["id"])
        stats["visible_pixels"][key] = int(mask.sum())
        stats["components"][key] = count - 1
        for component in range(1, count):
            binary = (components == component).astype(np.uint8)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is not None:
                stats["holes_filled"] += int(np.count_nonzero(hierarchy[0, :, 3] >= 0))
            external = [c for i, c in enumerate(contours) if hierarchy[0, i, 3] == -1]
            contour = max(external, key=cv2.contourArea).reshape(-1, 2)
            if len(contour) < 3 or cv2.contourArea(contour) == 0:
                x, y, w, h, _ = bounds[component]
                contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                stats["degenerate_components"] += 1
            normalized = contour.astype(float) / [width, height]
            rows.append(str(spec["id"]) + " " + " ".join(f"{v:.8f}" for v in normalized.flat))
    return "\n".join(rows) + ("\n" if rows else ""), stats


def _scenarios(root: Path) -> tuple[list[str], Path]:
    path = _relative_path(root, "effective_config.json")
    scenarios = json.loads(path.read_text(encoding="utf-8"))["scenarios"]
    names = [s["name"] for s in scenarios]
    if not names or len(set(names)) != len(names):
        raise ValueError("Source scenario inventory must be nonempty and unique")
    for name in names:
        _relative_path(root, name, file=False)
        if len(Path(name).parts) != 1:
            raise ValueError("Scenario must be a single directory name")
    actual = {p.parent.name for p in root.glob("*/render_metadata.json")}
    if actual != set(names):
        raise ValueError("Source metadata inventory differs from configured complete scenarios")
    return sorted(names), path


def prepare_dataset(
    config_path: Path | str, output: Path | str, *, repo_root: Path | str | None = None
) -> Path:
    """Preflight all inputs, then create a fresh dataset and deterministic manifest.

    Paths in configuration and manifest are repository relative. data.yaml has
    no absolute paths and no test split. Training resolves its root at runtime.
    """
    repo = _repo_root(repo_root)
    config_path = (repo / config_path).resolve()
    out = (repo / output).resolve()
    if not out.is_relative_to(repo) or out == repo:
        raise ValueError("Output must be inside the repository")
    config = _config(config_path, repo)
    roots = [
        _relative_path(repo, config["source_root"], file=False),
        _relative_path(repo, config["reserved_test_root"], file=False),
    ]
    if roots[0].is_relative_to(roots[1]) or roots[1].is_relative_to(roots[0]):
        raise ValueError("Source and reserved test roots must be disjoint")
    if any(out.is_relative_to(root) or root.is_relative_to(out) for root in roots):
        raise ValueError("Output must be outside both raw roots")
    if out.exists():
        raise FileExistsError(f"Choose a fresh training output directory: {out}")
    train_names, train_effective = _scenarios(roots[0])
    test_names, test_effective = _scenarios(roots[1])
    split_for = {n: s for s, names in config["splits"].items() for n in names}
    if set(split_for) != set(train_names) or set(train_names) & set(test_names):
        raise ValueError("Scenario partitions must cover the source and be disjoint from test")
    counts = {"train": 0, "val": 0, "test": 0}
    records, prepared = [], []
    seen_paths: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    seen_pixels: dict[str, str] = {}
    for root, names in zip(roots, [train_names, test_names]):
        for name in names:
            split = "test" if root == roots[1] else split_for[name]
            source = _relative_path(root, name, file=False)
            metadata_path = _relative_path(source, "render_metadata.json")
            metadata_hash = _sha(metadata_path)
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            frames = meta["frames"]
            expected = config["expected_frames_per_scenario"]
            if (
                len(frames) != expected
                or meta.get("expected_frames") != expected
                or [f["frame_index"] for f in frames] != list(range(expected))
                or meta.get("episode_id") != name
            ):
                raise ValueError(f"Incomplete or mismatched scenario: {name}")
            times = np.array([f["timestamp_sec"] for f in frames], dtype=float)
            if not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
                raise ValueError(f"Invalid source timestamps: {name}")
            for frame in frames:
                rgb_path = _relative_path(source, frame["rgb_path"])
                rgb_hash = _sha(rgb_path)
                rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
                if (
                    rgb is None
                    or rgb.dtype != np.uint8
                    or rgb.ndim != 3
                    or rgb.shape[2] != 3
                    or rgb.shape[:2] != (meta["height"], meta["width"])
                ):
                    raise ValueError(f"Invalid or misaligned RGB: {rgb_path}")
                pixel_digest = hashlib.sha256()
                pixel_digest.update(str(rgb.shape).encode("ascii"))
                pixel_digest.update(rgb.tobytes())
                pixel_hash = pixel_digest.hexdigest()
                for key, seen in [
                    (str(rgb_path), seen_paths),
                    (rgb_hash, seen_hashes),
                    (pixel_hash, seen_pixels),
                ]:
                    if key in seen and seen[key] != split:
                        raise ValueError(f"Duplicate RGB across {seen[key]}/{split}: {rgb_path}")
                    seen[key] = split
                record = {
                    "split": split,
                    "scenario": name,
                    "frame_index": frame["frame_index"],
                    "timestamp_sec": frame["timestamp_sec"],
                    "source_rgb": rgb_path.relative_to(repo).as_posix(),
                    "source_rgb_sha256": rgb_hash,
                    "source_rgb_pixels_sha256": pixel_hash,
                    "source_metadata": metadata_path.relative_to(repo).as_posix(),
                    "source_metadata_sha256": metadata_hash,
                }
                if split != "test":
                    labels_path = _relative_path(source, frame["labels_path"])
                    labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
                    if labels is None or labels.dtype != np.uint16 or labels.shape != rgb.shape[:2]:
                        raise ValueError(f"Requires aligned ushort label PNG: {labels_path}")
                    if (
                        labels_path.suffix.lower() != ".png"
                        or not np.isin(labels, [0, 1, 2, 3, 4, 5]).all()
                    ):
                        raise ValueError(f"Unknown object IDs or non-PNG labels: {labels_path}")
                    text, conversion = _polygons(labels)
                    stem = f"{name}_{frame['frame_index']:05d}"
                    image_name = f"dataset/images/{split}/{stem}.png"
                    label_name = f"dataset/labels/{split}/{stem}.txt"
                    record.update(
                        {
                            "source_labels": labels_path.relative_to(repo).as_posix(),
                            "source_labels_sha256": _sha(labels_path),
                            "dataset_image": image_name,
                            "dataset_label": label_name,
                            "conversion": conversion,
                            "dataset_label_sha256": hashlib.sha256(
                                text.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
                    prepared.append((rgb_path, image_name, label_name, text, rgb_hash))
                records.append(record)
                counts[split] += 1
    if counts != config["expected_counts"]:
        raise ValueError(f"Split counts differ from experiment bound: {counts}")
    manifest = {
        "schema_version": 1,
        "data_kind": "synthetic",
        "annotation_note": ANNOTATION_NOTE,
        "config": config_path.relative_to(repo).as_posix(),
        "config_sha256": _sha(config_path),
        "classes": CLASS_SPEC,
        "counts": counts,
        "scenario_partitions": {**config["splits"], "test": test_names},
        "source_configs": [
            {"path": p.relative_to(repo).as_posix(), "sha256": _sha(p)}
            for p in [train_effective, test_effective]
        ],
        "test_policy": "RGB hash audit only; never train, validate or tune thresholds on test.",
        "images": records,
    }
    out.mkdir(parents=True)
    for source_rgb, image_name, label_name, text, expected_hash in prepared:
        image_path, label_path = out / image_name, out / label_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_rgb, image_path)
        if _sha(image_path) != expected_hash:
            raise ValueError(f"Source RGB changed during preparation: {source_rgb}")
        label_path.write_text(text, encoding="utf-8")
    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {spec["id"]: spec["name"] for spec in CLASS_SPEC},
    }
    (out / "dataset/data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _write_json(out / "effective_config.json", config)
    _write_json(out / "manifest.json", manifest)
    return out / "manifest.json"


def _source_provenance(repo: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {"git_commit": commit, "git_dirty": bool(dirty), "git_status": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None, "git_status": None}


def _load_ultralytics():
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "Training requires optional ultralytics in the active environment"
        ) from error
    return YOLO, ultralytics.__version__


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False


def run_training(
    config_path: Path | str, output: Path | str, *, repo_root: Path | str | None = None
) -> dict:
    """Prepare a fresh bounded dataset, train actual YOLO26n-seg, retain failures."""
    repo = _repo_root(repo_root)
    manifest_path = prepare_dataset(config_path, output, repo_root=repo)
    out = manifest_path.parent
    config = json.loads((out / "effective_config.json").read_text(encoding="utf-8"))
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "data_kind": "synthetic",
        "annotation_note": ANNOTATION_NOTE,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_provenance(repo),
        "config": config,
        "config_sha256": _sha((repo / config_path).resolve()),
        "manifest": manifest_path.relative_to(repo).as_posix(),
        "manifest_sha256": _sha(manifest_path),
        "inference": config["inference"],
        "python_version": platform.python_version(),
        "runner_sha256": _sha(Path(__file__)),
        "log": (out / "training.log").relative_to(repo).as_posix(),
    }
    provenance_path = out / "training_provenance.json"
    _write_json(provenance_path, provenance)
    # Runtime settings, downloads and run products stay under the new ignored output.
    environment = {
        "YOLO_AUTOINSTALL": "false",
        "YOLO_CONFIG_DIR": str(out / "runtime_settings"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    previous = {k: os.environ.get(k) for k in environment}
    (out / "runtime_settings" / "Ultralytics").mkdir(parents=True)
    os.environ.update(environment)
    handlers = []
    try:
        with (out / "training.log").open("w", encoding="utf-8") as log:
            stdout, stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    YOLO, version = _load_ultralytics()
                    provenance["ultralytics_version"] = version
                    if "ultralytics.utils" in sys.modules:
                        logger = sys.modules["ultralytics.utils"].LOGGER
                        for handler in logger.handlers:
                            if hasattr(handler, "setStream"):
                                handlers.append((handler, handler.stream))
                                handler.setStream(stdout)
                    import torch

                    provenance.update(
                        {
                            "torch_version": torch.__version__,
                            "cuda_version": torch.version.cuda,
                            "cuda_available": torch.cuda.is_available(),
                            "device_name": torch.cuda.get_device_name(0)
                            if torch.cuda.is_available()
                            else None,
                        }
                    )
                    weights = out / "pretrained" / config["model"]
                    weights.parent.mkdir()
                    model = YOLO(str(weights), task="segment")
                    if not weights.is_file():
                        raise FileNotFoundError(
                            f"Pretrained checkpoint was not materialized: {weights}"
                        )
                    provenance.update(
                        {
                            "pretrained_checkpoint": weights.relative_to(repo).as_posix(),
                            "pretrained_sha256": _sha(weights),
                            "status": "training",
                        }
                    )
                    data = yaml.safe_load((out / "dataset/data.yaml").read_text(encoding="utf-8"))
                    # Resolve only the runtime YAML; the saved dataset YAML stays relocatable.
                    data["path"] = str(out / "dataset")
                    runtime_yaml = out / "runtime_data.yaml"
                    runtime_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")
                    parameters = {
                        **config["train"],
                        "data": str(runtime_yaml),
                        "project": str(out),
                        "name": "train",
                        "exist_ok": False,
                    }
                    provenance["train_arguments"] = parameters
                    _write_json(provenance_path, provenance)
                    model.train(**parameters)
                    trainer = model.trainer
                    epochs_completed = trainer.epoch + 1
                    if epochs_completed != config["train"]["epochs"]:
                        raise RuntimeError(f"Training ended after {epochs_completed} epochs")
                    best, last = Path(trainer.best), Path(trainer.last)
                    if not best.is_file() or not last.is_file():
                        raise FileNotFoundError("Training did not produce best.pt and last.pt")
                    final_link = out / "checkpoint.pt"
                    final_link.symlink_to(best.relative_to(out))
                    provenance.update(
                        {
                            "status": "completed",
                            "epochs_completed": epochs_completed,
                            "checkpoint": final_link.relative_to(repo).as_posix(),
                            "checkpoint_sha256": _sha(best),
                            "best_checkpoint": best.relative_to(repo).as_posix(),
                            "last_checkpoint": last.relative_to(repo).as_posix(),
                            "last_checkpoint_sha256": _sha(last),
                            "resolved_train_arguments": vars(trainer.args),
                            "validation_note": (
                                "Trainer metrics use synthetic val only; "
                                "external evaluation pending."
                            ),
                        }
                    )
                except BaseException as error:
                    provenance.update(
                        {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                    )
                    traceback.print_exc()
                    raise
                finally:
                    provenance["finished_at"] = datetime.now(timezone.utc).isoformat()
                    for handler, original in handlers:
                        handler.setStream(original)
                    _write_json(provenance_path, provenance)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        print(prepare_dataset(args.config, args.output))
    else:
        result = run_training(args.config, args.output)
        print(json.dumps({k: result[k] for k in ["status", "checkpoint", "checkpoint_sha256"]}))


if __name__ == "__main__":
    main()

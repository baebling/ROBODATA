"""Actual YOLO segmentation/tracking versus RGB calibrated planar localization.

Inference functions receive RGB, K, known fixture geometry and model weights.
Simulator labels and camera transforms enter evaluation only. Patch tracker
IDs do not assert the identity of a complete physical H-beam face.
"""

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .blender_input import camera_from_blender
from .export import write_json
from .geometry import estimate_pnp, inverse
from .pipeline import software_provenance
from .surface_tracking import PlanarSurfaceMap, detect_surface_markers, mask_scores

MODES = ("yolo_per_frame", "yolo_botsort", "yolo_botsort_map")


def reduce_masks(classes, masks, track_ids, shape):
    classes, masks = np.asarray(classes), np.asarray(masks)
    if masks.shape != (len(classes), *shape):
        raise ValueError("Masks must use original RGB coordinates, not letterboxed coordinates")
    if track_ids is not None and len(track_ids) != len(classes):
        raise ValueError("Track IDs do not match masks")
    web = masks[classes == 0].astype(bool).any(axis=0)
    tool = masks[classes == 2].astype(bool).any(axis=0)
    ids = [] if track_ids is None else [int(t) for c, t in zip(classes, track_ids) if c == 0]
    return web, tool, sorted(set(ids))


def ensure_disjoint_hashes(partitions, test_hashes):
    learned = {sha for hashes in partitions.values() for sha in hashes}
    if learned.intersection(test_hashes):
        raise ValueError("Evaluation RGB images overlap training/validation data")


def contained_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Raw input path leaves its source directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_frame_sequence(frames, expected):
    if not frames or len(frames) != expected:
        raise ValueError("Rendered frame count differs from declared count")
    if [f["frame_index"] for f in frames] != list(range(len(frames))):
        raise ValueError("Surface pilot requires original contiguous frame indices")
    times = np.asarray([f["timestamp_sec"] for f in frames], dtype=float)
    if not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be finite, nonnegative and increasing")


def pose_error(predicted, reference):
    if predicted is None:
        return {"position_error_m": None, "rotation_error_deg": None}
    delta = inverse(reference) @ predicted
    angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
    return {
        "position_error_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_error_deg": float(angle),
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read RGB: {path}")
    return image


def audit_training_inputs(root, training, checkpoint, inputs):
    provenance_path = training / "training_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest_path = training / "manifest.json"
    if provenance["status"] != "completed":
        raise ValueError("Training must be completed before holdout evaluation")
    if _sha(checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("Checkpoint is not bound to the declared training run")
    if _sha(manifest_path) != provenance["manifest_sha256"]:
        raise ValueError("Training manifest is not bound to the declared training run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    learned = [r for r in manifest["images"] if r["split"] in {"train", "val"}]
    reserved = [r for r in manifest["images"] if r["split"] == "test"]
    if not learned or not reserved:
        raise ValueError("Training and reserved test inventories must be recorded")
    records = {contained_file(root, r["source_rgb"]): r for r in reserved}
    actual = [p for _, _, paths in inputs for p in paths]
    if (
        len(records) != len(reserved)
        or len(set(actual)) != len(actual)
        or set(actual) != set(records)
    ):
        raise ValueError("Metadata-listed inference inventory differs from reserved test inventory")
    byte_hashes, pixel_hashes = [], []
    for source, metadata, paths in inputs:
        metadata_hash = _sha(source)
        for frame, path in zip(metadata["frames"], paths, strict=True):
            record = records[path]
            byte_hash = _sha(path)
            image = _load_image(path)
            pixel_hash = hashlib.sha256(
                str(image.shape).encode("ascii") + image.tobytes()
            ).hexdigest()
            if (
                contained_file(root, record["source_metadata"]) != source
                or record["source_metadata_sha256"] != metadata_hash
                or record["source_rgb_sha256"] != byte_hash
                or record["source_rgb_pixels_sha256"] != pixel_hash
                or record["frame_index"] != frame["frame_index"]
                or record["timestamp_sec"] != frame["timestamp_sec"]
            ):
                raise ValueError("Reserved test RGB, metadata or frame alignment changed")
            byte_hashes.append(byte_hash)
            pixel_hashes.append(pixel_hash)
    ensure_disjoint_hashes({"learned": [r["source_rgb_sha256"] for r in learned]}, byte_hashes)
    ensure_disjoint_hashes(
        {"learned": [r["source_rgb_pixels_sha256"] for r in learned]}, pixel_hashes
    )
    return {
        "training_provenance_sha256": _sha(provenance_path),
        "training_manifest_sha256": _sha(manifest_path),
        "evaluated_RGB_count": len(actual),
        "external_holdout_verified_by_RGB_hash": True,
        "external_holdout_verified_by_decoded_pixel_hash": True,
        "checkpoint_bound_to_completed_training": True,
    }


def _mask_result(result, shape):
    if result.masks is None:
        return reduce_masks([], np.empty((0, *shape)), None, shape)
    ids = result.boxes.id
    return reduce_masks(
        result.boxes.cls.cpu().numpy(),
        result.masks.data.cpu().numpy() > 0.5,
        ids.cpu().numpy() if ids is not None else None,
        shape,
    )


def _fixture_pose(image, mask, K, points, threshold):
    filtered = image if mask is None else np.where(mask[:, :, None], image, 0).astype(np.uint8)
    pixels = detect_surface_markers(filtered)
    return estimate_pnp(points, pixels, K, max_error_px=threshold).transform


def infer_surface_sequence(paths, reference_path, K, points, checkpoint, settings):
    """No simulator transforms, depth, keypoints or masks accepted at this boundary."""
    from ultralytics import YOLO

    direct, tracked = YOLO(str(checkpoint)), YOLO(str(checkpoint))
    if direct.names != {0: "web_patch", 1: "flange", 2: "tool"}:
        raise ValueError("Checkpoint has incompatible surface/tool class semantics")
    args = settings["inference"]
    mapper = PlanarSurfaceMap(K, **settings["map"])
    init: dict[str, Any] = {"valid": False, "failure_reason": None}
    reference = _load_image(reference_path)
    reference_web, _, _ = _mask_result(direct.predict(reference, **args)[0], reference.shape[:2])
    try:
        T = _fixture_pose(reference, None, K, points, settings["fixture_pnp_max_error_px"])
        init.update(mapper.initialize(reference, reference_web, T))
        init.update(valid=True, T_camera_surface=T.tolist())
    except ValueError as exc:
        init["failure_reason"] = str(exc)
    frames, preview = [], []
    for index, path in enumerate(paths):
        image = _load_image(path)
        a = _mask_result(direct.predict(image, **args)[0], image.shape[:2])
        b = _mask_result(
            tracked.track(image, persist=True, tracker=settings["tracker"], **args)[0],
            image.shape[:2],
        )
        evidence = []
        for mode, (web, tool, ids) in zip(MODES, [a, b, b], strict=True):
            failure, T, diagnostics = None, None, {}
            if mode.endswith("_map"):
                radius = settings["tool_mask_dilation_px"]
                blocked = cv2.dilate(
                    tool.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)
                ).astype(bool)
                T, diagnostics = mapper.update(image, web & ~blocked)
                failure = diagnostics["failure_reason"]
            else:
                try:
                    T = _fixture_pose(image, web, K, points, settings["fixture_pnp_max_error_px"])
                except ValueError as exc:
                    failure = str(exc)
            evidence.append(
                {
                    "mode": mode,
                    "frame_index": index,
                    "T_camera_surface": T.tolist() if T is not None else None,
                    "surface_valid": T is not None,
                    "failure_reason": failure,
                    "track_ids": ids,
                    "map_diagnostics": diagnostics,
                    "predicted_web_mask": web,
                }
            )
        frames.append(evidence)
        if index in {0, len(paths) // 2, len(paths) - 1}:
            overlay = image.copy()
            overlay[b[0]] = (0.6 * overlay[b[0]] + 0.4 * np.array([100, 220, 40])).astype(np.uint8)
            overlay[b[1]] = (0.6 * overlay[b[1]] + 0.4 * np.array([40, 80, 240])).astype(np.uint8)
            preview.append((index, overlay, [e["surface_valid"] for e in evidence]))
    return frames, init, preview


def _summarize(records):
    scores = [r["segmentation"]["iou"] for r in records if r["segmentation"]["iou"] is not None]
    valid = [r for r in records if r["surface_valid"]]
    return {
        "frames": len(records),
        "mask_detected_frames": sum(r["segmentation"]["predicted_pixels"] > 0 for r in records),
        "mean_visible_web_iou": float(np.mean(scores)) if scores else None,
        "surface_valid_frames": len(valid),
        "mean_position_error_m": float(np.mean([r["position_error_m"] for r in valid]))
        if valid
        else None,
        "mean_rotation_error_deg": float(np.mean([r["rotation_error_deg"] for r in valid]))
        if valid
        else None,
        "failure_counts": dict(
            Counter(r["failure_reason"] for r in records if r["failure_reason"])
        ),
    }


def run(config_path, checkpoint, output):
    config_path, checkpoint, out = map(
        lambda p: Path(p).resolve(), (config_path, checkpoint, output)
    )
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    render, references, training = [
        (root / settings[key]).resolve()
        for key in ("render_root", "reference_root", "training_root")
    ]
    if out.exists() or any(
        out.is_relative_to(p) or p.is_relative_to(out) for p in [render, references, training]
    ):
        raise ValueError("Choose a new evaluation output outside input trees")
    cv2.setRNGSeed(settings["seed"])
    sources = sorted(render.glob("*/render_metadata.json"))
    if not sources or not checkpoint.is_file():
        raise ValueError("Missing raw scenes or trained checkpoint")
    inputs = []
    for source in sources:
        metadata = json.loads(source.read_text(encoding="utf-8"))
        validate_frame_sequence(metadata["frames"], metadata["expected_frames"])
        paths = [contained_file(source.parent, f["rgb_path"]) for f in metadata["frames"]]
        inputs.append((source, metadata, paths))
    audit = audit_training_inputs(root, training, checkpoint, inputs)
    reference_manifest_path = references / "reference_manifest.json"
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    reference_records = {r["scenario"]: r for r in reference_manifest["records"]}
    provenance = {
        **software_provenance(config_path),
        "checkpoint_sha256": _sha(checkpoint),
        "source_sha256": {
            name: _sha(Path(__file__).parent / name)
            for name in ("yolo_surface_experiment.py", "surface_tracking.py", "geometry.py")
        },
        "ultralytics_version": importlib.metadata.version("ultralytics"),
        "reference_manifest_sha256": _sha(reference_manifest_path),
        **audit,
        "scope": settings["scope"],
        "metric_scale_source": "synthetic_known_patch_geometry_RGB_calibration",
        "track_id_note": "IDs identify visible patches, not proven physical face identity",
        "model_inputs": ["RGB", "intrinsics", "known patch geometry", "tool-free RGB reference"],
        "excluded_from_inference": ["depth", "GT transforms", "GT masks", "projected GT keypoints"],
        "evaluation_protocol": (
            "Training-held-out synthetic diagnostic pilot, not an untouched statistical test. "
            "An initial failed run exposed mask-clipped reference fixtures; calibration handling "
            "was corrected without changing model weights or thresholds."
        ),
    }
    out.mkdir(parents=True)
    rows, tiles = [], []
    all_records: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for source, metadata, paths in inputs:
        name = source.parent.name
        reference_path = references / name / "rgb.png"
        ref_record = reference_records[name]
        if (
            _sha(reference_path) != ref_record["rgb_sha256"]
            or _sha(source.parent / "hbeam_spray.blend") != ref_record["source_blend_sha256"]
        ):
            raise ValueError("Reference RGB or source scene hash changed")
        K = np.asarray(metadata["camera_matrix"])
        points = [metadata["surface_points_m"][key] for key in ("a", "b", "c", "d")]
        predictions, init, preview = infer_surface_sequence(
            paths, reference_path, K, points, checkpoint, settings
        )
        case = out / name
        case.mkdir()
        records: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
        for raw, evidence in zip(metadata["frames"], predictions, strict=True):
            labels = cv2.imread(
                str(contained_file(source.parent, raw["labels_path"])), cv2.IMREAD_UNCHANGED
            )
            reference_pose = camera_from_blender(raw["T_world_camera"]) @ np.asarray(
                raw["T_world_surface"]
            )
            for item in evidence:
                mask = item.pop("predicted_web_mask")
                mode = item["mode"]
                item["timestamp_sec"] = raw["timestamp_sec"]
                item["segmentation"] = mask_scores(mask, labels == 3)
                predicted = item["T_camera_surface"]
                item.update(
                    pose_error(
                        np.asarray(predicted) if predicted is not None else None, reference_pose
                    )
                )
                item["scenario"] = name
                records[mode].append(item)
                all_records[mode].append(item)
                cv2.imwrite(
                    str(case / f"{mode}_{raw['frame_index']:05d}.png"), mask.astype(np.uint8) * 255
                )
        write_json(
            case / "predictions.json",
            {
                "reference_initialization": init,
                "frames": records,
                "metadata_sha256": _sha(source),
                "RGB_sha256": [_sha(p) for p in paths],
            },
        )
        row = {
            "scenario": name,
            "map_initialized": init["valid"],
            "modes": {mode: _summarize(records[mode]) for mode in MODES},
        }
        rows.append(row)
        tiles.append((name, preview))
        print(
            name, {mode: row["modes"][mode]["surface_valid_frames"] for mode in MODES}, flush=True
        )
    summary = {mode: _summarize(all_records[mode]) for mode in MODES}
    write_json(
        out / "comparison.json", {"provenance": provenance, "summary": summary, "scenarios": rows}
    )
    write_json(out / "effective_config.json", settings)
    lines = [
        "# YOLO Surface Visibility Pilot",
        "",
        settings["scope"],
        "",
        "| Method | Masks | Web IoU | Valid poses | Position error m | Rotation error deg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, item in summary.items():

        def number(value):
            return "null" if value is None else f"{value:.4f}"

        lines.append(
            f"| {mode} | {item['mask_detected_frames']}/{item['frames']} | "
            f"{number(item['mean_visible_web_iou'])} | "
            f"{item['surface_valid_frames']}/{item['frames']} | "
            f"{number(item['mean_position_error_m'])} | {number(item['mean_rotation_error_deg'])} |"
        )
    lines += [
        "",
        "Coverage and conditional error must be considered together.",
        "All three conditions use the same trained checkpoint and held-out raw RGB frames.",
        "Map condition additionally uses a tool-free RGB calibration capture, not test GT poses.",
        "Known synthetic patch dimensions and colored calibration fixtures are explicit priors.",
        "The persistent map is planar reference-image localization, not full background SLAM.",
        "GT ID 3 includes web-body side faces and surface markers: a web-label proxy.",
        "Disconnected detections have patch IDs, not guaranteed physical work-face IDs.",
        "No pose is copied through failed localization. No gap filling/smoothing is performed.",
        "This is surface-only: tool and hand pose have NOT been upgraded or shown to improve.",
        "No complete canonical trajectory, real hardware accuracy or spray quality claim is made.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _contact_sheet(tiles, out / "contact_sheet.png")
    return summary


def _contact_sheet(tiles, destination):
    width, height, label = 360, 250, 190
    canvas = Image.new("RGB", (label + 3 * width, 50 + len(tiles) * height), "#f4f4f4")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text(
        (12, 12),
        "YOLO web = green / tool = red; surface validity: per-frame / tracked / map",
        fill="black",
        font=font,
    )
    for row, (name, preview) in enumerate(tiles):
        y = 50 + row * height
        draw.text((8, y + 10), name.replace("_", "\n"), fill="black", font=font)
        for col, (index, image, valid) in enumerate(preview):
            tile = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            tile.thumbnail((width - 4, height - 48))
            x = label + col * width
            canvas.paste(tile, (x, y + 30))
            draw.text((x + 4, y + 8), f"frame {index}: {valid}", fill="black", font=font)
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()

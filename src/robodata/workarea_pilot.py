"""Unmodified external RGB-PnP tracker diagnostic with one synthetic depth seed.

This tests camera motion, NOT S1-S5 recognition, semantic surface identity or
tool pose. Later frames receive RGB only; all first-frame depths are explicit
simulation priors, including hypothetical seeds for the smartphone views.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .blender_input import _read_depth, camera_from_blender, radial_to_axial
from .export import write_json
from .geometry import inverse, validate_transform
from .pipeline import software_provenance
from .yolo_surface_experiment import contained_file, pose_error, validate_frame_sequence

TRACKER_SOURCE = "integrations/zed_live_next_region_20260921/robust_rgb_pnp.py"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(root):
    root = Path(root).resolve()
    manifest_path = root / "MANIFEST_SHA256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("Empty source manifest")
    for name, item in manifest.items():
        path = contained_file(root, name)
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"External payload changed: {name}")
    return {"verified_files": len(manifest), "manifest_sha256": sha256(manifest_path)}


def camera_motion_reference(initial_world_camera, current_world_camera):
    return camera_from_blender(current_world_camera) @ inverse(
        camera_from_blender(initial_world_camera)
    )


def camera_motion_error(predicted, reference):
    if predicted is None:
        return pose_error(None, reference)
    return pose_error(inverse(predicted), inverse(reference))


def validate_depth_seed(depth, checks, atol_m, rtol):
    if depth.ndim != 2 or not checks or atol_m < 0 or rtol < 0:
        raise ValueError("Initial depth needs explicit independent sample checks")
    for check in checks:
        u, v, expected = check["u"], check["v"], check["expected_m"]
        if (
            type(u) is not int
            or type(v) is not int
            or not 0 <= u < depth.shape[1]
            or not 0 <= v < depth.shape[0]
            or not np.isfinite(expected)
            or expected <= 0
            or not np.isfinite(depth[v, u])
            or not np.isclose(depth[v, u], expected, atol=atol_m, rtol=rtol)
        ):
            raise ValueError("Initial depth disagrees with independent renderer samples")


def track_sequence(images, initial_depth, K, tracker_factory):
    """GT-free inference boundary: one depth seed, RGB sequence and intrinsics."""
    if not images or initial_depth.shape != images[0].shape[:2]:
        raise ValueError("Initial RGB/depth shape mismatch or empty sequence")
    if any(image.shape != images[0].shape or image.dtype != np.uint8 for image in images):
        raise ValueError("RGB sequence must keep its calibrated shape and uint8 format")
    K = np.asarray(K, float)
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or min(K[0, 0], K[1, 1]) <= 0
        or not np.allclose(K[2], [0, 0, 1], atol=1e-9, rtol=0)
        or abs(K[0, 1]) > 1e-9
        or abs(K[1, 0]) > 1e-9
    ):
        raise ValueError("Invalid camera calibration")
    tracker = tracker_factory(K)
    initialized, failure = False, None
    try:
        tracker.initialize(images[0], initial_depth)
        initialized = True
    except (ValueError, cv2.error) as exc:
        failure = str(exc)
    initialization = {
        "accepted": initialized,
        "failure_reason": failure,
        "initial_anchor_points": len(tracker.anchor_obj) if initialized else 0,
    }
    records: list[dict[str, Any]] = [
        {
            "frame_index": 0,
            "reference_capture": True,
            "accepted": False,
            "T_camera_reference": None,
            "failure_reason": "reference_only",
            "diagnostics": {},
            "runtime_ms": None,
        }
    ]
    for index, image in enumerate(images[1:], 1):
        start = perf_counter()
        T, stats = None, {"accepted": False, "reason": "not_initialized"}
        if initialized:
            try:
                T, stats = tracker.update(image)
                if T is not None:
                    T = validate_transform(T)
                    if not stats.get("accepted", False):
                        raise ValueError("External pose lacks explicit acceptance")
            except (ValueError, cv2.error) as exc:
                T = None
                stats = {"accepted": False, "reason": "invalid_pose", "error": str(exc)}
        records.append(
            {
                "frame_index": index,
                "reference_capture": False,
                "accepted": T is not None,
                "T_camera_reference": T.tolist() if T is not None else None,
                "failure_reason": stats.get("reason") if T is None else None,
                "diagnostics": stats,
                "runtime_ms": (perf_counter() - start) * 1000,
            }
        )
    return records, initialization


def summarize_camera_tracking(records):
    evaluated = [r for r in records if not r["reference_capture"]]
    accepted = [r for r in evaluated if r["accepted"]]
    return {
        "reference_frames": len(records) - len(evaluated),
        "evaluated_frames": len(evaluated),
        "accepted_frames": len(accepted),
        "mean_position_error_m": float(np.mean([r["position_error_m"] for r in accepted]))
        if accepted
        else None,
        "mean_rotation_error_deg": float(np.mean([r["rotation_error_deg"] for r in accepted]))
        if accepted
        else None,
        "max_position_error_m": max((r["position_error_m"] for r in accepted), default=None),
        "failure_counts": dict(
            Counter(r["failure_reason"] for r in evaluated if r["failure_reason"])
        ),
    }


def _contact_sheet(cases, destination):
    tile_w, tile_h, label_w = 340, 260, 190
    canvas = Image.new("RGB", (label_w + 3 * tile_w, 35 + len(cases) * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        "External RGB-PnP tracker: initial depth reference / middle / final RGB",
        fill="black",
    )
    for row, (name, preview) in enumerate(cases):
        y = 35 + row * tile_h
        draw.text((8, y + 8), name.replace("_", "\n"), fill="black")
        for col, (image, record) in enumerate(preview):
            x = label_w + col * tile_w
            tile = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            tile.thumbnail((tile_w - 8, tile_h - 55))
            canvas.paste(tile, (x, y + 48))
            label = (
                "reference, not scored"
                if record["reference_capture"]
                else (
                    f"accepted: {record['position_error_m']:.3f} m"
                    if record["accepted"]
                    else f"invalid: {record['failure_reason']}"
                )
            )
            draw.text((x, y + 8), f"frame {record['frame_index']}\n{label}", fill="black")
    canvas.save(destination)


def run(config_path, external_root, output):
    config_path, external_root, out = map(
        lambda p: Path(p).resolve(), (config_path, external_root, output)
    )
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[2]
    raw = (repo / settings["render_root"]).resolve()
    if out.exists() or any(
        out.is_relative_to(p) or p.is_relative_to(out) for p in [raw, external_root]
    ):
        raise ValueError("Choose a fresh output outside all raw/external inputs")
    audit = verify_bundle(external_root)
    sources = sorted(raw.glob("*/render_metadata.json"))
    if len(sources) != settings["expected_scenarios"]:
        raise ValueError("Scenario count differs from versioned configuration")
    source_path = contained_file(external_root, TRACKER_SOURCE)
    manifest = json.loads((external_root / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    if TRACKER_SOURCE not in manifest:
        raise ValueError("Tracker is not a verified external payload")
    out.mkdir(parents=True)
    snapshot = out / "upstream"
    snapshot.mkdir()
    copied = snapshot / "robust_rgb_pnp.py"
    shutil.copy2(source_path, copied)
    if sha256(copied) != manifest[TRACKER_SOURCE]["sha256"]:
        raise ValueError("Tracker changed during snapshot capture")
    spec = importlib.util.spec_from_file_location("workarea_external_rgb_pnp", copied)
    if spec is None or spec.loader is None:
        raise ValueError("External tracker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cv2.setRNGSeed(settings["seed"])
    records, summaries, tiles = [], [], []
    for source in sources:
        metadata = json.loads(source.read_text(encoding="utf-8"))
        validate_frame_sequence(metadata["frames"], settings["expected_frames_per_scenario"])
        images, paths = [], []
        for frame in metadata["frames"]:
            path = contained_file(source.parent, frame["rgb_path"])
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape != (metadata["height"], metadata["width"], 3):
                raise ValueError("Cannot decode calibrated RGB image")
            images.append(image)
            paths.append(path)
        depth_name = metadata["frames"][0]["depth_path"]
        depth = _read_depth(source.parent, depth_name, images[0].shape[:2])
        validate_depth_seed(
            depth,
            metadata["frames"][0].get("depth_checks"),
            settings["depth_check_atol_m"],
            settings["depth_check_rtol"],
        )
        K = np.asarray(metadata["camera_matrix"])
        if metadata["depth_convention"] == "ray_distance":
            depth = radial_to_axial(depth, K)
        elif metadata["depth_convention"] != "axial_z":
            raise ValueError("Unknown initial depth convention")
        predicted, init = track_sequence(images, depth, K, module.RobustRGBPnPTracker)
        # Simulator camera transforms enter only after RGB inference has finished.
        for frame, item in zip(metadata["frames"], predicted, strict=True):
            T = item["T_camera_reference"]
            item.update(
                camera_motion_error(
                    np.asarray(T) if T is not None else None,
                    camera_motion_reference(
                        metadata["frames"][0]["T_world_camera"], frame["T_world_camera"]
                    ),
                )
            )
            item.update(timestamp_sec=frame["timestamp_sec"], scenario=source.parent.name)
        case = out / source.parent.name
        case.mkdir()
        write_json(
            case / "predictions.json",
            {
                "initialization": init,
                "frames": predicted,
                "metadata_sha256": sha256(source),
                "RGB_sha256": [sha256(path) for path in paths],
                "initial_depth_sha256": sha256(contained_file(source.parent, depth_name)),
                "sensor_depth_available": metadata.get("sensor_profile", {}).get("depth_available"),
                "depth_source": "ideal_simulator_initial_depth_not_actual_sensor_measurement",
            },
        )
        records.extend(predicted)
        result = {
            "scenario": source.parent.name,
            "initialization": init,
            **summarize_camera_tracking(predicted),
        }
        summaries.append(result)
        tiles.append(
            (
                source.parent.name,
                [(images[i], predicted[i]) for i in [0, len(images) // 2, len(images) - 1]],
            )
        )
        print(
            source.parent.name,
            result["accepted_frames"],
            "/",
            result["evaluated_frames"],
            flush=True,
        )
    final_audit = verify_bundle(external_root)
    if final_audit != audit:
        raise ValueError("External manifest changed during evaluation")
    summary = summarize_camera_tracking(records)
    provenance = {
        **software_provenance(config_path),
        **audit,
        "runner_sha256": sha256(Path(__file__)),
        "upstream_tracker_sha256": sha256(copied),
        "source_package_unchanged": True,
        "inference_inputs": ["RGB sequence", "K", "first-frame synthetic axial depth"],
        "depth_import_validation": (
            "Independent renderer samples validate the seed; no scale fitting or GT pose inputs"
        ),
        "excluded_from_inference": [
            "later depth",
            "GT poses",
            "GT masks",
            "GT keypoints",
            "ZED VIO",
        ],
        "scope": settings["scope"],
        "weights_used": [],
        "FFS_or_five_face_recognition_executed": False,
        "comparison_warning": (
            "Camera motion, not semantic surface pose; not paired with YOLO surface-pose counts."
        ),
    }
    write_json(
        out / "comparison.json",
        {"provenance": provenance, "summary": summary, "scenarios": summaries},
    )
    write_json(out / "effective_config.json", settings)
    lines = [
        "# Workarea RGB Tracker Module Diagnostic",
        "",
        settings["scope"],
        "",
        "| Scenario | Initialized | Accepted/evaluated | Mean position error m |",
        "|---|---|---:|---:|",
    ]
    for case in summaries:
        value = case["mean_position_error_m"]
        error = "null" if value is None else f"{value:.4f}"
        lines.append(
            f"| {case['scenario']} | {case['initialization']['accepted']} | "
            f"{case['accepted_frames']}/{case['evaluated_frames']} | {error} |"
        )
    lines += [
        "",
        "Initial captures are not scored as successful poses.",
        "Later frames use RGB only; the metric seed is ideal synthetic depth, "
        "not RGB-only reconstruction.",
        "Phone views use a hypothetical initial RGB-D seed, not a claimed phone depth sensor.",
        "No semantic surface frame, five-face recognition, tool pose or canonical "
        "action is produced.",
        "Accepted poses can be wrong. Errors use independent simulator camera "
        "transforms after inference.",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _contact_sheet(tiles, out / "contact_sheet.png")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.external_root, args.output), indent=2))


if __name__ == "__main__":
    main()

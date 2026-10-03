"""Optional official-DITTO relative-motion probe, not an absolute pose detector."""

import argparse
import contextlib
import hashlib
import io
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def assess_fit(
    source,
    target,
    transform,
    *,
    upstream_failed=False,
    threshold_m=0.005,
    minimum_inliers=20,
    minimum_fraction=0.5,
    minimum_spread_m=0.001,
):
    source, target, transform = [np.asarray(x, dtype=float) for x in (source, target, transform)]
    result = {"valid": False, "failure_reason": None, "correspondences": len(source)}
    if upstream_failed:
        result["failure_reason"] = "upstream_fit_failed"
    elif source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        result["failure_reason"] = "invalid_point_shape"
    elif len(source) < minimum_inliers:
        result["failure_reason"] = "insufficient_correspondences"
    elif not all(np.isfinite(x).all() for x in (source, target, transform)):
        result["failure_reason"] = "nonfinite_fit"
    elif (
        transform.shape != (4, 4)
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6)
    ):
        result["failure_reason"] = "invalid_rigid_transform"
    else:
        residuals = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
        inliers = residuals < threshold_m
        result["inliers"] = int(inliers.sum())
        result["inlier_fraction"] = float(inliers.mean())
        if inliers.sum() < minimum_inliers or inliers.mean() < minimum_fraction:
            result["failure_reason"] = "insufficient_inliers"
        else:
            spreads = [
                np.linalg.svd(x[inliers] - x[inliers].mean(axis=0), compute_uv=False)
                / np.sqrt(inliers.sum())
                for x in (source, target)
            ]
            result["second_singular_value_m"] = float(min(x[1] for x in spreads))
            result["inlier_rmse_m"] = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
            if result["second_singular_value_m"] < minimum_spread_m:
                result["failure_reason"] = "degenerate_inliers"
            else:
                result["valid"] = True
    return result


def git_state(root):
    def command(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    return {"commit": command("rev-parse", "HEAD"), "dirty": bool(command("status", "--porcelain"))}


def read_observation(root, frame, matrix, depth_path):
    rgb = cv2.imread(str(root / frame["rgb_path"]), cv2.IMREAD_COLOR)
    depth = np.load(depth_path, allow_pickle=False)
    labels = cv2.imread(str(root / frame["labels_path"]), cv2.IMREAD_UNCHANGED)
    if rgb is None or labels is None:
        raise ValueError("Missing or unreadable RGB/depth/object labels")
    if depth.ndim != 2 or depth.shape != rgb.shape[:2] or labels.shape != depth.shape:
        raise ValueError("RGB, depth and labels must be aligned")
    v, u = np.indices(depth.shape)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(matrix).T
    xyz = rays * depth[:, :, None]
    xyz[~np.isfinite(depth) | (depth <= 0) | (depth >= 1e6)] = np.nan
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), xyz, labels


def filter_target_tool_matches(source_pixels, target_pixels, scores, target_mask):
    source_pixels, target_pixels, scores = [
        np.asarray(x) for x in (source_pixels, target_pixels, scores)
    ]
    inside = (
        (target_pixels[:, 0] >= 0)
        & (target_pixels[:, 0] < target_mask.shape[0])
        & (target_pixels[:, 1] >= 0)
        & (target_pixels[:, 1] < target_mask.shape[1])
    )
    keep = inside.copy()
    keep[inside] = target_mask[target_pixels[inside, 0], target_pixels[inside, 1]]
    return source_pixels[keep], target_pixels[keep], scores[keep]


def probe_pair(observation_a, observation_b, config, correspondence, step, warp):
    rgb_a, xyz_a, labels_a = observation_a
    rgb_b, xyz_b, labels_b = observation_b
    mask_a, mask_b = labels_a == 2, labels_b == 2
    if not mask_a.any():
        return {"valid": False, "failure_reason": "empty_source_tool_mask"}, None
    if not mask_b.any():
        return {"valid": False, "failure_reason": "empty_target_tool_mask"}, None
    rows, columns = np.where(mask_a | mask_b)
    if not len(rows):
        return {"valid": False, "failure_reason": "empty_tool_mask"}, None
    margin = config["crop_margin_px"]
    y0, y1 = max(0, int(rows.min()) - margin), min(len(mask_a), int(rows.max()) + margin + 1)
    x0, x1 = (
        max(0, int(columns.min()) - margin),
        min(mask_a.shape[1], int(columns.max()) + margin + 1),
    )
    rgb_a, rgb_b = [x[y0:y1, x0:x1] for x in (rgb_a, rgb_b)]
    xyz_a, xyz_b = [x[y0:y1, x0:x1] for x in (xyz_a, xyz_b)]
    mask_a, mask_b = [x[y0:y1, x0:x1] for x in (mask_a, mask_b)]
    radius = config["mask_erosion_px"]
    if radius:
        mask_a, mask_b = [
            cv2.erode(x.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)) > 0
            for x in (mask_a, mask_b)
        ]
    if not mask_a.any() or not mask_b.any():
        return {"valid": False, "failure_reason": "empty_eroded_tool_mask"}, None
    captured = {}

    def record_correspondences(*args, **kwargs):
        points_a, points_b, scores = correspondence(*args, **kwargs)
        captured["original_matches"] = len(points_a)
        points_a, points_b, scores = filter_target_tool_matches(points_a, points_b, scores, mask_b)
        captured["target_tool_matches"] = len(points_a)
        captured["source_pixels"], captured["target_pixels"] = points_a, points_b
        return points_a, points_b, scores

    log = io.StringIO()
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(log):
            transform = step(
                rgb_a, rgb_b, xyz_a, xyz_b, mask_a, correspondence_calculator=record_correspondences
            )
    except (ValueError, RuntimeError, IndexError, np.linalg.LinAlgError) as error:
        return {"valid": False, "failure_reason": type(error).__name__, "message": str(error)}, None
    a, b = captured["source_pixels"], captured["target_pixels"]
    inside = (b[:, 0] >= 0) & (b[:, 0] < len(xyz_b)) & (b[:, 1] >= 0) & (b[:, 1] < xyz_b.shape[1])
    source, target = xyz_a[a[inside, 0], a[inside, 1]], xyz_b[b[inside, 0], b[inside, 1]]
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    quality = assess_fit(
        source[finite],
        target[finite],
        transform,
        upstream_failed="Failed! Returning" in log.getvalue(),
        threshold_m=config["inlier_threshold_m"],
        minimum_inliers=config["minimum_inliers"],
        minimum_fraction=config["minimum_inlier_fraction"],
        minimum_spread_m=config["minimum_second_singular_value_m"],
    )
    quality.update({"runtime_sec": time.monotonic() - started, "upstream_stdout": log.getvalue()})
    quality["original_matches"] = captured["original_matches"]
    quality["target_tool_matches"] = captured["target_tool_matches"]
    quality["estimated_transform_source_to_target"] = (
        transform.tolist() if quality["valid"] else None
    )
    if quality["valid"]:
        # This checks the official warp API algebra, not new-scene task transfer.
        fixture = np.repeat(np.eye(4)[None], 3, axis=0)
        fixture[:, 0, 3] = [0, 0.02, 0.04]
        warped = warp(rgb_a, rgb_b, xyz_a, xyz_b, mask_a, fixture, lambda *a, **k: transform)
        quality["warp_api_max_abs_residual"] = float(np.max(np.abs(warped - transform @ fixture)))
    return quality, transform if quality["valid"] else None


def camera_tool_reference(frame):
    world_camera = np.asarray(frame["T_world_camera"])
    return np.diag([1, -1, -1, 1]) @ np.linalg.inv(world_camera) @ np.asarray(frame["T_world_tool"])


def validate_prepared_depth(manifest, metadata, metadata_sha256):
    provenance = manifest.get("provenance", {})
    camera = manifest.get("camera", {})
    if (
        provenance.get("render_metadata_sha256") != metadata_sha256
        or provenance.get("depth_convention_output") != "axial_z_meters"
        or manifest.get("episode_id") != metadata["episode_id"]
        or manifest.get("data_kind") != "synthetic"
        or manifest.get("depth_aligned_to_rgb") is not True
        or camera.get("width") != metadata["width"]
        or camera.get("height") != metadata["height"]
        or not np.allclose(camera.get("matrix"), metadata["camera_matrix"])
    ):
        raise ValueError("Prepared axial-meter depth must match its source render and calibration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ditto-root", type=Path, required=True)
    parser.add_argument("--flowcontrol-root", type=Path, required=True)
    parser.add_argument("--raft-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Existing blender-evaluate outputs with metric axial .npy depth",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output folder")
    config = json.loads(args.config.read_text())
    if config["schema_version"] != 1 or config["method"] != "flow_ransac":
        raise ValueError("Unsupported probe configuration")
    for path in [args.flowcontrol_root, args.raft_root / "src", args.ditto_root]:
        sys.path.insert(0, str(path.resolve()))
    import torch
    from DITTO.methods_2D import CorrespondencesInMaskMethod
    from DITTO.tracking_3D import Step3DMethod, warp_3D_trajectory

    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    report = {
        "data_kind": "synthetic_blender",
        "purpose": "relative_tool_motion_feasibility_probe",
        "inputs": "RGB + rendered aligned metric depth + oracle object masks",
        "functional_pose_seed": None,
        "automatic_segmentation": False,
        "camera_motion_removed": False,
        "warp_scope": "official API algebra check only; no new-scene correspondence benchmark",
        "config": config,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {
            "robodata": git_state(Path(__file__).resolve().parents[1]),
            "ditto": git_state(args.ditto_root),
            "flowcontrol": git_state(args.flowcontrol_root),
            "raft": git_state(args.raft_root),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
        "pairs": [],
    }
    args.output.mkdir(parents=True)
    for scenario in config["scenarios"]:
        root = args.render_root / scenario
        metadata_path = root / "render_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        matrix = np.asarray(metadata["camera_matrix"])
        prepared = args.prepared_root / scenario / "inputs"
        manifest = json.loads((prepared / "manifest.json").read_text())
        validate_prepared_depth(
            manifest, metadata, hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        )
        for ia, ib in config["pairs"]:
            a, b = metadata["frames"][ia], metadata["frames"][ib]
            depth_paths = [prepared / manifest["observations"][i]["depth_path"] for i in [ia, ib]]
            if any(manifest["observations"][i]["frame_index"] != i for i in [ia, ib]):
                raise ValueError("Prepared observations do not match original frame indices")
            observations = [
                read_observation(root, f, matrix, d) for f, d in zip([a, b], depth_paths)
            ]
            result, transform = probe_pair(
                *observations,
                config,
                CorrespondencesInMaskMethod.flow,
                Step3DMethod.flow_ransac,
                warp_3D_trajectory,
            )
            result.update(
                {
                    "scenario": scenario,
                    "frame_indices": [ia, ib],
                    "timestamps_sec": [a["timestamp_sec"], b["timestamp_sec"]],
                    "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                    "prepared_depth_sha256": [
                        hashlib.sha256(d.read_bytes()).hexdigest() for d in depth_paths
                    ],
                }
            )
            if transform is not None:
                # Simulator poses are used only here, after inference, for evaluation.
                truth_a, truth_b = camera_tool_reference(a), camera_tool_reference(b)
                reference = truth_b @ np.linalg.inv(truth_a)
                result["relative_rotation_error_deg"] = float(
                    np.degrees(
                        Rotation.from_matrix(transform[:3, :3] @ reference[:3, :3].T).magnitude()
                    )
                )
                transformed_tip = transform[:3, :3] @ truth_a[:3, 3] + transform[:3, 3]
                result["reference_tip_transfer_error_mm"] = float(
                    np.linalg.norm(transformed_tip - truth_b[:3, 3]) * 1000
                )
            report["pairs"].append(result)
            (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            print(
                json.dumps(
                    {
                        k: result.get(k)
                        for k in [
                            "scenario",
                            "frame_indices",
                            "valid",
                            "failure_reason",
                            "inlier_fraction",
                            "relative_rotation_error_deg",
                            "reference_tip_transfer_error_mm",
                        ]
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

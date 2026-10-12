"""Bounded public RGB-D preprocessing, separate from construction extraction."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .preflight import CAMERAS, check_assets, check_transform
from .runner import (
    align_next_pose,
    open_public,
    prepare_output,
    reproduction_revision,
    sha256_file,
    tree_digest,
    verify_source,
)

SUBMODULES = {
    "FoundationPose": "e7596fdfd2e34d103df4078330d760e678aba529",
    "Grounded-Segment-Anything": "05d97e6554d30ad7da17b4e3237e417a4b01021f",
}


def load_clip(root: Any, config: dict[str, Any]) -> dict[str, np.ndarray]:
    scale = config.get("depth_scale_m")
    if not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Explicit positive depth scale required")
    if config.get("intrinsics_dtype") != "float32":
        raise ValueError(
            "The released FoundationPose crop helper requires explicit float32 intrinsics"
        )
    timestamps = np.asarray(root["data/timestamp"][:])
    ends = np.asarray(root["meta/episode_ends"][:])
    align_next_pose(np.zeros((len(timestamps), 9)), ends)
    episode, count = config["episode"], config["frame_count"]
    if (
        isinstance(episode, bool)
        or not isinstance(episode, int)
        or not 0 <= episode < len(ends)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 2
    ):
        raise ValueError("Invalid episode or frame count")
    start = 0 if episode == 0 else int(ends[episode - 1])
    stop = start + count
    if stop > ends[episode]:
        raise ValueError("Clip must stay within one episode")
    selected_time = timestamps[start:stop]
    if not np.isfinite(selected_time).all() or np.any(np.diff(selected_time) <= 0):
        raise ValueError("Invalid clip timestamps")
    camera = config["camera"]
    if camera not in CAMERAS:
        raise ValueError("Unsupported camera")
    prefix = f"data/observations/images/{camera}"
    rgb = root[f"{prefix}_color"][start:stop]
    depth = root[f"{prefix}_depth"][start:stop].astype(np.float64) * scale
    intrinsics = root[f"{prefix}_intrinsics"][start:stop]
    if (
        intrinsics.shape != (count, 3, 3)
        or not np.isfinite(intrinsics).all()
        or not np.allclose(intrinsics, intrinsics[0])
        or intrinsics[0, 0, 0] <= 0
        or intrinsics[0, 1, 1] <= 0
    ):
        raise ValueError("Upstream tracker requires fixed valid intrinsics")
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or depth.shape != rgb.shape[:3]:
        raise ValueError("RGB/depth clip shape mismatch")
    return {
        "rgb": rgb,
        "depth_m": depth,
        "intrinsics": intrinsics[0].astype(np.float32),
        "depth_valid": np.isfinite(depth) & (depth > 0),
        "source_frame_index": np.arange(start, stop),
        "timestamp_sec": selected_time,
    }


def inspect_pose_video(path: Path, expected_frames: int) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    previews = {}
    count, shape = 0, None
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            valid, frame = capture.read()
            if not valid:
                break
            if shape is not None and shape != list(frame.shape):
                raise ValueError("Video frame dimensions changed")
            shape = list(frame.shape)
            if count == 0:
                previews["pose_first.png"] = frame.copy()
            if count == expected_frames // 2:
                previews["pose_middle.png"] = frame.copy()
            count += 1
    finally:
        capture.release()
    if count != expected_frames or not count:
        raise ValueError("Pose video frame count does not match estimated trajectory")
    for name, frame in previews.items():
        destination = path.with_name(name)
        if destination.exists():
            raise FileExistsError(destination)
        if not cv2.imwrite(str(destination), frame):
            raise OSError(f"Could not save {name}")
    return {"decoded_frames": count, "frame_shape": shape, "video_fps": fps}


def run(upstream: Path, dataset: Path, output: Path, config: dict) -> dict:
    import torch

    source = verify_source(upstream)
    for name, expected in SUBMODULES.items():
        directory = upstream / "third_party" / name
        actual = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        if expected != actual or dirty:
            raise ValueError(
                f"Preprocessing submodule {name} must be clean and pinned at {expected}"
            )
        source[name] = actual
    assets = check_assets(upstream, config["camera"])
    if not assets["ready"]:
        raise RuntimeError(f"Preprocessing assets not ready: {assets}")
    clip = load_clip(open_public(dataset), config)
    previous = Path.cwd()
    started = time.monotonic()
    try:
        os.chdir(upstream)
        from scripts import preprocess_human_play as official

        official.set_seed(config["seed"])
        mesh = official.trimesh.load("assets/mesh/hammer/textured_mesh.obj")
        to_origin, extents = official.trimesh.bounds.oriented_bounds(mesh)
        base = np.load("ti/real_world/robot_extrinsics/right_base_pose_in_world.npy")
        camera = np.load(f"ti/real_world/cam_extrinsics/{CAMERAS[config['camera']]}.npy")
        check_transform(base)
        check_transform(camera)
        estimator = official.FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=official.ScorePredictor(),
            refiner=official.PoseRefinePredictor(),
            glctx=official.dr.RasterizeCudaContext(),
            debug_dir=str(output / "debug"),
            debug=0,
        )
        # NumPy float64 otherwise promotes the released crop offsets to DoubleTensor.
        estimator.diameter = float(estimator.diameter)
        dino, sam = official.initialize_segmentation_models()
        text_revision = getattr(dino.model.bert.config, "_commit_hash", None)
        if text_revision != config["text_encoder_revision"]:
            raise ValueError(f"Unexpected BERT revision: {text_revision}")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            poses = official.track_tool_poses(
                clip["rgb"],
                clip["depth_m"],
                str(output / "segmentation.png"),
                str(output / "detection.png"),
                clip["intrinsics"],
                str(output / "pose.mp4"),
                segmentation_labels=["a nail hammer"],
                pose_model_info={
                    "to_origin": to_origin,
                    "bbox": np.stack([-extents / 2, extents / 2]),
                },
                base_pose_in_world=base,
                cam_extrinsics=camera,
                pose_model=estimator,
                rotation_transformer=official.RotationTransformer(
                    from_rep="matrix", to_rep="axis_angle"
                ),
                grounding_dino_model=dino,
                sam_predictor=sam,
            )
    finally:
        os.chdir(previous)
    if poses.shape != (config["frame_count"], 6) or not np.isfinite(poses).all():
        raise ValueError("Upstream pose output is nonfinite or incorrectly aligned")
    if not (output / "segmentation.png").is_file():
        raise ValueError("No segmentation output; not accepting pose-only fallback")
    np.savez_compressed(
        output / "estimated_poses.npz",
        pose_position_axis_angle=poses,
        timestamp_sec=clip["timestamp_sec"],
        source_frame_index=clip["source_frame_index"],
        numerical_valid=np.isfinite(poses).all(axis=1),
        depth_valid=clip["depth_valid"],
        confidence=np.full(len(poses), np.nan),
    )
    return {
        "stage": "official_grounded_sam_foundationpose_clip",
        "source": source,
        "config": config,
        "asset_checks": assets,
        "text_encoder_revision": text_revision,
        "pose_shape": list(poses.shape),
        "video_inspection": inspect_pose_video(output / "pose.mp4", len(poses)),
        "estimate_file_sha256": sha256_file(output / "estimated_poses.npz"),
        "pose_finite": True,
        "pose_frame": "inv(right_base_in_world) @ camera_in_world @ tool_mesh_pose_in_camera",
        "pose_provenance": (
            "new FoundationPose estimates; not independent GT or functional nozzle poses"
        ),
        "provided_pose_used_for_estimation": False,
        "sensor_depth_used": True,
        "confidence_available": False,
        "numerical_validity_is_not_accuracy": True,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_sec": time.monotonic() - started,
        "native_binary_sha256": {
            p.name: sha256_file(p)
            for p in (upstream / "third_party/FoundationPose/mycpp/build").glob("*.so")
        },
        "deviations": [
            "only a bounded clip, not all 40 episodes",
            "debug exports disabled",
            "explicit millimeter depth scale rather than upstream's max-depth heuristic",
            "no source-Zarr append; estimates are exported to a separate directory",
            "float32 intrinsics and Python-float mesh diameter to avoid mixed crop tensor dtypes",
        ],
        "metric_accuracy_validated": False,
        "construction_validated": False,
        "paper_results_reproduced": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inspect-video-only", type=Path)
    args = parser.parse_args()
    output = prepare_output(args.output, args.dataset)
    config = json.loads(args.config.read_text())
    if args.inspect_video_only is not None:
        if args.inspect_video_only.resolve().is_relative_to(args.dataset.resolve()):
            raise ValueError("Video previews must not be written in raw data")
        video_report = inspect_pose_video(args.inspect_video_only, config["frame_count"])
        (output / "video_inspection.json").write_text(json.dumps(video_report, indent=2) + "\n")
        print(json.dumps(video_report))
        return
    execution = {
        "reproduction": reproduction_revision(Path(__file__).resolve().parents[2]),
        "script_sha256": sha256_file(Path(__file__)),
        "config": config,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runner_sha256": sha256_file(Path(__file__).with_name("runner.py")),
    }
    before = tree_digest(args.dataset)
    try:
        report = run(args.upstream.resolve(), args.dataset, output, config)
    except Exception:
        failure = {
            "error": traceback.format_exc(),
            "execution": execution,
            "raw_unchanged": before == tree_digest(args.dataset),
        }
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    report.update(
        {
            "execution": execution,
            "raw_tree_sha256": before,
            "raw_unchanged": before == tree_digest(args.dataset),
            "software": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        }
    )
    (output / "preprocessing_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["raw_unchanged"]:
        raise RuntimeError("Raw data changed")
    print(json.dumps({"report": str(output / "preprocessing_smoke.json"), "raw_unchanged": True}))


if __name__ == "__main__":
    main()

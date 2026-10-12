"""Read-only public-data checks and an explicitly partial upstream execution test."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import numpy as np

UPSTREAM_COMMIT = "ec1c459ea27ca8e2a2f2787def1d1045ecb2a4ee"
ROBOMIMIC_COMMIT = "bb5a1cf4ff7b6040be46fc32e081e5b8712504e0"
_BYTECODE_CACHE: tempfile.TemporaryDirectory | None = None


def extract_archive(archive: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    with ZipFile(archive) as stream:
        for item in stream.infolist():
            path = PurePosixPath(item.filename.replace("\\", "/"))
            if (
                path.is_absolute()
                or ".." in path.parts
                or ":" in item.filename
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive path: {item.filename}")
        destination.mkdir(parents=True, exist_ok=False)
        stream.extractall(destination)


def align_next_pose(
    poses: np.ndarray, episode_ends: list[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Follow upstream's [:-1] observations / [1:] targets per episode."""
    ends = np.asarray(episode_ends)
    if (
        ends.ndim != 1
        or not ends.size
        or not np.issubdtype(ends.dtype, np.integer)
        or ends[-1] != len(poses)
        or np.any(np.diff(np.r_[0, ends]) <= 0)
    ):
        raise ValueError("Invalid episode ends")
    if poses.ndim != 2 or poses.shape[1] != 9 or not np.isfinite(poses).all():
        raise ValueError("Expected finite N x 9 poses; no filling is permitted")
    lengths = np.diff(np.r_[0, ends])
    if np.any(lengths < 2):
        raise ValueError("Each episode needs at least two frames")
    indices = np.concatenate(
        [np.arange(start, end - 1) for start, end in zip(np.r_[0, ends[:-1]], ends)]
    )
    return poses[indices], poses[indices + 1], np.cumsum(lengths - 1), indices


def prepare_output(output: Path, raw: Path) -> Path:
    output, raw = output.resolve(), raw.resolve()
    if output.is_relative_to(raw):
        raise ValueError("Output must not be inside raw data")
    output.mkdir(parents=True, exist_ok=False)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not accepted in raw data: {path}")
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def reproduction_revision(repository: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def inspect_scene_artifact(artifact: Path) -> dict[str, Any]:
    with np.load(artifact, allow_pickle=False) as data:
        points, views = data["points"], data["novel_views"]
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("Expected nonempty N x 3 scene points")
        if views.ndim != 4 or views.shape[0] != 10 or views.shape[-1] != 3:
            raise ValueError("Expected ten RGB novel views")
        if not all(np.isfinite(data[key]).all() for key in data.files):
            raise ValueError("Nonfinite reconstruction artifact")
        return {
            "point_count": len(points),
            "finite_point_fraction": float(np.isfinite(points).all(axis=1).mean()),
            "novel_view_shape": list(views.shape),
            "nonblank_views": int(
                sum(
                    (np.clip(v, 0, 1) * 255).astype(np.uint8).reshape(-1, 3).std(axis=0).max() > 1
                    for v in views
                )
            ),
            "near_black_pixel_fraction": [float((v.max(axis=-1) < 0.03).mean()) for v in views],
            "geometry_accuracy_validated": False,
            "quality_note": (
                "Nonblank is execution evidence only. Dark pixels can be background or holes."
            ),
        }


def profile_labels(root: Any) -> dict[str, Any]:
    ends = np.asarray(root["meta/episode_ends"][:])
    timestamps = np.asarray(root["data/timestamp"][:])
    poses = np.asarray(root["data/observations/robot_eef_pose"][:])
    if poses.ndim != 2 or poses.shape[1] != 6 or len(poses) != len(timestamps):
        raise ValueError("Expected aligned N x 6 supplied poses and timestamps")
    # Validate episode boundaries separately from pose validity, which is reported.
    align_next_pose(np.zeros((len(poses), 9), dtype=np.float32), ends)
    differences = np.concatenate(
        [np.diff(timestamps[start:end]) for start, end in zip(np.r_[0, ends[:-1]], ends)]
    )
    if not np.isfinite(timestamps).all() or np.any(differences <= 0):
        raise ValueError("Invalid within-episode timestamp order")
    return {
        "episodes": len(ends),
        "frames": len(poses),
        "episode_lengths": np.diff(np.r_[0, ends]).tolist(),
        "within_episode_dt_sec": {
            "min": float(differences.min()),
            "median": float(np.median(differences)),
            "max": float(differences.max()),
        },
        "nonfinite_pose_frames": int((~np.isfinite(poses).all(axis=1)).sum()),
        "pose_provenance": "supplied estimates; not independent ground truth",
        "pose_frame": "upstream calibrated frame; not ROBODATA's surface frame",
    }


def select_episode(root: Any, episode: int) -> tuple[int, int]:
    profile_labels(root)
    ends = np.asarray(root["meta/episode_ends"][:])
    if isinstance(episode, bool) or not isinstance(episode, int) or not 0 <= episode < len(ends):
        raise ValueError("Invalid episode selection")
    return (0 if episode == 0 else int(ends[episode - 1])), int(ends[episode])


def prediction_provenance(
    observation_indices: np.ndarray, timestamps: np.ndarray, n_obs_steps: int, n_action_steps: int
) -> dict[str, np.ndarray]:
    indices = np.asarray(observation_indices)
    first = n_obs_steps - 1
    last = first + n_action_steps
    if (
        indices.ndim != 2
        or not np.issubdtype(indices.dtype, np.integer)
        or n_obs_steps < 1
        or n_action_steps < 1
        or last > indices.shape[1]
        or indices.min() < 0
        or indices.max() + 1 >= len(timestamps)
    ):
        raise ValueError("Invalid sampled prediction indices")
    target = indices + 1
    action = target[:, first:last]
    return {
        "observation_frame_index": indices,
        "target_frame_index": target,
        "action_frame_index": action,
        "observation_timestamp_sec": timestamps[indices],
        "target_timestamp_sec": timestamps[target],
        "action_timestamp_sec": timestamps[action],
    }


def load_estimated_episode(
    path: Path, start: int, stop: int, timestamps: np.ndarray, raw_digest: str
) -> np.ndarray:
    report = json.loads(path.with_name("preprocessing_smoke.json").read_text())
    if (
        report.get("stage") != "official_grounded_sam_foundationpose_clip"
        or report.get("estimate_file_sha256") != sha256_file(path)
        or report.get("raw_tree_sha256") != raw_digest
        or report.get("raw_unchanged") is not True
        or report.get("provided_pose_used_for_estimation") is not False
    ):
        raise ValueError("Estimated pose provenance does not match this dataset")
    with np.load(path, allow_pickle=False) as data:
        poses = data["pose_position_axis_angle"].copy()
        indices = data["source_frame_index"]
        if (
            poses.shape != (stop - start, 6)
            or not np.isfinite(poses).all()
            or not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(indices, np.arange(start, stop))
            or not np.array_equal(data["timestamp_sec"], timestamps[start:stop])
            or data["numerical_valid"].dtype != bool
            or data["numerical_valid"].shape != (stop - start,)
            or not data["numerical_valid"].all()
        ):
            raise ValueError("Estimated episode alignment or numerical validity failed")
    return poses


def verify_source(upstream: Path, *, policy: bool = False) -> dict[str, str]:
    global _BYTECODE_CACHE
    result = {}
    sources = [("upstream", upstream, UPSTREAM_COMMIT)]
    if policy:
        sources.append(("robomimic", upstream / "third_party/robomimic", ROBOMIMIC_COMMIT))
    for name, path, expected in sources:
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
        )
        if actual != expected or dirty:
            raise ValueError(f"{name} must be the clean pinned source {expected}; got {actual}")
        untracked: list[str] = []
        for options in [[], ["--ignored"]]:
            names = (
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(path),
                        "ls-files",
                        "--others",
                        "--exclude-standard",
                        "-z",
                        *options,
                    ]
                )
                .decode()
                .split("\0")
            )
            untracked.extend(
                filename
                for filename in names
                if filename
                and Path(filename).suffix in {".py", ".pyw", ".so", ".pyd", ".pyc"}
                and "__pycache__" not in Path(filename).parts
            )
        if untracked:
            raise ValueError(f"{name} has untracked importable code: {untracked}")
        result[name] = actual
    # Ignore any pre-existing bytecode, including unchecked-hash caches.
    _BYTECODE_CACHE = tempfile.TemporaryDirectory(prefix="tai_verified_imports_")
    sys.pycache_prefix = _BYTECODE_CACHE.name
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "third_party/robomimic"))
    return result


def open_public(dataset: Path) -> Any:
    import zarr
    from ti.codecs.imagecodecs_numcodecs import register_codecs

    register_codecs()
    return zarr.open_group(str(dataset / "replay_buffer.zarr"), mode="r")


def audit_dataset(dataset: Path, output: Path) -> dict[str, Any]:
    import cv2
    import zarr

    root = open_public(dataset)
    report = profile_labels(root)
    arrays = {}

    def describe(name: str, item: Any) -> None:
        if isinstance(item, zarr.Array):
            arrays[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}

    root.visititems(describe)
    report["arrays"] = arrays
    views = sorted(key for key in arrays if key.endswith("_color"))
    report["rgb_views"] = views
    samples = []
    for key in views:
        rgb = root[key][0]
        filename = key.rsplit("/", 1)[-1] + ".png"
        if not cv2.imwrite(str(output / filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write {filename}")
        depth = root[key.replace("_color", "_depth")][0]
        samples.append(
            {
                "view": key,
                "preview": filename,
                "depth_nonzero_fraction": float(np.mean(depth > 0)),
                "depth_raw_median": float(np.median(depth[depth > 0])),
            }
        )
    report["decoded_first_frame"] = samples
    return report


def policy_smoke(
    dataset: Path, upstream: Path, output: Path, config: dict, pose_estimates: Path | None = None
) -> dict:
    import hydra
    import torch
    import torch.nn.functional as functional
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation
    from ti.codecs.imagecodecs_numcodecs import Jpeg2k
    from ti.common.cv2_util import ImgNorm_inv, load_images_mast3r
    from ti.common.replay_buffer import ReplayBuffer
    from ti.common.sampler import SequenceSampler
    from ti.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    from ti.model.common.rotation_transformer import RotationTransformer

    if config["batch_size"] < 1 or config["train_steps"] < 1:
        raise ValueError("Positive batch_size and train_steps required")
    if not torch.cuda.is_available():
        raise RuntimeError("This execution test requires CUDA")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with initialize_config_dir(config_dir=str(upstream / "conf/train_bc"), version_base="1.3"):
        cfg = compose(config_name="train_diffusion_policy")
    # Equivalent to the trusted upstream eval resolver with zero latency.
    cfg.policy.n_action_steps = int(cfg.n_action_steps + cfg.n_latency_steps)
    resolved = OmegaConf.to_container(cfg.policy, resolve=True)
    (output / "official_policy_config.yaml").write_text(OmegaConf.to_yaml(cfg.policy, resolve=True))

    root = open_public(dataset)
    start, stop = select_episode(root, config["episode"])
    pose6 = (
        root["data/observations/robot_eef_pose"][start:stop]
        if pose_estimates is None
        else load_estimated_episode(
            pose_estimates, start, stop, root["data/timestamp"][:], tree_digest(dataset)
        )
    )
    transformer = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    pose9 = np.concatenate([pose6[:, :3], transformer.forward(pose6[:, 3:])], axis=-1)
    roundtrip = RotationTransformer(from_rep="rotation_6d", to_rep="matrix").forward(pose9[:, 3:])
    rotation_error = float(
        np.max(np.abs(roundtrip - Rotation.from_rotvec(pose6[:, 3:]).as_matrix()))
    )
    if rotation_error > 1e-5:
        raise ValueError("Upstream rotation representation failed its independent roundtrip check")
    observations, actions, _, indices = align_next_pose(pose9, [len(pose9)])

    # Match upstream's center-crop -> area-resize -> uint8 feature image path.
    # Segmentation and novel-view synthesis are deliberately NOT represented here.
    rgb = root[f"data/observations/images/{config['view']}_color"][start : stop - 1]
    feature_images = []
    codec = Jpeg2k(level=50)
    for frame in rgb:
        loaded = load_images_mast3r([frame], size=224, verbose=False, device="cpu")[0]["img"]
        resized = functional.interpolate(ImgNorm_inv(loaded), size=(128, 128), mode="area")
        image = (resized[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        feature_images.append(codec.decode(codec.encode(image)))
    replay = ReplayBuffer.create_empty_numpy()
    replay.add_episode(
        {
            "robot_eef_pose": observations.astype(np.float32),
            "action": actions.astype(np.float32),
            "camera_features": np.stack(feature_images),
            "source_index": start + indices,
        }
    )
    sampler = SequenceSampler(
        replay,
        sequence_length=int(cfg.horizon),
        pad_before=int(cfg.n_obs_steps - 1),
        pad_after=int(cfg.n_action_steps - 1),
    )
    samples = [sampler.sample_sequence(i + 2) for i in range(config["batch_size"])]
    provenance = prediction_provenance(
        np.stack([s["source_index"] for s in samples]),
        root["data/timestamp"][:],
        int(cfg.n_obs_steps),
        int(cfg.n_action_steps),
    )
    batch = {
        "obs": {
            "robot_eef_pose": torch.from_numpy(
                np.stack([s["robot_eef_pose"] for s in samples])
            ).cuda(),
            "camera_features": torch.from_numpy(
                np.stack(
                    [
                        np.moveaxis(s["camera_features"], -1, 1).astype(np.float32) / 255
                        for s in samples
                    ]
                )
            ).cuda(),
        },
        "action": torch.from_numpy(np.stack([s["action"] for s in samples])).cuda(),
    }
    normalizer = LinearNormalizer()
    for key, values in [("robot_eef_pose", observations), ("action", actions)]:
        values = values.astype(np.float32)
        stats = {
            "min": values.min(0),
            "max": values.max(0),
            "mean": values.mean(0),
            "std": values.std(0),
        }
        max_abs = float(np.abs(values).max())
        if max_abs == 0:
            raise ValueError("Degenerate poses cannot be normalized")
        normalizer[key] = SingleFieldLinearNormalizer.create_manual(
            np.full(9, 1 / max_abs, dtype=np.float32), np.zeros(9, dtype=np.float32), stats
        )
    from ti.common.normalize_util import get_image_range_normalizer

    normalizer["camera_features"] = get_image_range_normalizer()
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    policy.cuda()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())
    first_parameter = next(policy.model.parameters())
    initial = first_parameter.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    started = time.monotonic()
    policy.train()
    for step in range(config["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy.compute_loss(batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite official policy loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters()):
            raise ValueError("Nonfinite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"Official policy smoke step {step + 1}: loss={losses[-1]:.6f}", flush=True)
    changed = not torch.equal(initial, first_parameter.detach())
    if not changed:
        raise RuntimeError("Optimizer did not change the official model")
    optimizer.zero_grad(set_to_none=True)
    policy.eval()
    with torch.no_grad():
        prediction = policy.predict_action(batch["obs"])
    finite = all(torch.isfinite(value).all().item() for value in prediction.values())
    if not finite:
        raise ValueError("Nonfinite denoised actions")
    np.savez_compressed(
        output / "smoke_predictions.npz",
        action=prediction["action"].cpu().numpy(),
        action_pred=prediction["action_pred"].cpu().numpy(),
        supplied_target=batch["action"].cpu().numpy(),
        conditioning_pose=batch["obs"]["robot_eef_pose"].cpu().numpy(),
        conditioning_rgb=batch["obs"]["camera_features"].cpu().numpy(),
        **provenance,
    )
    torch.save(
        {
            "purpose": "integration_only_not_a_trained_robot_policy",
            "state_dict": policy.state_dict(),
            "config": resolved,
            "smoke_config": config,
        },
        output / "integration_only.pt",
    )
    return {
        "stage": "official_policy_execution_smoke",
        "config": config,
        "official_model_parameters": sum(p.numel() for p in policy.parameters()),
        "losses": losses,
        "parameters_changed": changed,
        "action_shape": list(prediction["action"].shape),
        "prediction_finite": finite,
        "rotation_roundtrip_max_abs_error": rotation_error,
        "gpu": torch.cuda.get_device_name(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_sec": time.monotonic() - started,
        "pose_source": (
            "supplied public estimates; not recomputed or ground truth"
            if pose_estimates is None
            else "new FoundationPose episode estimates; not ground truth"
        ),
        "pose_estimate_file_sha256": None
        if pose_estimates is None
        else sha256_file(pose_estimates),
        "observation_source": "unsegmented public RGB; no novel views",
        "feature_storage_codec": "Jpeg2k(level=50) encode/decode in memory",
        "deviations": [
            "single episode and small fixed batch",
            "short optimization; no EMA or LR schedule",
            "no held-out evaluation",
            "no Grounded-SAM preprocessing",
            "no Gaussian augmentation",
        ],
        "paper_results_reproduced": False,
    }


def scene_smoke(dataset: Path, output: Path, config: dict | None = None) -> dict:
    import cv2
    import starster
    import torch
    from ti.common.data_utils import generate_novel_view, setup_scene_and_gs

    if config is None:
        config = json.loads(Path(__file__).with_name("scene_smoke.json").read_text())
    if config["gs_iterations"] != {"pruning": 500, "refinement": 100}:
        raise ValueError("GS iteration counts are fixed by the unmodified upstream helper")
    if len(config["views"]) != 2 or config["generated_views"] != 10:
        raise ValueError(
            "This reproduction uses the released two-view / ten-intermediate-view setting"
        )
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    root = open_public(dataset)
    colors = np.stack(
        [
            root[f"data/observations/images/{view}_color"][config["source_frame"]]
            for view in config["views"]
        ]
    )[None]
    for index, rgb in enumerate(colors[0]):
        if not cv2.imwrite(
            str(output / f"source_{index:02d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ):
            raise OSError("Could not save source view")
    model = starster.Mast3rModel.from_pretrained(
        config["model"], revision=config["model_revision"]
    ).cuda()
    model.eval()
    started = time.monotonic()
    # Unmodified official helper: 500 pruning + 100 refinement steps, 10 views.
    gs, _, matrices = setup_scene_and_gs(
        colors,
        config["image_size"],
        model,
        [f"{v}.png" for v in config["views"]],
        config["generated_views"],
        "cuda",
    )
    with torch.no_grad():
        novel = (
            generate_novel_view(
                gs, config["generated_views"], matrices, "cuda", config["image_size"]
            )
            .detach()
            .cpu()
            .numpy()
        )
        points, point_colors = gs.scene.pts_dense_flat()
        points, point_colors = points.detach().cpu().numpy(), point_colors.detach().cpu().numpy()
    if not np.isfinite(novel).all() or not np.isfinite(points).all():
        raise ValueError("Nonfinite scene or rendered views")
    nonblank = 0
    for index, rgb in enumerate(novel):
        image = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        if image.reshape(-1, 3).std(axis=0).max() > 1:
            nonblank += 1
        if not cv2.imwrite(
            str(output / f"novel_{index:02d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ):
            raise OSError("Could not save novel view")
    np.savez_compressed(
        output / "reconstruction.npz",
        points=points,
        colors=point_colors,
        novel_views=novel,
        novel_w2c=matrices.detach().cpu().numpy(),
        source_w2c=gs.scene.w2c().detach().cpu().numpy(),
        intrinsics=gs.scene.intrinsics().detach().cpu().numpy(),
    )
    if nonblank != 10:
        raise ValueError("Blank novel views; not accepting reconstruction as successful")
    return {
        "stage": "official_scene_and_gaussian_augmentation_single_pair",
        "config": config,
        "point_count": len(points),
        "finite_point_fraction": float(np.isfinite(points).all(axis=1).mean()),
        "novel_view_shape": list(novel.shape),
        "nonblank_views": nonblank,
        "elapsed_sec": time.monotonic() - started,
        "gs_iterations": config["gs_iterations"],
        "deviations": [
            "one image pair only",
            "no Grounded-SAM hand removal",
            "not connected to policy training in this test",
        ],
        "metric_accuracy_validated": False,
        "construction_validated": False,
        "paper_results_reproduced": False,
        "artifact_inspection": inspect_scene_artifact(output / "reconstruction.npz"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    extract = subcommands.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)
    audit = subcommands.add_parser("audit")
    audit.add_argument("--upstream", type=Path, required=True)
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    smoke = subcommands.add_parser("policy-smoke")
    smoke.add_argument("--upstream", type=Path, required=True)
    smoke.add_argument("--dataset", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--pose-estimates", type=Path)
    scene = subcommands.add_parser("scene-smoke")
    scene.add_argument("--upstream", type=Path, required=True)
    scene.add_argument("--dataset", type=Path, required=True)
    scene.add_argument("--output", type=Path, required=True)
    scene.add_argument("--config", type=Path, default=Path(__file__).with_name("scene_smoke.json"))
    args = parser.parse_args()
    if args.command == "extract":
        checksum = sha256_file(args.archive)
        extract_archive(args.archive, args.destination)
        print(json.dumps({"archive_sha256": checksum, "destination": str(args.destination)}))
    else:
        source = verify_source(args.upstream.resolve(), policy=args.command == "policy-smoke")
        output = prepare_output(args.output, args.dataset)
        execution = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "reproduction": reproduction_revision(Path(__file__).resolve().parents[2]),
            "runner_sha256": sha256_file(Path(__file__)),
        }
        before = tree_digest(args.dataset)
        try:
            if args.command == "audit":
                report = audit_dataset(args.dataset, output)
            elif args.command == "scene-smoke":
                report = scene_smoke(args.dataset, output, json.loads(args.config.read_text()))
            else:
                report = policy_smoke(
                    args.dataset,
                    args.upstream.resolve(),
                    output,
                    json.loads(args.config.read_text()),
                    args.pose_estimates,
                )
        except Exception:
            failure = {
                "command": args.command,
                "source": source,
                "error": traceback.format_exc(),
                "execution": execution,
                "raw_unchanged": before == tree_digest(args.dataset),
            }
            (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise
        report["source"] = source
        report["execution"] = execution
        report["raw_tree_sha256"] = before
        report["raw_unchanged"] = before == tree_digest(args.dataset)
        report["software"] = {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        }
        report["runner_sha256"] = execution["runner_sha256"]
        name = {
            "audit": "audit.json",
            "policy-smoke": "policy_smoke.json",
            "scene-smoke": "scene_smoke.json",
        }[args.command]
        (output / name).write_text(json.dumps(report, indent=2) + "\n")
        if not report["raw_unchanged"]:
            raise RuntimeError("Raw data changed during execution")
        print(json.dumps({"report": str(output / name), "raw_unchanged": report["raw_unchanged"]}))


if __name__ == "__main__":
    main()

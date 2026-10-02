"""Convert explicitly synthetic Blender renders to the existing input contract."""

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import RunConfig
from .export import save_episode, write_json
from .geometry import inverse, validate_transform
from .observations import Camera, Manifest
from .schema import Canonical, Episode, FrameRecord, Quality, Source
from .temporal import add_labels


def camera_from_blender(matrix_world) -> np.ndarray:
    """Map world points to OpenCV camera; Blender looks along local -Z."""
    return np.diag([1.0, -1, -1, 1]) @ inverse(matrix_world)


def radial_to_axial(depth, matrix) -> np.ndarray:
    raw = np.asarray(depth, dtype=float)
    if raw.ndim != 2:
        raise ValueError("Rendered depth must be a 2D metric array")
    K = Camera(matrix=np.asarray(matrix).tolist(), width=raw.shape[1], height=raw.shape[0])
    intrinsics = np.asarray(K.matrix)
    y, x = np.indices(raw.shape)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsics).T
    result = raw / np.linalg.norm(rays, axis=-1)
    result[~np.isfinite(raw) | (raw <= 0) | (raw >= 1e6)] = np.nan
    return result.astype(np.float32)


def _input_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Rendered input path leaves its source directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_depth(root: Path, name: str, shape) -> np.ndarray:
    path = _input_path(root, name)
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    raw = (
        np.load(path, allow_pickle=False)
        if path.suffix == ".npy"
        else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    )
    if raw is None:
        raise ValueError(f"Cannot decode rendered depth: {path}")
    if raw.ndim == 3 and np.allclose(raw, raw[:, :, :1], equal_nan=True):
        raw = raw[:, :, 0]
    if raw.shape != shape:
        raise ValueError("Rendered depth resolution differs from RGB")
    return raw.astype(np.float32)


def assemble_rendered(render_root: Path | str, output_root: Path | str) -> Path:
    root, out = Path(render_root).resolve(), Path(output_root).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new dataset directory: {out}")
    if out.is_relative_to(root):
        raise ValueError("Dataset output must be outside the raw render tree")
    metadata_path = root / "render_metadata.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    camera = Camera(matrix=meta["camera_matrix"], width=meta["width"], height=meta["height"])
    frames = meta["frames"]
    expected = meta.get("expected_frames", meta.get("settings", {}).get("config", {}).get("frames"))
    configured = meta.get("settings", {}).get("config", {}).get("frames")
    if configured is not None and (type(configured) is not int or configured != expected):
        raise ValueError("Conflicting rendered frame count declarations")
    if (
        type(expected) is not int
        or expected < 4
        or len(frames) != expected
        or [f["frame_index"] for f in frames] != list(range(len(frames)))
    ):
        raise ValueError("Rendered video needs at least four contiguous ordered frames")
    timestamps = np.asarray([f["timestamp_sec"] for f in frames], dtype=float)
    if not np.isfinite(timestamps).all() or timestamps[0] < 0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Rendered timestamps must be finite and strictly increasing")
    fps = float(meta["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid rendered FPS")
    convention = meta["depth_convention"]
    if convention not in {"ray_distance", "axial_z"}:
        raise ValueError("Unknown rendered depth convention")
    shape = (camera.height, camera.width)
    for f in frames:
        for field in ["rgb_path", "depth_path", "labels_path"]:
            _input_path(root, f[field])
        for field in ["T_world_tool", "T_world_surface", "T_world_camera"]:
            validate_transform(f[field])
        if not np.allclose(f["T_world_surface"], frames[0]["T_world_surface"], atol=1e-6, rtol=0):
            raise ValueError("Static workpiece surface frame must not be reanchored")
        for check in f.get("depth_checks", []):
            u, v, expected_m = check["u"], check["v"], check["expected_m"]
            if (
                type(u) is not int
                or type(v) is not int
                or not 0 <= u < camera.width
                or not 0 <= v < camera.height
                or not np.isfinite(expected_m)
                or not 0 < expected_m < 1e6
            ):
                raise ValueError("Invalid independent scene ray check")
    observations, truth = [], []
    out.mkdir(parents=True)
    for name in ["depth", "masks"]:
        (out / name).mkdir()
    video = cv2.VideoWriter(
        str(out / "rgb.avi"),
        cv2.VideoWriter.fourcc(*"MJPG"),
        fps,
        (camera.width, camera.height),
    )
    if not video.isOpened():
        raise RuntimeError("MJPG encoder unavailable")
    try:
        for f in frames:
            i = f["frame_index"]
            rgb = cv2.imread(str(_input_path(root, f["rgb_path"])), cv2.IMREAD_COLOR)
            if rgb is None or rgb.shape[:2] != shape:
                raise ValueError("Rendered RGB resolution differs from calibration")
            raw = _read_depth(root, f["depth_path"], shape)
            for check in f.get("depth_checks", []):
                value = raw[check["v"], check["u"]]
                if not np.isfinite(value) or abs(value - check["expected_m"]) > 0.005:
                    raise ValueError("Rendered depth disagrees with independent scene ray")
            depth = radial_to_axial(raw, camera.matrix) if convention == "ray_distance" else raw
            depth[~np.isfinite(depth) | (depth <= 0) | (depth >= 1e6)] = np.nan
            np.save(out / f"depth/{i:05d}.npy", depth)
            labels = cv2.imread(str(_input_path(root, f["labels_path"])), cv2.IMREAD_UNCHANGED)
            if labels is None or labels.shape != shape:
                raise ValueError("Rendered object labels must be single-channel aligned images")
            for name, index in [("tool", 2), ("surface", 3)]:
                cv2.imwrite(
                    str(out / f"masks/{name}_{i:05d}.png"), (labels == index).astype(np.uint8) * 255
                )
            video.write(rgb)
            observations.append(
                {
                    "frame_index": i,
                    "tool": {
                        "keypoints": f["tool_keypoints"],
                        "mask_path": f"masks/tool_{i:05d}.png",
                    },
                    "surface": {
                        "keypoints": f["surface_keypoints"],
                        "mask_path": f"masks/surface_{i:05d}.png",
                    },
                    "depth_path": f"depth/{i:05d}.npy",
                }
            )
            T = inverse(f["T_world_surface"]) @ np.asarray(f["T_world_tool"])
            truth.append(
                FrameRecord(
                    frame_index=i,
                    timestamp_sec=f["timestamp_sec"],
                    canonical=Canonical(
                        T_surface_tool=T.tolist(),
                        tip_position_surface_m=T[:3, 3].tolist(),
                        tool_axis_surface=T[:3, 2].tolist(),
                        standoff_distance_m=float(abs(T[2, 3])),
                        signed_surface_distance_m=float(T[2, 3]),
                        incidence_angle_rad=float(np.arccos(np.clip(-T[2, 2], -1, 1))),
                        surface_projected_path_m=T[:2, 3].tolist(),
                    ),
                    quality=Quality(
                        valid=True,
                        full_pose_valid=True,
                        confidence=1,
                        confidence_kind="synthetic_exact",
                    ),
                )
            )
    finally:
        video.release()
    provenance = {
        "renderer": "Blender",
        "blender_version": meta.get("blender_version"),
        "render_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "depth_convention_input": convention,
        "depth_convention_output": "axial_z_meters",
        "geometry_source": "synthetic_proxy_not_measured_real_tool",
        "landmark_note": "oracle projected marker centers with ray-cast visibility, not a detector",
        "render_settings": {
            key: value
            for key, value in meta.get("settings", {}).items()
            if key in {"config_sha256", "render_device", "proxy_note", "render_provenance"}
        },
        "render_time_source_recorded": bool(meta.get("settings", {}).get("render_provenance")),
    }
    manifest = Manifest.model_validate(
        {
            "episode_id": meta["episode_id"],
            "task_name": "synthetic_hbeam_spray_proxy",
            "video_path": "rgb.avi",
            "camera": camera.model_dump(),
            "data_kind": "synthetic",
            "timestamps_sec": timestamps.tolist(),
            "observations": observations,
            "landmark_source": "blender_oracle_projected_markers",
            "depth_aligned_to_rgb": True,
            "provenance": provenance,
        }
    )
    write_json(out / "manifest.json", manifest.model_dump(mode="json"))
    for mode in ["rgb", "rgbd"]:
        cfg = RunConfig(
            manifest_path="manifest.json",
            mode=mode,
            metric_scale_source="measured_geometry" if mode == "rgb" else "depth_sensor",
            camera_motion=meta["camera_motion"],
            surface_frame_id="synthetic_hbeam_top_patch",
            surface_method="pnp",
            tool={"points_m": meta["tool_points_m"]},
            surface={"points_m": meta["surface_points_m"]},
            depth_unit_m=1.0,
            horizon_sec=0.2,
            timestamp_tolerance_sec=0.01,
            seed=meta.get("seed", 42),
        )
        (out / f"{mode}.yaml").write_text(
            yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8"
        )
    gt = Episode(
        episode_id=manifest.episode_id,
        task_name=manifest.task_name,
        surface_frame_id="synthetic_hbeam_top_patch",
        frames=truth,
        source=Source(
            mode="external_pose", metric_scale_source="calibrated_external", data_kind="synthetic"
        ),
        provenance={
            "reference": "actual_simulator_object_transforms",
            **provenance,
            "render_settings": meta.get("settings", {}),
        },
    )
    add_labels(gt, 0.2, 0.01)
    save_episode(gt, out / "ground_truth.json")
    return out

"""RGB/RGB-D observations to validated robot-agnostic episodes."""

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import load_config, resolve
from .export import export_tensors, save_episode, write_json
from .geometry import backproject, fit_plane, inverse, unit
from .observations import (
    FlowTracker,
    FrameInput,
    Manifest,
    depth_tool,
    pnp_observation,
    read_depth,
    read_mask,
    video_frames,
)
from .schema import Canonical, Episode, Estimate, FrameRecord, Observation, Quality, Source
from .temporal import add_labels


def software_provenance(config_path: Path) -> dict:
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["robodata", "numpy", "scipy", "opencv-python-headless", "pydantic"]
        },
        "confidence_note": "quality heuristics; not calibrated probabilities",
    }


def run_episode(
    config_path: Path | str, output_dir: Path | str, manifest_override: Path | str | None = None
) -> Episode:
    path = Path(config_path).resolve()
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new run directory: {out}")
    cfg = load_config(path)
    if manifest_override is not None:
        cfg.manifest_path = str(Path(manifest_override).resolve())
    cv2.setRNGSeed(cfg.seed)
    manifest_path = resolve(path.parent, cfg.manifest_path)
    root = manifest_path.parent
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if cfg.mode == "rgbd" and (
        not manifest.depth_aligned_to_rgb or np.any(manifest.camera.distortion)
    ):
        raise ValueError("RGB-D requires aligned depth and a rectified RGB camera")
    inputs = {o.frame_index: o for o in manifest.observations}
    tool_tracker = FlowTracker(cfg.max_flow_fb_error_px)
    surface_tracker = FlowTracker(cfg.max_flow_fb_error_px)
    surface_reference = None
    frames = []
    video = resolve(root, manifest.video_path)
    for i, timestamp, rgb in video_frames(video, manifest.camera, manifest.timestamps_sec):
        obs = inputs.get(i, FrameInput(frame_index=i)).model_copy(deep=True)
        if cfg.tracking:
            obs.tool = tool_tracker.update(rgb, obs.tool)
            obs.surface = surface_tracker.update(rgb, obs.surface)
        mask = read_mask(root, obs.tool.mask_path, rgb.shape[:2])
        centroid = None
        if mask is not None and mask.any():
            ys, xs = np.nonzero(mask)
            centroid = [float(xs.mean()), float(ys.mean())]
        observation = Observation(
            rgb_path=f"{manifest.video_path}#frame={i}",
            depth_path=obs.depth_path,
            tool_mask_path=obs.tool.mask_path,
            surface_mask_path=obs.surface.mask_path,
            tool_centroid_pixel=centroid,
            landmark_source=manifest.landmark_source + ("+checked_klt" if cfg.tracking else ""),
        )
        surf, tool, canonical = None, None, Canonical()
        quality = Quality(occluded=obs.tool.occluded)
        try:
            depth = (
                read_depth(root, obs.depth_path, cfg, rgb.shape[:2]) if cfg.mode == "rgbd" else None
            )
            if cfg.surface_method == "pnp":
                surf = pnp_observation(obs.surface, cfg.surface, manifest.camera, cfg)
            else:
                assert depth is not None
                if surface_reference is None:
                    smask = read_mask(root, obs.surface.mask_path, rgb.shape[:2])
                    if smask is None:
                        raise ValueError("Plane reference needs a surface mask")
                    ys, xs = np.nonzero(smask & np.isfinite(depth))
                    cloud = backproject(np.c_[xs, ys], depth[ys, xs], manifest.camera.matrix)
                    plane = fit_plane(
                        cloud, cfg.plane_threshold_m, cfg.min_plane_inlier_ratio, cfg.seed
                    )
                    surface_reference = Estimate(
                        T_camera_frame=plane.transform.tolist(),
                        method="ransac_depth_plane",
                        confidence=plane.inlier_ratio,
                        residual=plane.residual,
                        residual_unit="meter",
                        origin_camera_m=plane.origin.tolist(),
                        axis_camera=plane.normal.tolist(),
                        frame_method="initial_visible_patch_centroid_camera_x_static_reference",
                    )
                surf = surface_reference.model_copy(deep=True)
            tool = (
                pnp_observation(obs.tool, cfg.tool, manifest.camera, cfg)
                if cfg.mode == "rgb"
                else (depth_tool(obs.tool, depth, manifest.camera, cfg))
            )
            S_C = inverse(surf.T_camera_frame)
            point = S_C[:3, :3] @ np.array(tool.origin_camera_m) + S_C[:3, 3]
            axis = unit(S_C[:3, :3] @ np.array(tool.axis_camera))
            T = S_C @ np.asarray(tool.T_camera_frame) if tool.T_camera_frame is not None else None
            canonical = Canonical(
                T_surface_tool=None if T is None else T.tolist(),
                tip_position_surface_m=point.tolist(),
                tool_axis_surface=axis.tolist(),
                standoff_distance_m=float(abs(point[2])),
                signed_surface_distance_m=float(point[2]),
                incidence_angle_rad=float(np.arccos(np.clip(-axis[2], -1, 1))),
                surface_projected_path_m=point[:2].tolist(),
            )
            assert surf.confidence is not None and tool.confidence is not None
            quality = Quality(
                valid=True,
                full_pose_valid=T is not None,
                occluded=obs.tool.occluded,
                confidence=min(surf.confidence, tool.confidence),
                estimated_fields=["surface", "tool", "canonical"],
            )
        except ValueError as exc:
            quality.failure_reason = str(exc)
        frames.append(
            FrameRecord(
                frame_index=i,
                timestamp_sec=timestamp,
                observation=observation,
                surface=surf,
                tool=tool,
                canonical=canonical,
                quality=quality,
                work_on=obs.work_on,
                primitive_label=obs.primitive_label,
                process_state_source=obs.process_state_source,
            )
        )
    if any(i >= len(frames) for i in inputs):
        raise ValueError("Observation index outside video")
    episode = Episode(
        episode_id=manifest.episode_id,
        task_name=manifest.task_name,
        surface_frame_id=cfg.surface_frame_id,
        source=Source(
            mode=cfg.mode,
            metric_scale_source=cfg.metric_scale_source,
            data_kind=manifest.data_kind,
            manifest_path=str(manifest_path),
            rgb_video_path=manifest.video_path,
            timestamp_source="explicit"
            if manifest.timestamps_sec is not None
            else "fps_assumed_constant",
        ),
        frames=frames,
        provenance=software_provenance(path),
    )
    episode.provenance["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    episode.provenance["observations"] = manifest.provenance
    add_labels(episode, cfg.horizon_sec, cfg.timestamp_tolerance_sec)
    out.mkdir(parents=True)
    save_episode(episode, out / "episode.json")
    export_tensors(episode, out / "tensors.npz")
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(
        out / "summary.json",
        {
            "frames": len(frames),
            "valid_frames": sum(f.quality.valid for f in frames),
            "full_pose_frames": sum(f.quality.full_pose_valid for f in frames),
            "valid_actions": sum(f.action.valid for f in frames),
            "data_kind": manifest.data_kind,
        },
    )
    return episode

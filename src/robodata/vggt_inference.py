"""Offline original-VGGT inference and an explicitly measured surface adapter."""

import inspect
from pathlib import Path

import cv2
import numpy as np
import yaml
from pydantic import Field, model_validator

from .config import resolve
from .export import write_json
from .observations import Manifest, read_mask, video_frames
from .pipeline import software_provenance
from .schema import Contract
from .vggt_bridge import (
    SurfaceCache,
    SurfaceSample,
    crop_affine,
    file_sha256,
    fit_metric_alignment,
    metric_camera_surface,
    sample_world_point,
    surface_estimate,
)


class VGGTRunConfig(Contract):
    manifest_path: str
    checkpoint_path: str
    device: str = "cuda"
    frame_indices: list[int] | None = None
    max_frames: int = Field(default=16, ge=1)
    surface_points_m: dict[str, list[float]] = Field(default_factory=dict)
    calibration_source: str = "unavailable"
    surface_frame_id: str = "reference_patch"
    alignment_frame_indices: list[int] | None = None
    min_raw_depth_confidence: float = Field(default=1.0, ge=0)
    min_keypoint_confidence: float = Field(default=0.5, ge=0, le=1)
    max_alignment_error_m: float = Field(default=0.02, gt=0)
    max_reprojection_error_px: float = Field(default=5.0, gt=0)
    exclude_tool_pixels: bool = True
    seed: int = 42

    @model_validator(mode="after")
    def specified(self):
        for ids in [self.frame_indices, self.alignment_frame_indices]:
            if ids is not None and (
                not ids or any(i < 0 for i in ids) or any(b <= a for a, b in zip(ids, ids[1:]))
            ):
                raise ValueError("Frame selections must be nonempty and strictly increasing")
        if self.frame_indices is not None and len(self.frame_indices) > self.max_frames:
            raise ValueError("Frame selection exceeds the configured memory budget")
        if any(len(p) != 3 or not np.isfinite(p).all() for p in self.surface_points_m.values()):
            raise ValueError("Measured surface points must be finite meter coordinates")
        if self.surface_points_m and (
            len(self.surface_points_m) < 4
            or not self.calibration_source.strip()
            or self.calibration_source == "unavailable"
        ):
            raise ValueError("Metric reconstruction needs four named points and calibration source")
        return self


def infer_rgb_images(
    image_paths: list[Path], checkpoint: Path, device: str = "cuda"
) -> tuple[dict, dict]:
    """Actual pretrained inference; no depth sensor or evaluation pose is consumed."""
    import torch
    import vggt.models.vggt as upstream
    import vggt.utils.load_fn as loader
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if not image_paths:
        raise ValueError("At least one RGB frame is required")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Configured CUDA device is unavailable")
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint), device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = upstream.VGGT()
    model.load_state_dict(state, strict=True)
    del state
    # Camera + depth is sufficient; do not spend memory on unused point/track heads.
    model.point_head = None
    model.track_head = None
    model = model.to(device).eval()
    images = loader.load_and_preprocess_images([str(p) for p in image_paths], mode="crop").to(
        device
    )
    dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=dtype,
            enabled=device.startswith("cuda"),
        ),
    ):
        predicted = model(images)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predicted["pose_enc"], images.shape[-2:]
        )
    result = {
        "depth": predicted["depth"][0, ..., 0].float().cpu().numpy(),
        "depth_conf": predicted["depth_conf"][0].float().cpu().numpy(),
        "extrinsics": extrinsics[0].float().cpu().numpy(),
        "intrinsics": intrinsics[0].float().cpu().numpy(),
    }
    metadata = {
        "model": "original_VGGT_1B",
        "checkpoint_sha256": file_sha256(checkpoint),
        "upstream_model_source_sha256": file_sha256(Path(inspect.getfile(upstream.VGGT))),
        "upstream_preprocess_sha256": file_sha256(Path(inspect.getfile(loader))),
        "torch_version": torch.__version__,
        "device": device,
        "dtype": str(dtype),
        "depth_units": "normalized_model_units_not_meters",
        "depth_confidence_note": "raw positive score, not probability",
        "extrinsics_convention": "OpenCV camera_from_normalized_model",
    }
    return result, metadata


def build_surface_cache(
    predictions: dict,
    indices: list[int],
    times: list[float],
    affine,
    manifest: Manifest,
    video: Path,
    cfg: VGGTRunConfig,
    provenance: dict,
) -> SurfaceCache:
    n = len(indices)
    depth, confidence = predictions["depth"], predictions["depth_conf"]
    extrinsics, intrinsics = predictions["extrinsics"], predictions["intrinsics"]
    if (
        len(times) != n
        or depth.ndim != 3
        or depth.shape[0] != n
        or confidence.shape != depth.shape
        or extrinsics.shape != (n, 3, 4)
        or intrinsics.shape != (n, 3, 3)
    ):
        raise ValueError("Prediction shapes do not match selected RGB frames")
    A = np.asarray(affine, float)
    if A.shape != (3, 3) or not np.isfinite(A).all():
        raise ValueError("Invalid original-to-model pixel transform")
    observations = {o.frame_index: o for o in manifest.observations}
    model_points, measured_points, used, rejected = [], [], [], []
    for j, index in enumerate(indices):
        obs = observations.get(index)
        if (
            obs is None
            or obs.surface.occluded
            or (
                cfg.alignment_frame_indices is not None and index not in cfg.alignment_frame_indices
            )
        ):
            continue
        for name, kp in obs.surface.keypoints.items():
            if (
                name not in cfg.surface_points_m
                or not kp.visible
                or kp.confidence < cfg.min_keypoint_confidence
            ):
                continue
            try:
                uv = (A @ [*kp.uv, 1])[:2]
                point = sample_world_point(
                    uv,
                    depth[j],
                    confidence[j],
                    intrinsics[j],
                    extrinsics[j],
                    cfg.min_raw_depth_confidence,
                )
            except ValueError as exc:
                rejected.append({"frame_index": index, "name": name, "reason": str(exc)})
                continue
            model_points.append(point)
            measured_points.append(cfg.surface_points_m[name])
            used.append({"frame_index": index, "name": name})
    fit = None
    failure = None
    try:
        if len({p["name"] for p in used}) < 4:
            raise ValueError("At least four distinct measured RGB anchors are required")
        fit = fit_metric_alignment(model_points, measured_points, cfg.max_alignment_error_m)
    except ValueError as exc:
        failure = str(exc)
    alignment: dict = {
        "accepted": fit is not None,
        "failure_reason": failure,
        "used_anchors": used,
        "rejected_anchors": rejected,
    }
    if fit is not None:
        alignment.update(
            {
                "scale_m_per_model_unit": fit.scale,
                "R_surface_model": fit.rotation.tolist(),
                "translation_surface_m": fit.translation.tolist(),
                "residual_m": fit.residual_m,
                "max_anchor_error_m": fit.max_error_m,
            }
        )
    samples, checks = [], []
    for j, (index, timestamp) in enumerate(zip(indices, times, strict=True)):
        estimate = None
        reason = failure
        try:
            if fit is None:
                raise ValueError(failure)
            T = metric_camera_surface(extrinsics[j], fit)
            obs = observations.get(index)
            check: dict = {
                "frame_index": index,
                "rgb_check_points": 0,
                "reprojection_rmse_px": None,
            }
            if obs is not None and not obs.surface.occluded:
                names = [
                    name
                    for name, kp in obs.surface.keypoints.items()
                    if name in cfg.surface_points_m
                    and kp.visible
                    and kp.confidence >= cfg.min_keypoint_confidence
                ]
                check["rgb_check_points"] = len(names)
                if names:
                    xyz = np.asarray([cfg.surface_points_m[name] for name in names])
                    if np.any((xyz @ T[:3, :3].T + T[:3, 3])[:, 2] <= 0):
                        raise ValueError("Measured surface behind predicted camera")
                    rv = cv2.Rodrigues(T[:3, :3])[0]
                    projected = cv2.projectPoints(
                        xyz,
                        rv,
                        T[:3, 3],
                        np.asarray(manifest.camera.matrix),
                        np.asarray(manifest.camera.distortion),
                    )[0]
                    pixels = np.asarray([obs.surface.keypoints[name].uv for name in names])
                    residual = float(
                        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1)))
                    )
                    check["reprojection_rmse_px"] = residual
                    if residual > cfg.max_reprojection_error_px:
                        raise ValueError("Surface reprojection exceeds calibrated RGB gate")
            checks.append(check)
            estimate = surface_estimate(
                T, fit.residual_m, cfg.max_alignment_error_m, cfg.calibration_source
            )
            reason = None
        except ValueError as exc:
            reason = str(exc)
        samples.append(
            SurfaceSample(
                frame_index=index, timestamp_sec=timestamp, estimate=estimate, failure_reason=reason
            )
        )
    return SurfaceCache(
        video_sha256=file_sha256(video),
        camera=manifest.camera,
        data_kind=manifest.data_kind,
        calibration_source=cfg.calibration_source,
        surface_frame_id=cfg.surface_frame_id,
        frames=samples,
        provenance={
            **provenance,
            "alignment": alignment,
            "per_frame_rgb_checks": checks,
            "confidence_note": "global anchor-fit heuristic; not calibrated pose accuracy",
            "whole_video_optimization": False,
            "gap_filling": False,
        },
    )


def probe_rgb_images(
    image_paths: list[Path], checkpoint: Path, output_dir: Path, device: str = "cuda"
) -> dict:
    out = output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new run directory: {out}")
    out.mkdir(parents=True)
    try:
        predictions, provenance = infer_rgb_images(image_paths, checkpoint, device)
    except Exception as exc:
        write_json(out / "failure.json", {"stage": "vggt_inference", "reason": str(exc)})
        raise
    np.savez_compressed(out / "reconstruction.npz", **predictions)
    provenance["input_rgb_sha256"] = {str(p.resolve()): file_sha256(p) for p in image_paths}
    report = {
        "rgb_frames": len(image_paths),
        "metric_alignment_performed": False,
        "tool_pose_estimated": False,
        "accuracy_evaluated": False,
        "finite_positive_depth_fraction": float(
            np.mean(np.isfinite(predictions["depth"]) & (predictions["depth"] > 0))
        ),
        "depth_shape": list(predictions["depth"].shape),
    }
    write_json(out / "provenance.json", provenance)
    write_json(out / "summary.json", report)
    return report


def run_vggt(config_path: Path | str, output_dir: Path | str) -> SurfaceCache:
    path, out = Path(config_path).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new run directory: {out}")
    cfg = VGGTRunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    manifest_path = resolve(path.parent, cfg.manifest_path)
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if np.any(manifest.camera.distortion):
        raise ValueError(
            "First VGGT bridge requires rectified RGB and matching rectified landmarks/K"
        )
    video = resolve(manifest_path.parent, manifest.video_path)
    if not resolve(path.parent, cfg.checkpoint_path).is_file():
        raise FileNotFoundError(cfg.checkpoint_path)
    count = sum(1 for _ in video_frames(video, manifest.camera, manifest.timestamps_sec))
    selected = (
        cfg.frame_indices
        if cfg.frame_indices is not None
        else np.linspace(0, count - 1, min(count, cfg.max_frames), dtype=int).tolist()
    )
    if selected[-1] >= count:
        raise ValueError("Selected frame outside original video")
    out.mkdir(parents=True)
    images_dir = out / "model_rgb"
    images_dir.mkdir()
    observations = {o.frame_index: o for o in manifest.observations}
    indices, times, images, masked = [], [], [], []
    for index, timestamp, rgb in video_frames(video, manifest.camera, manifest.timestamps_sec):
        if index not in selected:
            continue
        image = rgb.copy()
        obs = observations.get(index)
        if cfg.exclude_tool_pixels and obs is not None and obs.tool.mask_path is not None:
            mask = read_mask(manifest_path.parent, obs.tool.mask_path, image.shape[:2])
            image[mask] = 0
            masked.append(index)
        image_path = images_dir / f"{index:06d}.png"
        if not cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise ValueError("Could not write VGGT model input")
        indices.append(index)
        times.append(timestamp)
        images.append(image_path)
    import torch

    torch.manual_seed(cfg.seed)
    try:
        predictions, provenance = infer_rgb_images(
            images, resolve(path.parent, cfg.checkpoint_path), cfg.device
        )
    except Exception as exc:
        write_json(out / "failure.json", {"stage": "vggt_inference", "reason": str(exc)})
        raise
    affine, shape = crop_affine(manifest.camera.width, manifest.camera.height)
    if predictions["depth"].shape[1:] != shape:
        raise ValueError("Upstream preprocessing changed; pixel mapping must be revalidated")
    provenance.update(software_provenance(path))
    provenance.update(
        {
            "manifest_sha256": file_sha256(manifest_path),
            "masked_tool_frames": masked,
            "model_rgb_sha256": {p.name: file_sha256(p) for p in images},
            "pixel_affine_original_to_model": affine.tolist(),
            "input_sensor": "RGB_only",
            "timestamps_source": "explicit"
            if manifest.timestamps_sec is not None
            else "fps_assumed_constant",
        }
    )
    np.savez_compressed(
        out / "reconstruction.npz",
        **predictions,
        frame_indices=np.asarray(indices),
        timestamps_sec=np.asarray(times),
        pixel_affine=affine,
    )
    cache = build_surface_cache(
        predictions, indices, times, affine, manifest, video, cfg, provenance
    )
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(out / "surface_cache.json", cache.model_dump(mode="json"))
    write_json(
        out / "summary.json",
        {
            "selected_frames": len(indices),
            "metric_surface_frames": sum(s.estimate is not None for s in cache.frames),
            "alignment_accepted": cache.provenance["alignment"]["accepted"],
            "tool_pose_estimated_by_vggt": False,
            "whole_video_optimization": False,
            "data_kind": manifest.data_kind,
        },
    )
    return cache

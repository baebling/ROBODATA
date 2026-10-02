"""Rendered, explicitly synthetic RGB/RGB-D inputs with independent pose truth."""

from pathlib import Path

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .export import save_episode, write_json
from .schema import Canonical, Episode, FrameRecord, Quality, Source
from .temporal import add_labels


def create_demo(output_dir: Path | str, frames=40, seed=42, occluded=None) -> Path:
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Synthetic inputs already exist: {root}")
    if frames < 4:
        raise ValueError("Synthetic demo needs at least four frames")
    root.mkdir(parents=True)
    (root / "frames").mkdir()
    (root / "masks").mkdir()
    (root / "depth").mkdir()
    rng = np.random.default_rng(seed)
    occluded = set(occluded if occluded is not None else [frames // 2])
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    tool = {
        "tip": [0.0, 0, 0],
        "axis_base": [0.0, 0, -0.07],
        "side": [0.04, 0, 0],
        "top": [0.0, 0.03, 0],
        "rear_left": [-0.03, -0.025, -0.05],
        "rear_right": [0.03, -0.025, -0.05],
        "back_top": [0.0, 0.025, -0.06],
        "front_left": [-0.03, -0.02, 0],
    }
    surface = {
        "bottom_left": [-0.35, -0.25, 0.0],
        "bottom_right": [0.35, -0.25, 0.0],
        "top_right": [0.35, 0.25, 0.0],
        "top_left": [-0.35, 0.25, 0.0],
    }
    S = np.diag([1.0, -1, -1, 1])
    S[2, 3] = 1.0
    colors = [
        (240, 70, 70),
        (60, 210, 110),
        (70, 100, 240),
        (240, 190, 50),
        (190, 70, 220),
        (70, 210, 220),
        (235, 130, 60),
        (130, 100, 210),
    ]

    def project(points, pose):
        xyz = np.array(list(points.values()))
        rv = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        uv, _ = cv2.projectPoints(xyz, rv, pose[:3, 3], K, np.zeros(5))
        return uv.reshape(-1, 2), (xyz @ pose[:3, :3].T + pose[:3, 3])[:, 2]

    video = cv2.VideoWriter(
        str(root / "demo.avi"), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (320, 240)
    )
    if not video.isOpened():
        raise RuntimeError("MJPG encoder is unavailable")
    observations, gt_frames, training_samples = [], [], []
    su, _ = project(surface, S)
    try:
        for i in range(frames):
            t = i * 0.1
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("y", 0.5 + 0.05 * np.sin(t)).as_matrix()
            T[:3, 3] = [-0.12 + 0.24 * i / (frames - 1), 0.03 * np.sin(t), 0.7]
            uv, zs = project(tool, T)
            image = np.clip(rng.normal(36, 2, (240, 320, 3)), 0, 255).astype(np.uint8)
            smask: NDArray[np.uint8] = np.zeros((240, 320), np.uint8)
            tmask: NDArray[np.uint8] = np.zeros((240, 320), np.uint8)
            cv2.fillConvexPoly(smask, np.rint(su).astype(int), (255,))
            image[smask > 0] = [62, 68, 72]
            depth: NDArray[np.uint16] = np.zeros((240, 320), np.uint16)
            depth[smask > 0] = 1000
            for j, p in enumerate(su):
                cv2.circle(image, tuple(np.rint(p).astype(int)), 6, colors[j], -1)
            if i not in occluded:
                hull = cv2.convexHull(np.rint(uv).astype(int))
                cv2.fillConvexPoly(image, hull, (110, 116, 125))
                cv2.fillConvexPoly(tmask, hull, (255,))
                cv2.fillConvexPoly(depth, hull, (700,))
                for j, (p, z) in enumerate(zip(uv, zs)):
                    q = tuple(np.rint(p).astype(int))
                    cv2.circle(image, q, 3, colors[j], -1)
                    cv2.circle(depth, q, 2, (int(np.rint(z * 1000)),), -1)
            frame_name = f"frames/{i:05d}.png"
            cv2.imwrite(str(root / frame_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(root / f"masks/tool_{i:05d}.png"), tmask)
            cv2.imwrite(str(root / f"masks/surface_{i:05d}.png"), smask)
            cv2.imwrite(str(root / f"depth/{i:05d}.png"), depth)
            points = {
                n: {"uv": p.tolist(), "confidence": 1.0, "visible": i not in occluded}
                for n, p in zip(tool, uv)
            }
            spoints = {
                n: {"uv": p.tolist(), "confidence": 1.0, "visible": True}
                for n, p in zip(surface, su)
            }
            observations.append(
                {
                    "frame_index": i,
                    "tool": {
                        "keypoints": points,
                        "occluded": i in occluded,
                        "mask_path": f"masks/tool_{i:05d}.png",
                    },
                    "surface": {"keypoints": spoints, "mask_path": f"masks/surface_{i:05d}.png"},
                    "depth_path": f"depth/{i:05d}.png",
                }
            )
            ST = np.eye(4)
            ST[:3, :3] = S[:3, :3].T @ T[:3, :3]
            ST[:3, 3] = S[:3, :3].T @ (T[:3, 3] - S[:3, 3])
            gt_frames.append(
                FrameRecord(
                    frame_index=i,
                    timestamp_sec=t,
                    canonical=Canonical(
                        T_surface_tool=ST.tolist(),
                        tip_position_surface_m=ST[:3, 3].tolist(),
                        tool_axis_surface=ST[:3, 2].tolist(),
                        standoff_distance_m=0.3,
                        signed_surface_distance_m=0.3,
                        incidence_angle_rad=0.5 + 0.05 * np.sin(t),
                    ),
                    quality=Quality(
                        valid=True,
                        full_pose_valid=True,
                        confidence=1.0,
                        confidence_kind="synthetic_exact",
                    ),
                )
            )
            kp = {f"tool:{n}": p for n, p in points.items()}
            kp.update({f"surface:{n}": p for n, p in spoints.items()})
            training_samples.append({"image_path": frame_name, "keypoints": kp})
    finally:
        video.release()
    manifest = {
        "episode_id": "synthetic_spray",
        "task_name": "spray_proxy",
        "video_path": "demo.avi",
        "data_kind": "synthetic",
        "camera": {"matrix": K.tolist(), "width": 320, "height": 240},
        "timestamps_sec": [i * 0.1 for i in range(frames)],
        "observations": observations,
        "landmark_source": "synthetic_projected_landmarks",
        "depth_aligned_to_rgb": True,
    }
    write_json(root / "manifest.json", manifest)
    base = {
        "manifest_path": "manifest.json",
        "surface_frame_id": "synthetic_patch",
        "tool": {"points_m": tool},
        "surface": {"points_m": surface},
        "horizon_sec": 0.2,
        "timestamp_tolerance_sec": 0.01,
        "seed": seed,
    }
    for mode in ["rgb", "rgbd"]:
        cfg = dict(
            base,
            mode=mode,
            metric_scale_source="measured_geometry" if mode == "rgb" else "depth_sensor",
            camera_motion="moving" if mode == "rgb" else "static",
            surface_method="pnp" if mode == "rgb" else "depth_plane",
        )
        (root / f"{mode}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    gt = Episode(
        episode_id="synthetic_spray",
        task_name="spray_proxy",
        surface_frame_id="synthetic_patch",
        source=Source(
            mode="external_pose", metric_scale_source="calibrated_external", data_kind="synthetic"
        ),
        frames=gt_frames,
        provenance={"reference": "analytic_poses_used_by_renderer", "seed": seed},
    )
    add_labels(gt, 0.2, 0.01)
    save_episode(gt, root / "ground_truth.json")
    # Disjoint chronological holdout is only a smoke-test split, not a research split.
    split = max(1, int(frames * 0.8))
    for name, samples in [("train", training_samples[:split]), ("val", training_samples[split:])]:
        write_json(root / f"{name}.json", {"data_kind": "synthetic", "samples": samples})
    training = {
        "train_path": "train.json",
        "val_path": "val.json",
        "keypoint_names": list(training_samples[0]["keypoints"]),
        "image_size": 64,
        "width": 16,
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.002,
        "seed": seed,
        "device": "cpu",
    }
    (root / "train.yaml").write_text(yaml.safe_dump(training), encoding="utf-8")
    return root

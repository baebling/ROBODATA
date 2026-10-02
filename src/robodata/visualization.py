"""Scientific trajectory plots and representative RGB overlays."""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .export import read_episode
from .observations import Manifest


def plot_episode(episode_path, output_path, gt_path=None):
    episode = read_episode(episode_path)
    out = Path(output_path)
    times = np.array([f.timestamp_sec for f in episode.frames])
    pos = np.array(
        [
            f.canonical.tip_position_surface_m if f.quality.valid else [np.nan] * 3
            for f in episode.frames
        ]
    )
    standoff = [f.canonical.standoff_distance_m for f in episode.frames]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.suptitle(
        f"{episode.episode_id} | {episode.source.mode} | {episode.source.data_kind.upper()}"
    )
    axes[0, 0].plot(pos[:, 0], pos[:, 1], "o-", markersize=3, color="#16876d", label="estimated")
    axes[0, 0].set(xlabel="surface x (m)", ylabel="surface y (m)", title="Projected path")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 1].plot(times, standoff, color="#ba5141", label="estimated")
    axes[0, 1].set(xlabel="time (s)", ylabel="standoff (m)", title="Tool to surface")
    for j, color in enumerate(["#16876d", "#ba5141", "#376ba6"]):
        axes[1, 0].plot(times, pos[:, j], label="xyz"[j], color=color)
    axes[1, 0].set(xlabel="time (s)", ylabel="position (m)", title="Surface-frame trajectory")
    axes[1, 0].legend()
    axes[1, 1].step(times, [f.quality.valid for f in episode.frames], where="mid", label="state")
    axes[1, 1].step(times, [f.action.valid for f in episode.frames], where="mid", label="action")
    axes[1, 1].set(xlabel="time (s)", ylabel="validity", ylim=(-0.1, 1.2), title="Label coverage")
    axes[1, 1].legend()
    if gt_path:
        gt = read_episode(gt_path)
        points = np.array([f.canonical.tip_position_surface_m for f in gt.frames])
        axes[0, 0].plot(points[:, 0], points[:, 1], "--", color="black", label="reference")
        axes[0, 1].plot(
            [f.timestamp_sec for f in gt.frames],
            [f.canonical.standoff_distance_m for f in gt.frames],
            "--",
            color="black",
            label="reference",
        )
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[0, 1].legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def overlay_frames(episode, out_dir: Path, count=3):
    path = Path(episode.source.manifest_path)
    manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(path.parent / manifest.video_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ids = np.linspace(0, len(episode.frames) - 1, count, dtype=int)
        for i in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, image = cap.read()
            if not ok:
                continue
            f = episode.frames[i]
            if f.quality.full_pose_valid:
                for est in [f.surface, f.tool]:
                    T = np.array(est.T_camera_frame)
                    rv, _ = cv2.Rodrigues(T[:3, :3])
                    cv2.drawFrameAxes(
                        image,
                        np.array(manifest.camera.matrix),
                        np.array(manifest.camera.distortion),
                        rv,
                        T[:3, 3],
                        0.05,
                        2,
                    )
            text = f"{episode.source.data_kind.upper()} frame {i} valid={f.quality.valid}"
            cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(out_dir / f"frame_{i:05d}.png"), image)
    finally:
        cap.release()

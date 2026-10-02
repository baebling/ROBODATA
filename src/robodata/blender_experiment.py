"""Oracle-landmark geometry experiment, not an automatic perception benchmark."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .blender_input import assemble_rendered
from .evaluation import evaluate
from .export import read_episode, write_json
from .pipeline import run_episode
from .visualization import overlay_frames, plot_episode


def run_blender_experiment(render_root: Path | str, output_root: Path | str) -> Path:
    root, out = Path(render_root).resolve(), Path(output_root).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new experiment directory: {out}")
    if out.is_relative_to(root):
        raise ValueError("Experiment output must be outside the raw render tree")
    declared_frames = None
    if (root / "render_metadata.json").is_file():
        sources = [root]
    else:
        config = json.loads((root / "effective_config.json").read_text(encoding="utf-8"))
        declared_frames = config.get("frames")
        sources = [root / s["name"] for s in config["scenarios"]]
    if not sources:
        raise ValueError("No rendered scenarios")
    for source in sources:
        if not source.resolve().is_relative_to(root):
            raise ValueError("Scenario path leaves render directory")
        if not (source / "render_metadata.json").is_file():
            raise FileNotFoundError(source / "render_metadata.json")
        metadata = json.loads((source / "render_metadata.json").read_text(encoding="utf-8"))
        if declared_frames is not None and (
            type(declared_frames) is not int or len(metadata["frames"]) != declared_frames
        ):
            raise ValueError("Rendered frame count differs from effective configuration")
    rows, previews, durations = [], [], []
    out.mkdir(parents=True)
    for source in sources:
        name = source.name
        case = out / name
        dataset = assemble_rendered(source, case / "inputs")
        reference = dataset / "ground_truth.json"
        gt = read_episode(reference)
        gt_points = np.array([f.canonical.tip_position_surface_m for f in gt.frames])
        stationary = bool(np.max(np.linalg.norm(gt_points - gt_points[0], axis=1)) < 1e-6)
        for mode in ["rgb", "rgbd"]:
            folder = case / mode
            episode = run_episode(dataset / f"{mode}.yaml", folder)
            metrics = evaluate(episode, gt, tolerance_sec=0.01)
            failures = Counter(
                f.quality.failure_reason or "unspecified"
                for f in episode.frames
                if not f.quality.valid
            )
            points = np.array(
                [f.canonical.tip_position_surface_m for f in episode.frames if f.quality.valid]
            )
            span = None
            if stationary and len(points) > 1:
                span = float(np.max(np.linalg.norm(points - points[0], axis=1)))
            row = {
                "scenario": name,
                "mode": mode,
                **metrics,
                "failure_counts": dict(failures),
                "stationary_prediction_span_m": span,
            }
            rows.append(row)
            write_json(folder / "evaluation.json", row)
            plot_episode(folder / "episode.json", folder / "trajectory.png", reference)
            overlay_frames(episode, folder / "overlays")
        meta = json.loads((source / "render_metadata.json").read_text(encoding="utf-8"))
        for frame in meta["frames"]:
            image = Image.open(source / frame["rgb_path"]).convert("RGB")
            image.thumbnail((640, 360))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 35), fill="black")
            draw.text((8, 5), f"SYNTHETIC | {name}", fill="white")
            draw.text((8, 20), "ORACLE LANDMARKS - NO DETECTOR / SPRAY PHYSICS", fill="white")
            previews.append(image)
            durations.append(max(10, round(1000 / meta["fps"])))
        print(f"ROBODATA_EVALUATED {name}", flush=True)
    write_json(
        out / "comparison.json",
        {
            "data_kind": "synthetic",
            "landmarks": "oracle_not_detector",
            "scope": "calibrated geometry with rendered depth, no spray or coating physics",
            "results": rows,
        },
    )
    previews[0].save(
        out / "preview.gif", save_all=True, append_images=previews[1:], duration=durations, loop=0
    )
    lines = [
        "# Blender H-Beam Geometry Experiment",
        "",
        "SYNTHETIC ONLY. Oracle 2D landmarks and known proxy geometry; no automatic detector.",
        "RGB-D tool depth comes from the render, not ground-truth tool poses.",
        "Both modes use surface PnP. This does not test unknown-surface reconstruction,",
        "real RGB-only feasibility, spray particles, coating thickness, or robot execution.",
        "",
        "| Scenario | Mode | Valid / frames | Position RMSE (mm) | Axis MAE (deg) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        rmse, angle = row["position_rmse_m"], row["axis_mae_rad"]
        position = "N/A" if rmse is None else f"{1000 * rmse:.6f}"
        axis = "N/A" if angle is None else f"{np.degrees(angle):.6f}"
        lines.append(
            f"| {row['scenario']} | {row['mode']} | "
            f"{row['matched_valid_frames']} / {row['predicted_frames']} | {position} | {axis} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Near-zero RGB errors are expected with exact projected landmarks and known geometry.",
        "The RGB-D sampling baseline observes marker surfaces, not their semantic centers;",
        "marker radius, pixel sampling and the 3x3 median can produce systematic offsets.",
        "No trajectory alignment, smoothing or deletion of failed frames is performed.",
        "Null work_on / primitive labels remain unknown. Failure counts and stationary-control",
        "motion are recorded in comparison.json. Raw renders remain untouched.",
        "",
    ]
    if any(not row["reference_provenance"].get("render_time_source_recorded") for row in rows):
        lines += [
            "Legacy pilot renders lack a render-time renderer hash/Git snapshot.",
            "Evaluation-time provenance does not retroactively supply this missing record.",
            "",
        ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out

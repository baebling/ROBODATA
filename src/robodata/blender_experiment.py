"""Oracle-landmark geometry experiment, not an automatic perception benchmark."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .blender_geometry import nozzle_relation
from .blender_input import assemble_rendered, camera_from_blender
from .evaluation import evaluate
from .export import read_episode, write_json
from .pipeline import run_episode
from .visualization import overlay_frames, plot_episode


def relationship_diagnostics(episode, metadata) -> dict:
    if episode.surface_frame_id != metadata["surface_frame_id"]:
        raise ValueError("Relationship reference has a different surface frame")
    cfg = metadata["settings"]["config"]
    records, errors, evaluated, correct = [], [], 0, 0
    counts: Counter[str] = Counter()
    for frame, raw in zip(episode.frames, metadata["frames"], strict=True):
        if frame.frame_index != raw["frame_index"]:
            raise ValueError("Relationship frames are not aligned")
        reference = raw["nozzle_reference"]
        if reference.get("scene_ray_agrees") is not True:
            raise ValueError("Relationship reference lacks a verified scene ray")
        estimated, error = None, None
        if frame.quality.valid:
            estimated = nozzle_relation(
                cfg, frame.canonical.tip_position_surface_m, frame.canonical.tool_axis_surface
            )
            counts[estimated["status"]] += 1
            evaluated += 1
            correct += estimated["status"] == reference["status"]
            a, b = (item["ray_plane_intersection_surface_m"] for item in [estimated, reference])
            if a is not None and b is not None:
                error = float(np.linalg.norm(np.asarray(a) - b))
                errors.append(error)
        records.append(
            {
                "frame_index": frame.frame_index,
                "timestamp_sec": frame.timestamp_sec,
                "estimated": estimated,
                "reference": reference,
                "plane_intersection_error_m": error,
                "failure_reason": frame.quality.failure_reason,
            }
        )
    return {
        "scope": "known beam geometry; no perception, spray cone, deposition or physical collision",
        "reference_source": "actual simulator pose plus independently verified Blender beam ray",
        "summary": {
            "total_frames": len(records),
            "evaluated_frames": evaluated,
            "unknown_frames": len(records) - evaluated,
            "status_accuracy": correct / evaluated if evaluated else None,
            "status_counts": dict(counts),
            "reference_status_counts": dict(Counter(row["reference"]["status"] for row in records)),
            "plane_intersection_matched_frames": len(errors),
            "plane_intersection_rmse_m": float(np.sqrt(np.mean(np.square(errors))))
            if errors
            else None,
        },
        "frames": records,
    }


def _relation_preview(image, raw, metadata, predicted):
    draw = ImageDraw.Draw(image)
    scale = image.width / metadata["width"]
    K = np.asarray(metadata["camera_matrix"])
    C_S = camera_from_blender(raw["T_world_camera"]) @ np.asarray(raw["T_world_surface"])

    def project(point, transform):
        xyz = (transform @ [*point, 1])[:3]
        if xyz[2] <= 0:
            return None
        pixel = K @ xyz
        return tuple(pixel[:2] / pixel[2] * scale)

    origin = project([0, 0, 0], C_S)
    for point, color in [([0.12, 0, 0], "red"), ([0, 0.12, 0], "lime"), ([0, 0, 0.12], "cyan")]:
        end = project(point, C_S)
        if origin is not None and end is not None:
            draw.line([origin, end], fill=color, width=2)
    from .geometry import inverse

    T = inverse(raw["T_world_surface"]) @ np.asarray(raw["T_world_tool"])
    reference = raw["nozzle_reference"]
    endpoint = reference["first_hit_surface_m"]
    if endpoint is None:
        endpoint = (T[:3, 3] + 0.55 * T[:3, 2]).tolist()
    start, end = project(T[:3, 3], C_S), project(endpoint, C_S)
    if start is not None and end is not None:
        draw.line([start, end], fill="yellow", width=3)
        draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), outline="yellow", width=2)
    predicted_status = "unknown"
    if predicted.quality.valid:
        relation = nozzle_relation(
            metadata["settings"]["config"],
            predicted.canonical.tip_position_surface_m,
            predicted.canonical.tool_axis_surface,
        )
        end_point = relation["first_hit_surface_m"]
        predicted_status = relation["status"]
        if end_point is None:
            end_point = (
                np.asarray(predicted.canonical.tip_position_surface_m)
                + 0.55 * np.asarray(predicted.canonical.tool_axis_surface)
            ).tolist()
        transform = np.asarray(predicted.surface.T_camera_frame)
        p = project(predicted.canonical.tip_position_surface_m, transform)
        q = project(end_point, transform)
        if p is not None and q is not None:
            draw.line([p, q], fill="magenta", width=2)
    footer = image.height
    preview = Image.new("RGB", (image.width, image.height + 79), "black")
    preview.paste(image, (0, 0))
    draw = ImageDraw.Draw(preview)
    draw.text(
        (8, footer + 4),
        f"SYNTHETIC | {metadata['episode_id']} | {metadata['work_face']}",
        fill="white",
    )
    draw.text(
        (8, footer + 19),
        "ORACLE LANDMARKS / NO SPRAY PHYSICS | YELLOW: GT | MAGENTA: RGB",
        fill="white",
    )
    draw.text(
        (8, footer + 34),
        f"GT ray: {reference['status']} | first solid: {reference['first_beam_part']}",
        fill="yellow",
    )
    angle = np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))
    draw.text(
        (8, footer + 49),
        f"GT perpendicular gap: {abs(T[2, 3]):.3f} m | incidence: {angle:.1f} deg",
        fill="white",
    )
    draw.text(
        (8, footer + 64),
        f"RGB estimated ray: {predicted_status}",
        fill="magenta" if predicted.quality.valid else "white",
    )
    return preview


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
        meta = json.loads((source / "render_metadata.json").read_text(encoding="utf-8"))
        has_relations = all("nozzle_reference" in frame for frame in meta["frames"])
        by_mode = {}
        for mode in ["rgb", "rgbd"]:
            folder = case / mode
            episode = run_episode(dataset / f"{mode}.yaml", folder)
            by_mode[mode] = episode
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
            if has_relations:
                diagnostics = relationship_diagnostics(episode, meta)
                write_json(folder / "nozzle_relationship.json", diagnostics)
                row["nozzle_relationship"] = diagnostics["summary"]
            rows.append(row)
            write_json(folder / "evaluation.json", row)
            plot_episode(folder / "episode.json", folder / "trajectory.png", reference)
            overlay_frames(episode, folder / "overlays")
        for frame in meta["frames"]:
            image = Image.open(source / frame["rgb_path"]).convert("RGB")
            image.thumbnail((640, 360))
            if has_relations:
                image = _relation_preview(
                    image, frame, meta, by_mode["rgb"].frames[frame["frame_index"]]
                )
            else:
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 35), fill="black")
                draw.text((8, 5), f"SYNTHETIC | {name}", fill="white")
                draw.text((8, 20), "ORACLE LANDMARKS - NO DETECTOR / SPRAY PHYSICS", fill="white")
            image.save(case / f"relationship_{frame['frame_index']:05d}.png")
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
    if any("nozzle_relationship" in row for row in rows):
        lines += [
            "## Finite Beam / Nozzle Relationship",
            "",
            "The ray is the nozzle +Z axis, not orthogonal projection or a spray footprint.",
            "Predicted rays use estimated observations and configured known beam dimensions.",
            "References use simulator transforms checked against actual Blender beam mesh rays.",
            "No reference pose or ray label is passed to trajectory estimation.",
            "Unknown estimates remain unknown; classification accuracy excludes these frames.",
            "",
            "| Scenario | Mode | Evaluated / total | Status accuracy | GT ray statuses |",
            "|---|---|---:|---:|---|",
        ]
        for row in rows:
            if "nozzle_relationship" not in row:
                continue
            summary = row["nozzle_relationship"]
            accuracy = summary["status_accuracy"]
            score = "N/A" if accuracy is None else f"{accuracy:.3f}"
            lines.append(
                f"| {row['scenario']} | {row['mode']} | "
                f"{summary['evaluated_frames']} / {summary['total_frames']} | {score} | "
                f"{summary['reference_status_counts']} |"
            )
        lines += [
            "",
            "Per-frame intersections, first-hit solids and unknowns: nozzle_relationship.json.",
            "A valid geometric state does not imply that the spray reaches the working face.",
            "",
        ]
        for mode in ["rgb", "rgbd"]:
            target_rows = [
                row
                for row in rows
                if row["mode"] == mode
                and "nozzle_relationship" in row
                and set(row["nozzle_relationship"]["reference_status_counts"]) == {"work_face_hit"}
            ]
            if target_rows:
                total = sum(row["predicted_frames"] for row in target_rows)
                valid = sum(row["matched_valid_frames"] for row in target_rows)
                lines.append(
                    f"Target-facing work-face-hit conditions, including frontal control: "
                    f"{mode} valid {valid}/{total}. Unknowns are NOT successful predictions."
                )
        lines += [
            "",
            "Low target-facing coverage indicates occlusion limitations of this landmark baseline;",
            "Near-zero errors on a few oracle frames do not establish practical video extraction.",
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

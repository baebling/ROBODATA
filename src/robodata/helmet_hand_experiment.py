"""Compare synthetic helmet RGB baselines with optional MediaPipe hand fallback.

The fixed-grasp template calibration is a declared simulator prior, not a real
tool measurement. Only RGB and supplied tool masks reach hand inference; depth
and simulator poses are not used to predict hands. Existing oracle tool/surface
landmarks remain baseline constraints. GT is used for evaluation only.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .config import HandAssistanceSpec, RunConfig, load_config
from .evaluation import evaluate
from .export import read_episode, write_json
from .hand_assistance import PALM_KEYS, hand_tool_observation
from .hand_infer import KEYPOINT_SEMANTICS, infer_hands
from .observations import Detection, Manifest
from .pipeline import run_episode, software_provenance
from .schema import Episode

TILE_WIDTH, TILE_HEIGHT = 480, 432
ROW_LABEL_WIDTH, HEADER_HEIGHT = 164, 110
FINGER_COLORS = [(255, 80, 80), (255, 210, 60), (80, 220, 110), (60, 210, 255), (240, 100, 240)]
CALIBRATION_NOTE = (
    "Synthetic fixed-grasp template prior; not measured tool geometry or a measured real grasp."
)


@dataclass
class _Case:
    name: str
    source: Path
    baseline: Path
    metadata: dict
    config: RunConfig
    manifest_path: Path
    calibration: HandAssistanceSpec
    gt: Episode
    rgb: Episode
    rgbd: Episode | None


def _contained(root: Path, name, required=True) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Input path leaves its source directory: {name}")
    if required and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _aligned(episode: Episode, frames):
    if episode.source.data_kind != "synthetic":
        raise ValueError("Helmet hand experiment requires synthetic episodes")
    if len(episode.frames) != len(frames) or any(
        predicted.frame_index != raw["frame_index"]
        or not np.isclose(predicted.timestamp_sec, raw["timestamp_sec"], rtol=0, atol=1e-9)
        for predicted, raw in zip(episode.frames, frames)
    ):
        raise ValueError("Episode frames/timestamps do not match raw scenario frames")


def _preflight(root: Path, baseline: Path, out: Path, model: Path):
    if out.exists():
        raise FileExistsError(f"Choose a fresh helmet hand output directory: {out}")
    if any(out.is_relative_to(source) or source.is_relative_to(out) for source in (root, baseline)):
        raise ValueError("Hand output must be outside both raw render and baseline directories")
    if not model.is_file():
        raise FileNotFoundError(model)
    effective = json.loads(_contained(root, "effective_config.json").read_text(encoding="utf-8"))
    scenarios = effective.get("scenarios", [])
    if not scenarios:
        raise ValueError("No rendered scenarios in effective_config")
    cases, names, cells = [], set(), set()
    for scenario in scenarios:
        name = scenario["name"]
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or name in names
            or Path(name).is_absolute()
        ):
            raise ValueError("Scenario must have a unique contained directory name")
        names.add(name)
        source = _contained(root, name, required=False)
        case_root = _contained(baseline, name, required=False)
        metadata_path = _contained(source, "render_metadata.json")
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        settings = metadata["settings"]["scenario"]
        sensor, view = settings["sensor"], settings["view"]
        mount = settings.get("mount", "helmet")
        if settings.get("name", name) != name or any(
            not isinstance(value, str) or not value for value in (sensor, view, mount)
        ):
            raise ValueError("Scenario metadata needs matching name, sensor, view and mount")
        if any(
            key in scenario and scenario[key] != settings.get(key) for key in ("sensor", "view")
        ):
            raise ValueError("Scenario settings differ from effective_config")
        cell = (mount, view, sensor)
        if cell in cells:
            raise ValueError("Duplicate sensor/mount/view contact-sheet cell")
        cells.add(cell)
        frames = metadata.get("frames", [])
        declared = effective.get("frames")
        if not frames or (
            declared is not None and (type(declared) is not int or len(frames) != declared)
        ):
            raise ValueError("Rendered scenario frame count differs from effective_config")
        for frame in frames:
            _contained(source, frame["rgb_path"])
            for key in ("depth_path", "labels_path"):
                if frame.get(key):
                    _contained(source, frame[key], required=False)
        prior = metadata.get("hand_calibration")
        if prior is None:
            raise ValueError("Scenario is missing hand calibration")
        calibration = HandAssistanceSpec.model_validate(prior)
        calibration_source = calibration.calibration_source.lower()
        if (
            not calibration.grasp_verified
            or not all(
                word in calibration_source for word in ("synthetic", "fixed_grasp", "template")
            )
            or not PALM_KEYS.issubset(calibration.points_hand_m)
        ):
            raise ValueError(
                "Hand calibration must declare a verified synthetic fixed-grasp template"
            )
        config_path = _contained(case_root, "inputs/rgb.yaml")
        config = load_config(config_path)
        if config.mode != "rgb" or config.hand_assistance is not None:
            raise ValueError("Baseline configuration must be RGB without hand assistance")
        manifest_path = _contained(case_root, config_path.parent / config.manifest_path)
        manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.provenance.get("render_metadata_sha256")
            != hashlib.sha256(metadata_bytes).hexdigest()
        ):
            raise ValueError("Raw metadata SHA-256 does not match baseline manifest provenance")
        if manifest.data_kind != "synthetic":
            raise ValueError("Hand experiment input must be synthetic")
        if (
            manifest.camera.width != metadata["width"]
            or manifest.camera.height != metadata["height"]
            or not np.allclose(manifest.camera.matrix, metadata["camera_matrix"], rtol=0, atol=1e-6)
        ):
            raise ValueError("Raw render and baseline camera calibration/resolution differ")
        if manifest.timestamps_sec is not None and (
            len(manifest.timestamps_sec) != len(frames)
            or not np.allclose(
                manifest.timestamps_sec, [f["timestamp_sec"] for f in frames], rtol=0, atol=1e-9
            )
        ):
            raise ValueError("Manifest timestamps differ from raw scenario")
        _contained(case_root, manifest_path.parent / manifest.video_path)
        for observation in manifest.observations:
            for detection in (observation.tool, observation.surface, observation.hand):
                if detection.mask_path:
                    _contained(case_root, manifest_path.parent / detection.mask_path)
            if observation.depth_path:
                _contained(case_root, manifest_path.parent / observation.depth_path, required=False)
        gt = read_episode(_contained(case_root, "inputs/ground_truth.json"))
        rgb = read_episode(_contained(case_root, "rgb/episode.json"))
        rgbd_path = _contained(case_root, "rgbd/episode.json", required=False)
        rgbd = read_episode(rgbd_path) if rgbd_path.is_file() else None
        if (
            manifest.episode_id != gt.episode_id
            or metadata.get("episode_id", gt.episode_id) != gt.episode_id
        ):
            raise ValueError("Manifest/reference/raw episode identity mismatch")
        for episode in (gt, rgb, rgbd):
            if episode is not None:
                _aligned(episode, frames)
                if episode.episode_id != gt.episode_id:
                    raise ValueError("Baseline episode identity differs from its reference")
                if (
                    episode.surface_frame_id != config.surface_frame_id
                    or episode.task_name != gt.task_name
                ):
                    raise ValueError(
                        "Baseline/reference task or surface frame differs from RGB config"
                    )
        cases.append(
            _Case(
                name, source, case_root, metadata, config, manifest_path, calibration, gt, rgb, rgbd
            )
        )
    sensors = list(effective.get("sensor_profiles", {}))
    for case in cases:
        sensor = case.metadata["settings"]["scenario"]["sensor"]
        if sensor not in sensors:
            sensors.append(sensor)
    if len(sensors) > 4:
        raise ValueError("Contact sheet supports four sensor columns")
    sensors.extend([None] * (4 - len(sensors)))
    mounts = list(
        dict.fromkeys(
            (
                c.metadata["settings"]["scenario"].get("mount", "helmet"),
                c.metadata["settings"]["scenario"]["view"],
            )
            for c in cases
        )
    )
    return cases, sensors, mounts


def _metrics(episode: Episode, gt: Episode):
    return {
        **evaluate(episode, gt, tolerance_sec=0.01),
        "valid_frames": sum(frame.quality.valid for frame in episode.frames),
        "full_pose_valid_frames": sum(frame.quality.full_pose_valid for frame in episode.frames),
        "stage_counts": {
            "surface_unavailable": sum(
                frame.surface is None or frame.surface.T_camera_frame is None
                for frame in episode.frames
            ),
            "tool_unavailable": sum(
                frame.tool is None
                or frame.tool.origin_camera_m is None
                or frame.tool.axis_camera is None
                for frame in episode.frames
            ),
        },
        "failure_counts": dict(
            Counter(
                frame.quality.failure_reason or "unspecified"
                for frame in episode.frames
                if not frame.quality.valid
            )
        ),
    }


def _hand_pose_diagnostics(
    manifest: Manifest, raw_frames, spec: HandAssistanceSpec, cfg: RunConfig
):
    """Test selected hand PnP with only predicted hand, declared prior and camera K."""
    cv2.setRNGSeed(cfg.seed)
    observations = {frame.frame_index: frame for frame in manifest.observations}
    records = {}
    rejections: Counter[str] = Counter()
    for raw in raw_frames:
        record: dict = {
            "hand_pose_accepted": None,
            "hand_pose_rejection": None,
            "hand_pose_residual_px": None,
            "hand_pose_confidence": None,
        }
        if raw["selected_hand_index"] is not None:
            observation = observations.get(raw["frame_index"])
            detection = Detection() if observation is None else observation.hand
            try:
                estimate = hand_tool_observation(detection, spec, manifest.camera, cfg)
            except ValueError as error:
                reason = str(error)
                record.update(hand_pose_accepted=False, hand_pose_rejection=reason)
                rejections[reason] += 1
            else:
                record.update(
                    hand_pose_accepted=True,
                    hand_pose_residual_px=estimate.residual,
                    hand_pose_confidence=estimate.confidence,
                )
        records[raw["frame_index"]] = record
    return records, dict(rejections)


def _prediction_overlay(image: Image.Image, raw):
    """Draw only the selected model hand, never template or GT landmark positions."""
    image = image.copy()
    selected = raw.get("selected_hand_index")
    if selected is None:
        return image
    hands = raw["hands"]
    if type(selected) is not int or not 0 <= selected < len(hands):
        raise ValueError("Invalid selected model hand index")
    points = hands[selected]["normalized_landmarks"]
    if len(points) != 21:
        raise ValueError("Selected model hand must have 21 image landmarks")
    coordinates = {}
    for index, (x, y, _) in enumerate(points):
        if (
            x is not None
            and y is not None
            and np.isfinite([x, y]).all()
            and 0 <= x < 1
            and 0 <= y < 1
        ):
            coordinates[index] = (x * image.width, y * image.height)
    draw = ImageDraw.Draw(image)
    for finger, color in enumerate(FINGER_COLORS):
        indices = [0, *range(1 + finger * 4, 5 + finger * 4)]
        for a, b in zip(indices, indices[1:]):
            if a in coordinates and b in coordinates:
                draw.line([coordinates[a], coordinates[b]], fill=color, width=3)
        for index in indices[1:]:
            if index in coordinates:
                x, y = coordinates[index]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    if 0 in coordinates:
        x, y = coordinates[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="white")
    return image


def _font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _counts(metrics):
    return (
        "unavailable"
        if metrics is None
        else (f"{metrics['matched_valid_frames']}/{metrics['reference_frames']}")
    )


def _tile_text(draw, xy, text, font, fill):
    width = TILE_WIDTH - 16
    bounds = draw.textbbox((0, 0), text, font=font)
    if bounds[2] - bounds[0] > width:
        shortened = text
        while shortened:
            shortened = shortened[:-1]
            bounds = draw.textbbox((0, 0), shortened + "...", font=font)
            if bounds[2] - bounds[0] <= width:
                break
        text = shortened + "..."
    draw.text(xy, text, fill=fill, font=font)


def _tile(case: _Case, row, raw):
    frames = case.metadata["frames"]
    frame = frames[len(frames) // 2]
    with Image.open(_contained(case.source, frame["rgb_path"])) as source:
        image = source.convert("RGB")
    if image.size != (case.metadata["width"], case.metadata["height"]):
        raise ValueError("Midframe image dimensions differ from calibrated camera")
    image = _prediction_overlay(image, raw)
    image.thumbnail((TILE_WIDTH - 16, 288), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), (18, 20, 23))
    tile.paste(image, ((TILE_WIDTH - image.width) // 2, 26 + (288 - image.height) // 2))
    draw, font = ImageDraw.Draw(tile), _font()
    _tile_text(draw, (8, 5), case.name, font, "white")
    metrics = row["rgb_hand"]
    axis = metrics["axis_mae_rad"]
    text = [
        f"Canonical matched-valid {_counts(metrics)} | selected hands "
        f"{row['selected_hand_frames']}/{row['total_frames']}",
        f"Hand PnP {row['hand_pose_accepted_frames']}/{row['selected_hand_frames']} | "
        f"RGB {_counts(row['rgb'])} | RGB-D {_counts(row['rgbd'])}",
        f"Position RMSE {_number(metrics['position_rmse_m'])} m | "
        f"axis MAE {_number(None if axis is None else np.degrees(axis), 2)} deg",
        f"Midframe {frame['frame_index']} | "
        + (
            "selected MediaPipe hand"
            if raw["selected_hand_index"] is not None
            else "no selected hand"
        ),
        "SYNTHETIC / MediaPipe / NOT MEASURED TOOL",
        "Fixed-grasp template prior | heuristic quality scores",
    ]
    for index, line in enumerate(text):
        _tile_text(draw, (8, 319 + index * 18), line, font, (225, 230, 235))
    return tile


def _contact_sheet(tiles, sensors, mounts, path: Path):
    sheet = Image.new(
        "RGB",
        (ROW_LABEL_WIDTH + 4 * TILE_WIDTH, HEADER_HEIGHT + len(mounts) * TILE_HEIGHT),
        (12, 14, 17),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 10), "SYNTHETIC / MediaPipe / NOT MEASURED TOOL", fill="white", font=_font(20))
    draw.text(
        (14, 42),
        "Selected model hand only | colors: thumb, index, middle, ring, little",
        fill=(210, 220, 230),
        font=_font(),
    )
    for column, sensor in enumerate(sensors):
        draw.text(
            (ROW_LABEL_WIDTH + column * TILE_WIDTH + 8, 78),
            sensor or "No sensor",
            fill="white",
            font=_font(16),
        )
    for row, (mount, view) in enumerate(mounts):
        y = HEADER_HEIGHT + row * TILE_HEIGHT
        draw.text((10, y + 12), mount, fill="white", font=_font())
        draw.text((10, y + 35), view, fill=(200, 210, 220), font=_font())
        for column, sensor in enumerate(sensors):
            tile = tiles.get((mount, view, sensor))
            x = ROW_LABEL_WIDTH + column * TILE_WIDTH
            if tile is not None:
                sheet.paste(tile, (x, y))
            else:
                draw.text((x + 14, y + 30), "No rendered case", fill=(155, 165, 175), font=_font())
    sheet.save(path)


def _report(comparison):
    lines = [
        "# Synthetic Helmet Hand Experiment",
        "",
        "SYNTHETIC / MediaPipe / not measured tool. " + CALIBRATION_NOTE,
        "Baseline oracle tool/surface landmark constraints are retained; this is not a full "
        "automatic-perception benchmark. RGB+hand uses hand PnP only when baseline "
        "tool estimation fails.",
        "GT is the independent simulator evaluation reference. Depth and GT poses are not used "
        "for hand inference. Model-local world landmarks are not camera-metric coordinates.",
        "Confidence is heuristic, not an error probability. " + KEYPOINT_SEMANTICS,
        "Every failed frame remains in the episode and the coverage denominator. Errors describe "
        "matched valid frames only. Counts and errors below are measurements of this run; "
        "they do not establish real-world performance.",
        "",
        "![Four sensor columns, mount/view rows](contact_sheet.png)",
        "",
        "| Case | Method | Matched-valid / total | Selected hands / total | "
        "Position RMSE (m) | Axis MAE (deg) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison["results"]:
        for method, key in (("RGB", "rgb"), ("RGB-D", "rgbd"), ("RGB+hand", "rgb_hand")):
            metrics = row[key]
            selected = (
                f"{row['selected_hand_frames']}/{row['total_frames']}" if key == "rgb_hand" else "-"
            )
            position = None if metrics is None else metrics["position_rmse_m"]
            axis = None if metrics is None else metrics["axis_mae_rad"]
            lines.append(
                f"| {row['scenario']} | {method} | {_counts(metrics)} | {selected} | "
                f"{_number(position, 6)} | "
                f"{_number(None if axis is None else np.degrees(axis), 3)} |"
            )
    lines.extend(
        [
            "",
            "## Stage Diagnostics",
            "",
            "Hand detection/association is separate from pose acceptance and canonical validity. "
            "Hand PnP is checked independently of surface availability, using only predicted "
            "hand keypoints, the synthetic template prior and camera calibration. Accepted hand "
            "PnP is a geometry gate, not a ground-truth accuracy claim.",
            "Surface/tool counts below describe unavailable estimates in the actual episode. "
            "Tool unavailable also includes frames where surface failure prevented reaching "
            "the tool stage; the two counts can overlap.",
            "",
            "| Case | Selected hands / total | Hand PnP accepted / selected | "
            "Surface unavailable / total | Tool unavailable / total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison["results"]:
        stages, total = row["stage_counts"], row["total_frames"]
        lines.append(
            f"| {row['scenario']} | {row['selected_hand_frames']}/{total} | "
            f"{row['hand_pose_accepted_frames']}/{row['selected_hand_frames']} | "
            f"{stages['surface_unavailable']}/{total} | {stages['tool_unavailable']}/{total} |"
        )
    lines.extend(["", "Hand PnP rejection counts (selected frames only):", ""])
    for row in comparison["results"]:
        lines.append(
            f"- {row['scenario']}: "
            f"`{json.dumps(row['hand_pose_rejection_counts'], sort_keys=True)}`"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"Model: `{comparison['model_path']}`",
            f"Model SHA-256: `{comparison['model_sha256']}`",
            f"Created UTC: `{comparison['created_at_utc']}`",
            "Full model provenance, thresholds, calibration sources, per-frame failures "
            "and baseline metrics are retained in [comparison.json](comparison.json).",
        ]
    )
    for row in comparison["results"]:
        provenance = row["hand_source"]
        lines.append(
            f"- {row['scenario']}: MediaPipe {provenance.get('mediapipe_version', 'unknown')}; "
            f"threshold {provenance.get('min_confidence', 'unknown')}; "
            f"calibration `{row['calibration_source']}`; hand fallback "
            f"{row['hand_fallback_frames']}/{row['total_frames']} frames."
        )
    return "\n".join(lines) + "\n"


def run_helmet_hand_experiment(
    render_root: Path | str,
    baseline_root: Path | str,
    model_path: Path | str,
    output_root: Path | str,
    min_confidence=0.5,
) -> Path:
    """Preflight all inputs, then run RGB hand fallback/evaluation in fresh output."""
    root, baseline, model, out = map(Path, (render_root, baseline_root, model_path, output_root))
    root, baseline, model, out = (p.resolve() for p in (root, baseline, model, out))
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("Hand confidence threshold must lie in [0,1]")
    cases, sensors, mounts = _preflight(root, baseline, out, model)
    out.mkdir(parents=True, exist_ok=False)
    rows, tiles = [], {}
    for case in cases:
        folder = out / case.name
        predicted_path = Path(
            infer_hands(
                model,
                case.manifest_path,
                folder / "hand_predictions",
                min_confidence=min_confidence,
            )
        ).resolve()
        _contained((folder / "hand_predictions").resolve(), predicted_path)
        predicted = Manifest.model_validate_json(predicted_path.read_text(encoding="utf-8"))
        hand_source = predicted.provenance["hand_source"]
        raw_path = _contained(predicted_path.parent, hand_source["raw_detections_path"])
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_frames = raw["frames"]
        frames = case.metadata["frames"]
        if len(raw_frames) != len(frames) or any(
            model_frame["frame_index"] != frame["frame_index"]
            or not np.isclose(
                model_frame["timestamp_sec"], frame["timestamp_sec"], rtol=0, atol=1e-9
            )
            for model_frame, frame in zip(raw_frames, frames)
        ):
            raise ValueError("Raw model results must retain every scenario frame and timestamp")
        config = case.config.model_dump(mode="json")
        config.update(
            manifest_path=str(predicted_path), hand_assistance=case.calibration.model_dump()
        )
        checked = RunConfig.model_validate(config)
        hand_diagnostics, hand_rejections = _hand_pose_diagnostics(
            predicted, raw_frames, case.calibration, checked
        )
        config_path = folder / "rgb_hand.yaml"
        config_path.write_text(yaml.safe_dump(checked.model_dump(mode="json")), encoding="utf-8")
        episode = run_episode(config_path, folder / "rgb_hand")
        _aligned(episode, frames)
        metrics = _metrics(episode, case.gt)
        write_json(folder / "rgb_hand/evaluation.json", metrics)
        settings = case.metadata["settings"]["scenario"]
        row = {
            "scenario": case.name,
            "sensor": settings["sensor"],
            "view": settings["view"],
            "mount": settings.get("mount", "helmet"),
            "data_kind": "synthetic",
            "total_frames": len(frames),
            "selected_hand_frames": sum(f["selected_hand_index"] is not None for f in raw_frames),
            "hand_pose_accepted_frames": sum(
                record["hand_pose_accepted"] is True for record in hand_diagnostics.values()
            ),
            "hand_pose_rejection_counts": hand_rejections,
            "stage_counts": metrics["stage_counts"],
            "hand_fallback_frames": sum(
                f.tool is not None and f.tool.method == "calibrated_hand_grasp_pnp"
                for f in episode.frames
            ),
            "calibration_source": case.calibration.calibration_source,
            "calibration_handedness": case.calibration.handedness,
            "hand_source": hand_source,
            "rgb": _metrics(case.rgb, case.gt),
            "rgbd": None if case.rgbd is None else _metrics(case.rgbd, case.gt),
            "rgb_hand": metrics,
            "frames": [
                {
                    **hand_diagnostics[f.frame_index],
                    "frame_index": f.frame_index,
                    "timestamp_sec": f.timestamp_sec,
                    "selected_hand_index": r["selected_hand_index"],
                    "association_failure": r.get("association_failure"),
                    "valid": f.quality.valid,
                    "full_pose_valid": f.quality.full_pose_valid,
                    "failure_reason": f.quality.failure_reason,
                    "tool_method": None if f.tool is None else f.tool.method,
                }
                for f, r in zip(episode.frames, raw_frames, strict=True)
            ],
            "paths": {
                "rgb_hand_config": str(config_path),
                "hand_manifest": str(predicted_path),
                "hand_episode": str(folder / "rgb_hand/episode.json"),
                "ground_truth": str(case.baseline / "inputs/ground_truth.json"),
            },
        }
        rows.append(row)
        tiles[(row["mount"], row["view"], row["sensor"])] = _tile(
            case, row, raw_frames[len(frames) // 2]
        )
    comparison = {
        "data_kind": "synthetic",
        "scope": "RGB baseline with calibrated hand fallback",
        "calibration_note": CALIBRATION_NOTE,
        "confidence_note": "Pose confidence is an uncalibrated geometry/reprojection heuristic. "
        + KEYPOINT_SEMANTICS,
        "render_root": str(root),
        "baseline_root": str(baseline),
        "model_path": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": software_provenance(root / "effective_config.json"),
        "sensor_columns": sensors,
        "mount_rows": [{"mount": m, "view": v} for m, v in mounts],
        "results": rows,
    }
    write_json(out / "comparison.json", comparison)
    _contact_sheet(tiles, sensors, mounts, out / "contact_sheet.png")
    (out / "report.md").write_text(_report(comparison), encoding="utf-8")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args(argv)
    print(
        run_helmet_hand_experiment(
            args.render_root, args.baseline_root, args.model, args.output, args.min_confidence
        )
    )


if __name__ == "__main__":
    main()

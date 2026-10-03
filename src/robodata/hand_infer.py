"""Optional MediaPipe 0.10.21 RGB hand inference into a new portable manifest.

Only RGB video and supplied tool masks are read for predictions/association.
Local world landmarks have a hand-centered origin, never a calibrated camera
origin. All raw hands and all video frames are retained, including failures to
associate a grasp. No ground-truth pose, calibration fit or renderer is used.
"""

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .observations import Detection, Keypoint, Manifest, read_mask, video_frames

PALM_INDICES = (0, 5, 9, 13, 17)
KEYPOINT_SEMANTICS = (
    "confidence=1 marks an accepted model prediction, not calibrated joint confidence; "
    "visible means finite in-image coordinate availability, not GT visibility; "
    "handedness scores describe handedness classification only"
)


def _absolute_paths(value, root: Path):
    """Resolve original manifest path fields without opening their contents."""
    if isinstance(value, dict):
        for key, item in value.items():
            if (key == "path" or key.endswith("_path")) and isinstance(item, str):
                value[key] = str((root / item).resolve())
            elif key.endswith("_paths") and isinstance(item, list):
                value[key] = [
                    str((root / name).resolve()) if isinstance(name, str) else name for name in item
                ]
            else:
                _absolute_paths(item, root)
    elif isinstance(value, list):
        for item in value:
            _absolute_paths(item, root)


def _coordinates(landmarks):
    # Retain unavailable raw coordinates as JSON null rather than invented zeros.
    return [
        [
            float(v) if v is not None and np.isfinite(v) else None
            for v in (point.x, point.y, point.z)
        ]
        for point in landmarks
    ]


def _raw_hands(result):
    hands = []
    for index, landmarks in enumerate(result.hand_landmarks):
        if len(landmarks) != 21:
            raise ValueError("Hand model must return 21 image landmarks per hand")
        world = (
            result.hand_world_landmarks[index] if index < len(result.hand_world_landmarks) else []
        )
        if world and len(world) != 21:
            raise ValueError("Hand model must return 21 local world landmarks per hand")
        categories = result.handedness[index] if index < len(result.handedness) else []
        hands.append(
            {
                "normalized_landmarks": _coordinates(landmarks),
                "local_world_landmarks_m": _coordinates(world),
                "handedness": [
                    {
                        "category_name": c.category_name,
                        "display_name": c.display_name,
                        "score": float(c.score),
                        "index": int(c.index),
                    }
                    for c in categories
                ],
            }
        )
    return hands


def _associate(hands, mask, width, height):
    if mask is None or not np.any(mask):
        return None, None
    rows, columns = np.nonzero(mask)
    tree = cKDTree(np.column_stack([columns, rows]))
    threshold = max(8.0, width * 0.04)
    selected, nearest = None, None
    for index, hand in enumerate(hands):
        anchors = []
        for key in PALM_INDICES:
            x, y, _ = hand["normalized_landmarks"][key]
            if x is not None and y is not None and 0 <= x < 1 and 0 <= y < 1:
                anchors.append([x * width, y * height])
        if not anchors:
            continue
        distance = float(np.min(tree.query(anchors)[0]))
        if distance <= threshold and (nearest is None or distance < nearest):
            selected, nearest = index, distance
    return selected, nearest


def _detection(hand, width, height):
    points = {}
    for index, (x, y, _) in enumerate(hand["normalized_landmarks"]):
        if x is None or y is None:
            continue
        points[str(index)] = Keypoint(
            uv=[x * width, y * height],
            confidence=1.0,
            visible=0 <= x < 1 and 0 <= y < 1,
        )
    categories = hand.get("handedness", [])
    category = max(categories, key=lambda c: c["score"])["category_name"] if categories else None
    return Detection(
        keypoints=points, handedness=category if category in {"Left", "Right"} else None
    )


def _timestamp_ms(timestamp):
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Frame timestamp must be finite and nonnegative")
    return int(round(timestamp * 1000))


def _software_provenance():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        try:
            run = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return run.stdout.strip() if run.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def infer_hands(model_path, manifest_path, output_dir, min_confidence=0.5) -> Path:
    """Write manifest.json and hand_detections.json in a fresh external directory.

    VIDEO timestamps are rounded from the supplied seconds to milliseconds.
    Collisions/nonmonotonic times raise rather than fabricating tracker times.
    The threshold configures detection/presence/tracking gates, never handedness
    or joint confidence. Tool/surface landmarks and landmark_source are retained.
    """
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("Hand confidence threshold must lie in [0,1]")
    path, model, out = (
        Path(manifest_path).resolve(),
        Path(model_path).resolve(),
        Path(output_dir).resolve(),
    )
    if out.exists():
        raise FileExistsError(f"Choose a new hand inference directory: {out}")
    if out.is_relative_to(path.parent) or path.parent.is_relative_to(out):
        raise ValueError("Hand output directory must be outside the original manifest directory")
    manifest_bytes = path.read_bytes()
    data = json.loads(manifest_bytes)
    cloned = copy.deepcopy(data)
    _absolute_paths(cloned, path.parent)
    timestamps = data.get("timestamps_sec")
    if timestamps is not None:
        if not timestamps:
            raise ValueError("Explicit timestamps must be nonempty")
        previous_sec, previous_ms = -1.0, -1
        for timestamp in timestamps:
            ms = _timestamp_ms(timestamp)
            if timestamp <= previous_sec or ms <= previous_ms:
                raise ValueError("Nonmonotonic timestamp or millisecond timestamp collision")
            previous_sec, previous_ms = timestamp, ms
    camera = Manifest.model_validate(data).camera
    input_directories = {Path(cloned["video_path"]).parent}
    for observation in cloned.get("observations", []):
        if observation.get("depth_path"):
            input_directories.add(Path(observation["depth_path"]).parent)
        for group in ("tool", "surface", "hand"):
            mask_path = observation.get(group, {}).get("mask_path")
            if mask_path:
                input_directories.add(Path(mask_path).parent)
    if any(out.is_relative_to(directory) for directory in input_directories):
        raise ValueError("Hand output directory must be outside the original input directories")
    original = {}
    for observation in cloned.get("observations", []):
        index = observation["frame_index"]
        if not isinstance(index, int) or index < 0 or index in original:
            raise ValueError("Invalid or duplicate observation frame index")
        original[index] = observation
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()

    # Keep the optional Tasks runtime out of core imports and the main environment.
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError(
            "Hand inference requires optional MediaPipe 0.10.21; use ROBODATA_HAND environment"
        ) from exc
    if mp.__version__ != "0.10.21":
        raise RuntimeError("Hand inference requires the supported MediaPipe version 0.10.21")
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    observations, raw_frames, actual_timestamps = [], [], []
    previous_ms = -1
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for index, timestamp, rgb in video_frames(
            Path(cloned["video_path"]),
            camera,
            timestamps,
        ):
            timestamp_ms = _timestamp_ms(timestamp)
            if timestamp_ms <= previous_ms:
                raise ValueError("Nonmonotonic timestamp or millisecond timestamp collision")
            previous_ms = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            hands = _raw_hands(landmarker.detect_for_video(image, timestamp_ms))
            observation = copy.deepcopy(original.get(index, {"frame_index": index}))
            tool = Detection.model_validate(observation.get("tool", {}))
            mask = read_mask(path.parent, tool.mask_path, rgb.shape[:2])
            selected, distance = _associate(hands, mask, camera.width, camera.height)
            prediction = (
                Detection()
                if selected is None
                else _detection(hands[selected], camera.width, camera.height)
            )
            # Preserve a supplied hard occlusion constraint while replacing predictions.
            prediction.occluded = observation.get("hand", {}).get("occluded", False)
            prediction.mask_path = observation.get("hand", {}).get("mask_path")
            observation["hand"] = prediction.model_dump(mode="json")
            observations.append(observation)
            actual_timestamps.append(timestamp)
            raw_frames.append(
                {
                    "frame_index": index,
                    "timestamp_sec": timestamp,
                    "timestamp_ms": timestamp_ms,
                    "hands": hands,
                    "selected_hand_index": selected,
                    "association_distance_px": distance,
                    "association_failure": "no_wrist_mcp_tool_mask_association"
                    if selected is None
                    else None,
                }
            )
    if any(index >= len(observations) for index in original):
        raise ValueError("Observation frame index is outside the decoded video")
    provenance = {
        "method": "mediapipe_hand_landmarker_video",
        "model_path": str(model),
        "model_sha256": model_hash,
        "mediapipe_version": mp.__version__,
        "min_confidence": min_confidence,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_path": str(path),
        "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "timestamp_source": "explicit" if timestamps is not None else "fps_assumed_constant",
        "timestamp_conversion": "round(seconds * 1000); reject collisions",
        "association": "nearest wrist/MCP to actual tool-mask pixels; no mask means no grasp",
        "association_threshold_px": max(8.0, camera.width * 0.04),
        "keypoint_semantics": KEYPOINT_SEMANTICS,
        "world_coordinate_frame": "model-local hand geometric center; meters; not camera frame",
        "raw_detections_path": str(out / "hand_detections.json"),
        "data_kind": data.get("data_kind", "real"),
        "software": _software_provenance(),
    }
    cloned["observations"] = observations
    cloned["timestamps_sec"] = actual_timestamps if timestamps is not None else None
    cloned.setdefault("provenance", {})["hand_source"] = provenance
    Manifest.model_validate(cloned)
    raw_text = json.dumps(
        {"provenance": provenance, "frames": raw_frames}, indent=2, allow_nan=False
    )
    manifest_text = json.dumps(cloned, indent=2, allow_nan=False)
    out.mkdir(parents=True, exist_ok=False)
    (out / "hand_detections.json").write_text(raw_text, encoding="utf-8")
    result_path = out / "manifest.json"
    result_path.write_text(manifest_text, encoding="utf-8")
    return result_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args(argv)
    print(infer_hands(args.model, args.manifest, args.output, args.min_confidence))


if __name__ == "__main__":
    main()

"""Use trained RGB landmarks in the same extraction contract."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from ..export import write_json
from ..observations import Detection, FrameInput, Keypoint, Manifest, video_frames
from .model import LandmarkNet
from .train import LearningConfig


class RGBDetector:
    def __init__(self, checkpoint_path, min_confidence=0.5, device="cpu"):
        if not 0 <= min_confidence <= 1:
            raise ValueError("Confidence threshold must lie in [0,1]")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported detector checkpoint")
        self.config = LearningConfig.model_validate(checkpoint["config"])
        self.provenance = checkpoint["provenance"]
        self.model = LandmarkNet(len(self.config.keypoint_names), self.config.width).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.device, self.min_confidence = device, min_confidence

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        size = self.config.image_size
        resized = cv2.resize(rgb, (size, size))
        x = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(
            self.device
        )
        with torch.inference_mode():
            maps, logits = self.model(x, return_visibility=True)
        flat = maps[0].flatten(1)
        values, indices = flat.max(1)
        scores = torch.minimum(values, torch.sigmoid(logits[0]))
        output = {"tool": Detection(), "surface": Detection()}
        for j, name in enumerate(self.config.keypoint_names):
            group, key = name.split(":", 1)
            index = int(indices[j])
            uv = (np.array([index % size, index // size]) + 0.5) * [w / size, h / size] - 0.5
            confidence = float(scores[j])
            output[group].keypoints[key] = Keypoint(
                uv=uv.tolist(), confidence=confidence, visible=confidence >= self.min_confidence
            )
        return output


def infer_manifest(checkpoint_path, manifest_path, output_path, min_confidence=0.5):
    path, out = Path(manifest_path).resolve(), Path(output_path).resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    detector = RGBDetector(checkpoint_path, min_confidence)
    original = {f.frame_index: f for f in manifest.observations}
    observations = []
    for i, _, rgb in video_frames(
        path.parent / manifest.video_path, manifest.camera, manifest.timestamps_sec
    ):
        f = original.get(i, FrameInput(frame_index=i)).model_copy(deep=True)
        detections = detector.predict(rgb)
        # Only detections are replaced; depth, masks and process annotations retain provenance.
        for group in ["tool", "surface"]:
            old = getattr(f, group)
            new = detections[group]
            new.occluded = old.occluded
            if old.mask_path:
                new.mask_path = str((path.parent / old.mask_path).resolve())
            setattr(f, group, new)
        if f.depth_path:
            f.depth_path = str((path.parent / f.depth_path).resolve())
        observations.append(f)
    manifest.video_path = str((path.parent / manifest.video_path).resolve())
    manifest.observations = observations
    constraint_source = manifest.landmark_source
    manifest.landmark_source = "rgb_landmark_network_uncalibrated_scores"
    manifest.provenance["detector"] = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "training": detector.provenance,
        "min_confidence": min_confidence,
        "occlusion_constraint_source": constraint_source,
    }
    write_json(out, manifest.model_dump(mode="json"))
    write_json(
        out.with_suffix(".detector.json"),
        manifest.provenance["detector"],
    )
    return out

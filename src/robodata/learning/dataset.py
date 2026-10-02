"""Semantic RGB annotations for supervised detector training."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class LandmarkDataset(Dataset):
    def __init__(self, annotation_path, names, size):
        path = Path(annotation_path).resolve()
        self.root = path.parent
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("data_kind") not in ["real", "synthetic"]:
            raise ValueError("Annotation data_kind must be real or synthetic")
        self.data_kind = data["data_kind"]
        self.samples = data["samples"]
        if not self.samples:
            raise ValueError("Empty landmark dataset")
        self.names, self.size = names, size
        self.image_paths = [(self.root / s["image_path"]).resolve() for s in self.samples]
        if len(set(self.image_paths)) != len(self.image_paths):
            raise ValueError("Duplicate training images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        bgr = cv2.imread(str(self.image_paths[index]))
        if bgr is None:
            raise FileNotFoundError(self.image_paths[index])
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        uv: NDArray[np.float32] = np.zeros((len(self.names), 2), np.float32)
        visible: NDArray[np.bool_] = np.zeros(len(self.names), bool)
        for j, name in enumerate(self.names):
            p = self.samples[index]["keypoints"].get(name)
            if p is None or not p.get("visible", True):
                continue
            point = np.asarray(p["uv"], float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("Landmark must be a finite pixel coordinate")
            if 0 <= point[0] < w and 0 <= point[1] < h:
                uv[j] = (point + 0.5) * [self.size / w, self.size / h] - 0.5
                visible[j] = True
        return {
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0),
            "uv": torch.from_numpy(uv),
            "visible": torch.from_numpy(visible),
        }

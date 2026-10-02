"""Train/reload the RGB detector with held-out validation and run provenance."""

import hashlib
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import Field, model_validator
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..export import write_json
from ..pipeline import software_provenance
from ..schema import Contract
from .dataset import LandmarkDataset
from .model import LandmarkNet, heatmap_loss, heatmap_targets


class LearningConfig(Contract):
    train_path: str
    val_path: str
    keypoint_names: list[str] = Field(min_length=1)
    image_size: int = Field(default=128, ge=16)
    width: int = Field(default=16, ge=4)
    epochs: int = Field(default=20, gt=0)
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    seed: int = 42
    device: Literal["cpu", "cuda", "auto"] = "cpu"

    @model_validator(mode="after")
    def names(self):
        if len(set(self.keypoint_names)) != len(self.keypoint_names):
            raise ValueError("Duplicate landmark names")
        if any(not n.startswith(("tool:", "surface:")) for n in self.keypoint_names):
            raise ValueError("Landmarks use tool:name or surface:name")
        if self.width % 4:
            raise ValueError("Network width must be divisible by four")
        return self


def train(config_path: Path | str, checkpoint_path: Path | str) -> dict:
    path, out = Path(config_path).resolve(), Path(checkpoint_path).resolve()
    if out.exists() or out.with_suffix(".history.json").exists():
        raise FileExistsError(f"Choose a new checkpoint path: {out}")
    cfg = LearningConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable")
    datasets = [
        LandmarkDataset(path.parent / name, cfg.keypoint_names, cfg.image_size)
        for name in [cfg.train_path, cfg.val_path]
    ]
    if set(datasets[0].image_paths) & set(datasets[1].image_paths):
        raise ValueError("Training/validation image leakage")
    generator = torch.Generator().manual_seed(cfg.seed)
    loaders = [
        DataLoader(ds, batch_size=cfg.batch_size, shuffle=(i == 0), generator=generator)
        for i, ds in enumerate(datasets)
    ]
    model = LandmarkNet(len(cfg.keypoint_names), cfg.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    history = []
    best = float("inf")
    out.parent.mkdir(parents=True, exist_ok=True)
    provenance = software_provenance(path)
    provenance["torch"] = str(torch.__version__)
    provenance["data_kind"] = {"train": datasets[0].data_kind, "val": datasets[1].data_kind}
    provenance["annotation_sha256"] = {
        name: hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
        for name in [cfg.train_path, cfg.val_path]
    }
    for epoch in range(cfg.epochs):
        metrics: dict[str, Any] = {"epoch": epoch + 1}
        for split, loader in zip(["train", "val"], loaders):
            model.train(split == "train")
            loss_sum, count, pixel_error, visible_count = 0.0, 0, 0.0, 0
            for batch in loader:
                rgb, uv, visible = [batch[k].to(device) for k in ["rgb", "uv", "visible"]]
                targets = heatmap_targets(uv, visible, cfg.image_size)
                with torch.set_grad_enabled(split == "train"):
                    maps, visibility_logits = model(rgb, return_visibility=True)
                    loss = heatmap_loss(
                        maps, targets, visible
                    ) + 0.1 * F.binary_cross_entropy_with_logits(visibility_logits, visible.float())
                    if split == "train":
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                loss_sum += float(loss.detach()) * len(rgb)
                count += len(rgb)
                peaks = maps.detach().flatten(2).argmax(2)
                pixels = torch.stack([peaks % cfg.image_size, peaks // cfg.image_size], dim=-1)
                pixel_error += float(torch.linalg.vector_norm(pixels - uv, dim=-1)[visible].sum())
                visible_count += int(visible.sum())
            metrics[f"{split}_loss"] = loss_sum / count
            metrics[f"{split}_keypoint_mae_resized_px"] = (
                pixel_error / visible_count if visible_count else None
            )
        history.append(metrics)
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            torch.save(
                {
                    "format_version": 1,
                    "state_dict": model.state_dict(),
                    "config": cfg.model_dump(),
                    "epoch": epoch + 1,
                    "metrics": metrics,
                    "provenance": provenance,
                },
                out,
            )
    result = {"history": history, "best_val_loss": best, "provenance": provenance}
    write_json(out.with_suffix(".history.json"), result)
    return result

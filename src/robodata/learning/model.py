"""Compact U-Net heatmaps plus an explicit landmark visibility head."""

import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1),
        nn.GroupNorm(4, b),
        nn.ReLU(),
        nn.Conv2d(b, b, 3, padding=1),
        nn.GroupNorm(4, b),
        nn.ReLU(),
    )


class LandmarkNet(nn.Module):
    def __init__(self, num_keypoints: int, width=16):
        super().__init__()
        if num_keypoints < 1 or width < 4 or width % 4:
            raise ValueError("Landmark count must be positive and width a multiple of four")
        self.enc1 = block(3, width)
        self.enc2 = block(width, width * 2)
        self.enc3 = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.heatmaps = nn.Conv2d(width, num_keypoints, 1)
        self.visibility = nn.Linear(width * 4, num_keypoints)

    def forward(self, rgb, return_visibility=False):
        a = self.enc1(rgb)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        d = self.dec2(
            torch.cat(
                [F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b],
                dim=1,
            )
        )
        e = self.dec1(
            torch.cat(
                [F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a],
                dim=1,
            )
        )
        maps = torch.sigmoid(self.heatmaps(e))
        if return_visibility:
            return maps, self.visibility(c.mean(dim=(-2, -1)))
        return maps


def heatmap_targets(uv, visible, size: int, sigma=1.5):
    yy, xx = torch.meshgrid(
        torch.arange(size, device=uv.device), torch.arange(size, device=uv.device), indexing="ij"
    )
    dx = xx - uv[..., 0, None, None]
    dy = yy - uv[..., 1, None, None]
    return torch.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)) * visible[..., None, None]


def heatmap_loss(pred, target, visible):
    weights = visible[..., None, None].to(pred.dtype)
    loss = (pred - target).square() * (1 + 20 * target) * weights
    return loss.sum() / (weights.sum().clamp_min(1) * pred.shape[-1] * pred.shape[-2])

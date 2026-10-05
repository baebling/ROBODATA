"""RGB planar localization against a calibrated reference, not full-scene SLAM."""

from typing import Any

import cv2
import numpy as np

from .geometry import estimate_pnp, inverse, validate_transform


def detect_surface_markers(image):
    """Detect the four colored calibration fixtures in a tool-free RGB reference.

    Color identity is a synthetic calibration aid, not general surface perception.
    No projected simulator keypoints or visibility flags are consumed.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hues = hsv[:, :, 0]
    saturated = (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 60)
    selections = [
        (hues < 10) | (hues > 170),
        (hues >= 40) & (hues <= 85),
        (hues >= 100) & (hues <= 135),
        (hues >= 15) & (hues <= 38),
    ]
    points = []
    for selected in selections:
        n, _, stats, centers = cv2.connectedComponentsWithStats(
            (selected & saturated).astype(np.uint8)
        )
        ids = [i for i in range(1, n) if 1 <= stats[i, cv2.CC_STAT_AREA] <= 500]
        if not ids:
            raise ValueError("Reference lacks four detectable colored surface markers")
        index = max(ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        points.append(centers[index])
    return np.asarray(points)


def pixels_to_surface(pixels, K, camera_surface):
    T = inverse(camera_surface)
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    directions = rays @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = -T[2, 3] / directions[:, 2]
        xyz = T[:3, 3] + directions * distance[:, None]
    if not np.isfinite(xyz).all() or np.any(distance <= 0):
        raise ValueError("Reference rays do not intersect the physical plane in front of camera")
    xyz[:, 2] = 0
    return xyz


def mask_scores(predicted, reference):
    a, b = np.asarray(predicted, bool), np.asarray(reference, bool)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Masks must have matching image dimensions")
    intersection, union = int((a & b).sum()), int((a | b).sum())
    return {
        "iou": intersection / union if union else None,
        "reference_pixels": int(b.sum()),
        "predicted_pixels": int(a.sum()),
        "false_positive_pixels": int((a & ~b).sum()),
        "false_negative_pixels": int((~a & b).sum()),
    }


class PlanarSurfaceMap:
    """Immutable reference features with per-frame checked localization.

    Each frame is matched to the reference, allowing recovery without copying a
    previous pose. YOLO masks gate current visible evidence, not hidden geometry.
    """

    def __init__(
        self,
        K,
        max_fb_error_px=1.5,
        max_reprojection_px=3.0,
        min_inliers=6,
        max_features=300,
        min_spread_px=20.0,
        max_lk_error=30.0,
    ):
        self.K = np.asarray(K, dtype=float)
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("Map needs finite 3x3 intrinsics")
        if min(max_fb_error_px, max_reprojection_px, min_spread_px, max_lk_error) <= 0:
            raise ValueError("Localization thresholds must be positive")
        if min_inliers < 4 or max_features < min_inliers:
            raise ValueError("Insufficient configured map geometry")
        self.max_fb = max_fb_error_px
        self.max_reprojection = max_reprojection_px
        self.min_inliers = min_inliers
        self.max_features = max_features
        self.min_spread = min_spread_px
        self.max_lk_error = max_lk_error
        self.gray = None
        self.pixels = None
        self.xyz = None

    def initialize(self, image, visible_mask, camera_surface):
        T = validate_transform(camera_surface)
        mask = np.asarray(visible_mask, dtype=np.uint8)
        if mask.shape != image.shape[:2]:
            raise ValueError("Reference mask resolution mismatch")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        interior = cv2.erode(mask, np.ones((3, 3), np.uint8))
        pixels = cv2.goodFeaturesToTrack(
            gray, self.max_features, 0.005, 3, mask=interior * 255, blockSize=3
        )
        if pixels is None or len(pixels) < self.min_inliers:
            raise ValueError("Reference has insufficient visible surface texture")
        uv = pixels.reshape(-1, 2)
        xyz = pixels_to_surface(uv, self.K, T)
        self.gray, self.pixels, self.xyz = gray, pixels, xyz
        return {"features": len(uv), "method": "known_geometry_RGB_reference_plane"}

    def update(self, image, visible_mask):
        diagnostic: dict[str, Any] = {"inliers": 0, "failure_reason": None}
        if self.gray is None:
            diagnostic["failure_reason"] = "reference_not_initialized"
            return None, diagnostic
        assert self.pixels is not None and self.xyz is not None
        mask = np.asarray(visible_mask, bool)
        if mask.shape != image.shape[:2] or image.shape[:2] != self.gray.shape:
            raise ValueError("Current frame/mask resolution mismatch")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        q, forward, error = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.pixels, None, winSize=(21, 21), maxLevel=3
        )
        if q is None:
            diagnostic["failure_reason"] = "optical_flow_unavailable"
            return None, diagnostic
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, q, None, winSize=(21, 21), maxLevel=3
        )
        if back is None:
            diagnostic["failure_reason"] = "reverse_flow_unavailable"
            return None, diagnostic
        uv = q.reshape(-1, 2)
        ref = self.pixels.reshape(-1, 2)
        good = (
            (forward.ravel() != 0)
            & (reverse.ravel() != 0)
            & (np.linalg.norm(back.reshape(-1, 2) - ref, axis=1) <= self.max_fb)
            & (error.ravel() <= self.max_lk_error)
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 1] >= 0)
            & (uv[:, 0] < mask.shape[1])
            & (uv[:, 1] < mask.shape[0])
        )
        ids = np.flatnonzero(good)
        xy = uv[ids].astype(int)
        good[ids] &= mask[xy[:, 1], xy[:, 0]]
        if good.sum() < self.min_inliers:
            diagnostic["failure_reason"] = "insufficient_current_visible_matches"
            return None, diagnostic
        ids = np.flatnonzero(good)
        _, selected = cv2.findHomography(ref[ids], uv[ids], cv2.RANSAC, self.max_reprojection)
        if selected is None:
            diagnostic["failure_reason"] = "homography_failed"
            return None, diagnostic
        ids = ids[selected.ravel() != 0]
        diagnostic["inliers"] = len(ids)
        if len(ids) < self.min_inliers or np.any(np.ptp(uv[ids], axis=0) < self.min_spread):
            diagnostic["failure_reason"] = "insufficient_spatially_spread_inliers"
            return None, diagnostic
        try:
            estimate = estimate_pnp(
                self.xyz[ids], uv[ids], self.K, max_error_px=self.max_reprojection
            )
        except ValueError as exc:
            diagnostic["failure_reason"] = str(exc)
            return None, diagnostic
        diagnostic["residual_px"] = estimate.residual
        return estimate.transform, diagnostic

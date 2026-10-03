"""Simulator-only visibility diagnostics. Never use these references as model inputs."""

from itertools import groupby

import cv2
import numpy as np

from .geometry import inverse, validate_transform
from .observations import Camera


def _roi_inputs(matrix, camera_surface, shape, bounds):
    camera = Camera(matrix=np.asarray(matrix).tolist(), width=shape[1], height=shape[0])
    transform = validate_transform(camera_surface)
    box = np.asarray(bounds, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("ROI bounds must be finite xmin, ymin, xmax, ymax")
    return np.asarray(camera.matrix), transform, box


def project_roi(matrix, camera_surface, shape, bounds):
    """Intersect integer pixel-center rays with a fixed rectangle on surface Z=0."""
    K, C_S, box = _roi_inputs(matrix, camera_surface, shape, bounds)
    S_C = inverse(C_S)
    y, x = np.indices(shape)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(K).T
    directions = rays @ S_C[:3, :3].T
    origin = S_C[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = -origin[2] / directions[:, :, 2]
        points = origin + distance[:, :, None] * directions
    mask = (
        np.isfinite(distance)
        & (distance > 0)
        & (points[:, :, 0] >= box[0])
        & (points[:, :, 0] <= box[2])
        & (points[:, :, 1] >= box[1])
        & (points[:, :, 1] <= box[3])
    )
    # Ray Z is one in OpenCV, so the intersection parameter is axial camera depth.
    return mask, distance


def roi_out_of_view_fraction(matrix, camera_surface, shape, bounds):
    """Continuous projected area clipped to pixel footprint; unknown if behind camera."""
    K, C_S, b = _roi_inputs(matrix, camera_surface, shape, bounds)
    corners = (
        np.array([[b[0], b[1], 0, 1], [b[2], b[1], 0, 1], [b[2], b[3], 0, 1], [b[0], b[3], 0, 1]])
        @ C_S.T
    )
    if np.any(corners[:, 2] <= 0):
        return None
    projected = corners[:, :3] @ K.T
    polygon = (projected[:, :2] / projected[:, 2:3]).astype(np.float32)
    area = abs(cv2.contourArea(polygon))
    if area < 1e-9:
        return None
    h, w = shape
    image = np.array(
        [[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]], dtype=np.float32
    )
    clipped, _ = cv2.intersectConvexConvex(polygon, image)
    return float(np.clip(1 - clipped / area, 0, 1))


def occlusion_metrics(roi, expected_depth, labels, depth, tool_ids, tolerance_m):
    roi = np.asarray(roi, dtype=bool)
    expected, labels, depth = map(np.asarray, [expected_depth, labels, depth])
    if roi.ndim != 2 or any(a.shape != roi.shape for a in [expected, labels, depth]):
        raise ValueError("ROI, labels and depths must be aligned 2D arrays")
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Invalid occlusion depth tolerance")
    tool = np.isin(labels, tool_ids)
    usable = np.isfinite(depth) & (depth > 0) & (depth < 1e6)
    if np.any(roi & (~np.isfinite(expected) | (expected <= 0))):
        raise ValueError("ROI needs positive camera-axial reference depth")
    front = roi & usable & (depth < expected - tolerance_m)
    # Surface landmark spheres carry ID 3 and are an evaluation fixture, not an occluder.
    occluded = front & (labels != 3)
    tool_occluded = occluded & tool
    n, unknown = int(roi.sum()), int((roi & ~usable).sum())
    denominator = n if n and not unknown else None
    return {
        "roi_pixels": n,
        "invalid_roi_depth_pixels": unknown,
        "tool_image_occupancy": float(tool.mean()),
        "part_image_occupancy": {str(i): float((labels == i).mean()) for i in tool_ids},
        "tool_occluded_pixels": int(tool_occluded.sum()),
        "total_occluded_pixels": int(occluded.sum()),
        "tool_roi_occlusion": int(tool_occluded.sum()) / denominator if denominator else None,
        "total_roi_occlusion": int(occluded.sum()) / denominator if denominator else None,
        "hand_image_occupancy": float((labels == 6).mean()),
        "hand_roi_occlusion": int((occluded & (labels == 6)).sum()) / denominator
        if denominator
        else None,
    }


def validity_summary(valid, full_pose_valid, fps):
    """Frame durations for a complete, constant-FPS sequence; gaps are not filled."""
    if len(valid) != len(full_pose_valid) or not valid or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Validity summary needs aligned nonempty frames and positive FPS")
    if any(full and not base for base, full in zip(valid, full_pose_valid, strict=True)):
        raise ValueError("Full pose cannot be valid without a valid state")
    runs = [(bool(key), len(list(group))) for key, group in groupby(valid)]
    return {
        "valid_fraction": sum(valid) / len(valid),
        "full_pose_fraction": sum(full_pose_valid) / len(valid),
        "axis_only_frames": sum(base and not full for base, full in zip(valid, full_pose_valid)),
        "longest_valid_segment_sec": max((n for key, n in runs if key), default=0) / fps,
        "invalid_gap_lengths_sec": [n / fps for key, n in runs if not key],
        "duration_convention": "each rendered frame represents 1/fps seconds",
    }

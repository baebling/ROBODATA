"""SE(3), calibrated PnP and robust metric plane estimation."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class PoseEstimate:
    transform: np.ndarray
    residual: float
    confidence: float
    method: str


@dataclass
class PlaneEstimate:
    origin: np.ndarray
    normal: np.ndarray
    transform: np.ndarray
    residual: float
    inlier_ratio: float


def validate_transform(value) -> np.ndarray:
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("SE(3) must be a finite 4x4 matrix")
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous row")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(R), 1, atol=1e-5
    ):
        raise ValueError("Rotation must be orthonormal and right-handed")
    return T


def inverse(value) -> np.ndarray:
    T = validate_transform(value)
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return result


def unit(value) -> np.ndarray:
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-8:
        raise ValueError("Direction must be a finite nonzero 3D vector")
    return v / np.linalg.norm(v)


def frame_from_axis(origin, z_axis, x_hint) -> np.ndarray:
    z = unit(z_axis)
    hint = np.asarray(x_hint, dtype=float)
    x = unit(hint - z * np.dot(hint, z))
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = origin
    return validate_transform(T)


def estimate_pnp(points3d, pixels, K, distortion=None, max_error_px=4.0) -> PoseEstimate:
    xyz = np.asarray(points3d, dtype=np.float64)
    uv = np.asarray(pixels, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or uv.shape != (len(xyz), 2):
        raise ValueError("PnP needs corresponding Nx3 and Nx2 points")
    if len(xyz) < 4 or not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("PnP needs at least four visible finite landmarks")
    rank = np.linalg.matrix_rank(xyz - xyz.mean(axis=0), tol=1e-7)
    if rank < 2:
        raise ValueError("Collinear PnP geometry")
    if rank == 2:
        result = cv2.solvePnPGeneric(xyz, uv, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not result[0]:
            raise ValueError("Planar PnP failed")
        candidates = list(zip(result[1], result[2]))
        # IPPE has a frontal 180-degree rotation degeneracy in some OpenCV versions.
        ok, rv, tv = cv2.solvePnP(xyz, uv, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            candidates.append((rv, tv))
        inlier_ratio = 1.0
    else:
        if len(xyz) < 6:
            raise ValueError("Nonplanar PnP needs at least six landmarks")
        ok, rv, tv, inliers = cv2.solvePnPRansac(
            xyz,
            uv,
            K,
            dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(max_error_px),
            iterationsCount=100,
            confidence=0.99,
        )
        if not ok or inliers is None or len(inliers) < 6:
            raise ValueError("PnP RANSAC has insufficient inliers")
        ids = inliers.ravel()
        rv, tv = cv2.solvePnPRefineLM(xyz[ids], uv[ids], K, dist, rv, tv)
        candidates = [(rv, tv)]
        inlier_ratio = len(ids) / len(xyz)
    best = None
    for rv, tv in candidates:
        R, _ = cv2.Rodrigues(rv)
        if np.any((xyz @ R.T + tv.reshape(1, 3))[:, 2] <= 0):
            continue
        projected, _ = cv2.projectPoints(xyz, rv, tv, K, dist)
        residual = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - uv) ** 2, axis=1))))
        if best is None or residual < best[0]:
            best = (residual, R, tv)
    if best is None or best[0] > max_error_px:
        raise ValueError("PnP cheirality or reprojection threshold failed")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = best[1], best[2].ravel()
    return PoseEstimate(
        T, best[0], float(inlier_ratio * np.exp(-best[0] / max_error_px)), "calibrated_pnp"
    )


def fit_plane(
    points, threshold_m=0.01, min_inlier_ratio=0.7, seed=0, iterations=200
) -> PlaneEstimate:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Plane points must have shape Nx3")
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 6 or threshold_m <= 0:
        raise ValueError("Plane requires at least six finite points and a positive threshold")
    rng = np.random.default_rng(seed)
    if len(pts) > 4096:
        pts = pts[rng.choice(len(pts), 4096, replace=False)]
    best: NDArray[np.bool_] = np.zeros(len(pts), dtype=bool)
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-10:
            continue
        mask = np.abs((pts - a) @ unit(normal)) < threshold_m
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 6 or best.mean() < min_inlier_ratio:
        raise ValueError("Surface is sparse, collinear or insufficiently planar")
    cloud = pts[best]
    origin = cloud.mean(axis=0)
    _, singular, V = np.linalg.svd(cloud - origin, full_matrices=False)
    if singular[1] < 1e-7:
        raise ValueError("Surface points are collinear")
    normal = V[-1]
    if np.dot(normal, -origin) < 0:
        normal = -normal
    hint = np.array([1.0, 0, 0])
    if abs(normal @ hint) > 0.95:
        hint = np.array([0.0, 1, 0])
    T = frame_from_axis(origin, normal, hint)
    return PlaneEstimate(
        origin,
        normal,
        T,
        float(np.sqrt(np.mean(((cloud - origin) @ normal) ** 2))),
        float(best.mean()),
    )


def backproject(pixels, depth_m, K) -> np.ndarray:
    uv = np.asarray(pixels, dtype=float)
    K = np.asarray(K, dtype=float)
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    return rays * np.asarray(depth_m, dtype=float).reshape(-1, 1)

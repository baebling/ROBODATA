"""Nominal sensor optics and a rigid helmet mount for synthetic viewpoint studies."""

from copy import deepcopy

import numpy as np


def scenario_camera_config(cfg, scenario):
    result = deepcopy(cfg)
    if "sensor" in scenario:
        profile = deepcopy(cfg["sensor_profiles"][scenario["sensor"]])
        result.update(width=profile["width"], height=profile["height"], sensor_profile=profile)
    return result


def camera_intrinsics(cfg):
    w, h = cfg["width"], cfg["height"]
    profile = cfg.get("sensor_profile")
    if profile is None:
        fx = fy = cfg["camera_focal_px"]
    elif "diagonal_fov_deg" in profile:
        angle = profile["diagonal_fov_deg"]
        if not np.isfinite(angle) or not 0 < angle < 179:
            raise ValueError("Invalid nominal diagonal field of view")
        fx = fy = np.hypot(w, h) / (2 * np.tan(np.radians(angle / 2)))
    else:
        angles = np.array([profile["horizontal_fov_deg"], profile["vertical_fov_deg"]])
        if not np.isfinite(angles).all() or np.any((angles <= 0) | (angles >= 179)):
            raise ValueError("Invalid nominal sensor field of view")
        fx, fy = np.array([w, h]) / (2 * np.tan(np.radians(angles / 2)))
    return np.array([[fx, 0, (w - 1) / 2], [0, fy, (h - 1) / 2], [0, 0, 1]])


def blender_pixel_aspect(K):
    # Blender clamps each component to >= 1; normalize both, not just one axis.
    ratio = K[0, 0] / K[1, 1]
    return max(1.0, 1 / ratio), max(1.0, ratio)


def helmet_camera_pose(cfg, scenario, i):
    mount = cfg["helmet_mount"]
    phase = 2 * np.pi * i / max(1, cfg["frames"] - 1)
    head = np.asarray(scenario["head_position_surface_m"], dtype=float).copy()
    pitch = float(scenario["helmet_pitch_deg"])
    yaw = float(scenario.get("helmet_yaw_deg", 0))
    if scenario["camera_motion"] == "moving":
        head += mount["bob_m"] * np.array([np.sin(phase), np.sin(2 * phase), np.sin(phase)])
        yaw += mount["yaw_amplitude_deg"] * np.sin(phase)
        pitch += mount["pitch_amplitude_deg"] * np.sin(2 * phase)
    yaw, pitch = np.radians([yaw, pitch])
    direction = np.array(
        [np.sin(yaw) * np.cos(pitch), -np.sin(pitch), -np.cos(yaw) * np.cos(pitch)]
    )
    right = np.array([np.cos(yaw), 0, np.sin(yaw)])
    up = np.cross(right, direction)
    position = head + direction * mount["forward_offset_m"] + up * mount["up_offset_m"]
    if head.shape != (3,) or not np.isfinite(position).all() or direction[2] >= 0:
        raise ValueError("Helmet must face toward the work surface")
    return position, direction

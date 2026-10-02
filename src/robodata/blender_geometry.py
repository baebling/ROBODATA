"""Finite, known H-beam geometry diagnostics, not perception or spray physics."""

import numpy as np


def surface_frame_identity(cfg) -> str:
    defaults = {"top_flange": "synthetic_hbeam_top_patch", "web_front": "synthetic_hbeam_web_front"}
    face = cfg.get("work_face", "top_flange")
    if face not in defaults:
        raise ValueError("Unknown work face")
    identity = cfg.get("surface_frame_id", defaults[face])
    if (
        not isinstance(identity, str)
        or not identity.strip()
        or (identity in defaults.values() and identity != defaults[face])
    ):
        raise ValueError("Invalid work face frame identity")
    return identity


def _dimensions(cfg):
    names = [
        "surface_height_m",
        "beam_length_m",
        "beam_width_m",
        "beam_height_m",
        "flange_thickness_m",
        "web_thickness_m",
    ]
    values = np.asarray([cfg[name] for name in names], dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values <= 0)
        or values[4] * 2 >= values[3]
        or values[5] >= values[2]
        or values[0] <= values[3]
    ):
        raise ValueError("Invalid H-beam dimensions")
    return values


def work_surface_transform(cfg) -> np.ndarray:
    height, _, _, bh, _, wt = _dimensions(cfg)
    face = cfg.get("work_face", "top_flange")
    S = np.eye(4)
    if face == "top_flange":
        S[:3, 3] = [0, 0, height]
    elif face == "web_front":
        S[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        S[:3, 3] = [0, -wt / 2, height - bh / 2]
    else:
        raise ValueError("Unknown work face")
    return S


def beam_boxes(cfg):
    height, length, width, bh, ft, wt = _dimensions(cfg)
    return [
        (
            "top_flange",
            np.array([-length / 2, -width / 2, height - ft]),
            np.array([length / 2, width / 2, height]),
        ),
        (
            "web",
            np.array([-length / 2, -wt / 2, height - bh + ft]),
            np.array([length / 2, wt / 2, height - ft]),
        ),
        (
            "bottom_flange",
            np.array([-length / 2, -width / 2, height - bh]),
            np.array([length / 2, width / 2, height - bh + ft]),
        ),
    ]


def _ray_box(origin, axis, low, high):
    near, far = 0.0, float("inf")
    for j in range(3):
        if abs(axis[j]) < 1e-10:
            if origin[j] < low[j] or origin[j] > high[j]:
                return None
            continue
        a, b = sorted([(low[j] - origin[j]) / axis[j], (high[j] - origin[j]) / axis[j]])
        near, far = max(near, a), min(far, b)
        if near > far:
            return None
    return near


def nozzle_relation(cfg, point_surface, axis_surface) -> dict:
    """Intersect a unit nozzle ray with the selected face and all three beam solids."""
    point, axis = np.asarray(point_surface, dtype=float), np.asarray(axis_surface, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Invalid nozzle point")
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis) - 1) > 1e-6:
        raise ValueError("Invalid unit nozzle axis")
    S = work_surface_transform(cfg)
    _, length, width, bh, ft, _ = _dimensions(cfg)
    face = cfg.get("work_face", "top_flange")
    half_y = (bh - 2 * ft) / 2 if face == "web_front" else width / 2
    plane_point = None
    if abs(axis[2]) >= 1e-10:
        t = -point[2] / axis[2]
        if t >= 0:
            plane_point = point + t * axis
    within = bool(
        plane_point is not None
        and abs(plane_point[0]) <= length / 2 + 1e-9
        and abs(plane_point[1]) <= half_y + 1e-9
    )
    origin_world, axis_world = S[:3, :3] @ point + S[:3, 3], S[:3, :3] @ axis
    hits = []
    for part, low, high in beam_boxes(cfg):
        distance = _ray_box(origin_world, axis_world, low, high)
        if distance is not None:
            hits.append((distance, part))
    first = min(hits) if hits else None
    distance, part = first if first is not None else (None, None)
    hit_surface = None if distance is None else point + distance * axis
    selected = "web" if face == "web_front" else "top_flange"
    if within and part == selected and hit_surface is not None and abs(hit_surface[2]) < 1e-8:
        status = "work_face_hit"
    elif face == "web_front" and part in {"top_flange", "bottom_flange"}:
        status = "flange_blocked"
    elif abs(axis[2]) < 1e-10:
        status = "parallel"
    elif plane_point is None:
        status = "away"
    else:
        status = "miss"
    return {
        "work_face": face,
        "status": status,
        "ray_plane_intersection_surface_m": None if plane_point is None else plane_point.tolist(),
        "within_work_face": within,
        "first_beam_part": part,
        "first_hit_surface_m": None if hit_surface is None else hit_surface.tolist(),
        "ray_distance_m": None if distance is None else float(distance),
    }

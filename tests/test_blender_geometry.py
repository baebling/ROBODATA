"""Hand-derived work-frame and finite H-beam ray expectations."""

import importlib
import importlib.util

import numpy as np
import pytest


def geometry():
    assert importlib.util.find_spec("robodata.blender_geometry") is not None, (
        "Missing beam geometry"
    )
    return importlib.import_module("robodata.blender_geometry")


def beam():
    return {
        "work_face": "web_front",
        "surface_height_m": 0.8,
        "beam_length_m": 1.8,
        "beam_width_m": 0.4,
        "beam_height_m": 0.4,
        "flange_thickness_m": 0.02,
        "web_thickness_m": 0.012,
    }


def test_web_frame_has_fixed_face_center_and_right_handed_axes():
    S = geometry().work_surface_transform(beam())
    np.testing.assert_allclose(S[:3, 3], [0, -0.006, 0.6])
    np.testing.assert_allclose(S[:3, :3], [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.linalg.det(S[:3, :3]) == pytest.approx(1)


def test_legacy_top_frame_keeps_original_coordinates():
    cfg = beam()
    cfg.pop("work_face")
    S = geometry().work_surface_transform(cfg)
    np.testing.assert_allclose(S[:3, :3], np.eye(3))
    np.testing.assert_allclose(S[:3, 3], [0, 0, 0.8])


def test_normal_spray_hits_web_at_perpendicular_projection():
    result = geometry().nozzle_relation(beam(), [0, 0, 0.32], [0, 0, -1])
    assert result["status"] == "work_face_hit"
    assert result["first_beam_part"] == "web"
    np.testing.assert_allclose(result["ray_plane_intersection_surface_m"], [0, 0, 0])
    np.testing.assert_allclose(result["first_hit_surface_m"], [0, 0, 0], atol=1e-12)
    assert result["ray_distance_m"] == pytest.approx(0.32)


def test_oblique_spray_intersection_is_not_orthogonal_projection():
    result = geometry().nozzle_relation(beam(), [0, 0, 0.32], [0.5, 0, -np.sqrt(0.75)])
    assert result["status"] == "work_face_hit"
    np.testing.assert_allclose(
        result["ray_plane_intersection_surface_m"], [0.18475208614, 0, 0], atol=1e-10
    )
    assert result["ray_distance_m"] == pytest.approx(0.36950417228)


def test_top_flange_blocks_spray_before_infinite_web_plane():
    result = geometry().nozzle_relation(beam(), [0, 0.1, 0.32], [0, 0.5, -np.sqrt(0.75)])
    assert result["status"] == "flange_blocked"
    assert result["first_beam_part"] == "top_flange"
    assert not result["within_work_face"]
    np.testing.assert_allclose(result["first_hit_surface_m"], [0, 0.18, 0.18143593539])
    assert result["ray_distance_m"] == pytest.approx(0.16)


def test_bottom_flange_blocks_downward_spray():
    result = geometry().nozzle_relation(beam(), [0, -0.1, 0.32], [0, -0.5, -np.sqrt(0.75)])
    assert result["status"] == "flange_blocked"
    assert result["first_beam_part"] == "bottom_flange"


def test_spray_past_beam_end_is_not_a_valid_web_hit():
    result = geometry().nozzle_relation(beam(), [1.1, 0, 0.32], [0, 0, -1])
    assert result["status"] == "miss"
    assert not result["within_work_face"]
    assert result["first_hit_surface_m"] is None
    assert result["ray_distance_m"] is None


@pytest.mark.parametrize("axis,status", [([1, 0, 0], "parallel"), ([0, 0, 1], "away")])
def test_nonforward_spray_does_not_invent_surface_intersection(axis, status):
    result = geometry().nozzle_relation(beam(), [0, 0, 0.32], axis)
    assert result["status"] == status
    assert result["ray_plane_intersection_surface_m"] is None


@pytest.mark.parametrize("axis", [[0, 0, 0], [0, 0, -2], [0, float("nan"), -1]])
def test_invalid_spray_axis_is_rejected(axis):
    with pytest.raises(ValueError, match="axis"):
        geometry().nozzle_relation(beam(), [0, 0, 0.32], axis)


def test_unknown_face_is_not_silently_top_flange():
    cfg = beam()
    cfg["work_face"] = "unknown"
    with pytest.raises(ValueError, match="face"):
        geometry().work_surface_transform(cfg)


def test_impossible_flange_geometry_is_rejected():
    cfg = beam()
    cfg["flange_thickness_m"] = 0.3
    with pytest.raises(ValueError, match="dimensions"):
        geometry().work_surface_transform(cfg)


def test_web_default_identity_cannot_be_the_top_flange_identity():
    assert hasattr(geometry(), "surface_frame_identity"), "Missing face-specific identity"
    assert geometry().surface_frame_identity(beam()) == "synthetic_hbeam_web_front"
    cfg = beam()
    cfg.pop("work_face")
    assert geometry().surface_frame_identity(cfg) == "synthetic_hbeam_top_patch"


@pytest.mark.parametrize(
    "face,identity",
    [
        ("web_front", "synthetic_hbeam_top_patch"),
        ("top_flange", "synthetic_hbeam_web_front"),
    ],
)
def test_face_rejects_other_faces_reserved_identity(face, identity):
    assert hasattr(geometry(), "surface_frame_identity"), "Missing face-specific identity"
    cfg = beam()
    cfg["work_face"] = face
    cfg["surface_frame_id"] = identity
    with pytest.raises(ValueError, match="identity"):
        geometry().surface_frame_identity(cfg)

"""Occlusion uses a fixed amodal physical ROI, not a changing visible mask."""

import importlib

import numpy as np
import pytest


def module():
    return importlib.import_module("robodata.blender_occlusion")


def frontal():
    transform = np.diag([1.0, -1, -1, 1])
    transform[2, 3] = 2
    return [[2, 0, 1], [0, 2, 1], [0, 0, 1]], transform


def test_amodal_roi_includes_hidden_pixels_with_camera_axial_depth():
    K, transform = frontal()
    mask, depth = module().project_roi(K, transform, (3, 3), [-0.6, -0.6, 0.6, 0.6])
    assert mask.sum() == 1
    assert mask[1, 1]
    assert depth[1, 1] == pytest.approx(2)


def test_first_hit_depth_separates_occupancy_from_tool_occlusion():
    mask = np.ones((2, 3), bool)
    expected = np.full((2, 3), 2.0)
    labels = np.array([[4, 5, 2], [6, 3, 1]])
    depth = np.array([[1, 3, 1], [1, 2, 1]], dtype=float)
    result = module().occlusion_metrics(mask, expected, labels, depth, [2, 4, 5], 0.002)
    assert result["tool_image_occupancy"] == pytest.approx(3 / 6)
    assert result["tool_roi_occlusion"] == pytest.approx(2 / 6)
    assert result["total_roi_occlusion"] == pytest.approx(4 / 6)
    assert result["tool_occluded_pixels"] == 2


def test_unknown_roi_depth_is_not_reported_as_unoccluded():
    result = module().occlusion_metrics(
        np.ones((1, 2), bool),
        np.full((1, 2), 2.0),
        np.array([[4, 3]]),
        np.array([[np.nan, 2]]),
        [2, 4, 5],
        0.002,
    )
    assert result["tool_roi_occlusion"] is None
    assert result["invalid_roi_depth_pixels"] == 1


def test_empty_roi_is_unknown_not_zero_occlusion():
    result = module().occlusion_metrics(
        np.zeros((1, 2), bool),
        np.ones((1, 2)),
        np.zeros((1, 2), int),
        np.ones((1, 2)),
        [2, 4, 5],
        0.002,
    )
    assert result["tool_roi_occlusion"] is None
    assert result["roi_pixels"] == 0


def test_roi_out_of_view_fraction_is_separate_from_occlusion():
    K, transform = frontal()
    # ROI projects to [-1, 3]^2, image spans [-0.5, 2.5]^2: 9/16 in view.
    value = module().roi_out_of_view_fraction(K, transform, (3, 3), [-2, -2, 2, 2])
    assert value == pytest.approx(7 / 16)


def test_validity_summary_preserves_failure_gaps_and_pose_distinction():
    result = module().validity_summary(
        [True, True, False, True, False],
        [False, True, False, True, False],
        10,
    )
    assert result["valid_fraction"] == pytest.approx(3 / 5)
    assert result["full_pose_fraction"] == pytest.approx(2 / 5)
    assert result["longest_valid_segment_sec"] == pytest.approx(0.2)
    assert result["invalid_gap_lengths_sec"] == [0.1, 0.1]


def test_hopper_scaling_preserves_attachment_and_other_parts():
    from robodata.blender_geometry import scenario_tool_parts

    cfg = {
        "hopper_attachment_tool_m": [0, -0.04, -0.155],
        "proxy_parts": [
            {"name": "body", "group": "body", "center": [0, 0, -0.11], "size": [0.12, 0.08, 0.2]},
            {
                "name": "hopper",
                "group": "hopper",
                "center": [0, -0.18, -0.155],
                "length": 0.28,
                "radius": 0.035,
                "radius_top": 0.125,
            },
        ],
    }
    parts = scenario_tool_parts(cfg, {"hopper_scale": 1.25})
    assert parts[0] == cfg["proxy_parts"][0]
    assert cfg["proxy_parts"][1]["length"] == 0.28
    assert parts[1]["length"] == pytest.approx(0.35)
    assert parts[1]["center"][1] + parts[1]["length"] / 2 == pytest.approx(-0.04)
    assert parts[1]["radius_top"] == pytest.approx(0.15625)
    with pytest.raises(ValueError):
        scenario_tool_parts(cfg, {"hopper_scale": 0})

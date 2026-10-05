import numpy as np
import pytest
import yaml

from robodata.config import load_config
from robodata.observations import Camera, Manifest
from robodata.pipeline import run_episode
from robodata.synthetic import create_demo


def bridge():
    from robodata import vggt_bridge

    return vggt_bridge


def test_similarity_recovers_metric_surface_not_camera_centroid():
    mod = bridge()
    measured = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    # X_surface = 2 * R * X_model + [1, 2, 3].
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    model = (measured - [1, 2, 3]) @ rotation / 2
    fit = mod.fit_metric_alignment(model, measured, 0.001)
    assert fit.scale == pytest.approx(2)
    assert np.allclose(fit.rotation, rotation)
    assert np.allclose(fit.translation, [1, 2, 3])
    extrinsic = np.c_[np.eye(3), [0, 0, 5]]
    pose = mod.metric_camera_surface(extrinsic, fit)
    assert np.allclose(pose[:3, :3], rotation.T)
    assert np.allclose(pose[:3, 3], [-2, 1, 7])
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)


@pytest.mark.parametrize("kind", ["collinear", "deformed", "nan"])
def test_alignment_rejects_unobservable_or_inconsistent_geometry(kind):
    measured = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    model = measured.copy()
    if kind == "collinear":
        model[:, 1] = 0
    elif kind == "deformed":
        model[0, 0] += 1
    else:
        model[0, 0] = np.nan
    with pytest.raises(ValueError):
        bridge().fit_metric_alignment(model, measured, 0.001)


def test_crop_affine_accounts_for_resize_rounding_and_pixel_centers():
    affine, shape = bridge().crop_affine(1920, 1080)
    assert shape == (294, 518)
    uv = affine @ [959.5, 539.5, 1]
    assert np.allclose(uv, [258.5, 146.5, 1])
    portrait, shape = bridge().crop_affine(1080, 1920)
    assert shape == (518, 518)
    assert np.allclose(portrait @ [539.5, 959.5, 1], [258.5, 258.5, 1])


def test_raw_depth_confidence_is_not_a_probability():
    mod = bridge()
    K = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], float)
    E = np.c_[np.eye(3), [0, 0, 1]]
    depth = np.full((3, 3), 3.0)
    confidence = np.full((3, 3), 12.0)
    assert np.allclose(mod.sample_world_point([1, 1], depth, confidence, K, E, 5), [0, 0, 2])
    confidence[1, 1] = 0
    with pytest.raises(ValueError):
        mod.sample_world_point([1, 1], depth, confidence, K, E, 5)
    with pytest.raises(ValueError):
        mod.sample_world_point([-1, 1], depth, confidence, K, E, 0)


def test_vggt_cache_pipeline_preserves_sampling_gaps_and_refuses_wrong_video(tmp_path):
    inputs = create_demo(tmp_path / "inputs", frames=8, occluded=[])
    cfg_path = inputs / "rgb.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["surface_method"] = "vggt"
    cfg["vggt_surface_path"] = "surface.json"
    cfg_path.write_text(yaml.safe_dump(cfg))
    manifest_path = inputs / "manifest.json"
    manifest = Manifest.model_validate_json(manifest_path.read_text())
    # Analytic geometry fixture only, not a new simulation/perception experiment.
    surface = np.eye(4)
    surface[:3, :3] = np.diag([1, -1, -1])
    surface[:3, 3] = [0, 0, 1.0]
    mod = bridge()
    cache = mod.SurfaceCache(
        video_sha256=mod.file_sha256(inputs / manifest.video_path),
        camera=manifest.camera,
        data_kind=manifest.data_kind,
        calibration_source="analytic_fixture",
        surface_frame_id=cfg["surface_frame_id"],
        frames=[
            mod.SurfaceSample(
                frame_index=0,
                timestamp_sec=0,
                estimate=mod.surface_estimate(surface, 0, 0.01, "analytic_fixture"),
            )
        ],
        provenance={"model": "analytic_fixture_not_vggt_inference"},
    )
    cache_path = inputs / "surface.json"
    cache_path.write_text(cache.model_dump_json())
    episode = run_episode(cfg_path, tmp_path / "out")
    assert episode.frames[0].quality.valid
    assert episode.frames[0].canonical.standoff_distance_m == pytest.approx(0.3, abs=1e-4)
    assert all(not f.quality.valid for f in episode.frames[1:])
    assert all("sample" in f.quality.failure_reason.lower() for f in episode.frames[1:])
    assert not episode.frames[0].action.valid
    cache.video_sha256 = "0" * 64
    cache_path.write_text(cache.model_dump_json())
    with pytest.raises(ValueError, match="video"):
        run_episode(cfg_path, tmp_path / "wrong")

    cache.video_sha256 = mod.file_sha256(inputs / manifest.video_path)
    cache.surface_frame_id = "different_physical_web"
    cache_path.write_text(cache.model_dump_json())
    with pytest.raises(ValueError, match="frame"):
        run_episode(cfg_path, tmp_path / "wrong_frame")


def test_body_pose_is_transformed_to_measured_functional_nozzle(tmp_path):
    inputs = create_demo(tmp_path / "inputs", frames=8, occluded=[])
    path = inputs / "rgb.yaml"
    cfg = yaml.safe_load(path.read_text())
    T = np.eye(4)
    T[2, 3] = 0.1
    cfg["tool"]["T_body_tool"] = T.tolist()
    cfg["tool"]["calibration_source"] = "measured_fixture_body_to_nozzle"
    path.write_text(yaml.safe_dump(cfg))
    episode = run_episode(path, tmp_path / "out")
    assert episode.frames[0].quality.valid
    assert episode.frames[0].canonical.standoff_distance_m == pytest.approx(0.21224174381, abs=1e-4)
    assert episode.frames[0].tool.calibration_source == "measured_fixture_body_to_nozzle"
    assert episode.frames[0].tool.frame_method == "body_pose_measured_functional_transform"


def test_body_transform_requires_sourced_calibration(tmp_path):
    inputs = create_demo(tmp_path / "inputs", frames=8)
    path = inputs / "rgb.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["tool"]["T_body_tool"] = np.eye(4).tolist()
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="calibration"):
        load_config(path)


def test_cache_rejects_duplicate_indices_and_invalid_estimates():
    mod = bridge()
    camera = Camera(matrix=np.eye(3).tolist(), width=3, height=3)
    sample = mod.SurfaceSample(frame_index=0, timestamp_sec=0, failure_reason="no_alignment")
    with pytest.raises(ValueError):
        mod.SurfaceCache(
            video_sha256="0" * 64,
            camera=camera,
            data_kind="real",
            calibration_source="measurement",
            surface_frame_id="test_surface",
            frames=[sample, sample],
        )
    with pytest.raises(ValueError):
        mod.SurfaceSample(frame_index=0, timestamp_sec=0)


def test_cached_frame_timestamp_must_match_original_video(tmp_path):
    mod = bridge()
    inputs = create_demo(tmp_path / "inputs", frames=8)
    manifest = Manifest.model_validate_json((inputs / "manifest.json").read_text())
    cache = mod.SurfaceCache(
        video_sha256=mod.file_sha256(inputs / manifest.video_path),
        camera=manifest.camera,
        data_kind="synthetic",
        calibration_source="measurement",
        surface_frame_id="test_surface",
        frames=[mod.SurfaceSample(frame_index=0, timestamp_sec=1, failure_reason="no_alignment")],
    )
    with pytest.raises(ValueError, match="timestamp"):
        mod.cached_surface(cache, 0, 0)


def test_predicted_scene_builds_metric_cache_from_named_rgb_anchors(tmp_path):
    from robodata import vggt_inference as mod

    inputs = create_demo(tmp_path / "inputs", frames=8, occluded=[])
    cfg = load_config(inputs / "rgb.yaml")
    manifest = Manifest.model_validate_json((inputs / "manifest.json").read_text())
    K = np.asarray(manifest.camera.matrix)
    predictions = {
        "depth": np.ones((2, 240, 320)),
        "depth_conf": np.full((2, 240, 320), 10.0),
        "intrinsics": np.repeat(K[None], 2, axis=0),
        "extrinsics": np.repeat(np.c_[np.eye(3), np.zeros(3)][None], 2, axis=0),
    }
    settings = mod.VGGTRunConfig(
        manifest_path="manifest.json",
        checkpoint_path="weights.pt",
        surface_points_m=cfg.surface.points_m,
        calibration_source="analytic_fixture",
        max_alignment_error_m=0.001,
    )
    cache = mod.build_surface_cache(
        predictions,
        [0, 1],
        [0, 0.1],
        np.eye(3),
        manifest,
        inputs / manifest.video_path,
        settings,
        {"kind": "analytic"},
    )
    assert all(f.estimate is not None for f in cache.frames)
    assert np.allclose(
        cache.frames[0].estimate.T_camera_frame,
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]],
    )
    assert cache.provenance["alignment"]["scale_m_per_model_unit"] == pytest.approx(1)


def test_no_anchors_exports_failure_not_metric_poses(tmp_path):
    from robodata import vggt_inference as mod

    inputs = create_demo(tmp_path / "inputs", frames=8)
    manifest = Manifest.model_validate_json((inputs / "manifest.json").read_text())
    manifest.observations = []
    predictions = {
        "depth": np.ones((1, 240, 320)),
        "depth_conf": np.ones((1, 240, 320)),
        "intrinsics": np.asarray(manifest.camera.matrix)[None],
        "extrinsics": np.c_[np.eye(3), np.zeros(3)][None],
    }
    settings = mod.VGGTRunConfig(manifest_path="manifest.json", checkpoint_path="weights.pt")
    cache = mod.build_surface_cache(
        predictions, [0], [0], np.eye(3), manifest, inputs / manifest.video_path, settings, {}
    )
    assert cache.frames[0].estimate is None
    assert "anchor" in cache.frames[0].failure_reason.lower()
    assert cache.provenance["alignment"]["accepted"] is False


def test_frame_reprojection_gate_rejects_wrong_calibrated_camera(tmp_path):
    from robodata import vggt_inference as mod

    inputs = create_demo(tmp_path / "inputs", frames=8, occluded=[])
    cfg = load_config(inputs / "rgb.yaml")
    manifest = Manifest.model_validate_json((inputs / "manifest.json").read_text())
    E = np.c_[np.eye(3), [10, 0, 0]]
    predictions = {
        "depth": np.ones((2, 240, 320)),
        "depth_conf": np.full((2, 240, 320), 10),
        "intrinsics": np.repeat(np.asarray(manifest.camera.matrix)[None], 2, axis=0),
        "extrinsics": np.array([np.c_[np.eye(3), np.zeros(3)], E]),
    }
    # Only frame zero supplies alignment; frame one still carries measured RGB checks.
    settings = mod.VGGTRunConfig(
        manifest_path="manifest.json",
        checkpoint_path="weights.pt",
        surface_points_m=cfg.surface.points_m,
        calibration_source="analytic_fixture",
        alignment_frame_indices=[0],
        max_alignment_error_m=0.001,
    )
    cache = mod.build_surface_cache(
        predictions,
        [0, 1],
        [0, 0.1],
        np.eye(3),
        manifest,
        inputs / manifest.video_path,
        settings,
        {},
    )
    assert cache.frames[0].estimate is not None
    assert cache.frames[1].estimate is None
    assert "reprojection" in cache.frames[1].failure_reason.lower()


def test_rgb_probe_exports_normalized_outputs_without_claiming_metric_accuracy(
    tmp_path, monkeypatch
):
    import json

    from robodata import vggt_inference as mod

    image = tmp_path / "rgb.png"
    image.write_bytes(b"probe input digest fixture")
    predictions = {
        "depth": np.ones((1, 3, 3)),
        "depth_conf": np.full((1, 3, 3), 12),
        "intrinsics": np.eye(3)[None],
        "extrinsics": np.c_[np.eye(3), np.zeros(3)][None],
    }
    monkeypatch.setattr(
        mod, "infer_rgb_images", lambda *args: (predictions, {"model": "test_double"})
    )
    report = mod.probe_rgb_images([image], tmp_path / "checkpoint", tmp_path / "out")
    assert report["metric_alignment_performed"] is False
    assert report["tool_pose_estimated"] is False
    with np.load(tmp_path / "out" / "reconstruction.npz", allow_pickle=False) as saved:
        assert saved["depth"][0, 1, 1] == 1
    assert json.loads((tmp_path / "out" / "provenance.json").read_text())["model"] == "test_double"
    with pytest.raises(FileExistsError):
        mod.probe_rgb_images([image], tmp_path / "checkpoint", tmp_path / "out")


def test_offline_runner_exports_selected_original_frame_ids(tmp_path, monkeypatch):
    import json

    from robodata import vggt_inference as mod

    inputs = create_demo(tmp_path / "inputs", frames=8, occluded=[])
    checkpoint = inputs / "weights"
    checkpoint.write_bytes(b"not real weights: runner test substitutes only neural inference")
    settings = mod.VGGTRunConfig(
        manifest_path="manifest.json",
        checkpoint_path="weights",
        max_frames=2,
        exclude_tool_pixels=False,
    )
    config_path = inputs / "vggt.yaml"
    config_path.write_text(yaml.safe_dump(settings.model_dump()))
    shape = (2, 392, 518)
    predictions = {
        "depth": np.ones(shape),
        "depth_conf": np.full(shape, 12),
        "intrinsics": np.repeat(np.eye(3)[None], 2, axis=0),
        "extrinsics": np.repeat(np.c_[np.eye(3), np.zeros(3)][None], 2, axis=0),
    }
    monkeypatch.setattr(
        mod, "infer_rgb_images", lambda *args: (predictions, {"model": "test_double"})
    )
    cache = mod.run_vggt(config_path, tmp_path / "out")
    assert [s.frame_index for s in cache.frames] == [0, 7]
    assert [s.timestamp_sec for s in cache.frames] == pytest.approx([0, 0.7])
    assert all(s.estimate is None for s in cache.frames)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["metric_surface_frames"] == 0
    with pytest.raises(FileExistsError):
        mod.run_vggt(config_path, tmp_path / "out")


@pytest.mark.parametrize("height,expected_height", [(350, 238), (1310, 910)])
def test_crop_affine_matches_upstream_float_rounding_boundaries(height, expected_height):
    A, shape = bridge().crop_affine(740, height)
    assert A[1, 1] == pytest.approx(expected_height / height)
    assert shape == (min(518, expected_height), 518)


@pytest.mark.parametrize("visible_count", [1, 2, 3])
@pytest.mark.parametrize("translation", [[10, 0, 0], [0, 0, -10]])
def test_partial_anchors_still_reject_bad_camera_pose(tmp_path, visible_count, translation):
    from robodata import vggt_inference as mod

    inputs = create_demo(tmp_path / "inputs", frames=8, occluded=[])
    cfg = load_config(inputs / "rgb.yaml")
    manifest = Manifest.model_validate_json((inputs / "manifest.json").read_text())
    for k, kp in enumerate(manifest.observations[1].surface.keypoints.values()):
        kp.visible = k < visible_count
    predictions = {
        "depth": np.ones((2, 240, 320)),
        "depth_conf": np.full((2, 240, 320), 10),
        "intrinsics": np.repeat(np.asarray(manifest.camera.matrix)[None], 2, axis=0),
        "extrinsics": np.array([np.c_[np.eye(3), np.zeros(3)], np.c_[np.eye(3), translation]]),
    }
    settings = mod.VGGTRunConfig(
        manifest_path="manifest.json",
        checkpoint_path="weights.pt",
        surface_points_m=cfg.surface.points_m,
        calibration_source="analytic_fixture",
        alignment_frame_indices=[0],
        max_alignment_error_m=0.001,
    )
    cache = mod.build_surface_cache(
        predictions,
        [0, 1],
        [0, 0.1],
        np.eye(3),
        manifest,
        inputs / manifest.video_path,
        settings,
        {},
    )
    assert cache.frames[0].estimate is not None
    assert cache.frames[1].estimate is None

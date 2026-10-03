import json

import numpy as np
import pytest

from robodata.config import load_config
from robodata.export import export_tensors
from robodata.pipeline import run_episode
from robodata.schema import Episode
from robodata.synthetic import create_demo


@pytest.mark.parametrize("mode", ["rgb", "rgbd"])
def test_video_to_dataset_roundtrip_and_occlusion(tmp_path, mode):
    inputs = tmp_path / "inputs"
    create_demo(inputs, frames=12, occluded=[5])
    result = run_episode(inputs / f"{mode}.yaml", tmp_path / "run")
    first = result.frames[0]
    assert first.quality.valid
    assert first.canonical.standoff_distance_m == pytest.approx(0.3, abs=0.003)
    assert not result.frames[5].quality.valid
    assert result.frames[5].canonical.T_surface_tool is None
    saved = Episode.model_validate_json((tmp_path / "run" / "episode.json").read_text())
    assert len(saved.frames) == 12
    export_tensors(saved, tmp_path / "tensors.npz")
    with np.load(tmp_path / "tensors.npz") as tensor:
        assert tensor["state"].shape == (12, 12)
        assert not tensor["state_valid"][5]
        assert np.isfinite(tensor["state"]).all()


def test_rgb_without_known_scale_is_rejected(tmp_path):
    create_demo(tmp_path / "inputs", frames=8)
    path = tmp_path / "inputs" / "rgb.yaml"
    import yaml

    config = yaml.safe_load(path.read_text())
    config["metric_scale_source"] = "unknown"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_output_refuses_to_overwrite_existing_run(tmp_path):
    create_demo(tmp_path / "inputs", frames=8)
    run_episode(tmp_path / "inputs" / "rgb.yaml", tmp_path / "out")
    with pytest.raises(FileExistsError):
        run_episode(tmp_path / "inputs" / "rgb.yaml", tmp_path / "out")


def test_explicit_timestamps_must_match_video_length(tmp_path):
    create_demo(tmp_path / "inputs", frames=8)
    path = tmp_path / "inputs" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["timestamps_sec"] = [0, 0.1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="timestamp"):
        run_episode(tmp_path / "inputs" / "rgb.yaml", tmp_path / "out")


@pytest.mark.parametrize(
    "verified,hand_missing,tool_occluded",
    [(True, False, False), (False, False, False), (True, True, False), (True, False, True)],
)
@pytest.mark.parametrize("mode", ["rgb", "rgbd"])
def test_calibrated_hand_fallback_preserves_failed_frames(
    tmp_path, verified, hand_missing, tool_occluded, mode
):
    import yaml

    inputs = create_demo(tmp_path / "inputs", frames=8, occluded=[])
    path = inputs / f"{mode}.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["surface_method"] = "pnp"
    # Analytic fixture: hand-frame points coincide with the synthetic tool frame.
    names = ["0", "5", "9", "13", "17", "1", "2", "3"]
    correspondence = dict(zip(names, cfg["tool"]["points_m"], strict=True))
    cfg["hand_assistance"] = {
        "points_hand_m": dict(zip(names, cfg["tool"]["points_m"].values(), strict=True)),
        "T_hand_tool": np.eye(4).tolist(),
        "calibration_source": "analytic_test_fixture_not_real_hand",
        "grasp_verified": verified,
        "handedness": "Right",
    }
    path.write_text(yaml.safe_dump(cfg))
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for obs in manifest["observations"]:
        obs["hand"] = (
            {"keypoints": {}}
            if hand_missing
            else {
                "handedness": "Right",
                "keypoints": {
                    n: obs["tool"]["keypoints"][key] for n, key in correspondence.items()
                },
            }
        )
        obs["tool"]["keypoints"] = {}
        obs["tool"]["occluded"] = tool_occluded
    manifest_path.write_text(json.dumps(manifest))
    episode = run_episode(path, tmp_path / "out")
    assert len(episode.frames) == 8
    for frame in episode.frames:
        assert frame.quality.valid == (verified and not hand_missing)
        assert frame.quality.occluded == tool_occluded
        if frame.quality.valid:
            assert frame.tool.method == "calibrated_hand_grasp_pnp"
            assert frame.tool.metric_scale_source == "calibrated_hand_template"
            assert frame.canonical.standoff_distance_m == pytest.approx(0.3, abs=1e-5)
        else:
            assert "hand assistance" in frame.quality.failure_reason
            assert frame.canonical.T_surface_tool is None
    if mode == "rgbd" and verified and not hand_missing:
        assert episode.source.metric_scale_source == "mixed_depth_and_hand_calibration"

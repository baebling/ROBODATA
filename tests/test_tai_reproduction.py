import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from reproductions.tool_as_interface import runner


def test_extract_preserves_archive_and_refuses_overwrite(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("hammer/replay_buffer.zarr/.zgroup", "{}")
    original = archive.read_bytes()
    destination = tmp_path / "extracted"
    runner.extract_archive(archive, destination)
    assert (destination / "hammer/replay_buffer.zarr/.zgroup").read_text() == "{}"
    assert archive.read_bytes() == original
    with pytest.raises(FileExistsError):
        runner.extract_archive(archive, destination)


@pytest.mark.parametrize("name", ["../escape", "/escape", "a/../../escape", "C:/escape"])
def test_extract_rejects_unsafe_paths_without_partial_output(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "data.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("safe.txt", "safe")
        stream.writestr(name, "bad")
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe"):
        runner.extract_archive(archive, destination)
    assert not destination.exists()


def test_next_actions_do_not_cross_episode_boundaries() -> None:
    poses = np.arange(6 * 9, dtype=np.float32).reshape(6, 9)
    observations, actions, ends, indices = runner.align_next_pose(poses, [3, 6])
    np.testing.assert_array_equal(indices, [0, 1, 3, 4])
    np.testing.assert_array_equal(observations, poses[[0, 1, 3, 4]])
    np.testing.assert_array_equal(actions, poses[[1, 2, 4, 5]])
    np.testing.assert_array_equal(ends, [2, 4])


@pytest.mark.parametrize("ends", [[0, 6], [3, 3, 6], [3, 7], [3, 5], []])
def test_alignment_rejects_invalid_episode_ends(ends: list[int]) -> None:
    with pytest.raises(ValueError, match="episode"):
        runner.align_next_pose(np.zeros((6, 9), dtype=np.float32), ends)


def test_alignment_rejects_nonfinite_pose_without_filling_it() -> None:
    poses = np.zeros((3, 9), dtype=np.float32)
    poses[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        runner.align_next_pose(poses, [3])


def test_alignment_rejects_single_frame_episodes() -> None:
    with pytest.raises(ValueError, match="two"):
        runner.align_next_pose(np.zeros((3, 9), dtype=np.float32), [1, 3])


def test_output_cannot_be_written_inside_raw_data(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="raw"):
        runner.prepare_output(raw / "results", raw)
    assert list(raw.iterdir()) == []


def test_output_refuses_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    with pytest.raises(FileExistsError):
        runner.prepare_output(output, tmp_path / "raw")


def test_sha256_tracks_content_not_filename(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"abc")
    second.write_bytes(b"abc")
    assert runner.sha256_file(first) == runner.sha256_file(second)
    second.write_bytes(b"abcd")
    assert runner.sha256_file(first) != runner.sha256_file(second)


def test_tree_digest_detects_changed_raw_chunk(tmp_path: Path) -> None:
    (tmp_path / "chunk").write_bytes(b"first")
    before = runner.tree_digest(tmp_path)
    (tmp_path / "chunk").write_bytes(b"second")
    assert runner.tree_digest(tmp_path) != before


def test_reproduction_revision_records_uncommitted_code(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "fixture.txt").write_text("original")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    clean = runner.reproduction_revision(tmp_path)
    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False
    (tmp_path / "new_code.py").write_text("pass")
    dirty = runner.reproduction_revision(tmp_path)
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True


def test_scene_evidence_does_not_equate_nonblank_with_accuracy(tmp_path: Path) -> None:
    views = np.ones((10, 4, 4, 3), dtype=np.float32) * 0.5
    views[:, :2] = 0
    np.savez_compressed(
        tmp_path / "reconstruction.npz",
        points=np.ones((3, 3)),
        novel_views=views,
        source_w2c=np.stack([np.eye(4)] * 2),
        novel_w2c=np.stack([np.eye(4)] * 10),
        intrinsics=np.stack([np.eye(3)] * 2),
    )
    report = runner.inspect_scene_artifact(tmp_path / "reconstruction.npz")
    assert report["nonblank_views"] == 10
    assert report["finite_point_fraction"] == 1
    assert report["near_black_pixel_fraction"] == [0.5] * 10
    assert report["geometry_accuracy_validated"] is False


def test_scene_evidence_rejects_nonfinite_render(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "reconstruction.npz",
        points=np.ones((3, 3)),
        novel_views=np.full((10, 4, 4, 3), np.nan),
        source_w2c=np.stack([np.eye(4)] * 2),
        novel_w2c=np.stack([np.eye(4)] * 10),
        intrinsics=np.stack([np.eye(3)] * 2),
    )
    with pytest.raises(ValueError, match="Nonfinite"):
        runner.inspect_scene_artifact(tmp_path / "reconstruction.npz")


def test_uniform_colored_render_is_blank(tmp_path: Path) -> None:
    views = np.zeros((10, 4, 4, 3), dtype=np.float32)
    views[..., 0] = 1
    np.savez_compressed(
        tmp_path / "reconstruction.npz",
        points=np.ones((3, 3)),
        novel_views=views,
        source_w2c=np.stack([np.eye(4)] * 2),
        novel_w2c=np.stack([np.eye(4)] * 10),
        intrinsics=np.stack([np.eye(3)] * 2),
    )
    assert runner.inspect_scene_artifact(tmp_path / "reconstruction.npz")["nonblank_views"] == 0


def test_profile_excludes_time_gaps_between_episodes() -> None:
    root = {
        "meta/episode_ends": np.array([3, 6]),
        "data/timestamp": np.array([0.0, 0.1, 0.2, 10.0, 10.1, 10.2]),
        "data/observations/robot_eef_pose": np.zeros((6, 6), dtype=np.float32),
    }
    profile = runner.profile_labels(root)
    assert profile["episodes"] == 2
    assert profile["frames"] == 6
    assert profile["within_episode_dt_sec"]["max"] == pytest.approx(0.1)
    assert profile["pose_provenance"] == "supplied estimates; not independent ground truth"


def test_profile_flags_invalid_supplied_pose() -> None:
    poses = np.zeros((3, 6), dtype=np.float32)
    poses[1] = np.nan
    root = {
        "meta/episode_ends": np.array([3]),
        "data/timestamp": np.array([0.0, 0.1, 0.2]),
        "data/observations/robot_eef_pose": poses,
    }
    profile = runner.profile_labels(root)
    assert profile["nonfinite_pose_frames"] == 1


def test_profile_rejects_nonincreasing_timestamps() -> None:
    root = {
        "meta/episode_ends": np.array([3]),
        "data/timestamp": np.array([0.0, 0.1, 0.1]),
        "data/observations/robot_eef_pose": np.zeros((3, 6), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="timestamp"):
        runner.profile_labels(root)


@pytest.mark.parametrize("timestamps", [[0.0, 0.1, 0.1], [0.0, 0.1, np.nan]])
def test_episode_selection_validates_timestamps_before_model_execution(timestamps: list) -> None:
    root = {
        "meta/episode_ends": np.array([3]),
        "data/timestamp": np.asarray(timestamps),
        "data/observations/robot_eef_pose": np.zeros((3, 6), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="timestamp"):
        runner.select_episode(root, 0)


def test_episode_selection_rejects_fractional_boundaries() -> None:
    root = {
        "meta/episode_ends": np.array([3.5, 6.0]),
        "data/timestamp": np.arange(6, dtype=float),
        "data/observations/robot_eef_pose": np.zeros((6, 6), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="episode"):
        runner.select_episode(root, 0)


def test_sampled_prediction_indices_include_next_pose_and_action_slice() -> None:
    observation_indices = np.array([[3, 4, 5, 6, 7, 8, 9, 10]])
    timestamps = np.arange(20) / 15
    provenance = runner.prediction_provenance(observation_indices, timestamps, 2, 4)
    np.testing.assert_array_equal(provenance["target_frame_index"], observation_indices + 1)
    np.testing.assert_array_equal(provenance["action_frame_index"], [[5, 6, 7, 8]])
    np.testing.assert_array_equal(provenance["action_timestamp_sec"], timestamps[[[5, 6, 7, 8]]])


def estimated_episode(tmp_path: Path, indices=None):
    path = tmp_path / "estimated_poses.npz"
    np.savez_compressed(
        path,
        pose_position_axis_angle=np.ones((3, 6), dtype=np.float32),
        timestamp_sec=np.array([0.0, 0.1, 0.2]),
        source_frame_index=np.array([0, 1, 2] if indices is None else indices),
        numerical_valid=np.ones(3, dtype=bool),
    )
    report = {
        "stage": "official_grounded_sam_foundationpose_clip",
        "estimate_file_sha256": runner.sha256_file(path),
        "raw_tree_sha256": "known_raw",
        "raw_unchanged": True,
        "provided_pose_used_for_estimation": False,
    }
    (tmp_path / "preprocessing_smoke.json").write_text(json.dumps(report))
    return path


def test_policy_accepts_reestimated_episode_with_bound_provenance(tmp_path: Path):
    path = estimated_episode(tmp_path)
    poses = runner.load_estimated_episode(path, 0, 3, np.array([0.0, 0.1, 0.2]), "known_raw")
    np.testing.assert_array_equal(poses, np.ones((3, 6)))


def test_policy_rejects_reestimated_pose_from_different_frame_window(tmp_path: Path):
    path = estimated_episode(tmp_path, [0, 1, 5])
    with pytest.raises(ValueError, match="alignment"):
        runner.load_estimated_episode(path, 0, 3, np.array([0.0, 0.1, 0.2]), "known_raw")


def test_policy_rejects_reestimated_pose_from_different_raw_dataset(tmp_path: Path):
    path = estimated_episode(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        runner.load_estimated_episode(path, 0, 3, np.array([0.0, 0.1, 0.2]), "another_raw")


def test_policy_rejects_changed_estimate_file(tmp_path: Path):
    path = estimated_episode(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        runner.load_estimated_episode(path, 0, 3, np.array([0.0, 0.1, 0.2]), "known_raw")


@pytest.mark.parametrize("ignored", [False, True])
def test_pinned_source_rejects_untracked_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignored: bool
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "fixture.txt").write_text("official")
    if ignored:
        (tmp_path / ".gitignore").write_text("rogue.py\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(runner, "UPSTREAM_COMMIT", commit)
    (tmp_path / "rogue.py").write_text("raise RuntimeError('not official')")
    with pytest.raises(ValueError, match="untracked"):
        runner.verify_source(tmp_path)


def test_verified_source_does_not_execute_unchecked_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "pycache_prefix", None)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "trusted_probe.py"
    source.write_text("value = 'official'\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(runner, "UPSTREAM_COMMIT", commit)
    fake = tmp_path.parent / f"{tmp_path.name}_fake.py"
    fake.write_text("value = 'unverified'\n")
    cache = importlib.util.cache_from_source(str(source))
    py_compile.compile(
        str(fake),
        cfile=cache,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    runner.verify_source(tmp_path)
    spec = importlib.util.spec_from_file_location("trusted_probe", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.value == "official"


@pytest.mark.skipif(
    not os.environ.get("TAI_UPSTREAM"), reason="Needs isolated upstream GPU runtime"
)
def test_official_policy_smoke_on_public_hammer(tmp_path: Path) -> None:
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation

    upstream = Path(os.environ["TAI_UPSTREAM"])
    dataset = Path(os.environ["TAI_DATASET"])
    runner.verify_source(upstream, policy=True)
    config = {"episode": 0, "batch_size": 1, "train_steps": 1, "seed": 42, "view": "camera_3"}
    report = runner.policy_smoke(dataset, upstream, tmp_path, config)
    assert report["official_model_parameters"] > 200_000_000
    assert len(report["losses"]) == 1
    assert np.isfinite(report["losses"]).all()
    assert report["parameters_changed"]
    assert report["action_shape"] == [1, 4, 9]
    assert report["prediction_finite"]
    assert not report["paper_results_reproduced"]
    saved = np.load(tmp_path / "smoke_predictions.npz")
    assert saved["target_frame_index"].shape == saved["supplied_target"].shape[:2]
    assert saved["action_frame_index"].shape == saved["action"].shape[:2]
    np.testing.assert_array_equal(saved["target_frame_index"], saved["observation_frame_index"] + 1)
    root = runner.open_public(dataset)
    supplied = root["data/observations/robot_eef_pose"][:]
    expected = np.concatenate(
        [supplied[:, :3], Rotation.from_rotvec(supplied[:, 3:]).as_matrix()[:, :2].reshape(-1, 6)],
        axis=-1,
    )
    np.testing.assert_allclose(
        saved["supplied_target"], expected[saved["target_frame_index"]], atol=1e-5
    )
    np.testing.assert_allclose(
        saved["conditioning_pose"], expected[saved["observation_frame_index"]], atol=1e-5
    )
    official = OmegaConf.load(tmp_path / "official_policy_config.yaml")
    assert official.down_dims == [512, 1024, 2048]
    assert (official.horizon, official.n_obs_steps, official.n_action_steps) == (8, 2, 4)
    assert official.num_inference_steps == official.noise_scheduler.num_train_timesteps == 100
    assert report["feature_storage_codec"] == "Jpeg2k(level=50) encode/decode in memory"


@pytest.mark.skipif(
    not os.environ.get("TAI_RECON"), reason="Needs MASt3R weights and CUDA compiler"
)
def test_official_scene_augmentation_on_public_pair(tmp_path: Path) -> None:
    upstream = Path(os.environ["TAI_UPSTREAM"])
    runner.verify_source(upstream)
    report = runner.scene_smoke(Path(os.environ["TAI_DATASET"]), tmp_path)
    assert report["finite_point_fraction"] == 1.0
    assert report["point_count"] > 1000
    assert report["novel_view_shape"] == [10, 224, 224, 3]
    assert report["nonblank_views"] == 10
    assert not report["construction_validated"]

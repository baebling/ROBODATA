from pathlib import Path

import numpy as np
import pytest

from reproductions.tool_as_interface import preprocess_smoke


def public_clip():
    camera = "data/observations/images/camera_3"
    return {
        "meta/episode_ends": np.array([3, 6]),
        "data/timestamp": np.array([0, 0.1, 0.2, 10, 10.1, 10.2]),
        f"{camera}_color": np.zeros((6, 4, 4, 3), dtype=np.uint8),
        f"{camera}_depth": np.full((6, 4, 4), 1000, dtype=np.uint16),
        f"{camera}_intrinsics": np.stack([np.eye(3)] * 6),
    }


def config():
    return {
        "episode": 1,
        "frame_count": 3,
        "camera": "camera_3",
        "depth_scale_m": 0.001,
        "intrinsics_dtype": "float32",
    }


def test_model_intrinsics_use_explicit_float32_boundary():
    clip = preprocess_smoke.load_clip(public_clip(), config())
    assert clip["intrinsics"].dtype == np.float32


def test_clip_uses_explicit_depth_scale_and_does_not_need_provided_pose():
    clip = preprocess_smoke.load_clip(public_clip(), config())
    np.testing.assert_array_equal(clip["source_frame_index"], [3, 4, 5])
    np.testing.assert_allclose(clip["depth_m"], 1)
    np.testing.assert_array_equal(clip["timestamp_sec"], [10, 10.1, 10.2])


def test_clip_never_crosses_episode_boundary():
    settings = config()
    settings["frame_count"] = 4
    with pytest.raises(ValueError, match="episode"):
        preprocess_smoke.load_clip(public_clip(), settings)


def test_clip_preserves_invalid_depth_without_filling():
    root = public_clip()
    root["data/observations/images/camera_3_depth"][3, 1, 2] = 0
    clip = preprocess_smoke.load_clip(root, config())
    assert clip["depth_m"][0, 1, 2] == 0
    assert not clip["depth_valid"][0, 1, 2]


def test_clip_rejects_missing_scale():
    settings = config()
    del settings["depth_scale_m"]
    with pytest.raises(ValueError, match="scale"):
        preprocess_smoke.load_clip(public_clip(), settings)


def test_clip_rejects_camera_intrinsics_changing_during_tracking():
    root = public_clip()
    root["data/observations/images/camera_3_intrinsics"][4, 0, 0] = 2
    with pytest.raises(ValueError, match="intrinsics"):
        preprocess_smoke.load_clip(root, config())


def test_video_evidence_counts_actual_decoded_frames(tmp_path: Path):
    import cv2

    path = tmp_path / "poses.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (16, 16))
    assert writer.isOpened()
    for value in [0, 40, 80]:
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()
    report = preprocess_smoke.inspect_pose_video(path, 3)
    assert report["decoded_frames"] == 3
    assert report["frame_shape"] == [16, 16, 3]
    assert (tmp_path / "pose_first.png").is_file()
    with pytest.raises(ValueError, match="frame count"):
        preprocess_smoke.inspect_pose_video(path, 4)

import hashlib
import importlib
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from robodata import hand_infer
from robodata.observations import Manifest


def hand(x, y, handedness_score=0.99):
    landmarks = [SimpleNamespace(x=x, y=y, z=-0.02 * i) for i in range(21)]
    world = [SimpleNamespace(x=i * 0.001, y=0.01, z=-0.004) for i in range(21)]
    categories = [
        SimpleNamespace(
            category_name="Right",
            display_name="Right",
            score=handedness_score,
            index=1,
        )
    ]
    return landmarks, world, categories


def result(*hands):
    return SimpleNamespace(
        hand_landmarks=[h[0] for h in hands],
        hand_world_landmarks=[h[1] for h in hands],
        handedness=[h[2] for h in hands],
    )


@pytest.fixture
def original(tmp_path):
    root = tmp_path / "originals"
    root.mkdir()
    video = root / "rgb.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (100, 80))
    assert writer.isOpened()
    for value in [30, 60, 90]:
        writer.write(np.full((80, 100, 3), [value, 10, 200], np.uint8))
    writer.release()
    mask = np.zeros((80, 100), np.uint8)
    # Its centroid is far from the grasp point; real pixel proximity must win.
    mask[40, 5:96] = 255
    assert cv2.imwrite(str(root / "tool.png"), mask)
    manifest = {
        "episode_id": "virtual_grasp",
        "task_name": "synthetic",
        "data_kind": "synthetic",
        "video_path": "rgb.avi",
        "landmark_source": "original_tool_surface_source",
        "camera": {"matrix": [[100, 0, 50], [0, 100, 40], [0, 0, 1]], "width": 100, "height": 80},
        "timestamps_sec": [0.0, 0.04, 0.09],
        "observations": [
            {
                "frame_index": 0,
                "tool": {
                    "mask_path": "tool.png",
                    "keypoints": {"tip": {"uv": [25, 40], "confidence": 0.7}},
                },
                "surface": {"mask_path": "surface.png"},
                "depth_path": "depth.npy",
            },
            {"frame_index": 1, "tool": {"mask_path": "tool.png"}},
        ],
        "provenance": {"reference_path": "must_not_be_read.csv", "existing": "keep"},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    model = tmp_path / "model.task"
    model.write_bytes(b"test-model")
    return path, model, tmp_path / "predictions", manifest


def install_fake_mediapipe(monkeypatch, outputs, expected_shape=(80, 100, 3), check_color=True):
    class Landmarker:
        calls = []
        options = None

        @classmethod
        def create_from_options(cls, options):
            cls.options = options
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def detect_for_video(self, image, timestamp_ms):
            assert image.image_format == "SRGB"
            assert image.data.shape == expected_shape
            # Check decoded RGB, rather than a mask or BGR video buffer.
            if check_color:
                assert image.data[0, 0, 0] > 180 and image.data[0, 0, 2] < 110
            self.calls.append(timestamp_ms)
            return outputs[len(self.calls) - 1]

    vision = SimpleNamespace(
        HandLandmarker=Landmarker,
        HandLandmarkerOptions=lambda **kw: SimpleNamespace(**kw),
        RunningMode=SimpleNamespace(VIDEO="VIDEO"),
    )
    python = SimpleNamespace(BaseOptions=lambda **kw: SimpleNamespace(**kw), vision=vision)
    mp = SimpleNamespace(
        __version__="0.10.21",
        Image=lambda **kw: SimpleNamespace(**kw),
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
        tasks=SimpleNamespace(python=python),
    )
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", SimpleNamespace(python=python))
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", python)
    return Landmarker


def test_clones_manifest_and_retains_raw_predictions_for_every_frame(original, monkeypatch):
    path, model, output, original_data = original
    before = path.read_bytes()
    backend = install_fake_mediapipe(
        monkeypatch,
        [
            result(hand(0.1, 0.5, 0.12), hand(0.5, 0.1)),
            result(),
            result(hand(0.1, 0.5)),
        ],
    )
    new_path = hand_infer.infer_hands(model, path, output, min_confidence=0.6)
    cloned = json.loads(new_path.read_text())
    Manifest.model_validate(cloned)
    assert new_path.parent == output
    assert path.read_bytes() == before
    assert cloned["landmark_source"] == original_data["landmark_source"]
    assert cloned["video_path"] == str(path.parent / "rgb.avi")
    frames = cloned["observations"]
    assert [f["frame_index"] for f in frames] == [0, 1, 2]
    assert frames[0]["tool"]["keypoints"]["tip"]["confidence"] == 0.7
    assert frames[0]["tool"]["mask_path"] == str(path.parent / "tool.png")
    assert frames[0]["surface"]["mask_path"] == str(path.parent / "surface.png")
    assert frames[0]["depth_path"] == str(path.parent / "depth.npy")
    assert cloned["provenance"]["reference_path"] == str(path.parent / "must_not_be_read.csv")
    assert len(frames[0]["hand"]["keypoints"]) == 21
    assert frames[0]["hand"]["keypoints"]["0"] == {
        "uv": [10.0, 40.0],
        "confidence": 1.0,
        "visible": True,
    }
    assert not frames[1]["hand"]["keypoints"]
    assert not frames[2]["hand"]["keypoints"]  # no tool mask
    provenance = cloned["provenance"]["hand_source"]
    assert provenance["model_sha256"] == hashlib.sha256(b"test-model").hexdigest()
    assert provenance["mediapipe_version"] == "0.10.21"
    assert provenance["min_confidence"] == 0.6
    assert "confidence" in provenance["keypoint_semantics"]
    raw = json.loads((output / "hand_detections.json").read_text())
    assert [f["timestamp_ms"] for f in raw["frames"]] == [0, 40, 90]
    assert len(raw["frames"]) == 3
    assert raw["frames"][0]["selected_hand_index"] == 0
    assert len(raw["frames"][0]["hands"]) == 2
    assert raw["frames"][0]["hands"][0]["normalized_landmarks"][2] == [0.1, 0.5, -0.04]
    assert raw["frames"][0]["hands"][0]["local_world_landmarks_m"][2] == [0.002, 0.01, -0.004]
    assert raw["frames"][0]["hands"][0]["handedness"][0]["score"] == 0.12
    assert not raw["frames"][1]["hands"]
    assert len(raw["frames"][2]["hands"]) == 1
    assert backend.calls == [0, 40, 90]
    assert backend.options.running_mode == "VIDEO"
    assert backend.options.min_hand_detection_confidence == 0.6
    assert backend.options.min_hand_presence_confidence == 0.6
    assert backend.options.min_tracking_confidence == 0.6


@pytest.mark.parametrize("timestamps", [[0, 0.0004, 0.09], [0, 0.05, 0.04]])
def test_rejects_colliding_or_nonmonotonic_timestamps(original, monkeypatch, timestamps):
    path, model, output, data = original
    data["timestamps_sec"] = timestamps
    path.write_text(json.dumps(data))
    install_fake_mediapipe(monkeypatch, [result()] * 3)
    with pytest.raises(ValueError, match="timestamp"):
        hand_infer.infer_hands(model, path, output)
    assert not (output / "manifest.json").exists()


def test_refuses_existing_output_directory(original):
    path, model, output, _ = original
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        hand_infer.infer_hands(model, path, output)
    assert sentinel.read_text() == "original"


def test_refuses_output_inside_originals(original):
    path, model, _, _ = original
    with pytest.raises(ValueError, match="original"):
        hand_infer.infer_hands(model, path, path.parent / "predictions")
    assert not (path.parent / "predictions").exists()


def test_import_is_lazy_without_optional_mediapipe(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    importlib.reload(hand_infer)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_confidence_threshold(original, threshold):
    path, model, output, _ = original
    with pytest.raises(ValueError, match="confidence"):
        hand_infer.infer_hands(model, path, output, threshold)


def test_distant_palm_does_not_associate_even_if_fingertip_touches_tool(original, monkeypatch):
    path, model, output, _ = original
    prediction = hand(0.1, 0.1)
    prediction[0][8].y = 0.5
    install_fake_mediapipe(monkeypatch, [result(prediction), result(), result()])
    new_path = hand_infer.infer_hands(model, path, output)
    cloned = json.loads(new_path.read_text())
    assert not cloned["observations"][0]["hand"]["keypoints"]
    raw = json.loads((output / "hand_detections.json").read_text())
    assert len(raw["frames"][0]["hands"]) == 1
    assert raw["frames"][0]["selected_hand_index"] is None


def test_preserves_supplied_hand_mask_and_hard_occlusion(original, monkeypatch):
    path, model, output, data = original
    data["observations"][0]["hand"] = {
        "mask_path": "hand.png",
        "occluded": True,
        "keypoints": {"0": {"uv": [1, 2], "confidence": 0.1}},
    }
    path.write_text(json.dumps(data))
    install_fake_mediapipe(monkeypatch, [result(hand(0.1, 0.5)), result(), result()])
    cloned = json.loads(hand_infer.infer_hands(model, path, output).read_text())
    prediction = cloned["observations"][0]["hand"]
    assert prediction["mask_path"] == str(path.parent / "hand.png")
    assert prediction["occluded"] is True
    assert prediction["keypoints"]["0"]["uv"] == [10, 40]


def test_refuses_output_beside_external_original_video(original):
    path, model, _, data = original
    external = path.parent.parent / "other_originals"
    external.mkdir()
    data["video_path"] = str(external / "rgb.avi")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="original"):
        hand_infer.infer_hands(model, path, external / "predictions")
    assert not (external / "predictions").exists()


def test_unavailable_coordinates_are_never_invented_and_raw_is_preserved(original, monkeypatch):
    path, model, output, _ = original
    prediction = hand(0.1, 0.5)
    prediction[0][2].x = float("nan")
    prediction[0][3].x = 1.1
    prediction[0][4].z = float("inf")
    install_fake_mediapipe(monkeypatch, [result(prediction), result(), result()])
    cloned = json.loads(hand_infer.infer_hands(model, path, output).read_text())
    points = cloned["observations"][0]["hand"]["keypoints"]
    assert "2" not in points
    assert points["3"]["visible"] is False
    raw = json.loads((output / "hand_detections.json").read_text())
    normalized = raw["frames"][0]["hands"][0]["normalized_landmarks"]
    assert normalized[2][0] is None
    assert normalized[3][0] == 1.1
    assert normalized[4][2] is None


def test_nearest_palm_wins_instead_of_highest_handedness_score(original, monkeypatch):
    path, model, output, _ = original
    install_fake_mediapipe(
        monkeypatch,
        [
            result(hand(0.1, 0.55, 0.99), hand(0.1, 0.5, 0.01)),
            result(),
            result(),
        ],
    )
    cloned = json.loads(hand_infer.infer_hands(model, path, output).read_text())
    assert cloned["observations"][0]["hand"]["keypoints"]["0"]["uv"] == [10, 40]
    raw = json.loads((output / "hand_detections.json").read_text())
    assert raw["frames"][0]["selected_hand_index"] == 1


def test_reports_missing_optional_runtime_without_writing_outputs(original, monkeypatch):
    path, model, output, _ = original
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    with pytest.raises(RuntimeError, match="optional MediaPipe"):
        hand_infer.infer_hands(model, path, output)
    assert not output.exists()


def test_fps_fallback_records_actual_decoded_timestamps(original, monkeypatch):
    path, model, output, data = original
    del data["timestamps_sec"]
    path.write_text(json.dumps(data))
    install_fake_mediapipe(monkeypatch, [result(), result(), result()])
    cloned = json.loads(hand_infer.infer_hands(model, path, output).read_text())
    assert cloned["timestamps_sec"] is None
    raw = json.loads((output / "hand_detections.json").read_text())
    assert [f["timestamp_sec"] for f in raw["frames"]] == [0, 0.04, 0.08]
    assert cloned["provenance"]["hand_source"]["timestamp_source"] == "fps_assumed_constant"


def test_fps_assumption_survives_hand_inference_into_episode(tmp_path, monkeypatch):
    from robodata.pipeline import run_episode
    from robodata.synthetic import create_demo

    inputs = create_demo(tmp_path / "inputs", frames=4, occluded=[])
    path = inputs / "manifest.json"
    data = json.loads(path.read_text())
    data["timestamps_sec"] = None
    path.write_text(json.dumps(data))
    model = tmp_path / "hand.task"
    model.write_bytes(b"fake test runtime")
    install_fake_mediapipe(
        monkeypatch, [result() for _ in range(4)], expected_shape=(240, 320, 3), check_color=False
    )
    prediction = hand_infer.infer_hands(model, path, tmp_path / "hand_predictions")
    episode = run_episode(inputs / "rgb.yaml", tmp_path / "run", manifest_override=prediction)
    assert episode.source.timestamp_source == "fps_assumed_constant"
    assert (
        episode.provenance["observations"]["hand_source"]["timestamp_source"]
        == "fps_assumed_constant"
    )


def test_cli_writes_new_manifest_and_prints_its_path(original, monkeypatch, capsys):
    path, model, output, _ = original
    install_fake_mediapipe(monkeypatch, [result(), result(), result()])
    hand_infer.main(
        [
            "--model",
            str(model),
            "--manifest",
            str(path),
            "--output",
            str(output),
            "--min-confidence",
            "0.7",
        ]
    )
    new_path = output / "manifest.json"
    assert capsys.readouterr().out.strip() == str(new_path)
    checked = Manifest.model_validate_json(new_path.read_text())
    assert checked.provenance["hand_source"]["min_confidence"] == 0.7

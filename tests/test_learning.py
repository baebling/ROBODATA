import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from robodata.learning.infer import RGBDetector, infer_manifest  # noqa: E402
from robodata.learning.model import LandmarkNet, heatmap_loss, heatmap_targets  # noqa: E402
from robodata.learning.train import train  # noqa: E402
from robodata.observations import Manifest  # noqa: E402
from robodata.pipeline import run_episode  # noqa: E402
from robodata.synthetic import create_demo  # noqa: E402


def test_invisible_landmark_does_not_contribute_to_loss():
    uv = torch.tensor([[[16.0, 16.0], [0.0, 0.0]]])
    visible = torch.tensor([[True, False]])
    target = heatmap_targets(uv, visible, size=32)
    good = target.clone()
    good[:, 1] = 100
    assert heatmap_loss(good, target, visible).item() == pytest.approx(0)


def test_network_backward_and_state_dict_reload():
    torch.manual_seed(3)
    model = LandmarkNet(num_keypoints=4, width=8)
    x = torch.rand(2, 3, 32, 32)
    pred = model(x)
    assert pred.shape == (2, 4, 32, 32)
    visible = torch.ones(2, 4, dtype=torch.bool)
    targets = heatmap_targets(torch.full((2, 4, 2), 12.0), visible, 32)
    loss = heatmap_loss(pred, targets, visible)
    loss.backward()
    assert np.isfinite(loss.item())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    restored = LandmarkNet(num_keypoints=4, width=8)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    model.eval()
    torch.testing.assert_close(model(x), restored(x))


def test_training_reloading_inference_records_model_provenance(tmp_path):
    inputs = create_demo(tmp_path / "inputs", frames=8)
    config_path = inputs / "train.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update(epochs=1, width=4, image_size=32)
    config_path.write_text(yaml.safe_dump(config))
    checkpoint = tmp_path / "detector.pt"
    history = train(config_path, checkpoint)
    assert np.isfinite(history["best_val_loss"])
    assert RGBDetector(checkpoint).config.image_size == 32
    output = infer_manifest(checkpoint, inputs / "manifest.json", tmp_path / "inferred.json")
    manifest = json.loads(output.read_text())
    assert len(manifest["observations"]) == 8
    assert manifest["landmark_source"].startswith("rgb_landmark_network")
    assert len(manifest["provenance"]["detector"]["checkpoint_sha256"]) == 64


def test_training_validation_leakage_is_rejected(tmp_path):
    inputs = create_demo(tmp_path / "inputs", frames=8)
    config_path = inputs / "train.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update(epochs=1, val_path="train.json")
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="leakage"):
        train(config_path, tmp_path / "detector.pt")


def test_inference_preserves_hard_occlusion_constraints(tmp_path, monkeypatch):
    inputs = create_demo(tmp_path / "inputs", frames=6, occluded=[3])
    original = Manifest.model_validate_json((inputs / "manifest.json").read_text())

    class ConfidentDetector:
        provenance = {"test": "confident_fixture_predictions"}

        def __init__(self, *args):
            pass

        def predict(self, rgb):
            first = original.observations[0]
            return {
                "tool": first.tool.model_copy(deep=True),
                "surface": first.surface.model_copy(deep=True),
            }

    monkeypatch.setattr("robodata.learning.infer.RGBDetector", ConfidentDetector)
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    output = infer_manifest(checkpoint, inputs / "manifest.json", tmp_path / "inferred.json")
    predicted = Manifest.model_validate_json(output.read_text())
    assert predicted.observations[3].tool.occluded
    result = run_episode(inputs / "rgb.yaml", tmp_path / "run", output)
    assert not result.frames[3].quality.valid
    assert not result.frames[2].action.valid

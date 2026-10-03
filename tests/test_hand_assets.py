import hashlib
from io import BytesIO

import pytest

from robodata import hand_assets


def test_asset_checksum_failure_never_writes_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(hand_assets, "ASSETS", {"bad.glb": ("https://example.test", "invalid")})
    monkeypatch.setattr(hand_assets, "urlopen", lambda *a, **k: BytesIO(b"bad"))
    with pytest.raises(ValueError, match="checksum"):
        hand_assets.prepare_assets(tmp_path / "out")
    assert not (tmp_path / "out/bad.glb").exists()


def test_existing_asset_mismatch_is_not_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "right.glb"
    target.write_bytes(b"user asset")
    monkeypatch.setattr(
        hand_assets,
        "ASSETS",
        {"right.glb": ("https://example.test", hashlib.sha256(b"new").hexdigest())},
    )
    with pytest.raises(ValueError, match="checksum"):
        hand_assets.prepare_assets(tmp_path)
    assert target.read_bytes() == b"user asset"

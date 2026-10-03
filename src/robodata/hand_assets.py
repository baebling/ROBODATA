"""Download checksum-pinned optional assets; never overwrite existing files."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

ASSETS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "right.glb": (
        "https://cdn.jsdelivr.net/npm/@webxr-input-profiles/assets@1.0/"
        "dist/profiles/generic-hand/right.glb",
        "291790c14f7f88a7f9bd35330c47392ed8e8d395ae6728f4bb7089f1bc1f2b96",
    ),
}


def prepare_assets(output: Path | str):
    out = Path(output).resolve()
    for name, (url, expected) in ASSETS.items():
        target = out / name
        if target.exists():
            content = target.read_bytes()
        else:
            with urlopen(url, timeout=90) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Asset checksum mismatch: {name}; existing files are not replaced")
        out.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as file:
                file.write(content)
        print(f"VERIFIED_HAND_ASSET {name}", flush=True)
    license_source = Path(__file__).resolve().parents[2] / "docs/references/WEBXR_HAND_LICENSE.md"
    license_target = out / "WEBXR_HAND_LICENSE.md"
    if not license_target.exists():
        with license_target.open("xb") as file:
            file.write(license_source.read_bytes())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/hand_assets"))
    args = parser.parse_args()
    prepare_assets(args.output)


if __name__ == "__main__":
    main()

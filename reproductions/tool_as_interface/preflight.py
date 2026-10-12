"""Read-only readiness check; this is not segmentation or pose inference."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from .runner import prepare_output, reproduction_revision, sha256_file, verify_source

CAMERAS = {"camera_3": "947122060225", "camera_4": "947122061051"}
MODULES = [
    "groundingdino",
    "segment_anything",
    "supervision",
    "nvdiffrast",
    "warp",
    "kornia",
    "ruamel.yaml",
    "transformations",
    "psutil",
    "imageio",
]


def check_transform(matrix: np.ndarray) -> None:
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Calibration must be SE(3); no repair or identity fallback allowed")


def check_assets(upstream: Path, camera: str) -> dict[str, Any]:
    if camera not in CAMERAS:
        raise ValueError("Only the released camera_3 / camera_4 condition is supported")
    calibration_paths = [
        "ti/real_world/robot_extrinsics/right_base_pose_in_world.npy",
        f"ti/real_world/cam_extrinsics/{CAMERAS[camera]}.npy",
    ]
    required = [
        "assets/mesh/hammer/textured_mesh.obj",
        "ti/utils/Grounded_SAM/groundingdino_swinb_cogcoor.pth",
        "ti/utils/Grounded_SAM/sam_hq_vit_h.pth",
        *calibration_paths,
    ]
    for run in ["2023-10-28-18-33-37", "2024-01-11-20-02-45"]:
        required.extend(
            [
                f"third_party/FoundationPose/weights/{run}/config.yml",
                f"third_party/FoundationPose/weights/{run}/model_best.pth",
            ]
        )
    missing = [name for name in required if not (upstream / name).is_file()]
    hashes = {name: sha256_file(upstream / name) for name in required if name not in missing}
    invalid = []
    for name in calibration_paths:
        if name not in missing:
            try:
                check_transform(np.load(upstream / name, allow_pickle=False))
            except ValueError as error:
                invalid.append(f"{name}: {error}")
    return {
        "ready": not missing and not invalid,
        "missing": missing,
        "invalid_calibrations": invalid,
        "asset_sha256": hashes,
        "calibration_capture_correspondence_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=sorted(CAMERAS), default="camera_3")
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    source = verify_source(upstream)
    assets = check_assets(upstream, args.camera)
    modules = {}
    for name in MODULES:
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            modules[name] = False
    output = prepare_output(args.output, upstream)
    report = {
        "stage": "preprocessing_readiness_only",
        "source": source,
        "reproduction": reproduction_revision(Path(__file__).resolve().parents[2]),
        "assets": assets,
        "module_present": modules,
        "ready_for_import_test": assets["ready"] and all(modules.values()),
        "native_extensions_and_model_imports_validated": False,
        "segmentation_executed": False,
        "pose_reestimation_executed": False,
        "note": (
            "Module presence and proper matrices do not verify native imports "
            "or calibration provenance."
        ),
    }
    (output / "preprocessing_readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(output / "preprocessing_readiness.json"),
                "ready_for_import_test": report["ready_for_import_test"],
            }
        )
    )


if __name__ == "__main__":
    main()

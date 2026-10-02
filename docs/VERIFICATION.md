# Local Verification: 2026-10-02

This is execution evidence, not construction-video accuracy or scientific
validation. No real footage or calibrated real gold benchmark was available.

## Checks

- Format: `ruff format --check src tests`, 25 files already formatted.
- Lint: `ruff check src tests`, no errors.
- Types: `mypy --config-file pyproject.toml src/robodata`, 18 source files pass.
- Full tests with PyTorch: 37 passed, including detector train/reload/inference.
- Fresh isolated core install: 33 passed, 1 learning module skipped without
  PyTorch. `pip check` reports no broken requirements; both demo pipelines run.
- JSON Schema generated at `docs/episode.schema.json`.
- Trajectory comparison and RGB pose overlays visually inspected: nonblank,
  visible axes/units/legends, explicit SYNTHETIC label, invalid gaps preserved.

The learning environment used Python 3.10.12, NumPy 1.26.4, OpenCV 4.10.0,
SciPy 1.15.3, Pydantic 2.13.4 and PyTorch 2.12.0. The clean core environment
used NumPy 2.2.6, OpenCV 4.14.0 and Pydantic 2.13.5. Both used pytest 9.1.1,
Ruff 0.16.10 and mypy 2.4.0. Neural training used CPU, not GPU.
The learning venv shares existing system packages; its `pip check` reports an
unrelated pre-existing PyGObject/Pycairo dependency mismatch. The isolated core
environment does not inherit that mismatch. ROBODATA does not use those packages.

## Synthetic Geometry Fixture

`robodata demo --output outputs/final_demo` rendered 40 frames, one deliberately
occluded. Analytic renderer poses are the independent fixture reference. RGB
uses exact projected semantic landmarks, not neural predictions. RGB-D uses
aligned quantized depth and the initial surface mask.

| Pipeline | Valid frames | Position RMSE (m) | Axis MAE (rad) |
|---|---:|---:|---:|
| RGB with exact projected landmarks | 39/40 | 6.91e-08 | 1.19e-07 |
| RGB-D depth/keypoint fixture | 39/40 | 0.007974 | 0.001874 |

Near-zero RGB fixture error only verifies geometry/conventions. It is not
evidence that automatic RGB extraction is more accurate than RGB-D.

## Learned RGB Smoke Run

`configs/train_synthetic.yaml` trains 12 landmark heatmaps/visibility heads for
80 epochs, width 16, input 64x64, learning rate 0.002, batch size 8, seed 42.
It uses 32 training frames and an 8-frame chronological holdout from the same
synthetic episode. Final validation landmark MAE was 0.5584 resized pixels.
The saved best-loss checkpoint was reloaded and inference run on the rendered
MJPG video. Both detector and geometry thresholds were 0.5 for landmark scores;
PnP maximum reprojection error was 4 pixels. No test-trajectory fitting,
interpolation or smoothing was used.

Across the whole 40-frame episode (including training frames), 39 valid poses
had position RMSE 0.051405 m, axis MAE 0.110056 rad, full-rotation MAE
0.118835 rad and standoff MAE 0.034739 m. These are end-to-end smoke measurements,
not held-out real-data or independent-scene generalization results. Small
2D landmark errors still produced substantial depth/orientation error. More
real data, geometric observability analysis and detector validation are needed.

Local outputs are `outputs/learned_rgb_final/episode.json`, `evaluation.json`,
`comparison.png`, `tensors.npz`, and overlays. The checkpoint is
`outputs/demo/landmarks80.pt`; weights and generated data are intentionally
not committed. Recreate them using the README commands and a new output path.

## Remaining Gaps

Automatic segmentation and process/primitive inference, arbitrary-tool metric
reconstruction, confidence calibration, DITTO-style correspondence warping,
real gold measurements and robot execution are not verified or implemented.
Known-geometry RGB is a disclosed baseline assumption. Task selection and the
functional frame definition for each real tool still need researcher review.

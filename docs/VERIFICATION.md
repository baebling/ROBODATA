# Local Verification: 2026-10-02

This is execution evidence, not construction-video accuracy or scientific
validation. No real footage or calibrated real gold benchmark was available.

## Checks

- Format: `ruff format --check src tests`, 25 files already formatted.
- Lint: `ruff check src tests`, no errors.
- Types: `mypy --config-file pyproject.toml src/robodata`, 18 source files pass.
- Full tests with PyTorch: 46 passed, including detector train/reload/inference.
- Fresh isolated core install: 41 passed, 1 learning module skipped without
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
The input's explicit occlusion flag is preserved as a hard validity constraint;
this is not an automatic occlusion-detection benchmark.

Across the whole 40-frame episode (including training frames), 39 valid poses
had position RMSE 0.051405 m, axis MAE 0.110056 rad, full-rotation MAE
0.118835 rad and standoff MAE 0.034739 m. These are end-to-end smoke measurements,
not held-out real-data or independent-scene generalization results. Small
2D landmark errors still produced substantial depth/orientation error. More
real data, geometric observability analysis and detector validation are needed.

Local outputs are `outputs/learned_rgb_reviewed/episode.json`, `evaluation.json`,
`comparison.png`, `tensors.npz`, and overlays. The checkpoint is
`outputs/demo/landmarks80.pt`; weights and generated data are intentionally
not committed. Recreate them using the README commands and a new output path.

## Remaining Gaps

Automatic segmentation and process/primitive inference, arbitrary-tool metric
reconstruction, confidence calibration, DITTO-style correspondence warping,
real gold measurements and robot execution are not verified or implemented.
Known-geometry RGB is a disclosed baseline assumption. Task selection and the
functional frame definition for each real tool still need researcher review.

## Blender Pilot (Additional Run)

Blender 5.2.2 LTS rendered 7 x 30 frames with Cycles GPU: horizontal, vertical,
zigzag with fixed/moving cameras, and stationary-tool moving-camera control.
All 210 RGB frames have float32 axial depth and object labels. Depth samples
were checked against independent scene ray intersections before import.
Both existing pipelines ran on all 7 cases, with 30/30 valid states and 24
matched 0.2-second action targets per case/mode. No trajectory fitting was used.

RGB oracle position RMSE ranged from 0.000771 to 0.000874 mm. These numerical
geometry residuals are not detector accuracy. RGB-D rendered-depth position
RMSE ranged from 6.522 to 6.621 mm; semantic sphere-center annotations and
surface depth samples differ systematically. Both modes used surface PnP;
unknown-surface reconstruction is not tested. In the stationary control,
predicted span was 0.002715 mm (RGB) and 0.633607 mm (RGB-D).

Outputs: `outputs/blender_hbeam_20261002_01` (raw renders/scenes),
`outputs/blender_hbeam_results_20261002_02` (final episodes, reports and 210-frame GIF).
The native Blender window opened the saved fixed-horizontal scene; animation
was visually verified at different frames. The original open scene was saved
separately as `outputs/blender_original_before_simulation_20261002_01.blend`.
No add-ons were installed and no raw files were overwritten.

This initial 210-frame pilot started before renderer-source snapshot support
was added. Its missing render-time source hash/Git state is disclosed in the
report; evaluation provenance is not substituted for it. A fresh 4-frame
stationary/moving-camera smoke run in `outputs/blender_provenance_smoke_20261002_02`
verified the latest renderer hash (matching source bytes), effective config,
render-time commit `d85d257f7d5ce16e633a3bad1ea4e11cadaa999c` and dirty=true.
The selected-scenario parent directory evaluated successfully. This verifies
future provenance capture, not retroactive provenance for the full pilot.

Fresh checks after the extension: format covers 29 source/test/script files;
lint passes; mypy passes 20 source files; full tests pass 68/68 with PyTorch.
22 Blender-boundary tests cover axes, declared depth conventions, invalid
samples, truncation, static-surface reanchoring, invalid ray checks, timestamp
ordering, raw-tree protection, GT/generative-motion separation and stationary
tool recovery with a moving camera. A read-only independent code review was
used to identify and repair the data-integrity checks.
Conflicting per-episode and parent-configuration frame counts are rejected
before writing. The full 7-case import/extraction/evaluation was rerun after
the final validation changes and completed successfully.

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

## Web-Facing Follow-Up (2026-10-02)

The researcher approved switching the spraying example from the top flange to
the near web face. `configs/blender_web_simulation.json` renders 11 conditions,
12 frames at 10 FPS each, 1280x720, 16 Cycles samples, seed 42. All 132 frames
include RGB, axial depth, renderer labels and actual simulator transforms.
The web frame remains fixed: face center, +X beam length, +Y up, +Z outward.
The native Blender window opened `fixed_zigzag/hbeam_spray.blend`; animation
was verified at frames 1 and 6, then paused. No original/raw scene was overwritten.

Finite nozzle-axis/beam intersections agree with actual Blender mesh rays in
every rendered frame. GT includes 96 work-face hits, 12 top-flange blockages,
12 beam-end misses and 12 away-pointing rays. Reference ray labels/poses are
separate from inference; estimated relations use estimated points/axes and known
configured beam dimensions, not unknown geometry reconstructed from video.

RGB produced 41/132 valid states; RGB-D produced 12/132, all in the away control.
In the original seven target-facing movement/control conditions, RGB coverage
was only 5/84, RGB-D 0/84. The additional frontal control was invalid 12/12 in
both modes, so all target-facing web-hit conditions total RGB 5/96, RGB-D 0/96.
Every valid estimated ray status matched its reference, but unknown frames are
NOT successes. The moving-stationary control had no valid states; no invariance
claim is supported by this new view. Occlusion of functional/tool landmarks is
the main limitation, not demonstrated automatic perception performance.

The flange, miss and away controls have 12/12 valid RGB estimates each.
Away-control RGB-D position RMSE is 6.627442 mm, axis MAE 0.119826 degrees,
retaining the sphere-center versus sampled-depth offset. Oracle RGB residuals
are numerical geometry checks, never trained-detector or real-data accuracy.

Raw data: `outputs/blender_web_20261002_01`; final full evaluation and labelled
132-frame GIF: `outputs/blender_web_results_20261002_02`. The existing 210-frame
top-flange pilot also re-evaluated successfully in
`outputs/blender_legacy_compat_20261002_01` with both modes valid 30/30 per case.
Previous reports and inputs are preserved. A fresh current-renderer four-frame
run in `outputs/blender_web_source_smoke_20261002_01` and its separate evaluated
results verify the post-review renderer, rather than rewriting the full-run
provenance. The full run predates contact-edge fixes/formatting; each run retains
its actual renderer hash, parent commit c1126e9 and dirty=true.

Fresh checks: 91 pytest tests passed with PyTorch available. The 45 focused
Blender boundary/geometry tests cover web/top identity separation, handedness,
oblique versus orthogonal intersections, both flanges, miss/away/parallel,
invalid axes/dimensions, GT separation, unknown preservation and visible no-hit
estimated rays in a separate preview label band. Six actual Blender geometry
checks pass, including the reviewer-reported 5-micrometer standoff regression
which failed before the origin-offset fix. Format/lint and mypy are checked
separately before commit; independent read-only code review was used.

## Hopper Visibility Follow-Up: 2026-10-03

The [hopper pilot](HOPPER_SIMULATION_PILOT.md) rendered/evaluated 12 episodes,
96 frames and 24 paired RGB/RGB-D runs. All 96 nozzle rays independently agree
with actual Blender beam meshes. Renderer source hash matches every raw episode.
The three actual hopper mesh checks preserve attachment/body/nozzle dimensions
and part IDs; together with six beam-ray controls, nine Blender checks pass.

Fresh verification: 115 pytest tests pass with PyTorch available; 35 files pass
format checking (including renderer), lint passes, and mypy passes 22 source
files. Added tests cover amodal ROI, depth-aware first-hit attribution, missing/
empty ROI nulls, separate out-of-view area, validity gaps, hopper-only scaling,
multi-part tool masks and evaluation-reference exclusion from inference. An
independent read-only review found no actionable issues in the change. Raw
images, the labelled contact sheet and report were inspected. Boundary raster
aliasing is not exhaustively tested; no practical perception accuracy is claimed.

## Helmet Camera / Hand Assistance: 2026-10-03

Final raw/evaluation/model outputs are the version-02 folders recorded in
HELMET_SENSOR_HAND_PILOT.md. The rigid helmet mount and four nominal sensor
profiles render all 16 episodes / 128 frames; independent depth checks pass at
the import boundary. All renderer hashes match. Phone inference excludes depth.
Actual MediaPipe 0.10.21 VIDEO inference produces 24 associated hands but zero
accepted hand poses and zero canonical poses. RGB-D is 0/64; errors are null,
not zero. Failed frames, occlusions and evaluation-only GT are preserved.

Fresh verification: 193 pytest tests pass with the existing learning runtime;
format/lint and mypy pass (27 source files). Thirteen actual Blender checks pass,
including four camera projection/mount/hand-asset checks and the nine previous
beam/hopper checks. Optional hand-runtime `pip check` passes. Images and the
selected-prediction contact sheet were inspected.

Independent review found missing handedness binding, stale raw/baseline pairing,
nonrigid up offset, scale/timestamp provenance and overflowing labels. Fixes and
regressions cover each. An opt-in calibrated hand path may infer an occluded
tool while preserving its flag/method; an occluded hand still rejects inference.
Final read-only review of that behavior found no issue. Real calibration, worker
identity, grip changes, articulated hand templates, gloves, sensor error and
unknown-surface reconstruction remain unverified.

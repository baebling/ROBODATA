# ROBODATA

Research code for video-derived **surface-relative tool-motion datasets**.
The main record is `T_surface_tool(t)` with functional point/axis, standoff,
incidence, time derivatives, future targets, failures and provenance.

This version implements calibrated RGB landmark PnP, optional checked optical
flow, aligned RGB-D plane/tool estimation, a trainable RGB heatmap/visibility
network, independent GT import/evaluation, masked NPZ export and visual reports.
It is a runnable research baseline. Real construction-video accuracy is not
established; no real pretrained tool detector is bundled.

The isolated [Tool-as-Interface reproduction](reproductions/tool_as_interface/README.md)
uses public hammer data and a separate environment. Its upstream component checks
are not construction-video validation and do not replace this research pipeline.

## Run

Python 3.10+ (WSL/Linux or Windows). From this repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/robodata demo --output outputs/demo
```

On Windows use `.venv\Scripts\python.exe` and `.venv\Scripts\robodata.exe`.
The demo renders a 40-frame RGB video and depth, extracts both pipelines, saves
JSON/NPZ, compares against analytic truth, and creates trajectory/overlay figures.
One frame is deliberately occluded. Choose a new output folder for each run.

## Blender Simulation

Blender 5.2 renders an H-beam and a calibrated proxy spray tool without add-ons.
Seven fixed/moving-camera conditions export RGB, float EXR axial depth,
object-index masks, oracle 2D landmarks and separate simulator-transform truth.
The existing RGB/RGB-D pipelines produce errors, failure counts and previews:

```sh
.venv/bin/robodata blender-evaluate --render-root outputs/blender_renders --output outputs/blender_results
```

See [Blender experiment](docs/BLENDER_SIMULATION.md) for the rendering command,
coordinate conventions and limitations. This is a geometry integration test,
not a learned detector, real spraying, or an unknown-scale RGB-only benchmark.

The approved follow-up uses `configs/blender_web_simulation.json`: a persistent
web-facing frame, 11 motion/edge/occlusion conditions, and separate finite-beam
nozzle-ray diagnostics. Preview overlays distinguish GT rays from estimated RGB
rays. Nozzle occlusion remains an estimation failure, not a fabricated pose.

An optional [official-DITTO probe](docs/DITTO_INTEGRATION.md) now tests RAFT/RANSAC
relative motion on visible tool bodies in these renders. It needs the separate
DITTO runtime, metric depth and bootstrap object masks. It does not yet estimate
the initial nozzle frame or remove helmet-camera motion. ToolEENet's published
input requirements and checkpoint/category limitations are documented alongside it.

## Construction Research Positioning

The selected physical tool is Graco TexSpray G-TEX gravity-fed hopper gun,
25D497, for H-beam web spraying. The existing Blender tool remains a generic
proxy. [Research positioning](docs/RESEARCH_POSITIONING.md) compares DITTO,
EgoScaler/VLA, ORION and construction spraying research without claiming that
RGB-only extraction or task-relative tool frames are new.
The [G-TEX experiment protocol](docs/HOPPER_GUN_EXPERIMENT.md) separates physical
geometry, projected occupancy and tool-caused surface occlusion, with independent
GT and accuracy/coverage reporting. This is a proposed study, not measured
G-TEX performance. An approved [Blender hopper pilot](docs/HOPPER_SIMULATION_PILOT.md)
now implements the simulator-only visibility part: three unmeasured hopper sizes
and four viewpoint/control groups, with fixed body/nozzle/landmarks, first-hit
occlusion, occupancy, validity and temporal gaps. It is not a measured 25D497
mesh, a real-data pilot or an evaluation of learned perception.

## Train RGB Perception

The [helmet-camera / hand-assistance pilot](docs/HELMET_SENSOR_HAND_PILOT.md)
uses an optional pretrained MediaPipe hand model, calibrated fixed-grasp PnP,
and four nominal camera profiles across four helmet distance/pitch conditions.
It preserves failed frames and compares hand-assisted coverage with the existing
tool baseline. Synthetic hand calibration and nominal FoV are not real measurements.
Blender hand assets and model weights are checksum-pinned downloads, not bundled.

Install a platform-appropriate PyTorch build, or `pip install -e '.[learning]'`.
Then run the synthetic smoke example:

```sh
.venv/bin/robodata train --config configs/train_synthetic.yaml --checkpoint outputs/demo/landmarks.pt
.venv/bin/robodata infer --checkpoint outputs/demo/landmarks.pt --manifest outputs/demo/inputs/manifest.json --output outputs/demo/predicted_landmarks.json
.venv/bin/robodata run --config outputs/demo/inputs/rgb.yaml --manifest outputs/demo/predicted_landmarks.json --output outputs/neural_rgb
```

Detector confidence thresholds can cause low-coverage runs; inspect summary,
residuals and figures. Geometry tests with exact landmarks do not measure the
neural detector. Train on real semantic landmark annotations and hold out entire
workers/tools/scenes before reporting research accuracy. The smoke training uses
synthetic images and a small within-episode holdout.

## Real Input

Copy and edit `configs/rgb.yaml` or `configs/rgbd.yaml`, using measured geometry
and calibration. The manifest contains `episode_id`, `task_name`, `video_path`,
`camera` (`matrix`, `distortion`, `width`, `height`), `data_kind`, explicit
`timestamps_sec`, and frame-indexed `observations`. Each observation has
`tool`/`surface` named `keypoints` (`uv`, `visible`, `confidence`), optional masks,
aligned `depth_path`, and sourced process annotations. See generated
`outputs/demo/inputs/manifest.json` for a complete example.

RGB needs at least six visible nonplanar known tool landmarks (or four planar
points) and four measured surface landmarks. Geometry is expressed directly in
the intended functional frame. Camera motion is handled by estimating both
frames from persistent landmarks. Sparse manual initialization plus optical
flow is supported; a trained detector supplies automatic observations.

RGB-D needs rectified aligned depth, a surface mask and semantic functional
point/axis landmarks. A secondary landmark fixes roll. Plane-only references
require a static camera; moving-camera RGB-D must use surface PnP landmarks.
Masks/keypoints are bootstrap observations or model outputs, not assumed final
external research inputs. Their sources are retained with every frame.

```sh
.venv/bin/robodata run --config configs/rgb.yaml --output outputs/real_rgb_001
.venv/bin/robodata evaluate --prediction outputs/real_rgb_001/episode.json --reference data/gold/episode.json --output outputs/real_rgb_001/evaluation.json
.venv/bin/robodata export --episode outputs/real_rgb_001/episode.json --output outputs/real_rgb_001/learning.npz
```

`import-gt --csv measurements.csv --calibration calibration.yaml --output gold.json`
converts independent marker poses into the same physical surface/tool frame.
See [Data Schema](docs/DATA_SCHEMA.md) for required transforms and clock offset.
`configs/gt_calibration.yaml` is a calibration template, not a measured reference.
Tracker CSV must keep original `frame_index`, or calibration must specify the
verified `sample_period_sec`, so acquisition gaps block motion labels.
Keep raw videos/measurements under `data/raw/` or external storage; they are ignored by Git.

## Research and Checks

[Context](docs/RESEARCH_CONTEXT.md), [decisions](docs/DECISION_LOG.md),
[model](docs/SYSTEM_SPEC.md), [contract](docs/DATA_SCHEMA.md), [evaluation](docs/EVALUATION_PLAN.md),
[references](docs/references/RELATED_WORK.md), [open questions](docs/OPEN_QUESTIONS.md).
[Local verification](docs/VERIFICATION.md) separates synthetic geometry tests
from learned-detector results and real-data claims.

```sh
.venv/bin/python -m ruff format --check src tests
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy --config-file pyproject.toml src/robodata
.venv/bin/python -m pytest -q
```

Tests cover known-distance geometry, plane outliers/degeneracy, invalid SE(3),
timestamp/gap behavior, missing GT matches, RGB/RGB-D serialization, marker
calibration and neural gradients/reload. Scripts store configuration, source
commit/dirty state and software versions. Generated weights/data stay outside Git.

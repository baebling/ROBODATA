# Blender H-Beam Experiment

## Scope

Blender 5.2.2 LTS, built-in Cycles renderer, no extra add-ons. The checked-in
`configs/blender_simulation.json` defines 7 conditions: horizontal, vertical and
zigzag tool motion with fixed and moving cameras, plus a stationary-tool,
moving-camera control. Defaults: 30 frames at 30 FPS, 1280x720, 16 samples,
seed 42. These short episodes validate integration, not realistic work speed.

The illustrative spray gun has a functional nozzle frame and coloured geometry
landmarks. It is not a measured model of the researcher's actual spraying tool.
There is no human body, spray plume, fluid/material deposition, coating thickness,
process on/off detection, learned detector, IK, or robot execution in this phase.

## Render

Run this PowerShell command from the repository on Windows, replacing output
with a new folder. `blender` below denotes the installed executable; an example
installation is `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`.

```powershell
& blender --background --factory-startup --python-exit-code 1 --python scripts/blender_simulate.py -- --config configs/blender_simulation.json --output outputs/blender_renders
```

The script refuses existing output directories. `--scenario fixed_horizontal`
and `--frames 4` provide a quick initial probe; overrides are written to
`effective_config.json`, which declares only the selected scenarios. Rendering
uses CUDA where available and records the actual render device; CPU is fallback.
Existing open Blender scenes are not changed by the background process.

## Import and Evaluate

From the Python environment with the repository installed:

```sh
.venv/bin/robodata blender-evaluate --render-root outputs/blender_renders --output outputs/blender_results
```

In the local Windows/WSL setup, use UNC paths for Windows Blender, and Linux
paths for WSL Python. Do not pass Windows paths into the WSL configuration.

Each scenario exports `hbeam_spray.blend`, `render_metadata.json`, and raw
per-frame RGB PNG, float32 depth EXR, uint16 object-index PNG. Import checks
depth against independent scene ray intersections at selected pixels, assembles
video/depth/masks, and saves `manifest.json`, `rgb.yaml`, `rgbd.yaml`, and
`ground_truth.json`. Failures remain in the episode. Existing output folders
are refused, including partially completed runs; use a new folder to retry.

The evaluator saves per-mode JSON/NPZ, trajectory figures and RGB overlays,
`comparison.json`, `report.md`, and a labelled `preview.gif`. Raw renders are
read-only to the importer. Generated data, scenes and reports are Git-ignored.

## Frames and Depth

`T_A_B` maps B coordinates to A. All spatial coordinates use meters and
right-handed frames. Time is seconds and angles are radians. Blender camera
local -Z is the viewing direction and +Y is up; OpenCV camera +Z is viewing
and +Y is image-down. The basis conversion is `diag(1,-1,-1,1)`.

The persistent work frame is at the top-flange patch center: +X along the beam,
+Y across the flange, +Z outward/up. Its world height is configured. The nozzle
origin is the functional tip; tool +Z is the spray axis toward the surface, not
the camera or mask centroid. A secondary tool landmark defines roll.

The tested Blender 5.2 Cycles Depth pass is camera-axis Z depth, not off-axis
ray distance. Metadata explicitly declares `axial_z`. The importer also supports
explicit `ray_distance` from other renderers and converts that convention using
intrinsics. No convention is silently guessed. Independent ray intersections
are expressed in the declared convention for validation (5 mm absolute check).

## Interpretation

2D landmarks are exact projected marker centers, with actual scene ray-cast
visibility. They are oracle annotations, NOT outputs of a trained vision model.
Masks are renderer object indices, NOT inferred segmentation. Ground truth is
the actual simulator tool transform relative to the persistent surface frame;
these poses are stored separately and never passed to trajectory estimation.

RGB uses known metric proxy tool and surface geometry with PnP. Both modes
estimate the surface using the same persistent landmarks/PnP to isolate the
tool estimation branch. Thus the comparison does NOT establish unknown-scale
RGB feasibility or automatic work-surface reconstruction.

RGB-D samples actual rendered depth at semantic landmarks. A visible sphere's
surface is not its annotated center; finite marker radius, pixel sampling and
the estimator's 3x3 median create a systematic offset. This is retained, not
corrected using GT. Near-zero RGB error is expected in this oracle setting.
No real-data accuracy, sensor fidelity or process-quality claim is supported.

The stationary control reports predicted positional span without fitting or
aligning the trajectory to GT. Unknown process states and primitives remain
null. Reproducibility records include source commit/dirty state from the normal
pipeline, configuration hashes, rendered metadata hash, seed and Blender version.

The renderer also captures its source-file hash and render-time Git commit/dirty
state before rendering starts. If Git is unavailable, nulls and the failure reason
are recorded rather than a guessed version. Import rejects truncated declared
episodes, nonfinite/out-of-bounds ray checks, and changes to the static workpiece
surface transform. Output must be outside the raw render tree.

Opening a saved `.blend` permits interactive animation. Its render output is
redirected to `interactive_renders`, so F12 does not overwrite the dataset frames.

Blender 5 uses the updated File Output image media type and `ShaderNodeMath`;
see the [official compositor migration notes](https://developer.blender.org/docs/release_notes/5.0/migration/compositor_migration/).

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

## Web-Facing Follow-Up

Use `configs/blender_web_simulation.json` instead of the preserved top-flange
configuration. It defines 11 conditions with 12 frames at 10 FPS (132 frames):
the original seven motion/camera conditions plus top-flange blockage, beam-end
miss, away-pointing nozzle, and a front-view occlusion control. Beam dimensions,
camera placement and speed are illustrative configuration values, not measurements
of the real equipment or validated fireproofing settings.

The selected `web_front` face is the near web face at world Y = -web_thickness/2.
Its persistent origin is the physical face center, world Z = beam_top_height -
beam_height/2. Surface +X is world +X (beam length); +Y is world +Z (up); +Z is
world -Y (outward toward the worker). This is right-handed. The canonical
surface_frame_id is `synthetic_hbeam_web_front`, distinct from the top-flange pilot.
The camera is aimed at the web, with an oblique worker-view proxy and a separate
near-frontal control; it is not a calibrated helmet camera simulation.

`nozzle_relationship.json` is a separate diagnostic, not a replacement dataset
schema or a physical spray model. It records the nozzle-axis intersection with
the infinite selected plane, whether that point is on the finite work face,
the first beam solid hit, and statuses `work_face_hit`, `flange_blocked`, `miss`,
`away`, or `parallel`. The existing projected path is still orthogonal projection;
it must not be interpreted as an oblique nozzle ray or a spray footprint.

The renderer independently verifies analytic ray intersections with actual
Blender mesh rays. This check ignores the illustrative tool and calibration
markers, testing only the three beam solids. GT ray records are never included
in the inference manifest. Estimated diagnostics are computed after extraction
from estimated functional points/axes and the configured known beam dimensions.
They therefore do not test unknown-beam reconstruction or flange perception.
Unknown poses produce null diagnostics; classification accuracy is reported
alongside its evaluated-frame count, never as a claim about all frames.

The preview overlays surface axes, GT nozzle rays in yellow and valid RGB
estimated rays in magenta, with GT gap/angle/status labelled. The raw renders
remain unchanged. Geometry-valid frames may still point away or hit a flange.
Occluded landmarks remain invalid even in oracle annotation mode. In particular,
the direct RGB-D landmark method requires visibility of the functional tip and
axis endpoints; additional depth does not recover a hidden nozzle automatically.

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

## Hopper Size / Visibility Extension

`configs/blender_hopper_simulation.json` adds 12 eight-frame conditions: three
unmeasured hopper scales by fixed oblique, moving helmet-like, frontal stationary,
and moving-camera/stationary-tool groups. Body, nozzle and landmarks are fixed.
Only the hopper is scaled about its fixed attachment. Part labels 2/4/5 combine
into the input tool mask; the work-face mask remains label 3. Legacy label-2-only
renders retain their original behavior.

When `surface_roi_bounds_m` is present, `blender-evaluate` additionally writes
`occlusion_reference.json`, `hopper_comparison.png` and visibility/coverage fields
in `comparison.json`. The fixed physical ROI is projected with simulator camera
and surface transforms for evaluation only. Its denominator includes hidden
pixels within the image; camera out-of-view area is separate. First-hit labels
and rendered camera-axial depth determine tool-caused/total occlusion. A tool
behind the surface is not an occluder. Empty ROI or missing ROI depth yields
null, not zero occlusion. Surface calibration spheres are excluded as occluders.

No occlusion reference, simulator pose or ray label is supplied to inference.
Continuous validity and failure gaps use a declared constant-FPS sample duration
of 1/fps seconds. State/full-pose/action coverage remain separate.
See [pilot results and reproduction](HOPPER_SIMULATION_PILOT.md) for limitations.

# Hopper Visibility Simulation Pilot

## Scope

Executed 2026-10-03 in Blender 5.2.2 LTS using
`configs/blender_hopper_simulation.json`. Twelve episodes, eight frames each,
10 fps, 640 x 360, eight Cycles samples, seed 42. No add-ons were required.
Existing scenes and raw renders were not overwritten.

The new shape is an **unmeasured illustrative G-TEX-style proxy**, not a CAD
reconstruction or an actual dimensional measurement of Graco 25D497. Assumed
body dimensions are 0.12 x 0.08 x 0.20 m. The nominal hopper is a solid tapered
frustum, 0.28 m high with 0.035/0.125 m bottom/top radii. The real hopper opening,
housing, handle and nozzle details are not reproduced. These numbers are
engineering test inputs, not product specifications.

Hopper scale 0.75/1.0/1.25 changes only the hopper, about a fixed attachment;
body, nozzle, functional frame, oracle landmarks, path, camera sequence, lighting
and seed stay fixed within each view group. Nozzle standoff is 0.32 m, against
the front web. A persistent 0.8 x 0.4 m web ROI defines the amodal denominator.
The functional origin is the nozzle; mask centroid is stored separately.

Both pipelines use exact projected, visibility-checked calibration landmarks
and known geometry. RGB uses PnP; RGB-D uses sampled rendered depth at functional
landmarks, with surface PnP. This is NOT automatic tool/surface detection or
unknown-scale monocular reconstruction. No DITTO/EgoScaler/ORION learned-model
comparison was run on this new scene. The prior DITTO probe remains separate.

## Results

Percentages below are episode means. Each row represents eight frames. RGB-D
has zero valid states in every condition; its direct functional-depth baseline
cannot observe the hidden tip/axis pair, or fails the surface-landmark requirement.

| View / control | Hopper scale | Tool image occupancy % | Tool-caused web ROI occlusion % | Valid RGB frames |
|---|---:|---:|---:|---:|
| Fixed oblique | 0.75 | 2.91 | 11.88 | 3/8 |
| Fixed oblique | 1.00 | 3.98 | 12.95 | 2/8 |
| Fixed oblique | 1.25 | 5.37 | 13.82 | 2/8 |
| Moving helmet-like | 0.75 | 7.26 | 20.27 | 0/8 |
| Moving helmet-like | 1.00 | 10.52 | 22.97 | 0/8 |
| Moving helmet-like | 1.25 | 14.18 | 24.92 | 0/8 |
| Frontal stationary | 0.75 | 16.74 | 27.26 | 0/8 |
| Frontal stationary | 1.00 | 18.09 | 29.64 | 0/8 |
| Frontal stationary | 1.25 | 19.45 | 32.06 | 0/8 |
| Moving camera / stationary tool | 0.75 | 7.30 | 25.79 | 1/8 |
| Moving camera / stationary tool | 1.00 | 10.59 | 28.72 | 0/8 |
| Moving camera / stationary tool | 1.25 | 14.05 | 30.87 | 0/8 |

RGB state/full-pose coverage is **8/96 (8.33%)**; RGB-D is **0/96**. The longest
valid RGB run is two frames, or 0.20 s under the declared 1/fps sample-duration
convention. No valid temporal actions are available at the configured 0.2 s
horizon. No interpolation or hidden-frame deletion is applied.

All 96 simulator nozzle rays hit the web, independently verified against Blender
beam meshes. The eight valid RGB ray classifications agree with these references;
the other 88 frames are unknown, NOT successful web-hit predictions. Oracle RGB
position errors are numerical checks of projection/coordinate geometry, not
practical detector accuracy. The stationary small-hopper control has only one
valid state, so it cannot support a temporal camera-invariance claim.

All fixed web ROIs are in the image and have known rendered depth. Some hopper
tops are cropped by the camera; occupancy measures only visible in-image pixels.
Occlusion uses the first visible scene hit and positive depth in front of the
ROI plane, not mask overlap alone. Surface calibration spheres are excluded.
Visibility references stay outside the inference manifest.

## Interpretation

Larger illustrative hoppers increase image occupancy and web occlusion in all
four matched groups. Viewpoint changes also alter overlap substantially, so
physical size, projected occupancy and occlusion are not interchangeable.
These twelve short controlled episodes are descriptive, not independent
statistical evidence about commercial tools or construction workers.

The current pipeline loses both surface and tool support under occlusion, even
with perfect visible landmark coordinates. This is a useful failure baseline:
depth availability does not restore an invisible semantic nozzle landmark.
It does NOT show that RGB-D generally fails or that RGB is superior. A next
comparison should estimate visible body pose and apply an independently measured
rigid nozzle offset, retaining body/nozzle uncertainty and lost-surface frames.

Still untested: measured 25D497 geometry, hand/arm occlusion, material plume,
motion blur, learned perception, unknown camera calibration/metric scale,
nonplanar surfaces, new-scene trajectory warping, actual coating and independent
mocap/LiDAR GT. Process state and primitives remain unknown. No novelty or
fireproofing quality claim follows from these synthetic results.

## Outputs / Reproduction

Raw: `outputs/blender_hopper_20261003_01`. Evaluation:
`outputs/blender_hopper_results_20261003_01`, containing `report.md`,
`comparison.json`, `hopper_comparison.png`, labelled `preview.gif`, per-mode
trajectories/evaluations and per-frame `occlusion_reference.json`.
Each raw case contains a separate `hbeam_spray.blend`; interactive rendering
targets `interactive_renders`, not original dataset frames. Generated files are
ignored by Git; this summary and the code/configuration are tracked.

Run from the repository root with NEW output directories:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background --factory-startup --python-exit-code 1 --python scripts/blender_simulate.py -- --config configs/blender_hopper_simulation.json --output outputs/blender_hopper_new
```

```sh
.venv/bin/robodata blender-evaluate --render-root outputs/blender_hopper_new --output outputs/blender_hopper_results_new
```

The full raw run records parent commit `5eb7282` and `dirty=true` because this
implementation was tested before being committed. Its renderer SHA-256 is
`b59fe32640dbf59727185efeff9346075b400dfa9246728012c9be71313fef50`;
effective configuration SHA-256 is
`a176ce9a6db4639ca914d257ca6d818e7d90934e9d68fc3628f98efc08df29c6`.
Do not rewrite that provenance as a clean committed run.

# DITTO and ToolEENet Integration Probe

Checked 2026-10-03. This experiment invokes the local official DITTO source;
it does not replace ROBODATA's production perception pipeline.

## Roles and readiness

- [ToolEENet](https://arxiv.org/html/2404.04193v1) estimates functional EE pose
  from RGB-D affordance segmentation and partial point clouds. Ground-truth
  pose is a training label; inference initializes diffusion sampling from noise.
  The published categories cover hammer, screwdriver and wrench parts, not spray
  guns. [Its source README](https://github.com/yl-wang996/ToolEENet) still announces
  future checkpoint/data release. No trained checkpoint was found in the local
  checkout. A spray-gun implementation requires suitable training assets/weights.
- [DITTO](https://github.com/robot-learning-freiburg/DITTO) supplies learned RGB
  correspondences, depth-based rigid relative motion and trajectory warping.
  Its original experiments use static RGB-D cameras. Moving helmet cameras need
  a persistent surface reference to remove camera motion.

Our proposed combination is an initial functional tool pose, followed by
DITTO body tracking, then ROBODATA surface-relative conversion. Tracking visible
body points can carry an initialized hidden nozzle; it cannot identify the
initial functional origin/axis by itself. Drift and complete body occlusion
still require evaluation.

## Reproduce the probe

Use an isolated DITTO environment with its dependencies and CUDA PyTorch. Pass
existing source paths; no local absolute paths are embedded in the script.
FlowControl must use the upstream `RAFT_package` branch and RAFT the
`feature/package` branch specified by its README, with official model weights.
The former installed environment referenced removed `/tmp` checkouts. The trial
restored these sources under an ignored output directory without modifying
existing third-party changes or ROBODATA's environment.

```sh
python scripts/probe_ditto_blender.py \
  --ditto-root ../DITTO \
  --flowcontrol-root outputs/ditto_probe_20261003_01/flowcontrol \
  --raft-root outputs/ditto_probe_20261003_01/RAFT \
  --render-root outputs/blender_web_20261002_01 \
  --prepared-root outputs/blender_web_results_20261002_02 \
  --config configs/ditto_probe.json \
  --output outputs/ditto_probe_new/results
```

Prepared inputs must come from `robodata blender-evaluate`; its `.npy` depth is
already aligned axial depth in meters. This avoids OpenCV 5's failure to decode
these rendered EXR channels in the older DITTO environment.
The adapter requires matching render-metadata SHA-256, episode identity,
calibration, frame indices and the declared axial-meter depth convention before
using prepared inputs.

The probe calls official `Step3DMethod.flow_ransac` and
`warp_3D_trajectory`. Oracle source/target object masks define a shared RGB crop;
the eroded endpoint masks limit correspondences. We add a target-mask guard
before the official RANSAC to reject background/occluder matches, and retain
empty endpoint masks as failures. This guard is an adapter addition to DITTO.
The functional keypoints and
simulator poses are not inference inputs. This isolates tracking from automatic
segmentation. Renderer markers/textures remain present in the RGB images.

All parameters and pair selection are in `configs/ditto_probe.json`. Outputs
record source commits/dirty state, configuration/script/input hashes and software
versions. The restored official `raft-things.pth` used in this trial has SHA-256
`fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`.

## Interpretation and quality gates

The estimated transform maps source-camera tool points into target-camera tool
points. Inference never consumes simulator transforms. After fitting, independent
simulator transforms evaluate relative rotation and the transfer of the reference
nozzle location. That location is used only as an evaluation query: the resulting
error is not an automatically estimated absolute nozzle pose.

Upstream DITTO can return identity when RANSAC fails. We retain those failures,
require enough finite/inlier matches and reject collinear inlier sets. An observed
stationary identity can still be valid. Inlier scores are uncalibrated geometric
checks, not probabilities or guaranteed correctness.

The warp check passes arbitrary pose fixtures through the official API using the
estimated transform. It verifies transform composition only. It does not test
cross-scene correspondence, path adaptation, changing beam dimensions or robot
execution. Camera motion is included in the moving-camera relative transform;
this probe does not claim to have removed it.

To generate functional surface-relative poses, first obtain
`T_camera_tool(t0)` in the actual nozzle frame. Propagate it with the estimated
relative transform and combine with independently estimated
`T_camera_surface(t)`. For the same surface geometry, a downstream rigid transfer
can use `T_camera_new_surface @ T_surface_tool_demo(t)`. Adapting the path to a
different web shape or size needs a separately validated mapping that preserves
physical standoff and tool orientation constraints.

The test suite checks transform direction, true stationary identity versus failed
fits, insufficient/degenerate observations and metric depth back-projection.
The trial measurements and limitations are in the ignored run's `results.json`.

## Trial on 2026-10-03

Two preselected pairs, frames 0-1 and 5-6, were run in each episode:

| Condition | Fit checks accepted | Reference nozzle transfer error | Relative rotation error |
|---|---|---|---|
| Fixed camera, horizontally moving tool | 2/2 | 0.723-1.234 mm | 0.216-0.466 deg |
| Moving camera, stationary tool | 2/2 | 0.533-1.825 mm | 0.137-0.390 deg |
| Fixed front view, stationary occluded tool | 2/2 | Numerical zero | Numerical zero |

The last condition has no actual motion and tests observed identity only. The
first four pairs are the informative motion trials. The moving-camera results
compare camera-relative transformations, including camera motion; they do not
establish stationary surface-relative trajectories. Six selected pairs are not
an estimate of whole-video coverage, real-video accuracy or RGB-only feasibility.
All six official warp API composition checks had zero residual. Full repository
verification passed: 106 unit tests, formatting, lint and existing source typing.

Run: `outputs/ditto_probe_20261003_01/results_03/results.json`. Its script SHA-256
matched the verified source. The separate DITTO runtime and model/source downloads
remain outside versioned code.

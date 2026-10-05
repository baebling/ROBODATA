# YOLO Surface Visibility Pilot

## Question and Scope

Can learned work-surface segmentation, tracking and a persistent surface
reference improve helmet-camera localization when the hopper and hand obscure
the H-beam web? This follow-up executes actual YOLO inference, not oracle masks.
It does not upgrade the previous tool/hand estimator or produce a new complete
`T_surface_tool` trajectory. All scenes use an unmeasured G-TEX-like proxy.

The three conditions share one frozen checkpoint and the same 128 RGB frames:

1. YOLO26n-seg per frame, then RGB fixture PnP inside the predicted web mask.
2. YOLO26n-seg plus BoT-SORT, then the same fixture PnP.
3. YOLO26n-seg plus BoT-SORT and planar reference-image localization.

BoT-SORT IDs identify visible patches, not persistent physical work-face identity.
Its image-space tracking/camera compensation is not metric camera localization.
The map condition is a bounded planar implementation, not background SLAM.

## Inputs and Separation

The inference boundary receives RGB images, camera intrinsics, known metric
surface fixture coordinates, a tool-free RGB reference and trained weights.
It receives no simulator poses, depth, masks or projected GT keypoints. Four
colored fixtures are detected directly from RGB for calibration. Current-frame
fixture baselines require fixtures to survive the predicted web mask. The
tool-free reference uses full RGB for fixture detection and the predicted web
mask for reference feature selection.

These additional calibration inputs are explicit priors: this is not arbitrary
uncalibrated, markerless RGB reconstruction. The map recovers reference pose
from RGB PnP, intersects calibrated rays with the known plane, and matches
current RGB against that reference using LK, forward/backward checks,
homography RANSAC, spatial-spread gates and PnP residual/cheirality checks.
The current predicted tool mask is dilated and excluded. GT hand/tool masks
are never used for that exclusion. A failed localization returns no pose;
the last pose is not frozen, interpolated or silently copied.

Simulator labels and transforms enter only scoring after inference. Label 3
is a web-object proxy including its side faces and fixture markers, not a
measured physical work ROI. Training polygons use separate connected components
and external contours; filled holes and degenerate fragments approximate mask
topology. The score is visible-label IoU, not recovered amodal surface coverage.

## Training and Evaluation

Actual CUDA training: Ultralytics 8.4.78, YOLO26n-seg, RTX 3080 Ti, 40 epochs,
seed 42, image size 640, batch 8, AdamW, deterministic mode and AMP disabled.
All arguments and augmentation settings are in `configs/yolo_surface_pilot.json`.

- Train: 72 images, nine frontal/oblique/helmet hopper-size episodes.
- Validation: 24 images, three stationary hopper-size episodes.
- Evaluation: 128 images, sixteen helmet sensor/distance/pitch episodes with hand.
- Initialization: sixteen separately rendered tool-free RGB references.
- Classes: web patch, flange and tool. Hand is not a trained fourth class.

The reserved helmet images were not used for training or checkpoint selection.
The runner chooses validation `best.pt` and records source/decoded-pixel hashes.
Evaluation checks completed-training status, checkpoint and manifest hashes,
the exact metadata-listed test inventory, frame/time binding and RGB/metadata
hashes, and byte/pixel overlap against train/val. Hash checks exclude identical
images, not shared synthetic geometry or correlated scenes.

The first evaluation exposed reference fixtures being clipped by the learned
mask. That calibration bug was fixed without changing weights or thresholds;
old outputs remain preserved. These are training-held-out development
diagnostics, not an untouched final statistical test. Future development needs
a new reserved test set. Validation mask mAP is not helmet-test accuracy.

Checkpoint SHA-256:
`5ce51c81b648c304d2794f3909b86bcb766fa40a8642e6151abc924e94494378`.

## Results

The final guarded run is `outputs/yolo_surface_results_20261003_03`.
"Accepted" means passed the estimator's internal gates, not proved accurate.
Pose errors are conditional on accepted frames; the methods accept different
subsets, so their averages are not paired accuracy comparisons.

| Condition | Web mask present | Mean visible web IoU | Accepted surface poses | Mean position error | Mean rotation error |
|---|---:|---:|---:|---:|---:|
| Per-frame YOLO | 64/128 | 0.3237 | 6/128 | 0.0005 m | 0.1341 deg |
| YOLO + BoT-SORT | 52/128 | 0.2868 | 6/128 | 0.0005 m | 0.1341 deg |
| YOLO + BoT-SORT + planar map | 52/128 | 0.2868 | 9/128 | 0.0779 m | 4.3849 deg |

The map increases accepted coverage slightly, but does not demonstrate overall
accuracy improvement. In the Helios far/aimed synthetic view it accepts 8/8
frames versus 2/8 for fixture baselines. Seven map estimates there have position
errors between 0.0008 and 0.0115 m; another has 0.0670 m and 7.80 deg error.
This is an image-geometry result, not evidence of real Helios sensor superiority.

The only accepted ultrawide far/aimed map estimate is wrong by **0.5852 m and
27.54 deg** despite passing internal gates. This is a false geometric acceptance,
not a success. Per-frame/BoT-SORT also accept four main-camera far/aimed frames
where the map never initializes. Overall nine accepted map poses do not mean
nine correct poses. Missing estimates have null errors, never zero error.

Only two of sixteen maps initialize: the other references lose fixtures or
lack sufficient usable learned-mask features. Narrow/near views can lose the
fixtures even without the tool. The contact sheet shows predicted web masks
including floor/flange regions in some views and absent masks in many helmet
conditions. BoT-SORT also reports insufficient matching points. This small,
shared-asset training set does not generalize adequately to these views.

## Interpretation and Next Experiment

The bottlenecks remain learned web segmentation, calibration visibility,
usable texture and reliable camera-to-plane estimation. Tracking alone cannot
recover metric geometry when observations are missing. Low planar residuals
can still accept wrong pose on mismatched/nonplanar regions; the present gates
are insufficient. Do not reduce thresholds after inspecting this test and
report the result as independent evidence.

A subsequent experiment should collect more diverse helmet views, hands,
hopper occupancy and backgrounds; label web versus flange/floor consistently;
and reserve new scenes/materials for evaluation. Compare robust background
camera tracking and geometry verification against the present planar baseline.
Unknown real metric scale, actual sensor calibration and independent mocap/
LiDAR registration remain unresolved. Full occlusion should remain unknown
unless independent camera/background evidence supports a maintained map.

## Reproduction and Artifacts

Run from the repository root with a separate YOLO-capable Python environment.
The actual runtime is `/home/kmkm/.venvs/ROBODATA_YOLO`, separate from the
MediaPipe hand environment. The optional `yolo` extra pins Ultralytics 8.4.78
and lap 0.5.12. The executed environment reuses existing CUDA/PyTorch packages.
Its `pip check` reports an inherited unrelated `pygobject`/missing-`pycairo`
conflict; it is not a dependency-clean runtime, although training/inference ran.

```bash
python -m robodata.yolo_training --config configs/yolo_surface_pilot.json \
  --output outputs/yolo_surface_training_NEW
python -m robodata.yolo_surface_experiment \
  --config configs/yolo_surface_evaluation.json \
  --checkpoint outputs/yolo_surface_training_20261003_01/checkpoint.pt \
  --output outputs/yolo_surface_results_NEW
```

The evaluation configuration must point to the matching completed training run
and reference captures. Outputs must be fresh and outside input trees.
`scripts/blender_surface_reference.py` runs in background Blender against saved
raw scenes, hides tool/hand render objects, disables compositor output nodes
and writes new RGBs without saving or overwriting source scenes. Blender 5.2.2
rendered the sixteen reference captures. Their manifest records executed script
and source-scene hashes; later script formatting does not alter that history.
Existing interactive Blender edits were not touched.

Artifacts (generated images/weights remain Git-ignored):

- `outputs/yolo_surface_training_20261003_01`: split manifest, weights, log, provenance.
- `outputs/yolo_surface_references_20261003_01`: tool-free RGB, source-scene hashes.
- `outputs/yolo_surface_results_20261003_01`: preserved failed initialization run.
- `outputs/yolo_surface_results_20261003_02`: corrected calibration diagnostic.
- `outputs/yolo_surface_results_20261003_03`: guarded final diagnostic, per-frame
  masks/poses/errors, comparison JSON, configuration, report and contact sheet.

Each result records Git commit/dirty status, configuration, source and weight
hashes and runtime versions. Source changes are not yet committed. No real
construction accuracy, material thickness, coverage quality or policy success
is established.

Primary implementation references:
[YOLO26](https://docs.ultralytics.com/models/yolo26/),
[segmentation](https://docs.ultralytics.com/tasks/segment/), and
[tracking](https://docs.ultralytics.com/modes/track/).

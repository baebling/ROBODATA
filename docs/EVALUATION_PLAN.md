# Evaluation Plan

1. Collect held-out worker/tool/scene episodes with RGB and optional aligned
   sensor depth. Keep collection conditions, FPS, camera intrinsics, occlusion,
   illumination, tool identity and calibration files with the raw data.
2. Acquire an independent gold subset with mocap/calibrated tracking for the
   functional tool frame and surveyed/LiDAR surface geometry. Synchronize clocks
   and calibrate the marker-to-functional-point and surface-to-tracking transforms.
3. Apply identical temporal label rules to prediction and gold trajectories.
   Match timestamps one-to-one within an explicit tolerance. Report unmatched,
   invalid and orientation-unobserved frames instead of hiding them.
   Sparse minimum-cost assignment maximizes matched coverage first, then
   minimizes total timestamp error and orders tied matches chronologically.
4. Compare RGB manual-landmark, RGB tracked-landmark, learned RGB-landmark and
   aligned RGB-D variants on the same episodes. Manual landmarks isolate geometry
   errors but do not establish automatic extraction. Hold out entire episodes,
   workers, tool instances and scenes for research results.
5. Report position RMSE/MAE, functional-axis angle, full-rotation error only when
   observable, standoff/incidence MAE, action displacement error and valid-label
   coverage. Add segmentation/keypoint error, annotation time, failure rate and
   confidence reliability when real labels are available.

## Construction geometry and occlusion study

First hardware is G-TEX 25D497 on the H-beam web. Follow the proposed
[hopper-gun protocol](HOPPER_GUN_EXPERIMENT.md) and
[reference comparison](RESEARCH_POSITIONING.md) before collecting a benchmark.
Separate physical geometry, projected occupancy, nozzle visibility, tool-caused
web occlusion and other occluders. Use the same physical surface ROI and declare
how hidden area is measured. Keep GT/oracle masks out of automatic inference.
Existing synthetic proxies do not model the measured G-TEX.

Compare matched RGB frames with/without aligned depth; smartphone versus ZED
views are a separate camera-domain experiment. Include a calibrated object-body
baseline with the same body-to-nozzle prior. Test joint tool/surface validity
ablations using error/coverage curves and calibration-split thresholds.

Report full-pose, axis-only and temporal-label coverage, contiguous durations,
gaps, reacquisition delay and GT matching coverage alongside errors. Aggregate
uncertainty by independent episode/group and preserve failed frames. Ray-hit/
flange/miss diagnostics do not establish deposition or quality. The limited
[simulator pilot](HOPPER_SIMULATION_PILOT.md) implements first-hit fixed-ROI
occlusion, occupancy, full/state/action coverage, constant-FPS contiguous runs
and gaps as evaluation outputs. Learned-model ablations, reacquisition analysis,
risk/coverage curves and independent real GT remain future benchmark work.

## Current implementation and evidence

The included CLI implements calibrated reference import, point/axis/rotation and
temporal displacement evaluation. Automatic force, pressure, flow, material
quality and process ON/OFF inference have not been validated or implemented.
Process annotations can be stored with their source; absent values remain null.

Synthetic inputs supply exact projected 2D landmarks to test PnP and quantized
sensor-depth fixtures to test RGB-D. Their errors measure implementation behavior,
not real-video extraction accuracy. The neural smoke run uses a tiny chronological
holdout within one synthetic episode; it is not a generalization experiment.
The supplied synthetic occlusion flag is retained as a hard constraint during
inference. Coverage in that run does not validate automatic occlusion detection.

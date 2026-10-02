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

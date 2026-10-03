# Helmet Viewpoint and Hand-Assisted Tool Pose

## Scope

Approved follow-up: an RGB hand model assists the grasped spray gun, while the
camera moves with an explicit helmet-front mount. This is a synthetic visibility
and integration pilot, not real G-TEX accuracy or validated fireproofing practice.
The previous Blender scenes and raw outputs are preserved.

The H-beam web is the target. Its frame remains at the physical face center:
+x along the beam, +y up, +z outward. The tool origin is the nozzle and +z is the
spray direction. The hopper exterior is unmeasured illustrative geometry.

## Hand Model and Grasp Assumptions

[MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
0.10.21 provides 21 RGB landmarks in VIDEO mode. The official float16 task bundle
is checksum-pinned. Image coordinates, hand-centered model world coordinates and
handedness scores are retained. Local world coordinates are **not** a camera pose.
Accepted predictions have confidence=1 as an availability convention, not a
calibrated per-joint confidence. In-image availability does not prove visibility.

The implemented auxiliary path is:

```text
RGB hand landmarks + metric hand template -> T_camera_hand via PnP
T_camera_hand @ calibrated T_hand_tool -> T_camera_tool
inverse(T_camera_surface) @ T_camera_tool -> T_surface_tool
```

Direct tool estimation remains primary. The hand path runs only after it fails.
Association uses wrist/MCP proximity to actual tool-mask pixels; this is a
candidate grasp, not proof. The configuration must separately declare a sourced,
verified constant grasp and matching left/right handedness. Unverified calibration, missing hands, missing surface
pose or rejected PnP leave the frame invalid. An annotated occluded hand is rejected.
An occluded tool can receive an indirect calibrated-hand estimate, retaining its
occlusion flag and distinct method. No gap filling is performed.

The simulation uses a posed, skinned right hand from
[WebXR Input Profiles assets](https://github.com/immersive-web/webxr-input-profiles/blob/main/packages/assets/README.md),
with the [MIT notice](references/WEBXR_HAND_LICENSE.md). The arm is illustrative.
Its fixed 21-point template and hand-to-nozzle transform are explicit synthetic
calibration priors, **not per-frame GT poses**. No real worker measurements are
available. Real applications need hand dimensions, grasp calibration, and checks
for grip changes; an articulated hand is not generally a rigid object.

## Camera Conditions

The head reference is 0.42 m above the web center, which is at 1.35 m world height.
The virtual camera is 0.06 m forward along head direction and 0.04 m upward.
Both offsets rotate rigidly with the head, preserving their 72.1 mm combined
length across pitch/yaw. Head motion adds 0.01 m bob, 2 degrees yaw and 1.5 degrees pitch amplitudes.
The camera does not automatically look at the moving tool.

| Condition | Head perpendicular distance to web | Head pitch downward |
|---|---:|---:|
| near_aimed | 0.65 m | 50 degrees |
| normal_aimed | 0.85 m | 42 degrees |
| normal_level | 0.85 m | 10 degrees |
| far_aimed | 1.10 m | 32 degrees |

Distances refer to the head reference, not nozzle standoff or Euclidean camera
distance. Camera positions are available per frame. These are design assumptions,
not a measured helmet or a human ergonomic/safety assessment. The same gun path,
grasp and seed are used across camera profiles at each condition.

| Profile | Render grid | Nominal optics | Inference depth |
|---|---|---|---|
| Galaxy TriFold main | 960 x 540 | 85 degree FoV, assumed diagonal | No |
| Galaxy TriFold ultrawide | 960 x 540 | 120 degree FoV, assumed diagonal | No |
| ZED 2i 2.1 mm | 960 x 540 | Maximum 110 H / 70 V degrees | Ideal rendered depth |
| Helios2 Ray Chroma registered | 640 x 480 | ToF 67 H / 51 V degrees | Ideal rendered depth |

Sources: [Samsung specifications](https://samsungmobilepress.com/press-releases/introducing-galaxy-z-trifold-the-shape-of-whats-next-in-mobile-innovation/),
[Stereolabs specifications](https://www.stereolabs.com/docs/products/cameras/zed/specifications),
[LUCID Chroma specifications](https://thinklucid.com/product/helios2-chroma-3d-rgbd-camera/).
Samsung's source does not define the FoV axis: diagonal interpretation here is
provisional. Real video stabilization/crop and lens distortion require calibration.
ZED's maximum axes are a nominal preview, not simultaneous measured rectified K.
The Chroma assembly is HTR-RGB-001; HTR003S-001 is its Ray ToF component.
Chroma has a separate native Triton RGB imager: this pilot uses the registered
ToF grid, not a claimed 640 x 480 native RGB sensor. Confirm the actual assembly
and obtain both intrinsics and color/depth registration before real experiments.

The ZED polarizer, stereo matching, Helios ToF error, mist, exposure, motion blur,
rolling shutter, sensor mass/cabling and material deposition are not simulated.
The configured hardware capture plan is 30 fps, but the short pilot samples only
8 frames at 10 fps per condition. It does not test full-rate tracking behavior.
BLK360 G1 supplies static geometry references, not helmet RGB-D or moving-tool
GT: see [Leica G1 specification](https://shop.leica-geosystems.com/sites/default/files/2023-06/BLK360_G1_SpecSheet_EN.pdf).

## Reproduce

From the repository root, use a separate hand environment to avoid changing the
main research dependencies. Python 3.10 was tested:

```sh
python3 -m venv .venv-hand
.venv-hand/bin/python -m pip install -e '.[hand]' 'opencv-contrib-python==4.11.0.86'
.venv-hand/bin/python -m pip check
.venv/bin/python -m robodata.hand_assets --output outputs/hand_assets
```

The downloader verifies SHA-256 and never replaces existing files. The model and
mesh are ignored by Git. The renderer uses Blender's bundled glTF import and
Cycles; no additional add-on is needed.

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_simulate.py -- \
  --config configs/blender_helmet_sensors.json --output outputs/helmet_sensors_NEW
.venv/bin/robodata blender-evaluate \
  --render-root outputs/helmet_sensors_NEW --output outputs/helmet_baselines_NEW
.venv-hand/bin/python -m robodata.helmet_hand_experiment \
  --render-root outputs/helmet_sensors_NEW --baseline-root outputs/helmet_baselines_NEW \
  --model outputs/hand_assets/hand_landmarker.task --output outputs/helmet_hands_NEW
```

All output paths must be new. Baseline tool/surface landmarks are oracle projected
markers with ray-cast visibility, not detector outputs. Hand landmarks use the
actual pretrained model. RGB phone manifests do not receive the ideal depth used
for separate occlusion evaluation. Per-frame simulator poses, independent scene
rays, occlusion references and pose error GT remain outside the inference inputs.

Camera projection and depth are checked independently. Blender pixel-aspect
components are normalized above its minimum of 1, preserving non-square fx/fy.
The render metadata includes effective configuration hash and renderer hash/Git
snapshot. Hand outputs include model/input hashes, software, timestamps and
association failures. Report coverage together with pose error, not error alone.

## Results

Initial run (superseded helmet-offset calculation): `outputs/helmet_sensors_20261003_01` (16 conditions,
8 frames each). Baselines: `outputs/helmet_sensors_results_20261003_01`.
Actual model results: `outputs/helmet_hand_results_20261003_01`; the diagnostic
follow-up repeats the same inputs into a new output directory, preserving this run.

| Sensor profile | Associated model hand / 32 frames | Valid RGB+hand / 32 |
|---|---:|---:|
| Galaxy main | 6 | 0 |
| Galaxy ultrawide | 0 | 0 |
| ZED 2i | 0 | 0 |
| Helios registered | 14 | 0 |

Across 128 RGB frames, direct-tool and hand-assisted pipelines both have zero
valid canonical poses. The 64 RGB-D frames also have zero valid tool trajectories.
Position/orientation error is therefore **null**, not zero. Baseline RGB fails
before obtaining a surface frame on 90 frames; it obtains a surface but fails
tool estimation on 38. Only 20 frames have an associated model hand. Testing
hand-template PnP separately from surface availability rejects all 20: 11 lack
in-image wrist/MCP predictions and 9 fail cheirality/reprojection checks.

These outcomes establish that the inference runtime and validity gates execute,
not that hand assistance improves this scene. A grasped, partially hidden hand
requires more than an off-the-shelf landmark model plus a fixed hand template.
Narrower views produced more hand detections here because the rendered hand had
more image pixels. This is not evidence that Helios is the most accurate sensor.
The normal-level condition clips the hand; near views are dominated by the hopper.
Far views show more web but reduce hand pixel size.

Next experiments should separate surface registration (persistent web geometry,
camera motion, occluded landmarks) from hand/tool pose, add a positive-control
visible hand and multiple grip conditions, and then validate against a measured
real G-TEX/hand calibration. A static surface scan alone does not compensate
helmet motion. Gloves and spraying mist need real footage. Do not extrapolate
these synthetic results to real tools, material quality or sensor accuracy.

### Final Verified Run

The rigid-mount correction is rendered in `outputs/helmet_sensors_20261003_02`;
baseline evaluation is `outputs/helmet_sensors_results_20261003_02`, and actual
MediaPipe/diagnostic results are `outputs/helmet_hand_results_20261003_02`.
All 16 episodes contain 8 frames. Every renderer source hash matches the final
renderer (`de4d34e50fa806d5f38e196ec7a27aa9acf566f82587742946000d224fb447ee`).
The render snapshot records parent f984291 and dirty=true; it does not claim to
have been generated from the subsequent committed source.

| Sensor profile | Associated model hand / 32 | Accepted hand PnP | Valid canonical RGB+hand |
|---|---:|---:|---:|
| Galaxy main | 11 | 0 | 0 |
| Galaxy ultrawide | 0 | 0 | 0 |
| ZED 2i | 0 | 0 | 0 |
| Helios registered | 13 | 0 | 0 |

Total: 24/128 associated model hands, zero accepted calibrated-hand poses and
zero valid canonical poses. Direct RGB is also 0/128 and RGB-D 0/64.
The separate hand-only diagnostic rejects 9 frames for unavailable in-image
wrist/MCP predictions and 15 for cheirality/reprojection failure. No model-local
world coordinates or GT poses are substituted to turn these into successes.
Surface estimation fails first on 92 RGB frames; the remaining 36 reach tool
estimation and fail there. The report's `tool_unavailable=128` counts absent tool
estimates including the surface failures, not 128 independent tool-stage attempts.

Associated detections occur in main normal (3/8), main far (8/8), Helios normal
(6/8) and Helios far (7/8). Other conditions have none. Detection is not pose
accuracy, and all pose errors remain null for these zero-coverage runs. The
contact sheet draws selected model predictions, never template/GT landmarks.
Corrected Blender scenes were opened separately, preserving the original window's
unsaved changes. No sensor accuracy or real-world method improvement is claimed.

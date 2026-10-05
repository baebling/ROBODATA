# VGGT Integration And Bottlenecks

## What Is Connected

Original pretrained VGGT now runs through `robodata vggt`. It samples RGB
frames throughout an input video, predicts camera-from-model extrinsics,
intrinsics, normalized depth and raw depth confidence, and writes an NPZ
reconstruction plus a typed surface cache. Sampling is explicit and bounded by
the configured frame budget. This is not full-video trajectory optimization.
The original model was selected because its local runtime/checkpoint were
available. VGGT-Omega is not installed or integrated by this change.

`run_episode` accepts `surface_method: vggt` with `vggt_surface_path`. The cache
is bound to video SHA-256, calibrated camera, real/synthetic designation,
physical surface-frame ID, original frame indices and timestamps. Missing or
rejected samples remain invalid; no held pose, interpolation or stale reference
is substituted. The existing tool PnP/optional calibrated hand path is used;
VGGT does not generate semantic tool observations for it.

The tool configuration optionally accepts `T_body_tool` and `calibration_source`.
Then `tool.points_m` are body-frame landmark coordinates and the RGB path uses:

```text
T_camera_tool = T_camera_body @ T_body_tool
T_surface_tool = inverse(T_camera_surface) @ T_camera_tool
```

The functional tool origin is the nozzle outlet and +z its action direction.
The offset must describe the installed, measured nozzle. A second direction
fixes roll. Neither a 2D mask centroid nor the estimated hand-centered origin
is automatically a rigid, metric tool frame. Hand fallback still needs its own
verified `T_hand_tool`; body-to-tool calibration does not calibrate a grasp.

## Metric Surface Alignment

Provide >=4 distinct measured named surface points in `surface_points_m` and
corresponding visible RGB surface keypoints in the existing observation manifest.
An optional `alignment_frame_indices` limits initialization to selected frames.
The manifest camera must describe rectified RGB with zero distortion in this
first adapter. Original-to-model pixel mapping accounts for VGGT's resize,
14-pixel height rounding, pixel centers and center crop.

The adapter back-projects predicted depth with predicted intrinsics, converts
to the shared normalized model frame, and fits:

```text
X_surface_m = scale * R_surface_model * X_model + translation_surface_m
```

Collinear/nonfinite anchors, fewer than four distinct point identities and
excessive maximum anchor residual are rejected. Rotation is right-handed.
Each accepted alignment is composed with predicted per-frame camera motion.
Where any measured RGB correspondences are visible, calibrated-camera
reprojection and positive-depth checks also gate that frame. Frames with fewer
checks use the model camera prediction, not an independently validated pose;
check availability is recorded. These gates do not establish actual accuracy.
The global fit score is an uncalibrated heuristic, not a per-frame probability.
Raw depth confidence is a positive model score and may exceed one.

With no measured anchors, normalized reconstruction is retained but all metric
surface estimates remain null. Depth sensor files and GT trajectories in a
manifest are not consumed by VGGT. Scale comes from declared physical geometry,
not from fitting a predicted trajectory to evaluation GT.

## Why The Problem Remains Difficult

1. **Metric scale versus geometric distortion.** One similarity transform fixes
   an overall size/orientation/origin. It cannot fix local depth distortion,
   bending or time-varying errors. A visually convincing reconstruction can
   still give the wrong nozzle-to-web distance. Independent measured landmarks
   and unused geometric checks are needed, not only an attractive point cloud.
2. **Surface identity versus a visible plane.** A plane supplies a normal but
   does not locate a physical origin along the plane or determine its heading.
   Repeated H-beam geometry, unseen endpoints and symmetric faces create
   ambiguity. Semantic point identity and fixed axes must remain consistent.
   A visible-mask centroid moves when the hopper occludes the surface.
3. **Dynamic body pose versus static-scene reconstruction.** Camera motion and
   tool motion both move pixels. Background reconstruction is not a moving
   object's rigid 6D pose estimator. Known body shape/landmarks, segmentation,
   rigidity and independent tool observations must constrain this second path.
   Tool-mask exclusion is optional and only works when masks are supplied; this
   adapter does not automatically train a segmentation model.
4. **Functional nozzle pose versus body appearance.** A nozzle hidden behind a
   visible, localized rigid body can be inferred through measured calibration.
   If both body and nozzle are hidden, a temporal/hand prior may narrow the
   possibilities but cannot uniquely measure arbitrary motion. Grip changes,
   gloves and symmetry can break hand assistance. No invisible motion is
   silently accepted as GT.
5. **Temporal consistency and accumulated error.** Errors in surface pose, body
   rotation and nozzle calibration combine in the canonical trajectory. A long
   body-to-nozzle offset amplifies angular errors: for an illustrative 0.30 m
   lever arm and 5 degree rotation error, displacement is about 26 mm. These are
   explanatory numbers, not measured G-TEX dimensions or results. Smoothness
   alone does not prove correctness and may erase real rapid movements.

Future full-video refinement should constrain static geometry, rigid moving
body, known scale and image reprojections together, preserve original estimates
and record modified samples. It is not implemented here. Independent GT must
evaluate nozzle position, direction, standoff, temporal drift and valid coverage,
including failures; output density is not accuracy.

## Running

Install the optional runtime into a suitable research environment with
`pip install -e '.[vggt]'`, subject to the official model license. This change
used the existing runtime and cached weights without downgrading/installing
Torch, changing the external VGGT checkout or modifying raw recordings.

Edit `configs/vggt_rgb.yaml` for actual input paths and measured anchors:

```sh
.venv/bin/robodata vggt --config configs/vggt_rgb.yaml --output outputs/vggt_real_01
```

In a separately configured normal RGB extraction config, preserve the same
`surface_frame_id`, camera and physical geometry, set `surface_method: vggt`
and `vggt_surface_path: <relative path to outputs/vggt_real_01/surface_cache.json>`.
Then run the existing `robodata run --config ... --output ...` command. Tool
geometry and semantic RGB observations are still required for canonical poses.
The supplied template intentionally leaves measured points empty. The reported
MDF nominal dimensions (300 x 300 x 10 x 15, length 1200, presumed mm) are context,
not observed pixel correspondences or a calibrated full H-beam model.

For a checkpoint-only RGB check without a calibrated metric surface:

```sh
.venv/bin/robodata vggt-probe --images <RGB A> <RGB B> \
  --checkpoint <original VGGT checkpoint> --output outputs/vggt_probe_new
```

Outputs refuse overwrite. Model RGB copies/masks go only into the new output
directory. Whole-video sampling does not provide dense continuous motion labels:
unsampled frames are invalid in extraction and block future/velocity labels.
Use a short consecutive frame selection to test those labels; do not relabel
sparse samples as uninterrupted motion or hide gaps.

## Executed Validation

`outputs/vggt_rgb_probe_20261003_01` used existing real DITTO sponge-tray RGB
frames 00000 and 00005, not new simulator data and not H-beam spraying. Original
VGGT-1B inference ran on RTX 3080 Ti with bfloat16; camera/depth arrays were
generated with depth shape 2 x 294 x 518. Positive finite predicted depth at all
pixels is a numeric sanity check, NOT depth accuracy. Metric alignment, tool
pose estimation and accuracy evaluation were not performed in this probe.
Checkpoint SHA-256: `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.
Input, checkpoint and executed upstream source hashes are retained in provenance.
The upstream model emits a PyTorch autocast deprecation warning but completed.

Analytic unit fixtures test scale/frame composition, degeneracy, pixel mapping,
body-to-nozzle offset, sampling gaps, calibration/video identity, invalid outputs
and calibrated reprojection rejection. Runner unit tests substitute only neural
inference to avoid repeated multi-GB model loads; actual inference was tested
separately as above. These are engineering tests, not a new simulation study or
evidence of real construction trajectory accuracy.

## Sources

- [VGGT paper](https://arxiv.org/html/2503.11651v1): normalized coordinates,
  multi-view camera/depth prediction and limitations.
- [Official VGGT repository](https://github.com/facebookresearch/vggt): model
  API, confidence outputs, crop preprocessing and checkpoint license distinctions.
- [VGGT-Omega](https://vggt-omega.github.io/): static/dynamic reconstruction
  improvements, separate candidate not benchmarked or integrated here.

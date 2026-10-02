# ROBODATA Research Model v0.1

Authorization: the researcher requested implementation on 2026-10-02 and delegated
missing implementation criteria to the agent. This is an architectural task.

## Research boundary

The latest accessible BAE_ROBOT conversation, `선행연구 조사 계획`, and the researcher's
statements establish video-derived dataset generation and label validation as the
current study. RGB is the final input goal; RGB-D is a feasibility comparison.
Independent mocap/LiDAR measurements belong to evaluation. Policy training and
physical robot execution are downstream. No real footage or calibrated gold
benchmark has been provided to this repository.

## Selected approach

Implement a calibrated known-geometry RGB baseline using semantic landmarks,
OpenCV PnP and optional forward/backward optical flow. This makes scale provenance
explicit and supports moving cameras through a persistent surface landmark frame.
An optional trainable RGB heatmap network replaces manually supplied landmarks.
RGB-D uses aligned depth and a robust plane fit for the surface plus semantic
tool keypoints. A static camera is required for the plane-only reference frame;
moving-camera episodes use surface landmarks and PnP.

Alternatives: mesh-based FoundationPose needs a tool mesh, depth and model assets;
ToolEENet needs category-specific weights and functional-part annotations.
Multi-view MASt3R requires additional views and still needs a verified metric
reference. These can later supply pose observations through the common schema.
The selected baseline runs without unavailable pretrained checkpoints.

## Data flow and coordinates

Manifest -> timestamped RGB/depth frames -> semantic observations ->
T_camera_surface and T_camera_tool -> inverse(T_camera_surface) @ T_camera_tool
-> derivatives/future targets -> JSON plus masked NPZ learning tensors -> report.

T_A_B maps B coordinates into A. Matrices are 4x4 row-major; units are meters,
radians and seconds. Camera coordinates follow OpenCV (+x right, +y down, +z
forward). All physical frames are right-handed. The surface origin is the center
of a fixed reference patch, with +z toward the working side and x defined by
persistent landmark geometry. Tool origin is the functional point; +z is the
functional action direction. A mask centroid is stored separately because
viewpoint/occlusion changes its physical location. Full rotation needs a second
semantic direction; axis-only observations leave the full pose null.

## Modules and observable requirements

- Validated configuration/episode contracts reject malformed SE(3), timestamps,
  paths, confidence and unit metadata.
- PnP rejects insufficient, behind-camera and high-residual estimates.
- Depth fitting rejects invalid depth, sparse/nonplanar masks and degenerate axes.
- Every estimate records method, residual/validity, source and occlusion.
- Temporal labels require valid samples throughout the horizon; no interpolation
  or smoothing is implicit. Body-relative delta and surface delta are distinct.
- Evaluation aligns timestamps only within a configured tolerance, requires the
  same surface frame, reports matched coverage and position/axis/rotation errors.
- Synthetic RGB and RGB-D fixtures exercise real estimators and serialization.
- The RGB network has masked heatmap supervision, held-out validation,
  versioned checkpoints and an inference adapter. Synthetic training establishes
  executability only, not construction-video performance.

## Completion evidence

Run format/lint/type/test checks, both synthetic pipelines, JSON reread, tensor
export and GT evaluation. Train and reload the RGB network on synthetic RGB.
Commit source/config/docs/tests to the requested repository; exclude raw data,
weights, credentials and generated run directories.

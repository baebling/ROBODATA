# Released Source Audit

Reviewed upstream commit `ec1c459ea27ca8e2a2f2787def1d1045ecb2a4ee`.
This is an implementation audit, not a claim that the paper is incorrect.

## Inputs and Representations

- `scripts/preprocess_human_play.py::collect_states_from_demo` reads RGB,
  **sensor depth**, and camera intrinsics, and tracks only `view_names[:1]`.
- `track_tool_poses` passes depth to FoundationPose `register` and `track_one`.
  It converts the mesh pose with camera/base extrinsics to a six-dimensional
  position + axis-angle vector stored as `observations/robot_eef_pose`.
- This field name does not make human-demonstration data a measured robot EE
  pose. The public dataset supplies both this field and `camera_3_tool_poses`.
  Neither is an independent GT measurement, and neither defines a spray nozzle.
- `ti/real_world/real_data_conversion.py` makes actions from **the next pose
  within each episode**, not robot commands or task success labels.
- `ti/common/data_utils.py::_convert_actions` converts axis-angle to rotation-6D;
  the policy receives 3 position + 6 rotation numbers. Rotation-6D is a rotation
  representation, not six rotational degrees of freedom.

## Reconstruction and Augmentation

- MASt3R/3DGS are present in `ti/common/data_utils.py::setup_scene_and_gs`.
  They reconstruct a pair, optimize splats for 500 + 100 iterations and render
  intermediate viewpoints. They are not called by the pose-preprocessing script.
- `conf/train_bc/task/real_hammer_human.yaml` selects camera 3 and camera 4,
  `segment: True`, `aug_img_num: 11`, but **`novel_view: False`**.
- With augmentation disabled, `feature_proc` repeats the second real view;
  it does not produce genuine novel views. Enabling it is an explicit experimental
  override, not the default released hammer training configuration.
- `feature_proc` catches augmentation exceptions and falls back to repeated
  real views. A completed upstream run alone is not evidence of successful 3DGS.
  Our single-pair test directly calls the helper and fails on blank/nonfinite
  output without accepting the fallback.
- RGB scene reconstruction does not independently verify metric accuracy,
  surface-frame consistency, or functional nozzle pose.

## Safety and Reproducibility Concerns

- `generate_data` opens the source Zarr in append mode and writes poses there.
  Never execute it on the preserved public archive extraction.
- Missing extrinsics fall back to `np.ones((4, 4))`, despite an "identity" warning.
  This is not SE(3), and its inverse is singular. Do not silently replace it with
  identity and claim an accurate calibrated trajectory.
- `track_tool_poses` places the entire episode RGB tensor on the GPU at once.
  Long episodes may exceed memory even when model inference on one frame fits.
- `RealDataset` writes cache ZIPs inside `dataset_path`. Full preprocessing and
  training must use a separate working copy, not the protected raw location.
- The public data contains 5 camera streams, not helmet-mounted monocular video.
  We do not assume all streams are independently calibrated GT cameras.

## Local Execution Deviations

- PyTorch 2.0.0 + CUDA 11.8 retained; torchvision 0.15.1 is the matching wheel,
  whereas the upstream Conda file lists 0.15.2 alongside PyTorch 2.0.0.
- PyTorch3D 0.7.3 used from the official compatible wheel index; the upstream
  environment lists 0.7.5. Rotation conversion is independently checked against
  SciPy, but this does not prove every PyTorch3D operation is interchangeable.
- Compiler: system NVCC 11.5, distinct from PyTorch's bundled CUDA 11.8.
  Its GCC 11 build failed; a separate extracted GCC/G++ 10.5 toolchain is used
  for the GS CUDA extension without altering the system compiler.
- MASt3R uses the packaged `starst3r==0.3.0` source path and the pinned model
  revision in `scene_smoke.json`; compiled RoPE2D is absent, so its PyTorch fallback runs.
- Policy smoke retains the released 263,135,643-parameter policy and 100-step
  DDPM inference, but uses unsegmented RGB, supplied or re-estimated poses, one episode, batch 1,
  five optimizer updates, no EMA, no full schedule, and no held-out evaluation.
  It must not be used as evidence of policy skill or the paper's reported gains.
- Pose execution uses the released GroundingDINO/SAM-HQ and FoundationPose
  code/weights on a complete 72-frame public episode, without reading supplied
  poses for estimation. The inference input is RGB-D plus mesh/calibration,
  not monocular RGB. Missing 3D confidence stays unavailable.
- nvdiffrast is pinned to 0.3.1 rather than an unpinned Git HEAD. Boost/Eigen
  headers/libraries were extracted to an external directory, not system-installed.
- On WSL, Warp initially selected a Linux NVIDIA driver and failed with no CUDA
  device. Prioritizing the WSL driver directory fixes this without source edits.
- `compute_mesh_diameter` returns NumPy float64. This promotes crop offsets to
  DoubleTensor, conflicting with float32 intrinsics. Converting that scalar to
  Python float and making the intrinsics dtype explicit resolves the actual
  failure without changing the diameter value or upstream tracked code.
- JPEG2000 level-50 encode/decode is now reproduced before policy image loading.
  Older smoke evidence in `runs/2026-10-13-policy-smoke.json` predates this parity
  correction; it is historical, not the current reproduction evidence.
- Nonblank render checks use spatial variation per color channel. They do not
  certify geometry; inspected views still show holes and point-like artifacts.

## Still Required for Paper-Result Reproduction

1. Complete Grounded-SAM/SAM-HQ hand removal for policy images and augmentation,
   not just tool-mask initialization for the demonstrated pose branch.
2. Extend FoundationPose re-estimation to all episodes and verify mesh scale,
   calibration capture correspondence and independent GT, beyond numerical and
   visual execution checks.
3. Full multi-episode conversion, explicit augmentation condition, training,
   validation and comparable baselines/seeds. The released schedule is 3,050
   epochs, not the five-step integration test.
4. A specified UR5/Kinova setup and independent physical task trials for success
   rates. No robot is commanded by this reproduction tooling.
5. Separate construction adaptation and real spraying data after baseline
   reproduction. Do not conflate these with public hammer results.

# Tool-as-Interface: Isolated Reproduction

This directory reproduces released **Tool-as-Interface** components on public
hammer data. It is not a replacement for ROBODATA's construction pipeline and
does not claim helmet-RGB spraying accuracy or reproduced robot success rates.
No construction recordings or new simulator data are used.

References: [paper v2](https://arxiv.org/html/2504.04612v2),
[official project](https://tool-as-interface.github.io/),
[official source](https://github.com/Tool-as-Interface/Tool_as_Interface).
See [source audit](SOURCE_AUDIT.md) and [execution results](RESULTS.md) before
interpreting results.

## Separation

| Item | Location / meaning |
|---|---|
| Reproduction code | This directory only; no imports from `robodata` |
| Upstream code | Separate clean checkout pinned by `runner.py` |
| Python environment | Separate external virtual environment |
| Public raw dataset | External location; opened read-only |
| Large artifacts | New `outputs/tool_as_interface/...` directory per run, Git-ignored |
| Versioned evidence | Small reports in `runs/`; software lock in this directory |
| Construction data/model | Existing code and experiments remain unchanged |

The public archive contains 40 episodes / 2,626 frames with five RGB-D streams,
intrinsics, extrinsics, and estimated tool poses. Policy execution can use
**supplied estimates** or **newly re-estimated poses** with verified episode,
timestamp, file checksum and raw-data provenance. Neither is independent GT.
The policy receives next-pose targets in an upstream calibrated frame, not
`T_surface_tool` and not a spray-nozzle functional coordinate frame.

## Setup

Use Python 3.10, Linux x86_64, a CUDA-capable GPU and an external environment.
The main lock covers policy/reconstruction components. Additional preprocessing
dependencies are in `preprocessing_requirements.txt`; native setup scripts keep
compiler/Boost/Eigen libraries outside the system installation. This is not the
complete original training/hardware-control environment.

```sh
git clone https://github.com/Tool-as-Interface/Tool_as_Interface "$TAI_UPSTREAM"
git -C "$TAI_UPSTREAM" checkout ec1c459ea27ca8e2a2f2787def1d1045ecb2a4ee
git -C "$TAI_UPSTREAM" submodule update --init third_party/robomimic \
  third_party/FoundationPose third_party/Grounded-Segment-Anything
python3.10 -m venv "$TAI_ENV"
source "$TAI_ENV/bin/activate"
python -m pip install -r reproductions/tool_as_interface/requirements.lock.txt
```

Set `TAI_UPSTREAM`, `TAI_ENV`, `TAI_ARCHIVE`, and `TAI_RAW_ROOT` to separate
external paths. Set `TAI_DATASET` to the extracted `hammer_human` directory.
Do not place data or upstream checkouts in tracked source directories.

Download the archive linked by the official README:

```sh
gdown 1M1TkearRZcCR6FaAUzvGwQbhYrTLGmzz -O "$TAI_ARCHIVE"
python -m reproductions.tool_as_interface.runner extract \
  --archive "$TAI_ARCHIVE" --destination "$TAI_RAW_ROOT"
```

Verified archive SHA-256:
`b62e151d274884648c1f922d0117df4d3a378621bcf39255835f1a660ae61a47`.
Extraction rejects unsafe paths and existing destinations.

For segmentation/pose re-estimation, first prepare the compiler described below,
set `TAI_NATIVE` to an external native-library directory, and run:

```sh
python -m pip install -c reproductions/tool_as_interface/requirements.lock.txt \
  -r reproductions/tool_as_interface/preprocessing_requirements.txt
bash reproductions/tool_as_interface/install_segmentation.sh
bash reproductions/tool_as_interface/install_pose_runtime.sh
bash reproductions/tool_as_interface/download_preprocessing_models.sh
```

The last script downloads the official Swin-B, SAM-HQ and FoundationPose weights.
GroundingDINO also fetches `bert-base-uncased` on first initialization. Its loaded
revision must match the versioned preprocessing configuration; a different
revision is rejected. The renderer is pinned to nvdiffrast 0.3.1, explicitly a
compatibility choice rather than upstream's mutable Git URL.

## Executable Checks

Run from the ROBODATA reproduction checkout with the separate environment
activated. Outputs must be new directories outside the raw dataset.

```sh
python -m reproductions.tool_as_interface.runner audit \
  --upstream "$TAI_UPSTREAM" --dataset "$TAI_DATASET" \
  --output outputs/tool_as_interface/audit_new

python -m reproductions.tool_as_interface.runner policy-smoke \
  --upstream "$TAI_UPSTREAM" --dataset "$TAI_DATASET" \
  --config reproductions/tool_as_interface/policy_smoke.json \
  --output outputs/tool_as_interface/policy_new

MAX_JOBS=2 python -m reproductions.tool_as_interface.runner scene-smoke \
  --upstream "$TAI_UPSTREAM" --dataset "$TAI_DATASET" \
  --output outputs/tool_as_interface/scene_new
```

- `audit`: decodes each camera's first RGB/depth frame, records shapes, supplied
  label validity and within-episode timestamp intervals, and hashes the raw tree.
- `policy-smoke`: unmodified released policy architecture, normalization and
  episode-bounded sampler, including JPEG2000 level-50 storage roundtrip;
  optimizer updates and full 100-step DDPM inference. Inputs are unsegmented
  real RGB and supplied poses, or `--pose-estimates` from re-estimation.
  The checkpoint is explicitly
  labeled `integration_only`, not a deployable trained policy.
- `scene-smoke`: official `setup_scene_and_gs` and `generate_novel_view` helpers
  on camera 3/4 frame 0, including the original 500 + 100 GS iterations. Blank
  or nonfinite output is an error, not an accepted fallback.

To execute the pose branch and link **new estimates** into the policy test:

```sh
TAI_OUTPUT=outputs/tool_as_interface/pose_new \
TAI_PREPROCESS_CONFIG=reproductions/tool_as_interface/preprocessing_episode.json \
  bash reproductions/tool_as_interface/run_preprocessing_check.sh

TAI_POSE_ESTIMATES="$PWD/outputs/tool_as_interface/pose_new/estimated_poses.npz" \
TAI_OUTPUT=outputs/tool_as_interface/linked_new \
  bash reproductions/tool_as_interface/run_offline_checks.sh
```

The episode configuration covers the first complete 72-frame demonstration.
`preprocessing_smoke.json` instead covers three frames for dependency diagnosis.
The linked policy command requires a complete selected episode and rejects
changed/misaligned estimates. It does not quietly fall back to supplied labels.

`preprocess_smoke.py` calls the released Grounded-SAM/SAM-HQ initialization and
`track_tool_poses` (10 registration iterations, 5 tracking iterations), using
sensor depth, the provided mesh and extrinsics. It never calls the append-mode
upstream dataset writer. Estimates, timestamps, source indices, missing confidence
and numerical validity are saved separately, along with a decoded-video check.
Numerical validity does not establish accuracy or visibility.

The scene condition and pretrained model revision are versioned in
`scene_smoke.json`. Policy window indices and timestamps are saved alongside
predictions, including the next-pose offset and the four-action inference slice.

Activating the environment matters: CUDA builds must find its `ninja` executable.
System NVCC 11.5 with GCC 11 failed locally. We extracted Ubuntu GCC/G++ 10.5
packages into a separate toolchain without changing the system compiler. For
that configuration set `CC` and `CXX` to the extracted GCC/G++ 10 binaries.
This compiler workaround does not change model or reconstruction code.

The pose wrapper also prioritizes `/usr/lib/wsl/lib` on WSL. Without this, Warp
loaded the Linux driver installed with a CUDA toolkit instead of the Windows
driver and could not see the GPU. No driver was installed or changed.
The pose adapter uses float32 intrinsics and a Python-float mesh diameter to
prevent NumPy float64 crop offsets from creating mixed tensor dtypes. These
compatibility deviations are recorded; upstream tracked source remains clean.

The local compiler workaround can be prepared without administrator access on
Ubuntu 22.04. Choose an external `TAI_TOOLCHAIN` directory, then:

```sh
mkdir -p "$TAI_TOOLCHAIN"
cd "$TAI_TOOLCHAIN"
apt-get download g++-10 gcc-10 cpp-10 libgcc-10-dev libstdc++-10-dev gcc-10-base
for archive in *.deb
do
  dpkg-deb -x "$archive" .
done
export CC="$TAI_TOOLCHAIN/usr/bin/gcc-10"
export CXX="$TAI_TOOLCHAIN/usr/bin/g++-10"
```

The tested package version was `10.5.0-1ubuntu1~22.04.3`. Return to the
reproduction checkout before running its commands.

Each completed run records source commits, configuration, package versions,
reproduction commit/dirty state, script hash, and the raw-data digest before/after.
Pre-existing upstream bytecode caches are not consumed. Failures are saved to
`failure.json`. All stages are offline; none commands a physical robot.

## Tests

```sh
python -m pytest tests/test_tai_reproduction.py -q
TAI_UPSTREAM="$TAI_UPSTREAM" TAI_DATASET="$TAI_DATASET" \
  python -m pytest tests/test_tai_reproduction.py -q
TAI_RECON=1 TAI_UPSTREAM="$TAI_UPSTREAM" TAI_DATASET="$TAI_DATASET" \
  MAX_JOBS=2 python -m pytest tests/test_tai_reproduction.py -q
```

Without external paths, GPU tests are deliberately skipped. They are not replaced
with synthetic outputs. Do not interpret a skipped check as a passing model test.

## Next Reproduction Steps

Extend pose re-estimation beyond the tested episode, then complete hand removal,
multi-episode policy training and held-out evaluation with the original settings and explicit
augmentation conditions. Preprocessing and cache creation require a **working
copy** because the upstream scripts write into their dataset path. Full training
and physical hammer success rates are not covered by the smoke checks. The
new-pose-to-policy test and the MASt3R/3DGS pair test are distinct paths; this is
not a claim of a fully trained end-to-end RGB-only policy.

Only after this baseline should we adapt functional tool frames, work-surface
references and uncertainty to real construction videos. The absence of real
spraying footage does not prevent public hammer component tests, but it prevents
claims about construction performance.

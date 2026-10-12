# Execution Results: 2026-10-13

## Status

**Released-component execution reproduction, not paper-result reproduction.**
The public hammer data was used; no construction videos or new simulator data
were used. Existing `src/robodata` and construction experiments are unchanged.
Work is isolated on `reproduce/tool-as-interface`, in a separate worktree and
Python environment. No physical robot was commanded.

## Executed Stages

| Stage | Actual input / execution | Evidence | Interpretation |
|---|---|---|---|
| Public-data audit | 40 episodes, 2,626 frames, five RGB-D streams; decoded first RGB/depth frame per stream | [audit report](runs/2026-10-13-final-audit.json) | Contracts and source integrity checked, not ground-truth accuracy |
| Tool segmentation and tracking | GroundingDINO/SAM-HQ + FoundationPose; first complete 72-frame episode, camera 3, sensor depth, scanned hammer mesh and calibration | [pose report](runs/2026-10-13-final-preprocessing.json) | 72 finite position/axis-angle estimates, not measured GT or functional nozzle poses |
| New-pose-to-policy connection | New estimates from the preceding stage, real RGB, JPEG2000 level-50 roundtrip; official 263,135,643-parameter policy | [linked policy report](runs/2026-10-13-final-linked-policy.json) | Five optimizer updates and full 100-step DDPM inference; finite `[1, 4, 9]` predictions and changed parameters establish execution, not skill |
| Scene / view augmentation | Official MASt3R + 3DGS helpers; camera 3/4 at frame 0; 500 pruning + 100 refinement iterations | [scene report](runs/2026-10-13-final-scene.json) | 6,424 finite points and 10 nonblank 224-pixel views; geometry quality is not validated |

Provided pose labels were **not used to estimate** the 72-frame trajectory.
The policy uses the newly estimated trajectory, with its checksum, episode
indices and timestamps checked. Its observations are unsegmented RGB; scene
augmentation is tested separately, not fed into this policy run.

The pose video was actually decoded: 72 frames at 15 fps, 640 x 480 pixels.
Inspection of a middle frame shows an overlaid box/axes near the tool, but
visual plausibility does not establish 3D accuracy.

## Important Limitations

- This pose path uses **RGB-D, a provided mesh and calibration**, not RGB-only
  helmet footage. It neither proves nor disproves the final RGB research goal.
- The pose reference is the upstream calibrated base/world frame. It is not
  ROBODATA's work-surface frame or a functional nozzle frame.
- Pose confidence is unavailable and retained as unavailable. Finite values
  do not establish visibility, tracking correctness or metric accuracy.
- Calibration matrices pass numerical SE(3) checks; correspondence to the
  specific physical capture and their measurement accuracy remain unverified.
- Generated views have visible holes, point-like artifacts and large dark areas.
  The final run's near-black pixel fractions range from 65.2% to 93.4%; these
  pixels can represent background or missing content. A nonblank render is
  execution evidence, **not an acceptable-quality augmentation conclusion**.
- Hand removal for policy images/scene augmentation, all-episode conversion,
  the full 3,050-epoch schedule, held-out evaluation, baseline/seed comparisons
  and robot success-rate measurements have not been performed.
- There is no real spraying evaluation. The reproduction is not a deployable
  trained policy and must not be described as a construction-model result.

## Local Artifacts

Large files remain Git-ignored under the isolated worktree:

```text
/home/kmkm/ROBODATA-repro-tai/outputs/tool_as_interface/
  final_preprocessing_20261013/
    estimated_poses.npz
    pose.mp4
    pose_first.png
    pose_middle.png
    segmentation.png
    preprocessing_smoke.json
  final_linked_20261013/
    audit/audit.json
    policy/policy_smoke.json
    policy/smoke_predictions.npz
    policy/integration_only.pt
    scene/scene_smoke.json
    scene/reconstruction.npz
    scene/source_*.png
    scene/novel_*.png
```

The checkpoint is integration-only. Do not use it for robot execution.
Earlier successful and failed runs are retained separately. The older versioned
`2026-10-13-policy-smoke.json` predates JPEG2000 parity and is historical evidence;
the four `final` reports above describe the current execution.

## Reproducibility

- Official source pin: `ec1c459ea27ca8e2a2f2787def1d1045ecb2a4ee`.
- Archive SHA-256:
  `b62e151d274884648c1f922d0117df4d3a378621bcf39255835f1a660ae61a47`.
- Raw tree SHA-256 before/after every final stage:
  `b0a2dcbc5e53c9792828289b24074fe44b2124b8ef6e2acd0a869c620373c1a5`.
- New pose file SHA-256:
  `2165345d208f57d57d626dad6d3d87fc85e2fb0673ab8806fe8653686922add3`.
- Hardware: RTX 3080 Ti, 12 GB; Python 3.10.12 on WSL2 Ubuntu 22.04.
  Model revisions, asset hashes, package versions, seeds and compatibility
  deviations are recorded in the reports and [source audit](SOURCE_AUDIT.md).
- The runs occurred before commit, so their metadata correctly records the
  baseline commit plus a dirty working tree. Startup source hashes identify the
  executed reproduction scripts; do not reinterpret that baseline commit as
  containing this reproduction implementation.

## Verification

Full repository tests: **322 passed, 2 skipped**. Those two skips are the explicit
external-GPU checks in the original environment. The focused reproduction suite
was then executed in the separate environment with both GPU checks enabled:
**52 passed, no skips**, including official policy and scene execution.
Pinned upstream dependencies produced 86 deprecation/API warnings; these are
not hidden or interpreted as model-quality evidence.

`ruff check`, `ruff format --check` (66 files), `mypy` (33 existing source files),
and reproduction-environment `pip check` passed. The existing `robodata demo`
integration also completed without error; its synthetic metrics are regression
checks, not construction-video accuracy measurements. Independent read-only
review found no blocking issues within this bounded execution scope.

## Next Milestone

Continue the isolated public-data reproduction: hand removal, all-episode
conversion on a working copy, complete original training settings, then held-out
evaluation. Treat construction adaptation as a later, separate experiment when
real spraying recordings are available.

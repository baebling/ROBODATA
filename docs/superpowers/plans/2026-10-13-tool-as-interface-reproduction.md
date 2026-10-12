# Tool-as-Interface Reproduction Plan

User decision: reproduce the public hammer experiment first, separately from
ROBODATA. No construction footage is available. Do not resume Blender tests.

## Boundaries

- Upstream: https://github.com/Tool-as-Interface/Tool_as_Interface
- Pin: `ec1c459ea27ca8e2a2f2787def1d1045ecb2a4ee`.
- Paper: https://arxiv.org/html/2504.04612v2
- Implement only reproduction tooling in `reproductions/tool_as_interface/`.
- Do not modify `src/robodata`, existing experiments, or raw data.
- Keep upstream checkout, virtual environment, data, and large run artifacts
  outside the tracked repository. Record exact versions and deviations.

## Tasks

1. Audit official preprocessing, observation conversion, policy, public data,
   calibration, and pretrained-weight requirements.
2. Test-first: safe archive extraction, read-only dataset audit, episode-bounded
   next-pose action alignment, and source/config verification.
3. Download the public hammer archive, hash it, extract without overwrite, and
   audit all array contracts and supplied pose fields.
4. Execute the unmodified official policy's loss, gradient update, and denoising
   on actual public samples. Retain the official architecture. Explicitly label
   unsegmented input and shortened optimization as an integration smoke test.
5. Execute released reconstruction/augmentation modules and attempt released
   segmentation/pose preprocessing with official weights. Link one re-estimated
   public episode into the policy check. Record failures and compatibility
   adaptations rather than silently substitute.
6. Record executed/not-executed stages, upstream mismatches, raw-data integrity,
   hardware, software, configuration, logs, and artifacts.
7. Verify focused tests and all existing ROBODATA checks before committing the
   separate reproduction branch. Do not merge into the construction pipeline.

## Acceptance Levels

- Source/data audit: verified contracts, checksums, actual decoded RGB/depth.
- Execution reproduction: official modules execute on public samples.
- Training reproduction: complete original training configuration and schedule,
  validated held-out episodes (not satisfied by smoke tests).
- Paper-result reproduction: independent robot trials and comparable success
  rates (not possible without the specified hardware and protocol).

The public preprocessing uses sensor depth for FoundationPose; MASt3R and
Gaussian splatting are in the separate observation-conversion path. The default
hammer config has `novel_view: False`. Preserve and report these distinctions.

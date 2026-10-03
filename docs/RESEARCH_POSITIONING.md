# Construction-Specific Research Positioning

Reviewed 2026-10-03. Proposed research argument, not established novelty or
performance. Researcher decisions remain separate in DECISION_LOG.md.

## Research question

Can helmet-view RGB videos of human spraying on H-beams produce metrically
validated, surface-relative functional nozzle trajectories, and under what
tool-geometry and occlusion conditions are those labels reliable enough to use?

RGB is the final inference target. RGB-D is a paired feasibility/depth comparison.
Camera metadata and declared geometry/scale priors are allowed but must be
disclosed. Independent evaluation instruments must not leak into inference.
The selected tool is Graco TexSpray G-TEX gravity-fed hopper gun, 25D497; the
target is the H-beam web. Material compatibility/quality is a separate question.
See [hardware and experiments](HOPPER_GUN_EXPERIMENT.md).

## What the closest references already do

| Reference and primary evidence | Established relevant contribution | Proposed distinction to test, not a proven advantage |
|---|---|---|
| [DITTO, III-A/B](https://arxiv.org/html/2403.15203v2) | RGB-D relative object motion, manipulated/secondary-object references, trajectory warping and robot execution. | Functional nozzle motion relative to a persistent finite work surface; continuous process geometry and usable-label coverage. |
| [EgoScaler, sections 3-5](https://arxiv.org/html/2506.03605v1) | Ordinary egocentric video, estimated depth, object trajectories, camera-motion compensation and language-conditioned trajectory generation. | Independent metric validation of nozzle/surface labels under construction-tool occlusion. RGB-only or ego-motion compensation alone is not a distinction. |
| [EgoScaler-based VLA, III-B/C](https://arxiv.org/html/2509.21986v1) | Curates object-centroid/rotation trajectories from egocentric datasets and uses them for VLA pre-training. | Construction-process supervision with explicit unreliable intervals, without claiming a new general-purpose VLA. |
| [ORION v2, sections 2-3 and appendix A.2](https://arxiv.org/html/2405.20321v2) | RGB-D/RGB-only demonstrations, object/grasp/contact graphs, continuous trajectories and reference-conditioned warping. RGB uses estimated mesh/depth and adapted FoundationPose. | Validate functional nozzle/finite-web labels and reliability under coupled tool/surface occlusion. RGB-only, continuous/non-contact motion, relations and surface alignment alone are not new. |
| [Tool-as-Interface, sections 3-4](https://arxiv.org/html/2504.04612v2) | Two-view RGB tool/task-frame actions; policy robustness and tool-size variation in pan tasks. | Coupled functional-part and work-surface observability in construction dataset extraction, not tool-centric framing or size robustness in general. |

Here EgoScaler/VLA means Yoshida et al.'s CVPR 2025 EgoScaler and its VLA
follow-up, not every egocentric VLA paper. A broader survey remains necessary
before claiming priority. These source descriptions do not establish that the
papers could not be adapted to construction.

EgoScaler also uses HOT3D's measured object trajectories to evaluate trajectory
generation. Independent GT itself is not our novelty; the proposed benchmark
validates extracted functional nozzle/surface labels in construction conditions.

Use ORION's 2025-09-04 expanded v2, not only the original RGB-D v1. Its discussion
also identifies metrically consistent dynamic reconstruction from moving cameras
as future work; this motivates evaluation, not a claim that our baseline solves it.
ORION retains motion between keyframes and includes Pour-juice. Do not portray
it as contact-only or incapable of continuous motion; our candidate distinction
must rest on nozzle/finite-web process labels and demonstrated reliability gains.

Construction robots already use spray geometry. [Ikeda et al., ISARC 2020,
section 2.3](https://www.iaarc.org/publications/fulltext/ISARC_2020_Paper_204.pdf)
describe beam-part-specific distance, speed, pitch and angle selected through
advance experiments. [Nakamura et al., 2023](https://www.jstage.jst.go.jp/article/aijt/29/72/29_639/_article/-char/en/)
report robot spraying methods for columns, beams and joints. Neither robotic
fireproofing nor geometric spray-path planning is the proposed new contribution.
The proposed contribution is extracting and validating human-video supervision.

## Candidate contribution bundle

1. **Construction-process labels.** Stable functional nozzle point and direction
   in a persistent physical web frame, with distance, incidence, velocity, stroke
   structure and finite-face ray diagnostics. Process ON/OFF stays unknown unless
   supported by observations or sourced annotations.
2. **Occlusion-aware extraction and abstention.** Propagate an initialized nozzle
   frame from visible rigid-body evidence, independently maintain the surface
   reference, and flag unsupported poses. Test whether joint checks improve the
   error/coverage trade-off, rather than merely removing difficult samples.
3. **A paired RGB/RGB-D, independent-GT benchmark.** Separate physical geometry,
   projected tool occupancy, functional-part visibility and tool-caused surface
   occlusion; report error together with remaining contiguous supervision.

The bundle needs a method and supporting experiments. Transform composition,
choosing a nozzle origin, adding schema fields, applying an existing network to
one product, or documenting tool-size sensitivity alone is insufficient evidence
of methodological novelty.

## Construction-specific geometry

For nozzle point p and unit action axis a in the surface frame, with the web at
z=0 and +z outward toward the worker:

```text
d_normal = abs(p_z)
incidence = acos(a dot [0, 0, -1])
lambda = -p_z / a_z
q_nominal = p + lambda * a
```

Reject near-parallel rays or non-positive lambda. A forward plane hit also needs
finite-web and earlier-flange intersection checks. d_normal is ROBODATA's existing
standoff; lambda is the distinct axial nozzle-to-plane distance. For a front-side
ray, lambda = d_normal / cos(incidence).

q_nominal is an ideal geometric intersection, not measured material impact.
Swept rays do not prove deposited coverage, thickness, overlap quality or fire
resistance: a calibrated spray footprint, process/material measurements and
separate validation are required.

An easily tracked large hopper does not guarantee a visible nozzle or a stable
surface reference. Losing the latter can corrupt T_surface_tool while the
camera-relative tool path looks smooth. A plane normal alone cannot recover
persistent in-plane position/heading from a moving view of a featureless plane.

## Transfer inspired by DITTO and ORION

For the same physical surface geometry, rigid transfer can use:

```text
T_camera_new_tool(t) = T_camera_new_surface @ T_surface_tool_demo(t)
```

Different web dimensions require a separately validated tangential path mapping.
Do not scale normal distance, spray footprint or timing merely because beam
length changes. Preserve declared metric standoff/incidence and check flange/
finite-face constraints. Recompute velocity and overlap-related descriptors:
an affine path warp does not automatically preserve them. This is a proposed
later experiment, not completed retargeting or a robot motion planner.

## Hypotheses and falsification

| Hypothesis | Required evidence | Result that would weaken it |
|---|---|---|
| Larger visible bodies can help tracking while hiding the nozzle or web. | Matched hopper-envelope-only simulations; error and coverage by part visibility. | No change after controlling viewpoint, texture and depth. |
| Functional/surface labels are useful beyond centroid paths. | Independent nozzle GT and a fair centroid baseline with calibrated body-to-nozzle offset. | The calibrated object-motion baseline gives equivalent labels/robustness. |
| Joint tool/surface reliability checks improve usable supervision. | Same predictions with/without checks; risk/coverage curves. | Checks only reject hard frames without improving the trade-off. |
| RGB-D diagnoses depth/scale failures but not optical occlusion. | Same RGB frames with depth enabled/disabled; separate body/nozzle/web strata. | Apparent gain depends on unmatched observations or leaked GT. |

A hidden nozzle requires an initialized functional frame and calibrated rigid
body-to-nozzle transform for full-pose propagation. This is not zero-shot semantic
discovery; report initialization cost and drift. Axis-only labels are not full
observable SE(3).

## Fair comparisons and current evidence

- Compare calibrated RGB PnP, RGB-D functional landmarks and body tracking with a
  measured functional offset under identical relevant geometry priors.
- Add EgoScaler-style RGB depth/tracking/registration as a candidate, with an
  explicit metric-scale treatment and confirmed executable assets.
- Name DITTO/ORION components as adaptations. Their robot-execution success rates
  are not directly comparable to our label RMSE.
- Separate oracle/manual observations from automatic perception. Disclose masks,
  CAD, scale, initialization and manual rejection; identify unequal-prior tests.
- Retain failures in coverage denominators. Never fit scale or functional offsets
  using held-out test trajectories.

Current code implements surface-relative labels and synthetic geometry checks,
not this full benchmark or an automatic G-TEX model. The [DITTO probe](DITTO_INTEGRATION.md)
tested six selected synthetic pairs with oracle masks and renderer metric depth;
it did not estimate the initial nozzle frame, remove camera motion or validate
transfer to changed webs. The old spray tool is a generic proxy. The approved
[hopper simulation pilot](HOPPER_SIMULATION_PILOT.md) adds an unmeasured G-TEX-style
shape and descriptive matched size/visibility tests, not a measured 25D497 model
or a new learned-model comparison.

No real G-TEX accuracy, geometry/occlusion effect size, unknown-tool metric RGB
generalization, VLA gain or coating-quality improvement is established. The next
deliverable is measured geometry and a small paired benchmark, not a larger
foundation model or a complete autonomous robot.

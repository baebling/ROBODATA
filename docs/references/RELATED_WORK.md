# Primary References and Implementation Choices

Checked 2026-10-03. Source claims and our engineering choices are distinct.

| Source | Verified relevant content | Use in ROBODATA |
|---|---|---|
| [ToolEENet](https://arxiv.org/abs/2404.04193), abstract | RGB-D functional-part segmentation, category-level 6D EE pose and symmetry-aware representation. | Functional point/axis, separate full-rotation validity. |
| [ToolEENet source](https://github.com/yl-wang996/ToolEENet), README | Repository still states clean code, checkpoints and dataset will be released. | Do not require unavailable weights or claim this repo runs its trained model. |
| [Tool-as-Interface](https://arxiv.org/html/2504.04612v2), section 3, Figures 2-3 | Two RGB views, reconstruction, tool pose in camera/task frames, task-space representation. | Camera-to-surface transform and geometry/perception module separation. |
| [DITTO](https://arxiv.org/html/2403.15203v2), III-A/B | RGB-D relative object trajectories and manipulated/secondary-object-conditioned warping. | Motion/transfer reference, not proof of nozzle initialization or moving-camera surface compensation. |
| [EgoScaler](https://arxiv.org/html/2506.03605v1), sections 3-5 | Monocular estimated depth, egocentric object tracking, reference registration and trajectory generation. | Candidate RGB baseline; RGB-only extraction and camera compensation have precedent. |
| [EgoScaler-based VLA](https://arxiv.org/html/2509.21986v1), III-B/C | Curated egocentric object trajectories for VLA pre-training. | Supervision precedent; video-to-action data generation itself is not new. |
| [ORION v2](https://arxiv.org/html/2405.20321v2), sections 2-3 and appendix A.2 | RGB-D/RGB-only demonstrations, object/contact graphs and reference-conditioned warping; RGB estimates mesh/depth and adapts FoundationPose. | Latest expanded version, not RGB-D-only v1; study continuous spraying rather than generic graph novelty. |
| [ISARC fireproofing robot](https://www.iaarc.org/publications/fulltext/ISARC_2020_Paper_204.pdf), 2.3 | Beam-part-specific distance, speed, pitch and angle from advance experiments. | Distinguish video-label extraction from existing spray-path programming. |
| [OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html), pose definition and methods | Known 3D points and corresponding image points produce object-to-camera pose. | Calibrated metric RGB baseline using measured tool/surface geometry. |

The compact heatmap/visibility network is our transparent supervised baseline,
not an implementation or reproduction of ToolEENet's diffusion model. Fixed
geometry and annotation initialization are disclosed additional requirements of
the first RGB baseline. Mask centroid alone and one action axis cannot resolve
scale, depth and full roll; this is a geometry constraint, not model performance.

Multi-view geometry can recover structure but an absolute meter scale still
needs a physical reference (measured baseline/object/depth). We do not treat a
paper's broad reconstruction description as proof of metric scale for an
arbitrary monocular construction video.

See [research positioning](../RESEARCH_POSITIONING.md) and
[G-TEX protocol](../HOPPER_GUN_EXPERIMENT.md) for claims and controlled tests.
Tool-as-Interface also tests pan sizes; size sensitivity or task-relative tool
frames alone do not establish novelty. EgoScaler's
[official source](https://github.com/Biscue5/EgoScaler) and its
[VLA follow-up source](https://github.com/Biscue5/egoVLA) are candidate integrations,
not currently executed ROBODATA modules.

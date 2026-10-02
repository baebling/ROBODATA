# Primary References and Implementation Choices

Checked 2026-10-02. Source claims and our engineering choices are distinct.

| Source | Verified relevant content | Use in ROBODATA |
|---|---|---|
| [ToolEENet](https://arxiv.org/abs/2404.04193), abstract | RGB-D functional-part segmentation, category-level 6D EE pose and symmetry-aware representation. | Functional point/axis, separate full-rotation validity. |
| [ToolEENet source](https://github.com/yl-wang996/ToolEENet), README | Repository still states clean code, checkpoints and dataset will be released. | Do not require unavailable weights or claim this repo runs its trained model. |
| [Tool-as-Interface](https://arxiv.org/html/2504.04612v2), section 3, Figures 2-3 | Two RGB views, reconstruction, tool pose in camera/task frames, task-space representation. | Camera-to-surface transform and geometry/perception module separation. |
| [DITTO](https://arxiv.org/abs/2403.15203), abstract | Extracts object motion from human video and transforms it for robot imitation. | Trajectory extraction/transformation reference; no imported robot controller. |
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

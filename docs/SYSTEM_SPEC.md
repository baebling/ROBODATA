# Implemented Research Model

## Executable pipeline

```text
RGB video + intrinsics + measured tool/surface landmark geometry
    -> manual/checked-KLT or learned semantic landmarks
    -> calibrated T_camera_tool and T_camera_surface (PnP)
    -> T_surface_tool = inverse(T_camera_surface) @ T_camera_tool
    -> functional point/axis, standoff, incidence and velocities
    -> future pose and body-relative motion labels
    -> validated JSON + masked NPZ + trajectory/overlay figures
```

`LandmarkNet` is an RGB encoder/decoder with three resolution levels, skip
connections, GroupNorm and separate heatmap/visibility heads. The 12-landmark
synthetic example uses width 16 and 64x64 input. Training uses Gaussian landmark
heatmaps, foreground-weighted masked MSE, and 0.1 times visibility BCE. Invisible
landmarks do not contribute to coordinate loss. The best validation-loss
checkpoint records configuration, annotation hashes and training provenance.
Peak coordinates map back to the original image using the same half-pixel
resize convention. Scores are uncalibrated; neither high confidence nor small
reprojection residual guarantees correct metric depth.

RGB is the primary path. Measured geometry is an explicit scale reference for
this baseline, not a claim that metric scale is observable from arbitrary RGB.
PnP does not need a pre-specified tool pose. It estimates the pose from known
landmark geometry and detected image correspondences. Low-confidence,
insufficient, degenerate or high-residual observations produce invalid records.

## RGB-D comparison

Aligned rectified depth back-projects functional semantic landmarks. A tool
point and axis are retained even when no secondary landmark fixes roll. Full
SE(3) and rotation-dependent future labels are then unavailable, not invented.

The static-camera path robustly fits the first valid masked surface point cloud
and freezes its centroid/normal/axes. Camera-x projected onto the plane fixes x.
For moving cameras, a plane alone cannot fix the in-plane origin and heading;
persistent measured surface landmarks and per-frame PnP are required instead.

## Module boundaries

| Module | Responsibility |
|---|---|
| `observations.py` | Calibrated video/depth, semantic observations, checked KLT |
| `learning/` | Trainable RGB landmark detector and inference adapter |
| `geometry.py` | SE(3), PnP, plane estimation, projection conventions |
| `pipeline.py` | Canonical state construction, provenance, failure preservation |
| `schema.py` | Machine-validated data contract |
| `temporal.py` | Velocities, horizon targets, missing/invalid-frame boundaries |
| `reference.py` | Independently calibrated mocap/tracker CSV import |
| `evaluation.py` | Timestamp-only matching, coverage and geometric errors |
| `export.py` | JSON and masked robot-agnostic learning tensors |
| `synthetic.py` | Rendered engineering fixtures with analytic reference poses |

## Implemented vs Pending

Masks are accepted as bootstrap/model observations and their centroids are
stored; this version does not train an automatic tool/surface segmentation
network. The RGB landmark network is trainable, but only synthetic smoke weights
have been produced locally. Real semantic annotations and measured geometry are
needed for a real-data experiment. Geometry-based symmetry disambiguation for
unseen tools remains open.

Process ON/OFF and primitive labels are optional sourced annotations. Automatic
process/primitive classification, force/material estimation, arbitrary monocular
surface reconstruction, DITTO-style correspondence-based trajectory warping,
robot IK/planning and robot policy learning are not implemented. DITTO informs
the representation/transfer discussion; this repository is not a reproduction
of its robot controller.

Independent tracker CSV poses require surveyed/calibrated transforms and clock
offset. A LiDAR surface survey may supply calibration, but there is no
vendor-specific LiDAR or mocap acquisition driver. Inference never consumes
ground-truth poses or calibrates itself by fitting the test GT trajectory.
Original tracker indices or verified sampling cadence preserve acquisition gaps.
Externally supplied occlusion constraints survive detector inference and remain
distinguishable from learned keypoint visibility predictions.

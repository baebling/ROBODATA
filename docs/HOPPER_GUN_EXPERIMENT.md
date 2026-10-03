# G-TEX Geometry and Occlusion Experiment

Proposed protocol, 2026-10-03; no real-data run or measured CAD is supplied here.
Numeric levels below are engineering test conditions, not manufacturer operating
recommendations or fireproofing quality limits.

## Selected hardware and missing measurements

The researcher selected **Graco TexSpray G-TEX gravity-fed hopper gun, 25D497**.
The [manufacturer](https://www.graco.com/kr/ko/contractor/product/25d497-texspray-g-tex.html)
lists a 5.7 L hopper, metal body, included 4/6/8 mm metal nozzles and a round spray
pattern. This is not pressure-boosted FastFinish. Record the installed nozzle in
each episode; the listed alternatives do not establish which one is used.
A round nominal pattern does not establish its surface footprint.

Before a geometry-calibrated real run, collect:

- Hopper/body envelope dimensions in meters and a versioned mesh or measured
  semantic landmark set. Do not infer dimensions from product photographs.
- Nozzle exit center/outgoing axis and a stable second body direction for roll.
  Keep body, hopper and functional-part annotations separate.
- T_body_tool for each nozzle/assembly and independent T_marker_tool if tracking
  markers are used; verify attachment rigidity.
- Installed nozzle, hopper fill, hands/grip and hose layout. Fill is a possible
  motion/appearance covariate, not an assumed effect.
- H-beam web patch identity/dimensions, flange geometry, persistent surface
  landmarks and independent surface-to-tracking calibration.
- Camera intrinsics/distortion, mounting, timestamps and RGB/depth alignment
  matching the actual lens/crop/resolution mode.

These are additional priors of the first baseline. A future unknown-tool RGB
method must disclose how it obtains geometry and meter scale. LiDAR can survey
stationary geometry, not a moving tool's synchronized 6D trajectory; mocap or
calibrated tracking provides the latter reference.

The product page does not establish compatibility with the selected fireproofing
material. Separate dry motion, appropriate non-structural spraying tests and any
later qualified material study. For this proposed spraying protocol, work_on
means actual material discharge, not nozzle motion or trigger position. Record
known no-material clips as OFF with protocol provenance; do not infer air/trigger
state from it. They do not test plume visibility, flow or coating performance.

## Distinct variables and occlusion denominators

| Quantity | Definition |
|---|---|
| Physical geometry | Measured or explicitly synthetic hopper/body shape, dimensions and nozzle offset. |
| Projected tool occupancy | Visible tool-mask pixels / image pixels; also retain bounding box and part visibility. |
| Tool-caused surface occlusion | Fraction of a fixed physical web ROI's camera projection hidden specifically by the tool. |
| Functional-part visibility | Nozzle exit and direction-defining part visibility, not whole-tool confidence. |
| Surface-reference observability | Persistent landmark/registration support, degeneracy, residual and loss durations. |
| Other occluders | Hands, hose, flange and plume, separately attributed where possible. |

Physical size, image occupancy and surface occlusion are not interchangeable.
Viewpoint, camera-to-tool distance and field of view also change occupancy.
Changing nozzle standoff affects both perception and process geometry, so it is
not a pure size experiment.

For simulator evaluation, A_full is the same physical web ROI projected inside
the current image with occluders removed. A_tool is its subset where the first
scene intersection in front of the ROI is a tool-body/hopper/nozzle instance,
using depth and instance IDs. A hand hiding both tool and ROI is attributed to
the hand, not also counted as directly tool-caused occlusion.

```text
tool_surface_occlusion = area(A_tool) / area(A_full)
tool_image_occupancy = area(visible_tool_mask) / (image_width * image_height)
```

Empty/unknown A_full makes occlusion unavailable, not zero. Report out-of-image
ROI fraction separately. Do not use the entire beam as denominator for one local
working patch. A visible mask alone cannot reveal hidden area: real ROI occlusion
needs independently registered geometry or amodal annotation with uncertainty.
These evaluation-only annotations must not be passed to perception. Report union
occlusion separately from tool-only occlusion.

## Stage 1: measured-geometry simulation

Replace the existing web-facing Blender proxy only after obtaining measurements;
keep old renders unchanged. Register the nozzle frame, render body/hopper/nozzle
and occluder instance IDs, metric depth, transforms and separate visibility GT.

1. Repeat identical nozzle/camera motions, lighting and seeds, varying only the
   hopper envelope at 0.75/1.00/1.25 times measured dimensions. Preserve gun body,
   nozzle geometry/frame and their relation. Scaled hoppers are synthetic stress
   conditions, not commercial products.
2. Use fixed oblique and helmet-like moving views, with stationary and sweeping
   nozzle controls. Stationary controls diagnose false motion, not actual-motion
   tracking performance.
3. Cross nozzle visibility and surface-reference visibility where feasible:
   both visible, nozzle hidden/reference visible, nozzle visible/reference hidden,
   both hidden. Record realized occlusion, not assumed percentages.
4. Include center sweeps, flange-adjacent sweeps and web-end misses. Away/parallel
   rays remain invalid for spraying onto the web.
5. Compare synthetic RGB/RGB-D with identical disclosed initialization/masks;
   evaluate automatic observations separately. Oracle inputs are named controls,
   not inputs to the automatic method.

Vary camera viewpoint/distance separately to sample occupancy. Report shape,
occupancy and occlusion jointly. Counterfactual simulation isolates geometry;
real repeated motions are not identical and cannot alone establish a causal
tool-size effect.

## Stage 2: independent-GT real pilot

Use actual 25D497, a surveyed web patch, calibrated helmet camera and independent
tool tracker. For a controlled depth comparison, enable/disable aligned depth
on the same RGB frames from one RGB-D stream. Separately test smartphone RGB:
different lenses/views/times constitute a cross-camera comparison, not a pure
depth ablation.

Combined T_surface_tool GT needs surveyed surface-to-tracking calibration and
synchronized tool tracking, but not camera tracking. To decompose tool and surface
errors in the moving camera frame, additionally acquire independent
T_tracking_camera(t), a calibrated camera-marker/extrinsic transform and clock
alignment. Then transform tool/surface GT into the camera frame independently.
Without this extra camera reference, restrict separate camera-frame error
decomposition to simulation and report only combined real surface-relative error.

Do not give tracker poses to inference or use test poses to tune calibration.
Record whether GT markers are visible in RGB: marker-assisted tracking can be
easier than unmarked tool tracking. Disclose this condition and obtain a separate
unmarked/marker-out-of-view subset where independent tracking remains valid
before claiming marker-free video performance.

Proposed dry-motion pilot: two viewpoints (fixed oblique/helmet) by three sweeps
(horizontal/vertical/reversing-zigzag) by three conditions (reference mostly
visible/partial tool overlap/prolonged reference loss), with three repetitions:
**54 episodes**. Final clip duration, feasible visibility combinations, distance
and speed follow measured setup. This is a debugging pilot, not a powered final
dataset or a worker/tool generalization study. Record realized visibility.

Start with one installed, recorded nozzle. Add other nozzles after checking their
visible geometry difference, rather than multiplying every pilot condition.
Process-ON clips require an appropriate documented material/procedure and form
a separate plume/process-observability study.

Later hold out whole workers, days, tool instances and web patches. Adjacent
frames are not independent train/test data. One G-TEX instance cannot establish
generalization across tool shapes or commercial families.

## Baselines and reporting

- Body motion plus calibrated fixed nozzle offset versus functional landmarks.
  A raw mask-centroid path without that offset is only a diagnostic.
- Calibrated RGB/RGB-D; DITTO body tracking with disclosed nozzle initialization;
  candidate EgoScaler-style RGB tracking/depth/registration.
- Independent surface registration versus a static-camera reference. The latter
  is a deliberately invalid-assumption control under a moving camera.
- Joint functional/surface validity versus tool-only checks on the same
  predictions, measuring the rejection/error trade-off.
- Record oracle observations, CAD/scale priors, calibration, initialization cost
  and manual rejections for every baseline.

Report nozzle-position MAE/RMSE/tail error, axis angle, observable full rotation,
normal-distance/incidence error, nominal ray-hit position error on comparable
valid hits, and web-hit/flange-block/miss classification. Separate tool and surface
pose errors only where independent camera-frame GT is available. Do not count
away or flange-blocked rays as successful web hits.

Also report full-pose/axis-only coverage separately, temporal label coverage at
the configured horizon, longest continuous valid segment, gap durations and
reacquisition delay. Preserve every original frame/time gap. Give sampled-frame
and duration-weighted coverage for variable-rate streams. Distinguish independent
GT availability from prediction availability and count unmatched observations.
Use per-episode/group distributions and group-level confidence intervals, not
frame-independent confidence intervals.

Stratify by occupancy, nozzle visibility, surface occlusion and viewpoint. Show
risk/coverage curves; choose empirical thresholds on a calibration split, not
the held-out test set. Existing heuristic scores are not probabilities.

For normal aim on a plane, angular perturbation delta produces lateral nominal
hit displacement d_normal * tan(delta). This is a geometric illustration,
not measured performance, a coating model or an accuracy claim.

## Implementation boundary

Detailed occlusion metrics, measured G-TEX mesh, ablations and this real pilot are
**not implemented or collected**. The current schema/CLI does not export all the
proposed fields. Version a separate benchmark manifest/renderer annotations and
test geometry, visibility attribution and denominator handling before evaluating.
Keep raw footage and GT immutable; retain original estimates when later testing
smoothing or gap recovery.

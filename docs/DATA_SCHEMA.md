# Data and Coordinate Contract

Authoritative machine models: src/robodata/schema.py and observations.py.
Generate JSON Schema with `robodata schema --output docs/episode.schema.json`.

T_A_B is a row-major 4x4 matrix mapping coordinates from B into A.
T_surface_tool = inverse(T_camera_surface) @ T_camera_tool. Camera is OpenCV:
+x right, +y down, +z forward. All physical frames are right-handed. Position is
meters, angles radians, timestamps seconds. Geometry points in the configuration
must be expressed in their intended functional/physical reference frames.

The tool origin is its functional point (nozzle/impact point). Tool +z is the
action direction; a second semantic direction fixes roll. An axis-only RGB-D
estimate retains point and +z but leaves SE(3) null and full_pose_valid false.
The mask centroid is a 2D diagnostic, not a persistent 3D functional origin.

Optional frame input `hand` uses MediaPipe landmark index strings (`0` wrist,
`5/9/13/17` MCPs). Hand inference stores all 21 normalized image/local world
landmarks separately. Local world values are hand-centered estimates, NOT a
camera-frame pose. Keypoint confidence=1 denotes an accepted model prediction;
`visible` denotes in-image availability, not actual unoccluded joint visibility.
Handedness score is stored separately and is not a joint confidence.

The optional run configuration `hand_assistance` supplies metric `points_hand_m`,
row-major `T_hand_tool`, `calibration_source`, and explicit `grasp_verified=true`.
It also declares `handedness` (Left/Right); unknown or opposite model handedness
is rejected. Handedness agreement does not prove worker identity or a stable grasp.
After direct tool estimation fails, calibrated hand-template PnP estimates
`T_camera_hand`; composition gives `T_camera_tool = T_camera_hand @ T_hand_tool`.
This assumes a fixed calibrated grasp and hand template. A proximity association
to the tool mask is only a candidate, not proof of grasp or calibrated scale.
Calibration is a declared prior, not a per-frame GT tool-pose input. Missing hands,
unverified calibration or rejected PnP leave frames invalid. Synthetic calibration
cannot be applied to a real hand/tool without independent measurement.
The hand estimate records its actual `metric_scale_source=calibrated_hand_template`
and calibration source. RGB-D episodes using this path declare
`mixed_depth_and_hand_calibration`, not depth-only tool scale. FPS-derived hand
timestamps remain marked assumed constant even though raw tracker times are stored.
An occluded tool can be inferred only through this explicitly enabled calibrated
hand path; `quality.occluded` remains true and `tool.method` identifies the indirect
estimate. An annotated occluded hand is still rejected. Without this opt-in prior,
tool occlusion remains invalid as before.

For moving cameras, surface origin/axes belong to a fixed measured landmark
patch, normally its center. For plane-only static RGB-D the first fitted visible
patch centroid defines the origin and camera +x projected into the plane defines
x (camera +y fallback). The reference remains fixed during the episode. It is
view-dependent across episodes; independently calibrate it before GT comparison.
Do not reset a centroid/reference every frame and call the result invariant.

Signed surface distance is tool point z in the surface frame. Standoff is its
absolute value. Incidence is acos(dot(tool +z, -surface +z)): zero means directed
normally into the working side. This convention avoids normal-sign ambiguity.

Each frame stores input paths, estimates/method/residuals, canonical geometry,
quality, process annotations if available and temporal targets. Quality scores
are uncalibrated heuristics, not probabilities. Invalid fields are JSON null.
Explicit timestamps are preferred; FPS fallback declares a constant-rate
assumption. RGB-D requires aligned, rectified depth; unit conversion is explicit.

Action target is T_surface_tool(t+H), snapped only within configured timestamp
tolerance. delta_T_tool_current_tool_future = inverse(T_current) @ T_future.
Its translation and rotation vector are in the current tool frame. The separate
delta_position_surface_m is a surface-frame displacement. Angular velocity is
expressed in the surface frame. Future labels require all intervening full poses.
Skipped frame indices also block these labels and adjacent-frame derivatives.
Direction vectors must be unit length; full poses must agree with their stored
functional point and +z axis. Valid action records require a complete future target.

NPZ tensors: state/target are Nx12 [position(3), row-major rotation(9)];
delta_tool is Nx6 [body translation(3), body rotation vector(3)]. axis_state is
Nx6 [point(3), direction(3)]. Invalid entries are zeros and must be ignored using
state_valid/action_valid/axis_valid. Use `np.load(..., allow_pickle=False)`.

Reference CSV requires timestamp_sec,x_m,y_m,z_m,qx,qy,qz,qw in the tracking
frame and original integer frame_index, preserving acquisition gaps. If original
indices are unavailable, calibration must supply a verified sample_period_sec;
indices are reconstructed on that grid within sample_time_tolerance_sec. Missing
slots are not renumbered away. A calibration YAML supplies T_surface_tracking, T_marker_tool,
time_offset_sec (added to reference time), instrument and surface_frame_id.
T_surface_tool = T_surface_tracking @ T_tracking_marker @ T_marker_tool.
Never obtain reference calibration by fitting the estimated test trajectory.
CSV rows must be complete valid instrument measurements; malformed rows are
rejected. Vendor validity/occlusion exports need an explicit adapter rather than
silently replacing missing measurements with zero poses.

Neural landmark manifests retain the checkpoint SHA-256, training provenance
and score threshold. Supplied occlusion flags retain their source; direct tool
estimation rejects them. Only the explicit calibrated hand exception above can
produce an indirect occluded-tool estimate. Extraction propagates this provenance.

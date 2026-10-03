# Open Research Questions

- For selected G-TEX 25D497/web spraying, confirm installed nozzle, material/procedure
  and observable semantic nozzle/body landmarks. Tool family and part number are decided.
- Obtain measured G-TEX geometry/CAD, body-to-nozzle calibration, intrinsics and
  an agreed persistent physical web patch with flange geometry.
- Determine whether arbitrary-tool geometry can be recovered from RGB without manual measurements.
- Resolve symmetry/roll observability for each tool; an action axis alone is not full rotation.
- Acquire representative helmet-view footage and a calibrated independent gold subset.
- Determine availability and accuracy of mocap/LiDAR/tracker equipment and synchronization.
- Choose target dataset scale, task labels and episode-level holdout protocols.
- Calibrate quality scores against gold errors and select thresholds empirically.
- Compare a real pretrained semantic perception model once usable assets and licenses are available.
- Establish novelty and real-data performance relative to DITTO, EgoScaler/VLA,
  ORION and ToolEENet/Tool-as-Interface, including construction spraying work.
- Determine a reproducible amodal surface ROI and tool-only occlusion reference;
  visible masks alone cannot measure hidden surface area.
- Choose practical physical/synthetic geometry variation, camera conditions and
  holdouts; one G-TEX cannot establish cross-tool-shape generalization.
- Define material-specific requirements before deposition/quality claims;
  nominal nozzle-ray hits alone are geometric diagnostics.
- Measure worker-specific hand template and hand-to-nozzle calibration; determine
  how grip changes, gloves, finger occlusion and tool symmetry affect hand assistance.
- Calibrate helmet mount pose and each video-mode camera; confirm native Triton
  RGB intrinsics versus Helios registered RGB-D grid and actual Chroma assembly.

These questions do not prevent running the current baseline, but they constrain
the scientific claims that can be made from its outputs.

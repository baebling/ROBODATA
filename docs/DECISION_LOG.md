# Decision Log

| ID | Status/source | Decision |
|---|---|---|
| D001 | Researcher | RGB final goal; RGB-D used to investigate feasibility. |
| D002 | Researcher/BAE_ROBOT | Dataset generation and label validation are the current study. |
| D003 | Researcher | Functional +z direction depends on tool use (impact/nozzle). |
| D004 | Researcher | Mocap/LiDAR are evaluation references, not mandatory inference inputs. |
| D005 | Researcher | DITTO, especially trajectory transformation, is a reference. |
| D006 | Researcher 2026-10-02 | Missing implementation criteria may be selected with research support. |
| D007 | Researcher 2026-10-02 | Follow-up spraying simulation faces the H-beam web, not the top flange. |
| D008 | Researcher 2026-10-03 | First physical spraying tool: Graco TexSpray G-TEX gravity-fed hopper gun, part 25D497. Installed nozzle, measured geometry and material remain unconfirmed. |
| D009 | Researcher 2026-10-03 | Positioning must compare DITTO, EgoScaler/VLA and ORION; investigate tool geometry/size and surface occlusion. Effects and novelty are not yet established. |
| D010 | Researcher 2026-10-03 | Use hand-pose estimation as auxiliary evidence for a grasped tool; correct Blender to helmet-mounted egocentric view and compare available sensors. |
| D011 | Approved follow-up 2026-10-03 | Test YOLO segmentation, BoT-SORT and persistent surface localization on the preserved helmet simulations. A tracker ID alone is not a metric surface frame. |
| D012 | Approved follow-up 2026-10-03 | Test local fireproof_workarea components and included SVO where the runtime permits, preserving the source, weights, data and gates. Module outputs are not full-pipeline success. |
| D013 | Researcher | Measured tool/H-beam shape and dimensions are allowed priors; runtime inference remains helmet RGB only. |
| D014 | Researcher | Stop additional simulation experiments; acquire real data and prefer offline full-video processing over assumed realtime operation. Full-video optimization is not yet implemented. |
| D015 | Approved follow-up | Connect VGGT scene reconstruction to measured surface coordinates and the canonical tool trajectory path, with an explanation of remaining bottlenecks. |
| A001 | Agent default | Use calibrated known-geometry PnP as an executable RGB baseline. |
| A002 | Agent default | Functional point is the stable action origin; preserve mask centroid separately. |
| A003 | Agent default | Persistent measured surface landmarks define a moving-camera reference. |
| A004 | Agent default | Plane-only RGB-D reference is frozen from its first valid fit, static camera only. |
| A005 | Agent default | Right-handed OpenCV camera; meters/radians/seconds; T_A_B maps B to A. |
| A006 | Agent default | Future pose and body-relative delta at a configurable 0.2s horizon. |
| A007 | Agent default | Do not interpolate or smooth; invalid samples block temporal labels. |
| A008 | Agent baseline | Compact RGB heatmap/visibility U-Net is trainable, without real pretrained weights. |
| A009 | Agent example | Existing synthetic spray proxy is an engineering fixture, not the selected G-TEX geometry or a validated material process. |
| A010 | Approved follow-up | Web frame: face center, +x along beam, +y up, +z outward; ray/finite-face diagnostics remain separate from automatic perception. |
| A011 | Approved follow-up 2026-10-03 | Run a simulator-only hopper size/visibility pilot using explicitly unmeasured exterior assumptions. Scale only the hopper around its attachment; preserve body/nozzle/landmarks and prior raw renders. This does not establish actual 25D497 dimensions or performance. |
| A012 | Agent baseline 2026-10-03 | Optional MediaPipe 0.10.21 VIDEO landmarks assist through metric hand-template PnP and an explicitly verified fixed T_hand_tool calibration, only after direct tool estimation fails. Local model world landmarks are not a camera pose. |
| A013 | Agent simulation default 2026-10-03 | Synthetic fixed grasp and nominal sensor FoV; four helmet distance/pitch conditions. Smartphone published FoV is provisionally interpreted as diagonal. ZED max H/V axes and Helios registered ToF grid are not calibrated video intrinsics. No real sensor noise or spray physics is claimed. |
| A014 | Agent bounded pilot 2026-10-03 | YOLO26n-seg fine-tuning on 72 synthetic train / 24 val images; 128 helmet images reserved from training. Planar RGB localization uses a tool-free reference, known metric patch geometry and color fixtures. Evaluation exposed a reference-mask bug; subsequent runs are development diagnostics, not untouched statistical tests. |
| A015 | Agent bounded pilot 2026-10-03 | Unmodified local RGB-PnP tracker receives an ideal first-frame synthetic depth seed and later RGB only; camera-center scoring is separate from semantic web pose. The original five-face recognizer is tested independently on 16 initial frames with the original checkpoint. Neither test produces T_surface_tool; SVO counts remain unavailable without ZED SDK. See WORKAREA_ALGORITHM_PILOT.md. |
| A016 | Agent VGGT adapter | Use locally available original VGGT checkpoint/runtime, explicit named measured RGB anchors and right-handed Sim3 alignment. Reject missing/poor alignment; preserve sampled-frame gaps. Add measured body-to-nozzle RGB PnP composition. Test actual inference on existing DITTO real RGB, not a new simulation or construction accuracy experiment. |

Thresholds are configuration defaults for engineering tests, not standards for
fireproofing quality or validated operating limits. Revise them using the gold
benchmark and keep old experimental configurations for reproducibility.

Contribution hypotheses and the 54-episode debugging pilot in
RESEARCH_POSITIONING.md and HOPPER_GUN_EXPERIMENT.md are agent proposals, not
researcher-approved academic claims or collected real data. The limited synthetic
visibility implementation is recorded separately in HOPPER_SIMULATION_PILOT.md.

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

Thresholds are configuration defaults for engineering tests, not standards for
fireproofing quality or validated operating limits. Revise them using the gold
benchmark and keep old experimental configurations for reproducibility.

Contribution hypotheses and the 54-episode debugging pilot in
RESEARCH_POSITIONING.md and HOPPER_GUN_EXPERIMENT.md are agent proposals, not
researcher-approved academic claims or collected real data. The limited synthetic
visibility implementation is recorded separately in HOPPER_SIMULATION_PILOT.md.

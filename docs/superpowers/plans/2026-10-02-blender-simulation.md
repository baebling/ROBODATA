# Blender Simulation Implementation Plan

> **For agentic workers:** Use executing-plans and test-driven-development to execute the approved simulation scope.

**Goal:** Render a repeatable H-beam/tool proxy experiment and evaluate RGB and RGB-D against simulation truth.

**Architecture:** Blender produces images, depth, object labels and scene transforms without accessing ROBODATA predictions. A separate Python adapter validates those outputs and creates existing ROBODATA manifests and reference episodes. The running Blender window opens a separate saved scene; existing work is backed up first.

**Tech Stack:** Installed Blender 5.2.2 LTS, Blender Python API, existing NumPy/OpenCV/Pydantic/pytest stack. No add-ons or new external services.

**Spec:** The researcher's approved scope in this conversation: Blender scene -> RGB/depth/truth -> ROBODATA -> error report.

## Global Constraints

- Synthetic proxy geometry and oracle landmarks are not real gun perception or human demonstrations.
- Meters, seconds, right-handed frames; OpenCV camera and functional nozzle origin/+z axis.
- RGB and RGB-D use identical rendered RGB and persistent surface landmarks.
- Preserve raw rendered depth; convert declared depth semantics into axial camera Z explicitly.
- No overwriting existing output directories, raw data or user scenes.
- Process/particle physics, robot IK and policy learning are outside this slice.
- Configuration, source hashes, runtime version and Git state accompany results.

## Tasks

### 1. Adapter and geometry tests

Files: `src/robodata/blender_input.py`, `tests/test_blender_input.py`.

- [x] Add failing tests for Blender/OpenCV axes, radial-to-axial depth, missing samples, and existing output rejection.
- [x] Run `.venv/bin/python -m pytest tests/test_blender_input.py -q` and observe missing adapter failure.
- [x] Implement `camera_from_blender(matrix_world)` and `radial_to_axial(depth, K)` with validated inputs.
- [x] Implement `assemble_rendered(render_root, output_root)` consuming `render_metadata.json`, PNG RGB/labels and float EXR depth. Produce `manifest.json`, RGB/RGB-D YAML and a separate `ground_truth.json`.
- [x] Re-run focused tests and exercise malformed/invalid data paths.

### 2. Blender renderer

Files: `scripts/blender_simulate.py`, `configs/blender_simulation.json`.

- [x] Run the absent renderer path to establish the missing executable artifact.
- [x] Build a separate H-beam top-face reference and a labelled calibration spray proxy, not an assumed measured real gun.
- [x] Animate fixed/moving cameras and horizontal/vertical/zigzag paths plus stationary-tool moving-camera control.
- [x] Render RGB, native float depth and object IDs with Cycles. Export actual scene transforms, projected marker centers and ray-cast visibility.
- [x] Save a `.blend` file and settings for each scenario in a new directory.
- [x] Check a short render against ray-cast depth samples before the full pilot.

### 3. Evaluation, display and documentation

Files: `src/robodata/blender_experiment.py`, `src/robodata/cli.py`, `docs/BLENDER_SIMULATION.md`, `README.md`. Evaluation uses the existing CLI pattern rather than a separate wrapper script.

- [x] Assemble rendered inputs and run both existing extraction pipelines without exposing truth to them.
- [x] Report matched coverage, position/axis/standoff/incidence error; preserve invalid frames.
- [x] Generate RGB previews and existing trajectory/overlay reports.
- [x] Back up the original running Blender scene and open the experiment scene with Computer Use.
- [x] Run format, lint, type checking, all unit tests and a real Blender end-to-end pilot.
- [x] Review source and depth/frame conventions before publishing source changes; keep rendered data out of Git.

## Verified Outcome

The 7-condition, 210-frame pilot and final RGB/RGB-D evaluation completed.
All 68 tests, format/lint, and type checks pass. Independent review found no
remaining important issues after validation fixes. Native Blender animation,
original-scene backup, and fresh 4-frame source-provenance capture were verified.
See `docs/VERIFICATION.md` for measurements and the initial pilot's explicitly
disclosed missing render-time source snapshot.

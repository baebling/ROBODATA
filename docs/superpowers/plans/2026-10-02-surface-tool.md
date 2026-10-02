# Surface Tool Model Implementation Plan

> Agentic workers: execute inline using executing-plans; verify each deliverable.

Goal: runnable RGB/RGB-D dataset generation with a trainable RGB perception model.
Architecture: independent geometry, observation, temporal and evaluation modules
share a validated episode contract. CLI commands compose the modules.
Tech stack: Python 3.10+, NumPy, SciPy, OpenCV, Pydantic, PyTorch (optional).
Spec: ../specs/2026-10-02-surface-tool-design.md

## Global Constraints

Meters/radians/seconds, right-handed frames, T_A_B maps B to A. RGB final target;
RGB-D comparison. No implicit metric scale, GT, interpolation or full rotation.
Configurations and source revisions accompany outputs. Keep raw data read-only.

## Task 1: Geometry and contracts

Files: src/robodata/{geometry,schema,config}.py; tests/test_geometry.py.
Interface: estimate_pnp(points3d, pixels, K) -> PoseEstimate; fit_plane(points,
config) -> PlaneEstimate; inverse(T) -> T; Episode validates complete records.
Test: project a known 0.7m tool pose, recover it; add plane outliers; reject
reflection matrices and unsupported scale metadata. Run tests before and after.

## Task 2: Frames and extraction

Files: src/robodata/{observations,pipeline}.py; tests/test_pipeline.py.
Interface: run_episode(config_path, output_dir) -> Episode; frame loader yields
timestamps/RGB/aligned depth; detection observations are named pixels/visibility.
Test: a generated video yields the hand-defined 0.3m standoff; an occluded frame
has null pose; missing scale and moving plane-only references are rejected.

## Task 3: Labels and evaluation

Files: src/robodata/{temporal,evaluation,export}.py; tests/test_labels.py.
Interface: add_labels(episode, horizon, tolerance); evaluate(pred, gt, tolerance);
export_tensors(episode, path). Hand-check 0.1m displacement and 0.2s horizon;
invalid intermediate frames must invalidate the action. Never align GT by fitting
the predicted trajectory. Unmatched frames remain in coverage denominators.

## Task 4: Learnable RGB detector

Files: src/robodata/learning/{model,dataset,train,infer}.py; tests/test_learning.py.
Interface: train(config_path, checkpoint_path); predict(checkpoint, RGB) -> named
pixels plus confidence. Mask missing landmarks, prevent train/val overlap, reload
weights and verify coordinate scaling and invalid invisible targets.

## Task 5: Runnable examples and handoff

Files: src/robodata/{synthetic,visualization,cli}.py; configs/; docs/; README.md;
.github/workflows/tests.yml. Generate synthetic RGB/RGB-D, extract, evaluate,
export and visualize. Train a small synthetic model and reload it. Run all
checks; commit and push to baebling/ROBODATA with measured verification evidence.

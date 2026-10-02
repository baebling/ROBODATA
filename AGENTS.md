# Repository Instructions

Read docs/RESEARCH_CONTEXT.md, docs/DATA_SCHEMA.md, docs/DECISION_LOG.md,
docs/OPEN_QUESTIONS.md and TASK.md before changes. Current task and the
researcher's direct decisions override agent-selected implementation defaults.
Summarize assumptions and intended edits before implementation.

- RGB is the final research target; RGB-D is a feasibility comparison.
- The canonical record is T_surface_tool(t), not robot joint commands.
- Functional point/+z axis and mask centroid are different quantities.
- Never infer metric scale, physical frame identity or full roll from insufficient input.
- Mocap/LiDAR/tracker measurements are independent evaluation references.
- Pose sequence is state; future target and relative motion are derived labels.
- Keep raw data read-only and retain failures, timestamps, units and provenance.
- Do not silently fill gaps, smooth, calibrate scores or generate process states.
- Keep robot policies, IK and physical execution outside the current research scope.
- Every accuracy claim must identify real vs synthetic data and the reference source.
- Version experiment configuration and record source commit/dirty state with results.

Commands (from repository root):

```sh
.venv/bin/python -m ruff format --check src tests
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy --config-file pyproject.toml src/robodata
.venv/bin/python -m pytest -q
.venv/bin/robodata demo --output outputs/new_demo
```

Learning tests skip only when PyTorch is unavailable. Run them explicitly with
the learning extra when modifying the detector. Do not commit data, credentials,
weights or generated run directories.

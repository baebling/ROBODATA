# Current Task

Implement a runnable first research model from the BAE_ROBOT context in
https://github.com/baebling/ROBODATA. Missing engineering criteria were delegated
to the coding agent by the researcher on 2026-10-02.

Deliver: calibrated RGB baseline, RGB-D comparison, optional trainable RGB
landmark network, validated episode/action schema, uncertainty/missingness,
masked learning export, independent GT import/evaluation, figures, tests and
reproducible CLI examples. Verify on synthetic inputs and document what remains
unverified on real construction video. Track source in Git.

The researcher's latest clarified scope is dataset generation and label accuracy;
physical robot task success and policy training are downstream.

Research refinement on 2026-10-03: compare DITTO, EgoScaler/VLA and ORION for
construction-specific positioning; selected Graco G-TEX 25D497 on the H-beam
web. Document geometry/size/occlusion experiments and claim boundaries in
docs/RESEARCH_POSITIONING.md and docs/HOPPER_GUN_EXPERIMENT.md. These protocols
do not authorize fabricated tool dimensions, modify prior raw renders or imply
that the new benchmark is implemented.

Approved follow-up on 2026-10-03: commit the research refinement, extend Blender
to study hopper size, image occupancy and fixed physical web-ROI occlusion,
then run paired RGB/RGB-D simulations. With no measured 25D497 exterior yet,
the new scene is an explicitly unmeasured illustrative proxy, not product CAD.
Preserve the old scene/renders; do not change learned perception or policy scope.
Record implementation, realized visibility, validity/temporal gaps and failures.

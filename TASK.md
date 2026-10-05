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

Latest approved follow-up: use an RGB hand-pose model to assist the grasped tool,
move the camera to an explicit helmet-front head-mounted frame, and test Galaxy
TriFold, ZED 2i 2.1mm and Helios2 Ray Chroma viewpoints. BLK360 G1 remains a static
surface-reference device. Use nominal FoV only as a labeled simulation assumption;
real camera intrinsics and measured hand/tool calibration remain required.
Preserve all previous renders and report actual model failures as well as coverage.

Latest approved surface follow-up: compare actual YOLO segmentation per frame,
YOLO plus BoT-SORT, and YOLO plus BoT-SORT plus persistent surface localization.
Fine-tune on separate existing synthetic hopper episodes; reserve the helmet
episodes from training. Use a tool-free RGB reference and known metric fixture
geometry as explicit calibration priors. This first implementation is planar
reference-image localization, not full background SLAM or markerless RGB.
Keep test GT outside inference, preserve failed earlier runs, and report both
coverage and conditional pose error. Do not claim tool/hand-pose improvements
from this surface-only test. See docs/YOLO_SURFACE_PILOT.md.

Latest approved local-algorithm follow-up: test the transferred
`fireproof_workarea` without changing its source, checkpoint or acceptance
gates. Run its RGB tracker independently on the preserved helmet sequences
with one explicitly ideal RGB-D seed, and its original five-face recognizer
on initial RGB-D captures. Separate these diagnostics from a successful
end-to-end chain or metric RGB-only reconstruction. Attempt included SVO
replay only with an available compatible SDK; record missing runtime as
unavailable counts, not zero measured performance. Results and limitations
are in docs/WORKAREA_ALGORITHM_PILOT.md.

Current researcher direction supersedes further simulator experiments: acquire
real helmet RGB data and pursue offline whole-video processing. Measured tool
and H-beam shape/dimensions are allowed priors; inference remains RGB-only.
The red MDF H-beam is nominally 300 x 300 x 10 x 15 with length 1200 (presumed
mm); actual geometry, axes, calibrated camera and nozzle relation remain to be
measured. Do not fabricate semantic observations from these nominal dimensions.

Approved VGGT follow-up: connect original pretrained RGB camera/depth outputs
to measured persistent surface coordinates and the existing canonical trajectory
path; explain metric scale, physical reference, dynamic tool/nozzle pose,
occlusion and temporal-consistency bottlenecks. Preserve all earlier data/runs.
The current implementation is a sparse offline bridge with explicit measured
anchor bootstrap, not completed whole-video optimization or automatic G-TEX
tracking. Implementation and actual RGB runtime probe: docs/VGGT_INTEGRATION.md.

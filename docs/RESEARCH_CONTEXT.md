# Research Context

The study converts human construction-work video into structured, robot-agnostic
tool-motion supervision. The main record is the time sequence of functional tool
position and direction relative to the working surface. Derived targets are
future pose and relative motion; they do not represent measured human control
commands or robot joint commands.

RGB is the final target. RGB-D provides an initial feasibility reference and
sensor-depth comparison. Helmet viewpoint is a candidate collection setting.
Representative work, dataset size and actual sensor availability remain open.
The executable example is an explicitly synthetic planar spray proxy.

Context consulted: the researcher's messages in this Codex conversation and
BAE_ROBOT's `선행연구 조사 계획` (conversation
6a82ec73-c058-83e9-92b2-7ddeb253f033), including the professor's scope clarification
and the latest research-structure discussion. Other project conversations and
uploaded project-plan attachments have not been audited as part of this version.

Mocap or a calibrated tracker can measure gold tool trajectories. LiDAR can
provide a surface reference; a static scan alone does not supply a moving tool's
time-aligned 6D trajectory. Calibration and synchronization must be recorded.
The current study evaluates generated data, with policy learning, IK, robot
execution and construction quality verification reserved for downstream work.

The expected contribution is a tested data-generation method with surface/tool
geometry, temporal targets and explicit reliability. Model selection itself is
not established as the academic novelty. Real data and independent evaluation
are required before making claims about accuracy or generalization.

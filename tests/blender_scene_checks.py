"""Run with Blender --background --python to test actual scene geometry."""

import json
import runpy
from pathlib import Path

import bpy
from mathutils import Vector

root = Path(__file__).resolve().parents[1]
renderer = runpy.run_path(str(root / "scripts/blender_simulate.py"))
cfg = json.loads((root / "configs/blender_simulation.json").read_text())
cfg.update(work_face="web_front", surface_frame_id="synthetic_hbeam_web_front")
scenario = {"name": "geometry_test", "path": "stationary", "camera_motion": "static"}
scene, tool, camera, S, *_ = renderer["setup_scene"](cfg, scenario)
assert (S.translation - Vector((0, -0.006, 0.6))).length < 1e-6, "Scene still uses top flange"
assert (S.to_3x3() @ Vector((0, 1, 0)) - Vector((0, 0, 1))).length < 1e-6
checks = [
    ([0, 0, 0.32], [0, 0, -1], "work_face_hit", "web"),
    ([0, 0.1, 0.32], [0, 0.5, -(0.75**0.5)], "flange_blocked", "top_flange"),
    ([0, -0.1, 0.32], [0, -0.5, -(0.75**0.5)], "flange_blocked", "bottom_flange"),
    ([1.1, 0, 0.32], [0, 0, -1], "miss", None),
    ([0, 0, 0.32], [0, 0, 1], "away", None),
    ([0, 0, 0.000005], [0, 0, -1], "work_face_hit", "web"),
]
for i, (point, axis, status, part) in enumerate(checks):
    case = {**scenario, "nozzle_position_surface_m": point, "nozzle_axis_surface": axis}
    renderer["set_pose"](cfg, case, tool, camera, S, i)
    bpy.context.view_layer.update()
    relation = renderer["nozzle_reference"](scene, cfg, tool, S)
    assert relation["status"] == status, relation
    assert relation["scene_ray_first_beam_part"] == part, relation
    assert relation["scene_ray_agrees"], relation
    print("ROBODATA_SCENE_CHECK", status, part, flush=True)
print("ROBODATA_SCENE_CHECKS_PASSED", len(checks), flush=True)

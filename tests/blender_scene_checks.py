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
hopper_cfg = json.loads((root / "configs/blender_hopper_simulation.json").read_text())
for scale in [0.75, 1.0, 1.25]:
    case = {**scenario, "name": f"hopper_geometry_{scale}", "hopper_scale": scale}
    scene, tool, camera, S, *_ = renderer["setup_scene"](hopper_cfg, case)
    renderer["set_pose"](hopper_cfg, case, tool, camera, S, 0)
    bpy.context.view_layer.update()
    hopper = next(o for o in scene.objects if o.name.startswith("IllustrativeHopper"))
    body = next(o for o in scene.objects if o.name.startswith("GunBody"))
    nozzle = next(o for o in scene.objects if o.name.startswith("Nozzle"))
    local_vertices = [
        hopper.location + hopper.rotation_euler.to_matrix() @ v.co for v in hopper.data.vertices
    ]
    assert abs(max(v.y for v in local_vertices) + 0.04) < 1e-6, "Attachment moved"
    assert abs(max(v.x for v in local_vertices) - 0.125 * scale) < 1e-6
    assert abs(body.dimensions.x - 0.12) < 1e-6, "Body was scaled with hopper"
    assert abs(nozzle.dimensions.z - 0.046) < 1e-6, "Nozzle was scaled with hopper"
    assert hopper.pass_index == 4 and nozzle.pass_index == 5 and body.pass_index == 2
    relation = renderer["nozzle_reference"](scene, hopper_cfg, tool, S)
    assert relation["status"] == "work_face_hit" and relation["scene_ray_agrees"]
    print("ROBODATA_HOPPER_SCENE_CHECK", scale, flush=True)
helmet_cfg = json.loads((root / "configs/blender_helmet_sensors.json").read_text())
helmet_cfg = {**hopper_cfg, **helmet_cfg}
for sensor in helmet_cfg["sensor_profiles"]:
    case = next(s for s in helmet_cfg["scenarios"] if s["sensor"] == sensor)
    effective = renderer["scenario_camera_config"](helmet_cfg, case)
    scene, tool, camera, S, *_ = renderer["setup_scene"](effective, case)
    renderer["set_pose"](effective, case, tool, camera, S, 0)
    bpy.context.view_layer.update()
    projection = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=effective["width"],
        y=effective["height"],
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    K = renderer["camera_intrinsics"](effective)
    assert abs(projection[0][0] * effective["width"] / 2 - K[0, 0]) < 0.001
    assert abs(projection[1][1] * effective["height"] / 2 - K[1, 1]) < 0.001
    position, direction = renderer["helmet_camera_pose"](effective, case, 0)
    assert (camera.location - S @ Vector(position)).length < 1e-6
    forward = camera.matrix_world.to_3x3() @ Vector((0, 0, -1))
    assert (forward - S.to_3x3() @ Vector(direction)).length < 1e-6
    assert any(o.pass_index == 6 for o in scene.objects if o.type == "MESH")
    calibration = json.loads(tool["hand_calibration_json"])
    assert len(calibration["points_hand_m"]) == 21 and calibration["grasp_verified"]
    print("ROBODATA_HELMET_SCENE_CHECK", sensor, flush=True)
print("ROBODATA_SCENE_CHECKS_PASSED", len(checks) + 7, flush=True)

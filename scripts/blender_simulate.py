"""Run inside Blender 5.2; no add-ons and no modification of existing scenes."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robodata.blender_geometry import (  # noqa: E402
    nozzle_relation,
    surface_frame_identity,
    work_surface_transform,
)


def material(name, color, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = 0.38
    return mat


def mesh_object(
    name, kind, center, size=None, radius=None, length=None, parent=None, mat=None, index=0
):
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.object
        obj.dimensions = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=length)
        obj = bpy.context.object
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=radius)
        obj = bpy.context.object
    obj.name = name
    obj.parent = parent
    obj.location = center
    obj.pass_index = index
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def file_output(group, directory, name, image_format, color_depth, socket):
    node = group.nodes.new("CompositorNodeOutputFile")
    node.directory = str(directory)
    node.file_name = name
    node.file_output_items.new("FLOAT", "Value")
    node.format.media_type = "IMAGE"
    node.format.file_format = image_format
    node.format.color_depth = color_depth
    node.format.color_mode = "BW" if image_format == "PNG" else "RGB"
    node.save_as_render = False
    group.links.new(socket, node.inputs[0])
    return node


def setup_scene(cfg, scenario):
    surface_transform = work_surface_transform(cfg)
    surface_frame_identity(cfg)
    face = cfg.get("work_face", "top_flange")
    scene = bpy.data.scenes.new("ROBODATA_" + scenario["name"])
    bpy.context.window.scene = scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.samples = cfg["samples"]
    scene.cycles.seed = cfg["seed"]
    scene.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == "CUDA"
        scene.cycles.device = "GPU" if any(d.use for d in prefs.devices) else "CPU"
    except (TypeError, RuntimeError):
        scene.cycles.device = "CPU"
    scene.render.resolution_x, scene.render.resolution_y = cfg["width"], cfg["height"]
    scene.render.resolution_percentage = 100
    scene.render.fps = cfg["fps"]
    scene.frame_start, scene.frame_end = 1, cfg["frames"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = cfg["exposure"]
    scene.world = bpy.data.worlds.new("ROBODATA_World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.22, 0.25, 0.28, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.5
    mats = {
        "metal": material("BeamSteel", (0.38, 0.43, 0.47), 0.65),
        "red": material("GunRed", (0.65, 0.035, 0.045), 0.2),
        "dark": material("Grip", (0.055, 0.065, 0.07)),
        "brass": material("NozzleBrass", (0.72, 0.48, 0.10), 0.7),
        "floor": material("Floor", (0.17, 0.19, 0.21)),
    }
    height = cfg["surface_height_m"]
    length, width, bh = cfg["beam_length_m"], cfg["beam_width_m"], cfg["beam_height_m"]
    ft = cfg["flange_thickness_m"]
    mesh_object(
        "HBeam_Top_WorkFace",
        "box",
        (0, 0, height - ft / 2),
        size=(length, width, ft),
        mat=mats["metal"],
        index=3 if face == "top_flange" else 1,
    )
    mesh_object(
        "HBeam_Web",
        "box",
        (0, 0, height - bh / 2),
        size=(length, cfg["web_thickness_m"], bh - 2 * ft),
        mat=mats["metal"],
        index=3 if face == "web_front" else 1,
    )
    mesh_object(
        "HBeam_Bottom",
        "box",
        (0, 0, height - bh + ft / 2),
        size=(length, width, ft),
        mat=mats["metal"],
        index=1,
    )
    for x in [-0.65, 0.65]:
        mesh_object(
            "BeamSupport",
            "box",
            (x, 0, (height - bh) / 2),
            size=(0.12, 0.30, height - bh),
            mat=mats["dark"],
        )
    mesh_object("Floor", "box", (0, 0, -0.025), size=(8, 8, 0.05), mat=mats["floor"])
    tool = bpy.data.objects.new("FunctionalNozzleFrame", None)
    scene.collection.objects.link(tool)
    tool.empty_display_type, tool.empty_display_size = "ARROWS", 0.10
    tool.rotation_mode = "QUATERNION"
    for part in cfg["proxy_parts"]:
        mesh_object(
            part["name"],
            part["kind"],
            part["center"],
            size=part.get("size"),
            radius=part.get("radius"),
            length=part.get("length"),
            parent=tool,
            mat=mats[part["material"]],
            index=2,
        )
    colors = [
        (0.9, 0.06, 0.06),
        (0.05, 0.85, 0.15),
        (0.05, 0.25, 0.95),
        (0.95, 0.7, 0.04),
        (0.75, 0.05, 0.9),
        (0.04, 0.85, 0.9),
        (0.9, 0.3, 0.04),
        (0.5, 0.4, 0.95),
    ]
    tool_markers, surface_markers = {}, {}
    for j, (name, point) in enumerate(cfg["tool_points_m"].items()):
        obj = mesh_object(
            "ToolMarker_" + name,
            "sphere",
            point,
            radius=cfg["landmark_radius_m"],
            parent=tool,
            mat=material("Marker_" + name, colors[j % len(colors)]),
            index=2,
        )
        tool_markers[name] = obj
    S = Matrix(surface_transform.tolist())
    for j, (name, point) in enumerate(cfg["surface_points_m"].items()):
        obj = mesh_object(
            "SurfaceMarker_" + name,
            "sphere",
            S @ Vector(point),
            radius=cfg["surface_landmark_radius_m"],
            mat=material("PatchMarker_" + name, colors[j]),
            index=3,
        )
        surface_markers[name] = obj
    data = bpy.data.cameras.new("RGB_Camera")
    camera = bpy.data.objects.new("RGB_Camera", data)
    scene.collection.objects.link(camera)
    data.sensor_width = 36.0
    data.sensor_fit = "HORIZONTAL"
    data.lens = cfg["camera_focal_px"] * data.sensor_width / cfg["width"]
    data.clip_start, data.clip_end = 0.01, 100.0
    scene.camera = camera
    for name, pos, power, size in [
        ("Key", (0.6, -1.0, 3.0), 800, 3),
        ("Fill", (-2, 1.0, 2.0), 500, 2),
    ]:
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy, light_data.shape, light_data.size = power, "DISK", size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = pos
        light.rotation_euler = (
            (Vector((0, 0, height)) - light.location).to_track_quat("-Z", "Y").to_euler()
        )
    layer = scene.view_layers[0]
    layer.use_pass_z = True
    layer.use_pass_object_index = True
    group = bpy.data.node_groups.new("ROBODATA_Outputs", "CompositorNodeTree")
    group.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = group
    render = group.nodes.new("CompositorNodeRLayers")
    render.scene = scene
    output = group.nodes.new("NodeGroupOutput")
    group.links.new(render.outputs["Image"], output.inputs["Image"])
    return scene, tool, camera, S, tool_markers, surface_markers, group, render


def set_pose(cfg, scenario, tool, camera, S, i):
    u = i / max(cfg["frames"] - 1, 1)
    a, b = cfg["path_half_length_m"], cfg["path_half_width_m"]
    path = scenario["path"]
    x, y = 0.0, 0.0
    if path == "horizontal":
        x = a * (2 * u - 1)
    elif path == "vertical":
        y = b * (2 * u - 1)
    elif path == "zigzag":
        x, y = a * (2 * u - 1), b * math.sin(4 * math.pi * u)
    pose = (
        Euler((math.pi, cfg["axis_tilt_rad"] * math.sin(2 * math.pi * u), 0), "XYZ")
        .to_matrix()
        .to_4x4()
    )
    if path == "stationary":
        pose = Euler((math.pi, 0, 0), "XYZ").to_matrix().to_4x4()
    pose.translation = Vector((x, y, cfg["standoff_m"]))
    if "nozzle_axis_surface" in scenario:
        axis = Vector(scenario["nozzle_axis_surface"])
        if abs(axis.length - 1) > 1e-6:
            raise ValueError("Scenario nozzle axis must be unit length")
        pose = Vector((0, 0, 1)).rotation_difference(axis).to_matrix().to_4x4()
        pose.translation = Vector((x, y, cfg["standoff_m"]))
    if "nozzle_position_surface_m" in scenario:
        pose.translation = Vector(scenario["nozzle_position_surface_m"])
    tool.matrix_world = S @ pose
    position = Vector(scenario.get("camera_position_surface_m", cfg["camera_position_surface_m"]))
    if scenario["camera_motion"] == "moving":
        amplitude = cfg["camera_motion_amplitude_m"]
        position += Vector((amplitude * math.sin(2 * math.pi * u), amplitude * (2 * u - 1), 0))
    camera.location = S @ position
    target = S @ Vector(cfg["camera_target_surface_m"])
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    for obj, fields in [
        (tool, ["location", "rotation_quaternion"]),
        (camera, ["location", "rotation_euler"]),
    ]:
        for field in fields:
            obj.keyframe_insert(data_path=field, frame=i + 1)


def keypoints(scene, camera, markers):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = {}
    for name, obj in markers.items():
        point = obj.matrix_world.translation
        ndc = world_to_camera_view(scene, camera, point)
        uv = [
            ndc.x * scene.render.resolution_x - 0.5,
            (1 - ndc.y) * scene.render.resolution_y - 0.5,
        ]
        delta = point - camera.location
        hit = scene.ray_cast(
            depsgraph, camera.location, delta.normalized(), distance=delta.length + 0.02
        )
        visible = bool(
            ndc.z > 0
            and 0 <= uv[0] < scene.render.resolution_x
            and 0 <= uv[1] < scene.render.resolution_y
            and hit[0]
            and hit[4].name == obj.name
        )
        result[name] = {"uv": uv, "visible": visible, "confidence": 1.0 if visible else 0.0}
    return result


def depth_checks(scene, camera, cfg):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    w, h, focal = cfg["width"], cfg["height"], cfg["camera_focal_px"]
    checks = []
    for u, v in [(w // 2, h // 2), (w // 3, h // 2), (2 * w // 3, h // 2), (w // 2, 2 * h // 3)]:
        local_ray = Vector(((u - (w - 1) / 2) / focal, -(v - (h - 1) / 2) / focal, -1)).normalized()
        direction = camera.matrix_world.to_3x3() @ local_ray
        hit = scene.ray_cast(depsgraph, camera.location, direction, distance=100)
        if hit[0]:
            camera_point = camera.matrix_world.inverted() @ hit[1]
            checks.append({"u": u, "v": v, "expected_m": -camera_point.z})
    return checks


def nozzle_reference(scene, cfg, tool, S):
    pose = S.inverted() @ tool.matrix_world
    point = pose.translation
    axis = pose.to_3x3() @ Vector((0, 0, 1))
    relation = nozzle_relation(cfg, list(point), list(axis))
    origin = tool.matrix_world.translation
    direction = tool.matrix_world.to_3x3() @ Vector((0, 0, 1))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hit_part, hit_point, distance = None, None, None
    cursor = origin.copy()
    # Ignore proxy/marker geometry; independently check only actual beam solids.
    for _ in range(64):
        hit = scene.ray_cast(depsgraph, cursor, direction, distance=10)
        if not hit[0]:
            break
        for prefix, part in [
            ("HBeam_Top", "top_flange"),
            ("HBeam_Web", "web"),
            ("HBeam_Bottom", "bottom_flange"),
        ]:
            if hit[4].name.startswith(prefix):
                hit_part, hit_point = part, S.inverted() @ hit[1]
                distance = (hit[1] - origin).length
                break
        if hit_part is not None:
            break
        cursor = hit[1] + direction * 1e-5
    agrees = hit_part == relation["first_beam_part"]
    if distance is not None:
        agrees = agrees and abs(distance - relation["ray_distance_m"]) < 1e-5
    relation.update(
        scene_ray_first_beam_part=hit_part,
        scene_ray_hit_surface_m=None if hit_point is None else list(hit_point),
        scene_ray_distance_m=distance,
        scene_ray_agrees=agrees,
    )
    if not agrees:
        raise ValueError("Analytic nozzle ray disagrees with actual Blender beam geometry")
    return relation


def render_scenario(cfg, scenario, root, config_hash, render_provenance):
    folder = root / scenario["name"]
    folder.mkdir()
    scene, tool, camera, S, tm, sm, group, render = setup_scene(cfg, scenario)
    depth_node = file_output(group, folder, "depth", "OPEN_EXR", "32", render.outputs["Depth"])
    divide = group.nodes.new("ShaderNodeMath")
    divide.operation = "DIVIDE"
    divide.inputs[1].default_value = 65535
    group.links.new(render.outputs["Object Index"], divide.inputs[0])
    labels_node = file_output(group, folder, "labels", "PNG", "16", divide.outputs[0])
    frames = []
    for i in range(cfg["frames"]):
        scene.frame_set(i + 1)
        set_pose(cfg, scenario, tool, camera, S, i)
        bpy.context.view_layer.update()
        frame_dir = folder / "frames" / f"{i:05d}"
        frame_dir.mkdir(parents=True)
        depth_node.directory = labels_node.directory = str(frame_dir)
        scene.render.filepath = str(frame_dir / "rgb.png")
        observations = {
            "tool_keypoints": keypoints(scene, camera, tm),
            "surface_keypoints": keypoints(scene, camera, sm),
            "depth_checks": depth_checks(scene, camera, cfg),
            "nozzle_reference": nozzle_reference(scene, cfg, tool, S),
        }
        bpy.ops.render.render(write_still=True)
        depth_files = list(frame_dir.glob("depth*.exr"))
        label_files = list(frame_dir.glob("labels*.png"))
        if len(depth_files) != 1 or len(label_files) != 1:
            raise RuntimeError("Compositor did not write one depth and label image")
        frames.append(
            {
                "frame_index": i,
                "timestamp_sec": i / cfg["fps"],
                "rgb_path": (frame_dir / "rgb.png").relative_to(folder).as_posix(),
                "depth_path": depth_files[0].relative_to(folder).as_posix(),
                "labels_path": label_files[0].relative_to(folder).as_posix(),
                "T_world_tool": [list(row) for row in tool.matrix_world],
                "T_world_camera": [list(row) for row in camera.matrix_world],
                "T_world_surface": [list(row) for row in S],
                **observations,
            }
        )
        print("ROBODATA_FRAME", scenario["name"], i + 1, "/", cfg["frames"], flush=True)
    scene.frame_set(1)
    preview_dir = folder / "interactive_renders"
    depth_node.directory = labels_node.directory = str(preview_dir)
    scene.render.filepath = str(preview_dir / "rgb.png")
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "hbeam_spray.blend"))
    f, w, h = cfg["camera_focal_px"], cfg["width"], cfg["height"]
    metadata = {
        "episode_id": scenario["name"],
        "blender_version": bpy.app.version_string,
        "width": w,
        "height": h,
        "fps": cfg["fps"],
        "expected_frames": cfg["frames"],
        "seed": cfg["seed"],
        "camera_matrix": [[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]],
        "camera_motion": scenario["camera_motion"],
        "work_face": cfg.get("work_face", "top_flange"),
        "surface_frame_id": surface_frame_identity(cfg),
        "depth_convention": "axial_z",
        "tool_points_m": cfg["tool_points_m"],
        "surface_points_m": cfg["surface_points_m"],
        "settings": {
            "render_provenance": render_provenance,
            "config_sha256": config_hash,
            "config": cfg,
            "render_device": scene.cycles.device,
            "proxy_note": "coloured marker centers on an illustrative calibration gun",
        },
        "frames": frames,
    }
    (folder / "render_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--frames", type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.frames is not None:
        cfg["frames"] = args.frames
    if cfg["frames"] < 4:
        raise ValueError("At least four frames are required")
    root = Path(args.output)
    if root.exists():
        raise FileExistsError("Render output already exists; select a new directory")
    scenarios = [s for s in cfg["scenarios"] if args.scenario is None or s["name"] == args.scenario]
    if not scenarios:
        raise ValueError("Unknown scenario")
    cfg["scenarios"] = scenarios
    root.mkdir(parents=True)
    config_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    (root / "effective_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    provenance = {
        "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "effective_config_sha256": hashlib.sha256(
            (root / "effective_config.json").read_bytes()
        ).hexdigest(),
        "git_commit": None,
        "git_dirty": None,
    }
    try:
        repo = Path(__file__).resolve().parent.parent
        # Trust only this script's repository for this read, without global config changes.
        git = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            [*git, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        provenance.update(git_commit=commit.stdout.strip(), git_dirty=bool(status.stdout.strip()))
    except (OSError, subprocess.SubprocessError) as exc:
        provenance["git_unavailable_reason"] = str(exc)
    for scenario in scenarios:
        render_scenario(cfg, scenario, root, config_hash, provenance)
    print("ROBODATA_RENDER_COMPLETE", str(root), flush=True)


if __name__ == "__main__":
    main()

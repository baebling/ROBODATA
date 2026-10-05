"""Render tool-free initialization RGBs without changing source .blend files."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    root, out = args.render_root.resolve(), args.output.resolve()
    if out.exists() or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Choose a new output outside the preserved raw render tree")
    sources = sorted(root.glob("*/hbeam_spray.blend"))
    if not sources:
        raise ValueError("No Blender scenes found")
    out.mkdir(parents=True)
    records = []
    for source in sources:
        bpy.ops.wm.open_mainfile(filepath=str(source))
        scene = bpy.context.scene
        scene.frame_set(1)
        for obj in bpy.data.objects:
            if obj.pass_index in {2, 4, 5, 6}:
                obj.hide_render = True
        # Remove compositor file outputs; only the new RGB reference is written.
        scene.use_nodes = False
        scene.render.use_compositing = False
        destination = out / source.parent.name
        destination.mkdir()
        scene.render.filepath = str(destination / "rgb.png")
        bpy.ops.render.render(write_still=True)
        records.append(
            {
                "scenario": source.parent.name,
                "rgb_path": f"{source.parent.name}/rgb.png",
                "source_blend_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "rgb_sha256": hashlib.sha256((destination / "rgb.png").read_bytes()).hexdigest(),
            }
        )
    (out / "reference_manifest.json").write_text(
        json.dumps(
            {
                "scope": "synthetic tool-free RGB calibration capture, not a per-frame GT pose",
                "reference_camera": "first recorded helmet pose; recovered from RGB fixtures",
                "blender_version": bpy.app.version_string,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

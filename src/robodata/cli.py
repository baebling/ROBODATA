"""Command-line research workflows."""

import argparse
import json
from pathlib import Path

from .evaluation import evaluate
from .export import export_tensors, read_episode, write_json
from .pipeline import run_episode
from .schema import Episode
from .synthetic import create_demo
from .visualization import overlay_frames, plot_episode


def main(argv=None):
    parser = argparse.ArgumentParser(prog="robodata")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Render and extract synthetic RGB/RGB-D episodes")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--frames", type=int, default=40)
    demo.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("run", help="Extract a real or synthetic video episode")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--manifest", type=Path)
    ev = sub.add_parser("evaluate", help="Compare with an independent calibrated reference")
    ev.add_argument("--prediction", type=Path, required=True)
    ev.add_argument("--reference", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--tolerance-sec", type=float, default=0.02)
    exp = sub.add_parser("export", help="Export masked observation/action NPZ tensors")
    exp.add_argument("--episode", type=Path, required=True)
    exp.add_argument("--output", type=Path, required=True)
    plot = sub.add_parser("plot", help="Save a trajectory figure")
    plot.add_argument("--episode", type=Path, required=True)
    plot.add_argument("--reference", type=Path)
    plot.add_argument("--output", type=Path, required=True)
    tr = sub.add_parser("train", help="Train the optional RGB landmark detector")
    tr.add_argument("--config", type=Path, required=True)
    tr.add_argument("--checkpoint", type=Path, required=True)
    infer = sub.add_parser("infer", help="Predict RGB landmarks with a trained detector")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--min-confidence", type=float, default=0.5)
    schema = sub.add_parser("schema")
    schema.add_argument("--output", type=Path, required=True)
    reference = sub.add_parser("import-gt", help="Import calibrated mocap/tracker CSV")
    reference.add_argument("--csv", type=Path, required=True)
    reference.add_argument("--calibration", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    blender = sub.add_parser(
        "blender-evaluate", help="Import and evaluate synthetic Blender renders"
    )
    blender.add_argument("--render-root", type=Path, required=True)
    blender.add_argument("--output", type=Path, required=True)
    vggt = sub.add_parser("vggt", help="Run offline VGGT and measured surface alignment")
    vggt.add_argument("--config", type=Path, required=True)
    vggt.add_argument("--output", type=Path, required=True)
    probe = sub.add_parser(
        "vggt-probe", help="Check pretrained VGGT on RGB images, without metric claims"
    )
    probe.add_argument("--images", type=Path, nargs="+", required=True)
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            root = args.output.resolve()
            create_demo(root / "inputs", args.frames, args.seed)
            reference = root / "inputs" / "ground_truth.json"
            reports = {}
            for mode in ["rgb", "rgbd"]:
                folder = root / mode
                episode = run_episode(root / "inputs" / f"{mode}.yaml", folder)
                reports[mode] = evaluate(episode, read_episode(reference))
                write_json(folder / "evaluation.json", reports[mode])
                plot_episode(folder / "episode.json", folder / "trajectory.png", reference)
                overlay_frames(episode, folder / "overlays")
            write_json(root / "comparison.json", reports)
            print(
                json.dumps(
                    {
                        m: {
                            k: r[k]
                            for k in [
                                "matched_valid_frames",
                                "valid_coverage",
                                "position_rmse_m",
                                "axis_mae_rad",
                            ]
                        }
                        for m, r in reports.items()
                    },
                    indent=2,
                )
            )
        elif args.command == "run":
            episode = run_episode(args.config, args.output, args.manifest)
            plot_episode(args.output / "episode.json", args.output / "trajectory.png")
            overlay_frames(episode, args.output / "overlays")
            print(
                json.dumps(
                    {
                        "frames": len(episode.frames),
                        "valid": sum(f.quality.valid for f in episode.frames),
                    }
                )
            )
        elif args.command == "evaluate":
            report = evaluate(
                read_episode(args.prediction), read_episode(args.reference), args.tolerance_sec
            )
            write_json(args.output, report)
            print(json.dumps(report, indent=2))
        elif args.command == "export":
            export_tensors(read_episode(args.episode), args.output)
        elif args.command == "plot":
            plot_episode(args.episode, args.output, args.reference)
        elif args.command == "train":
            from .learning.train import train

            result = train(args.config, args.checkpoint)
            print(
                json.dumps(
                    {
                        "best_val_loss": result["best_val_loss"],
                        "final_epoch": result["history"][-1],
                    },
                    indent=2,
                )
            )
        elif args.command == "infer":
            from .learning.infer import infer_manifest

            infer_manifest(args.checkpoint, args.manifest, args.output, args.min_confidence)
        elif args.command == "schema":
            write_json(args.output, Episode.model_json_schema())
        elif args.command == "import-gt":
            from .reference import import_reference

            import_reference(args.csv, args.calibration, args.output)
        elif args.command == "blender-evaluate":
            from .blender_experiment import run_blender_experiment

            print(run_blender_experiment(args.render_root, args.output))
        elif args.command == "vggt":
            from .vggt_inference import run_vggt

            result = run_vggt(args.config, args.output)
            print(
                json.dumps(
                    {
                        "selected_frames": len(result.frames),
                        "metric_surface_frames": sum(f.estimate is not None for f in result.frames),
                    }
                )
            )
        elif args.command == "vggt-probe":
            from .vggt_inference import probe_rgb_images

            print(
                json.dumps(probe_rgb_images(args.images, args.checkpoint, args.output, args.device))
            )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"robodata: {exc}\n")


if __name__ == "__main__":
    main()

"""Command-line interface.

    python -m src.cli image --input orchard.jpg
    python -m src.cli video --input input_orchard.mp4 --output output_annotated.mp4
    python -m src.cli evaluate-counts --csv results.csv --pred predicted --truth manual_count [--group lighting]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .errors import FruitCounterError
from .logging_utils import setup_logging

logger = logging.getLogger("fruit_counter.cli")


def _summary(res) -> str:
    c, h = res.counts, res.harvest
    lines = [f"analysis_id: {res.analysis_id}", f"mode: {res.mode}",
             f"raw_frame_detections: {c.raw_detections}"]
    if c.confirmed_tracks is not None:
        lines.append(f"confirmed_tracks: {c.confirmed_tracks}")
    lines += [f"estimated_unique_visible_fruit: {c.estimated_unique_visible_fruit}  ({c.estimate_basis})",
              f"by_species: {json.dumps(c.by_species)}",
              f"visible_harvest_weight_kg: {h.visible_harvest_weight_kg}",
              f"estimated_total_tree_fruit: {h.estimated_total_tree_fruit}"]
    lines += [f"WARNING: {w}" for w in res.report["warnings"]]
    if res.annotated_path:
        lines.append(f"annotated: {res.annotated_path}")
    lines += [f"{k}: {v}" for k, v in res.report_paths.items()]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fruit-counter", description="Fruit Counting & Harvest Estimator")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("image", "video"):
        s = sub.add_parser(name)
        s.add_argument("--input", help="input file (default for video: video.input_path in config)")
        s.add_argument("--model", help="override model.path")
        s.add_argument("--device", help="override model.device")
        s.add_argument("--conf", type=float, help="override model.conf")
        s.add_argument("--demo", action="store_true", help="demo mode with a generic COCO model (labelled DEMO)")
        if name == "video":
            s.add_argument("--output", help="annotated video path (default: video.output_path)")
            s.add_argument("--stride", type=int, help="override video.frame_stride")
            s.add_argument("--camera", choices=["fixed", "moving"])
    e = sub.add_parser("evaluate-counts")
    e.add_argument("--csv", required=True)
    e.add_argument("--pred", required=True)
    e.add_argument("--truth", required=True)
    e.add_argument("--group")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "evaluate-counts":
            from .evaluation import evaluate_csv
            print(evaluate_csv(args.csv, args.pred, args.truth, args.group).to_string(index=False))
            return 0
        cfg = load_config(args.config)
        if args.model:
            cfg.model.path = args.model
        if args.device:
            cfg.model.device = args.device
        if args.conf is not None:
            cfg.model.conf = args.conf
        if args.demo:
            cfg.model.demo_mode = True
            if not args.model:
                cfg.model.path = "yolov8n.pt"
        if args.command == "video":
            if args.stride:
                cfg.video.frame_stride = args.stride
            if args.camera:
                cfg.tracking.camera_mode = args.camera
        cfg.validate()
        setup_logging(cfg.logging.level)
        from .pipeline import build_pipeline
        pipe = build_pipeline(cfg)
        if args.command == "image":
            if not args.input:
                print("--input is required for image mode", file=sys.stderr)
                return 2
            res = pipe.analyze_image(args.input)
        else:
            res = pipe.analyze_video(args.input or cfg.video.input_path, args.output or cfg.video.output_path)
        print(_summary(res))
        return 0
    except FruitCounterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

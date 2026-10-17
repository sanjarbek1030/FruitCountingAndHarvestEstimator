"""Train / validate an Ultralytics fruit detector.

    python scripts/train_detector.py train --data data/fruit.yaml --weights yolov8s.pt --epochs 100 --imgsz 960
    python scripts/train_detector.py val   --data data/fruit.yaml --weights runs/fruit/exp/weights/best.pt --split test

Metrics printed by ``val`` come from Ultralytics on YOUR dataset. Nothing is hard-coded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def resolve_data_yaml(data: str) -> Path:
    """Write a copy of the dataset YAML with an absolute ``path`` (avoids datasets_dir resolution surprises)."""
    src = Path(data).resolve()
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    root = Path(cfg["path"]) if cfg.get("path") else src.parent
    cfg["path"] = str((root if root.is_absolute() else src.parent / root).resolve())
    for split in ("train", "val"):
        if not (Path(cfg["path"]) / cfg[split]).exists():
            raise SystemExit(f"Dataset folder missing: {Path(cfg['path']) / cfg[split]}")
    out = Path("runs") / "_resolved_data.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/fruit.yaml")
    t.add_argument("--weights", default="yolov8s.pt", help="pretrained start point (fine-tuning)")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=960)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--device", default=None)
    t.add_argument("--project", default="runs/fruit")
    t.add_argument("--name", default="exp")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--patience", type=int, default=30)
    v = sub.add_parser("val")
    v.add_argument("--data", default="data/fruit.yaml")
    v.add_argument("--weights", required=True)
    v.add_argument("--split", default="val", choices=["val", "test"])
    v.add_argument("--imgsz", type=int, default=960)
    v.add_argument("--device", default=None)
    a = ap.parse_args()

    from ultralytics import YOLO
    data = resolve_data_yaml(a.data)
    model = YOLO(a.weights)
    if a.cmd == "train":
        model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                    project=a.project, name=a.name, seed=a.seed, patience=a.patience)
        print(f"Best checkpoint: {Path(a.project) / a.name / 'weights' / 'best.pt'}  -> copy to models/best.pt")
    else:
        m = model.val(data=str(data), split=a.split, imgsz=a.imgsz, device=a.device)
        inf_ms = m.speed.get("inference")
        print(f"precision={m.box.mp:.4f} recall={m.box.mr:.4f} mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f}")
        if inf_ms:
            print(f"inference {inf_ms:.1f} ms/img (~{1000 / inf_ms:.1f} FPS) on this machine/device")


if __name__ == "__main__":
    main()

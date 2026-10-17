"""Group-aware train/val/test split (no leakage between frames/views of the same tree or session).

Input CSV columns: image (file name in --src-images), group (tree id / recording session / orchard block).

    python scripts/split_dataset.py --manifest manifest.csv --src-images raw/images --src-labels raw/labels
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import assign_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--src-images", required=True)
    ap.add_argument("--src-labels", required=True)
    ap.add_argument("--dst", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = pd.read_csv(a.manifest)
    split = assign_images(dict(zip(df["image"], df["group"].astype(str))), seed=a.seed)
    for img, s in split.items():
        for kind, src_dir, suffix in (("images", a.src_images, None), ("labels", a.src_labels, ".txt")):
            name = img if suffix is None else Path(img).with_suffix(suffix).name
            src = Path(src_dir) / name
            if not src.exists():
                if kind == "labels":
                    continue      # image without label = background image (allowed)
                raise SystemExit(f"Missing image: {src}")
            dst = Path(a.dst) / kind / s
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / name)
    print(pd.Series(split).value_counts().to_string())


if __name__ == "__main__":
    main()

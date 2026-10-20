"""Count / weight error metrics against manual ground truth (no fabricated results).

Detector metrics (precision, recall, mAP@50, mAP@50-95) come from ``scripts/train_detector.py val``
(Ultralytics). This module covers count and harvest-weight errors from a CSV you supply.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .errors import FruitCounterError


def count_errors(predicted: Sequence[float], truth: Sequence[float]) -> dict[str, float]:
    """MAE, RMSE, bias and mean absolute percentage error (only over truth > 0)."""
    p, t = np.asarray(predicted, float), np.asarray(truth, float)
    if p.shape != t.shape or p.size == 0:
        raise FruitCounterError("predicted and truth must be non-empty and equal length")
    err = p - t
    pos = t > 0
    return {"n": int(p.size), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "mape_pct": float(np.mean(np.abs(err[pos]) / t[pos]) * 100) if pos.any() else float("nan")}


def evaluate_csv(path: str | Path, pred_col: str, truth_col: str, group_col: str | None = None) -> pd.DataFrame:
    """Evaluate a CSV; optionally split metrics by a column (lighting, canopy density, ...)."""
    df = pd.read_csv(path)
    for c in (pred_col, truth_col) + ((group_col,) if group_col else ()):
        if c not in df.columns:
            raise FruitCounterError(f"Column '{c}' not in CSV (columns: {list(df.columns)})")
    rows = [{"group": "ALL", **count_errors(df[pred_col], df[truth_col])}]
    if group_col:
        for g, sub in df.groupby(group_col):
            rows.append({"group": str(g), **count_errors(sub[pred_col], sub[truth_col])})
    return pd.DataFrame(rows)

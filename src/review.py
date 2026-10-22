"""Helpers for manual review: flag suspicious detections/tracks so a human can reject them."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from .schemas import Detection, TrackSummary

MIN_SIDE_PX = 8.0
EXTREME_ASPECT = 2.5


def flag_detection(d: Detection, low_conf: float, min_side: float = MIN_SIDE_PX) -> list[str]:
    """Reasons a detection deserves a second look (heuristics, not errors)."""
    reasons = []
    if d.confidence < low_conf:
        reasons.append("low_confidence")
    if min(d.width, d.height) < min_side:
        reasons.append("very_small_box")
    ar = d.width / d.height if d.height > 0 else float("inf")
    if ar > EXTREME_ASPECT or ar < 1 / EXTREME_ASPECT:
        reasons.append("extreme_aspect_ratio")
    return reasons


def detections_review_df(dets: Sequence[Detection], low_conf: float) -> pd.DataFrame:
    rows = []
    for d in dets:
        rows.append({"reject": False, "detection_id": d.detection_id, "class_name": d.class_name,
                     "confidence": round(d.confidence, 3), "maturity_label": d.maturity_label,
                     "bbox": f"{d.bbox_x1:.0f},{d.bbox_y1:.0f},{d.bbox_x2:.0f},{d.bbox_y2:.0f}",
                     "suspicious": ", ".join(flag_detection(d, low_conf))})
    return pd.DataFrame(rows, columns=["reject", "detection_id", "class_name", "confidence",
                                       "maturity_label", "bbox", "suspicious"])


def tracks_review_df(tracks: Sequence[TrackSummary], low_conf: float, min_confirm: int) -> pd.DataFrame:
    rows = []
    for t in tracks:
        if not t.confirmed:
            continue
        flags = []
        if t.mean_confidence < low_conf:
            flags.append("low_mean_confidence")
        if t.n_observations <= min_confirm:
            flags.append("barely_confirmed")
        rows.append({"reject": False, "track_id": t.track_id, "class_name": t.class_name,
                     "n_observations": t.n_observations, "first_frame": t.first_frame,
                     "last_frame": t.last_frame, "mean_confidence": round(t.mean_confidence, 3),
                     "maturity_label": t.maturity_label, "suspicious": ", ".join(flags)})
    return pd.DataFrame(rows, columns=["reject", "track_id", "class_name", "n_observations", "first_frame",
                                       "last_frame", "mean_confidence", "maturity_label", "suspicious"])

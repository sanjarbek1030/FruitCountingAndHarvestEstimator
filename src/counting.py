"""Counting policies for images and videos.

Design rules (see README "Counting definitions"):

* Video counts are **never** the sum of per-frame detections.
* A track id is an observation grouping, not proof of a globally unique fruit.
* Nearby fruit are never merged by centre distance. Image-mode duplicate suppression only removes
  same-class boxes whose IoU is >= ``duplicate_iou_threshold`` (default 0.85: near-identical boxes).
"""
from __future__ import annotations

import dataclasses
import logging
import math
from collections import Counter
from typing import Iterable, Optional, Sequence

import numpy as np

from .config import CountingConfig
from .geometry import iou_xyxy
from .schemas import (CountResult, Detection, NOT_ASSESSED, RejectedDetection, ReviewCorrections,
                      TrackSummary, UNAVAILABLE, UNCERTAIN)

logger = logging.getLogger(__name__)

MIN_BOX_AREA_PX = 4.0   # clipped boxes smaller than this are treated as degenerate


# --------------------------------------------------------------------------- validation
def validate_detections(detections: Iterable[Detection], width: int, height: int
                        ) -> tuple[list[Detection], list[RejectedDetection]]:
    """Reject malformed detections and clip boxes that extend past the image border.

    Fruit partially outside the frame is legitimate, so boxes are clipped to ``[0,w] x [0,h]``
    rather than rejected. Rejected: non-finite values, inverted/zero-size boxes, boxes that do not
    overlap the image (or are degenerate after clipping), confidence outside [0, 1].
    """
    valid: list[Detection] = []
    rejected: list[RejectedDetection] = []
    for d in detections:
        coords = (d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2)
        reason: Optional[str] = None
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in coords):
            reason = "non_finite_bbox"
        elif not (isinstance(d.confidence, (int, float)) and math.isfinite(d.confidence)
                  and 0.0 <= d.confidence <= 1.0):
            reason = "confidence_out_of_range"
        elif d.bbox_x2 <= d.bbox_x1 or d.bbox_y2 <= d.bbox_y1:
            reason = "inverted_or_empty_bbox"
        else:
            x1, y1 = max(0.0, d.bbox_x1), max(0.0, d.bbox_y1)
            x2, y2 = min(float(width), d.bbox_x2), min(float(height), d.bbox_y2)
            if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < MIN_BOX_AREA_PX:
                reason = "outside_image_or_degenerate"
            else:
                valid.append(dataclasses.replace(d, bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2))
                continue
        logger.warning("Rejected detection %s: %s", d.detection_id, reason)
        rejected.append(RejectedDetection(d.detection_id, reason))
    return valid, rejected


def filter_by_confidence(detections: Iterable[Detection], min_confidence: float
                         ) -> tuple[list[Detection], list[RejectedDetection]]:
    """Keep detections with ``confidence >= min_confidence``."""
    kept: list[Detection] = []
    rejected: list[RejectedDetection] = []
    for d in detections:
        if d.confidence >= min_confidence:
            kept.append(d)
        else:
            rejected.append(RejectedDetection(d.detection_id, "below_confidence_threshold"))
    return kept, rejected


def suppress_duplicates(detections: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    """Class-aware suppression of near-identical boxes (IoU >= threshold). Keeps highest confidence."""
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda x: -x.confidence):
        box = (d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2)
        if any(k.class_name == d.class_name
               and iou_xyxy(box, (k.bbox_x1, k.bbox_y1, k.bbox_x2, k.bbox_y2)) >= iou_threshold
               for k in kept):
            continue
        kept.append(d)
    return kept


# --------------------------------------------------------------------------- zones
def zone_for_point(cx: float, cy: float, width: int, height: int,
                   zones: Sequence[dict]) -> Optional[str]:
    """Return the first zone (normalised box) containing the point, else ``None``."""
    if width <= 0 or height <= 0:
        return None
    nx, ny = cx / width, cy / height
    for z in zones:
        x1, y1, x2, y2 = z["box"]
        if x1 <= nx <= x2 and y1 <= ny <= y2:
            return str(z["name"])
    return None


def assign_zones(detections: Sequence[Detection], width: int, height: int,
                 zones: Sequence[dict]) -> list[Detection]:
    """Set ``observation_zone`` on each detection. No-op when no zones are configured."""
    if not zones:
        return list(detections)
    return [dataclasses.replace(d, observation_zone=zone_for_point(d.center_x, d.center_y, width, height, zones))
            for d in detections]


# --------------------------------------------------------------------------- aggregation helpers
def _confidence_stats(values: Sequence[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "min": float(arr.min()), "max": float(arr.max())}


def _maturity_key(label: Optional[str]) -> str:
    return UNAVAILABLE if label is None else label


def _condition_key(label: Optional[str]) -> str:
    return UNAVAILABLE if label is None else label


def _apply_manual_additions(by_species: dict[str, int], corrections: Optional[ReviewCorrections]) -> int:
    added = 0
    if corrections:
        for species, n in corrections.added_by_species.items():
            if n and n > 0:
                by_species[species] = by_species.get(species, 0) + int(n)
                added += int(n)
    return added


# --------------------------------------------------------------------------- image counting
def count_image(detections: Sequence[Detection], width: int, height: int,
                counting_cfg: CountingConfig, raw_detections: Optional[int] = None,
                corrections: Optional[ReviewCorrections] = None) -> tuple[CountResult, list[Detection]]:
    """Count visible fruit in a single image.

    Pipeline: geometry validation -> confidence filter -> near-identical duplicate suppression ->
    zone assignment -> optional zone restriction -> manual review rejections.

    Returns the :class:`CountResult` and the list of detections that were counted.
    """
    raw_n = len(detections) if raw_detections is None else raw_detections
    valid, rej_geo = validate_detections(detections, width, height)
    valid, rej_conf = filter_by_confidence(valid, counting_cfg.min_confidence)
    n_before_dedup = len(valid)
    valid = suppress_duplicates(valid, counting_cfg.duplicate_iou_threshold)
    n_dedup = n_before_dedup - len(valid)
    valid = assign_zones(valid, width, height, counting_cfg.zones)
    rejected_n = len(rej_geo) + len(rej_conf) + n_dedup

    warnings: list[str] = []
    if counting_cfg.count_only_in_zones:
        outside = [d for d in valid if d.observation_zone is None]
        valid = [d for d in valid if d.observation_zone is not None]
        rejected_n += len(outside)

    manual_rej = 0
    if corrections and corrections.rejected_detection_ids:
        before = len(valid)
        valid = [d for d in valid if d.detection_id not in corrections.rejected_detection_ids]
        manual_rej = before - len(valid)

    valid.sort(key=lambda d: (d.bbox_y1, d.bbox_x1))
    by_species = dict(Counter(d.class_name for d in valid))
    by_maturity = dict(Counter(_maturity_key(d.maturity_label) for d in valid))
    by_condition = dict(Counter(_condition_key(d.condition_label) for d in valid))
    by_zone = dict(Counter(d.observation_zone or "outside_zones" for d in valid)) if counting_cfg.zones else {}
    uncertain = sum(1 for d in valid if d.maturity_label == UNCERTAIN)
    model_count = len(valid)
    added = _apply_manual_additions(by_species, corrections)

    if by_maturity.get(UNAVAILABLE) == model_count and model_count:
        warnings.append("Maturity classification unavailable: maturity breakdown is not reported.")

    result = CountResult(
        mode="image", raw_detections=raw_n, rejected_detections=rejected_n, confirmed_tracks=None,
        estimated_unique_visible_fruit=model_count + added,
        by_species=by_species, by_maturity=by_maturity, by_condition=by_condition, by_zone=by_zone,
        uncertain_classifications=uncertain, confidence=_confidence_stats([d.confidence for d in valid]),
        estimate_basis="single_image_visible_fruit (one viewpoint; occluded fruit are not counted)",
        manual_additions=added, manual_rejections=manual_rej, warnings=warnings)
    return result, valid


# --------------------------------------------------------------------------- video counting
def count_video(summaries: Sequence[TrackSummary], raw_frame_detections: int, width: int, height: int,
                camera_mode: str, counting_cfg: CountingConfig,
                track_diagnostics: Optional[dict] = None,
                corrections: Optional[ReviewCorrections] = None
                ) -> tuple[CountResult, list[TrackSummary]]:
    """Count fruit in a video from track summaries (never from the sum of frame detections).

    * ``confirmed_tracks``: tracks whose ``confirmed`` flag is set (>= min_confirm_observations).
    * ``estimated_unique_visible_fruit``: equals ``confirmed_tracks`` after zone filtering and manual
      review. For ``camera_mode == "moving"`` it is explicitly a best-effort figure: ID switches and
      re-entries can inflate it, track merges can deflate it.
    """
    warnings: list[str] = []
    tracks = [dataclasses.replace(
        t, observation_zone=zone_for_point(t.mean_center_x, t.mean_center_y, width, height, counting_cfg.zones)
        if counting_cfg.zones else t.observation_zone) for t in summaries]
    confirmed = [t for t in tracks if t.confirmed]
    confirmed_n = len(confirmed)

    if counting_cfg.count_only_in_zones:
        confirmed = [t for t in confirmed if t.observation_zone is not None]

    manual_rej = 0
    if corrections and corrections.rejected_track_ids:
        before = len(confirmed)
        confirmed = [t for t in confirmed if t.track_id not in corrections.rejected_track_ids]
        manual_rej = before - len(confirmed)

    by_species = dict(Counter(t.class_name for t in confirmed))
    by_maturity = dict(Counter(_maturity_key(t.maturity_label) for t in confirmed))
    by_condition = dict(Counter(_condition_key(t.condition_label) for t in confirmed))
    by_zone = dict(Counter(t.observation_zone or "outside_zones" for t in confirmed)) if counting_cfg.zones else {}
    uncertain = sum(1 for t in confirmed if t.maturity_label == UNCERTAIN)
    model_count = len(confirmed)
    added = _apply_manual_additions(by_species, corrections)

    if camera_mode == "moving":
        basis = ("best_effort_confirmed_tracks (moving camera: ID switches and re-entries can inflate, "
                 "track merges can deflate; NOT a globally unique count)")
        warnings.append("Moving camera: the same fruit may receive several track IDs. The estimate is "
                        "best-effort and not a globally unique count without multi-view association.")
    else:
        basis = "confirmed_tracks (fixed camera; one track = one fruit assumed; ID switches possible)"
    if confirmed_n == 0 and summaries:
        warnings.append("No track reached the confirmation threshold; consider lowering "
                        "tracking.min_confirm_observations or checking detector confidence.")
    if track_diagnostics and track_diagnostics.get("untracked_detections"):
        warnings.append(f"{track_diagnostics['untracked_detections']} detections had no track id and were "
                        "not counted as confirmed observations.")

    result = CountResult(
        mode="video", raw_detections=raw_frame_detections, rejected_detections=0,
        confirmed_tracks=confirmed_n, estimated_unique_visible_fruit=model_count + added,
        by_species=by_species, by_maturity=by_maturity, by_condition=by_condition, by_zone=by_zone,
        uncertain_classifications=uncertain,
        confidence=_confidence_stats([t.mean_confidence for t in confirmed]),
        estimate_basis=basis, manual_additions=added, manual_rejections=manual_rej,
        track_diagnostics=track_diagnostics, warnings=warnings)
    return result, tracks

"""Report generation (JSON + CSV). Raw detections and unique-fruit estimates are always separate fields."""
from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .config import AppConfig
from .schemas import (CountResult, DETECTION_COLUMNS, Detection, HarvestEstimate, TRACK_COLUMNS,
                      TrackSummary)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

DEFINITIONS = {
    "raw_frame_detections": "Every box the detector emitted in every processed frame (images: boxes before "
                            "filtering). NOT a fruit count for videos.",
    "confirmed_tracks": "Video only: tracks seen in >= min_confirm_observations processed frames.",
    "estimated_unique_visible_fruit": "Policy-based estimate of distinct fruit visible in the footage "
                                      "(image: counted detections; video: confirmed tracks after review). "
                                      "Not ground truth.",
    "estimated_total_tree_fruit": "estimated_unique_visible_fruit / visibility_fraction. Present only when a "
                                  "visibility fraction is configured; the fraction is an assumption.",
    "visible_harvestable_count": "unique visible fruit * harvestable_probability (unrounded).",
    "visible_harvest_weight_kg": "visible_harvestable_count * average_weight_grams / 1000.",
}

LIMITATIONS = [
    "Counts cover only fruit visible from the recorded viewpoint(s); occluded fruit are not counted.",
    "A track id is not proof of a globally unique fruit: ID switches and merges occur.",
    "Maturity labels describe visible appearance, not physiological maturity or operational harvest readiness.",
    "Harvest values depend entirely on the configured calibration assumptions (probability, mean weight, "
    "visibility fraction).",
    "No accuracy figure is implied; validate against manual counts and measured harvest weights.",
]

SUMMARY_COLUMNS = [
    "schema_version", "analysis_id", "input_filename", "analysis_timestamp_utc", "mode", "demo_mode",
    "model_checkpoint", "model_sha256_prefix", "raw_frame_detections", "rejected_detections",
    "confirmed_tracks", "estimated_unique_visible_fruit", "manual_additions", "manual_rejections",
    "counts_by_species_json", "counts_by_maturity_json", "counts_by_condition_json", "counts_by_zone_json",
    "uncertain_classifications", "estimate_basis", "camera_mode",
    "visible_harvestable_count", "visible_harvest_weight_kg", "visibility_fraction",
    "estimated_total_tree_fruit", "tree_harvest_weight_kg", "average_weight_grams_json",
    "harvestable_probability_json", "estimation_method", "uncertainty_lower_kg", "uncertainty_upper_kg",
    "uncertainty_coverage", "uncertainty_n_samples", "processing_duration_seconds", "warnings",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_report(analysis_id: str, input_name: str, mode: str, cfg: AppConfig, counts: CountResult,
                 harvest: HarvestEstimate, model_info: dict[str, Any], demo_mode: bool,
                 processing_seconds: float, video: Optional[dict[str, Any]] = None,
                 extra_warnings: Sequence[str] = (), timestamp: Optional[str] = None) -> dict[str, Any]:
    """Assemble the JSON-serialisable report."""
    warnings = list(dict.fromkeys([*counts.warnings, *harvest.warnings, *extra_warnings]))
    if demo_mode:
        warnings.insert(0, "DEMO MODE: generic model, not validated for orchard fruit. Not for decisions.")
    unc = harvest.uncertainty
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "input_filename": Path(input_name).name,
        "analysis_timestamp_utc": timestamp or utc_now_iso(),
        "mode": mode,
        "demo_mode": demo_mode,
        "model": model_info,
        "inference_config": {
            "imgsz": cfg.model.imgsz, "detector_conf": cfg.model.conf, "nms_iou": cfg.model.iou,
            "counting_min_confidence": cfg.counting.min_confidence,
            "duplicate_iou_threshold": cfg.counting.duplicate_iou_threshold,
            "tiling_enabled": cfg.model.tiling_enabled, "tracking_backend": cfg.tracking.backend,
            "tracker_type": cfg.tracking.tracker_type, "camera_mode": cfg.tracking.camera_mode,
            "min_confirm_observations": cfg.tracking.min_confirm_observations,
            "frame_stride": cfg.video.frame_stride if mode == "video" else None,
            "maturity_enabled": cfg.maturity.enabled,
            "maturity_confidence_threshold": cfg.maturity.confidence_threshold,
        },
        "video": video,
        "counts": {
            "raw_frame_detections": counts.raw_detections,
            "rejected_detections": counts.rejected_detections,
            "confirmed_tracks": counts.confirmed_tracks,
            "estimated_unique_visible_fruit": counts.estimated_unique_visible_fruit,
            "manual_additions": counts.manual_additions,
            "manual_rejections": counts.manual_rejections,
            "by_species": counts.by_species, "by_maturity": counts.by_maturity,
            "by_condition": counts.by_condition, "by_zone": counts.by_zone,
            "uncertain_classifications": counts.uncertain_classifications,
            "confidence": counts.confidence, "estimate_basis": counts.estimate_basis,
            "track_diagnostics": counts.track_diagnostics,
        },
        "harvest": {
            "visible_harvestable_count": harvest.visible_harvestable_count,
            "visible_harvestable_count_rounded_for_display": harvest.visible_harvestable_count_rounded,
            "visible_harvest_weight_kg": harvest.visible_harvest_weight_kg,
            "visibility_fraction": harvest.visibility_fraction,
            "estimated_total_tree_fruit": harvest.estimated_total_tree_fruit,
            "tree_harvestable_count": harvest.tree_harvestable_count,
            "tree_harvest_weight_kg": harvest.tree_harvest_weight_kg,
        },
        "assumptions": {
            "average_weight_grams": harvest.average_weight_grams_used,
            "harvestable_probability": harvest.harvestable_probability_used,
            "visibility_fraction": harvest.visibility_fraction,
            "note": "Manually configured assumptions; not measured by this system.",
        },
        "estimation_method": harvest.method,
        "uncertainty": None if unc is None else {
            "lower": unc.lower, "upper": unc.upper, "unit": unc.unit, "coverage": unc.coverage,
            "method": unc.method, "n_samples": unc.n_samples},
        "uncertainty_note": harvest.uncertainty_note,
        "processing_duration_seconds": round(processing_seconds, 3),
        "warnings": warnings,
        "limitations": LIMITATIONS,
        "definitions": DEFINITIONS,
    }


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    """One CSV row with stable column names (``SUMMARY_COLUMNS``)."""
    c, h, a = report["counts"], report["harvest"], report["assumptions"]
    unc = report.get("uncertainty") or {}
    m = report.get("model") or {}
    row = {
        "schema_version": report["schema_version"], "analysis_id": report["analysis_id"],
        "input_filename": report["input_filename"], "analysis_timestamp_utc": report["analysis_timestamp_utc"],
        "mode": report["mode"], "demo_mode": report["demo_mode"],
        "model_checkpoint": m.get("checkpoint_name"), "model_sha256_prefix": m.get("checkpoint_sha256_prefix"),
        "raw_frame_detections": c["raw_frame_detections"], "rejected_detections": c["rejected_detections"],
        "confirmed_tracks": c["confirmed_tracks"],
        "estimated_unique_visible_fruit": c["estimated_unique_visible_fruit"],
        "manual_additions": c["manual_additions"], "manual_rejections": c["manual_rejections"],
        "counts_by_species_json": json.dumps(c["by_species"], sort_keys=True),
        "counts_by_maturity_json": json.dumps(c["by_maturity"], sort_keys=True),
        "counts_by_condition_json": json.dumps(c["by_condition"], sort_keys=True),
        "counts_by_zone_json": json.dumps(c["by_zone"], sort_keys=True),
        "uncertain_classifications": c["uncertain_classifications"], "estimate_basis": c["estimate_basis"],
        "camera_mode": report["inference_config"]["camera_mode"],
        "visible_harvestable_count": h["visible_harvestable_count"],
        "visible_harvest_weight_kg": h["visible_harvest_weight_kg"],
        "visibility_fraction": h["visibility_fraction"],
        "estimated_total_tree_fruit": h["estimated_total_tree_fruit"],
        "tree_harvest_weight_kg": h["tree_harvest_weight_kg"],
        "average_weight_grams_json": json.dumps(a["average_weight_grams"], sort_keys=True),
        "harvestable_probability_json": json.dumps(a["harvestable_probability"], sort_keys=True),
        "estimation_method": report["estimation_method"],
        "uncertainty_lower_kg": unc.get("lower"), "uncertainty_upper_kg": unc.get("upper"),
        "uncertainty_coverage": unc.get("coverage"), "uncertainty_n_samples": unc.get("n_samples"),
        "processing_duration_seconds": report["processing_duration_seconds"],
        "warnings": " | ".join(report["warnings"]),
    }
    assert list(row) == SUMMARY_COLUMNS, "flatten_report out of sync with SUMMARY_COLUMNS"
    return row


def detections_to_df(detections: Sequence[Detection]) -> pd.DataFrame:
    return pd.DataFrame([d.to_dict() for d in detections], columns=DETECTION_COLUMNS)


def tracks_to_df(tracks: Sequence[TrackSummary]) -> pd.DataFrame:
    return pd.DataFrame([t.to_dict() for t in tracks], columns=TRACK_COLUMNS)


def report_to_json_bytes(report: dict[str, Any]) -> bytes:
    return json.dumps(report, indent=2, default=str).encode("utf-8")


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def summary_csv_bytes(report: dict[str, Any]) -> bytes:
    return df_to_csv_bytes(pd.DataFrame([flatten_report(report)], columns=SUMMARY_COLUMNS))


def save_reports(report: dict[str, Any], detections: Sequence[Detection], tracks: Sequence[TrackSummary],
                 out_dir: str | Path) -> dict[str, Path]:
    """Write JSON + summary CSV (+ tracks / detections CSV when present). Returns the written paths."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    aid = report["analysis_id"]
    paths = {"json": d / f"{aid}_report.json", "summary_csv": d / f"{aid}_summary.csv"}
    paths["json"].write_bytes(report_to_json_bytes(report))
    paths["summary_csv"].write_bytes(summary_csv_bytes(report))
    if tracks:
        paths["tracks_csv"] = d / f"{aid}_tracks.csv"
        paths["tracks_csv"].write_bytes(df_to_csv_bytes(tracks_to_df(tracks)))
    if detections:
        paths["detections_csv"] = d / f"{aid}_detections.csv"
        paths["detections_csv"].write_bytes(df_to_csv_bytes(detections_to_df(detections)))
    logger.info("Reports written: %s", ", ".join(p.name for p in paths.values()))
    return paths

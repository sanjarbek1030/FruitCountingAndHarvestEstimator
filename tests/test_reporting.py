import json

import pandas as pd

from src.config import AppConfig, FruitConfig, HarvestConfig
from src.counting import count_image
from src.reporting import (DETECTION_COLUMNS, SUMMARY_COLUMNS, TRACK_COLUMNS, build_report, flatten_report,
                           report_to_json_bytes, save_reports, summary_csv_bytes)
from src.yield_estimation import estimate_harvest
from tests.helpers import det

REQUIRED_TOP = {"schema_version", "analysis_id", "input_filename", "analysis_timestamp_utc", "model",
                "inference_config", "counts", "harvest", "assumptions", "estimation_method", "uncertainty",
                "uncertainty_note", "processing_duration_seconds", "warnings", "limitations", "definitions"}


def make_report(demo=False):
    cfg = AppConfig()
    cfg.fruit = FruitConfig(species=["apple"], average_weight_grams={"apple": 150})
    cfg.harvest = HarvestConfig(harvestable_probability=0.9)
    counts, kept = count_image([det(did="a"), det(60, 60, 90, 90, did="b")], 100, 100, cfg.counting)
    h = estimate_harvest(counts.by_species, cfg.harvest, cfg.fruit)
    rep = build_report("id1", "/secret/path/img.jpg", "image", cfg, counts, h, {"checkpoint_name": "best.pt"},
                       demo, 0.5, extra_warnings=["w1"])
    return rep, kept


def test_report_has_required_fields():
    rep, _ = make_report()
    assert REQUIRED_TOP <= set(rep)
    assert rep["input_filename"] == "img.jpg"          # no local path leaked
    for k in ("raw_frame_detections", "estimated_unique_visible_fruit", "by_species", "by_maturity", "by_condition"):
        assert k in rep["counts"]
    assert rep["assumptions"]["average_weight_grams"] == {"apple": 150.0}


def test_raw_and_unique_are_separate_fields_and_json_roundtrips():
    rep, _ = make_report()
    parsed = json.loads(report_to_json_bytes(rep))
    assert "raw_frame_detections" in parsed["counts"] and "estimated_unique_visible_fruit" in parsed["counts"]
    assert parsed["harvest"]["estimated_total_tree_fruit"] is None


def test_demo_mode_is_flagged():
    rep, _ = make_report(demo=True)
    assert rep["demo_mode"] and rep["warnings"][0].startswith("DEMO MODE")


def test_summary_csv_columns_stable():
    rep, _ = make_report()
    df = pd.read_csv(pd.io.common.BytesIO(summary_csv_bytes(rep)))
    assert list(df.columns) == SUMMARY_COLUMNS and len(df) == 1
    assert list(flatten_report(rep)) == SUMMARY_COLUMNS


def test_save_reports_writes_files(tmp_path):
    rep, kept = make_report()
    paths = save_reports(rep, kept, [], tmp_path)
    assert paths["json"].exists() and paths["summary_csv"].exists()
    assert list(pd.read_csv(paths["detections_csv"]).columns) == DETECTION_COLUMNS
    assert "tracks_csv" not in paths


def test_uncalibrated_uncertainty_not_presented_as_interval():
    rep, _ = make_report()
    assert rep["uncertainty"] is None and "No uncertainty interval" in rep["uncertainty_note"]

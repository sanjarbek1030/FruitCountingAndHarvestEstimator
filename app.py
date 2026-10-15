"""Streamlit dashboard. Presentation only: all logic lives in ``src/``.

Run:  streamlit run app.py
Optional: set FRUIT_ADMIN_PASSWORD to require a password for editing calibration settings.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import AppConfig, load_config
from src.errors import ConfigError, FruitCounterError
from src.logging_utils import setup_logging
from src.pipeline import AnalysisResult, build_pipeline
from src.reporting import (df_to_csv_bytes, detections_to_df, report_to_json_bytes, summary_csv_bytes,
                           tracks_to_df)
from src.review import detections_review_df, tracks_review_df
from src.schemas import ReviewCorrections

CONFIG_PATH = os.environ.get("FRUIT_CONFIG", "config.yaml")
SPECIES_CHOICES = ["apple", "orange", "strawberry", "mango", "pear", "peach"]
MAX_INLINE_VIDEO_MB = 100

st.set_page_config(page_title="Fruit Counting & Harvest Estimator", layout="wide")


# --------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading detector (once)...")
def load_manager(model_path: str, device: str, demo_mode: bool, species: tuple, demo_classes: tuple,
                 imgsz: int):
    """Load the detector once per (checkpoint, device, mode, species). Thresholds don't need a reload."""
    from src.config import ModelConfig
    from src.model_manager import ModelManager
    mc = ModelConfig(path=model_path, device=device, demo_mode=demo_mode, demo_class_names=list(demo_classes),
                     imgsz=imgsz)
    return ModelManager(mc, list(species)).load()


def get_base_config() -> AppConfig:
    try:
        cfg = load_config(CONFIG_PATH)
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    setup_logging(cfg.logging.level)
    return cfg


# --------------------------------------------------------------------------- calibration tab
def can_edit(base: AppConfig) -> bool:
    pw = os.environ.get(base.dashboard.admin_password_env, "")
    if not pw:
        return True      # local single-user use; set the env var to restrict editing
    return st.text_input("Admin password (required to edit calibration)", type="password") == pw


def calibration_tab(base: AppConfig) -> AppConfig | None:
    st.subheader("Calibration settings")
    st.caption("Values below are **manually configured assumptions**, not measurements made by this system. "
               "They are written into every report.")
    editable = can_edit(base)
    cfg = base.copy()
    c1, c2 = st.columns(2)
    with c1:
        species = st.multiselect("Fruit species counted", sorted(set(SPECIES_CHOICES) | set(base.fruit.species)),
                                 default=base.fruit.species, disabled=not editable,
                                 help="The loaded detector must have been trained for these classes.")
        weights = {}
        for s in species:
            w = st.number_input(f"Average fruit weight - {s} (g)", min_value=0.0, step=5.0,
                                value=float(base.fruit.average_weight_grams.get(s, 0.0)), disabled=not editable,
                                help="0 = unknown (weight not estimated). Use measured mean weights.")
            if w > 0:
                weights[s] = w
        use_p = st.checkbox("Apply harvestable probability", value=base.harvest.harvestable_probability is not None,
                            disabled=not editable)
        p = st.slider("Harvestable probability", 0.0, 1.0, float(base.harvest.harvestable_probability or 0.8),
                      0.01, disabled=not editable or not use_p, help="Must come from field calibration.")
        use_v = st.checkbox("Apply visibility correction (extrapolate to whole tree)",
                            value=base.harvest.visibility_fraction is not None, disabled=not editable)
        v = st.slider("Visibility fraction", 0.05, 1.0, float(base.harvest.visibility_fraction or 0.5), 0.01,
                      disabled=not editable or not use_v, help="Fraction of tree fruit visible. An assumption.")
    with c2:
        conf = st.slider("Detector confidence", 0.01, 0.99, float(base.model.conf), 0.01, disabled=not editable)
        iou = st.slider("NMS IoU", 0.1, 0.95, float(base.model.iou), 0.05, disabled=not editable)
        min_conf = st.slider("Counting min confidence", 0.01, 0.99, float(base.counting.min_confidence), 0.01,
                             disabled=not editable)
        confirm = st.number_input("Min observations to confirm a track (video)", 1, 100,
                                  base.tracking.min_confirm_observations, disabled=not editable)
        camera = st.selectbox("Camera mode (video)", ["fixed", "moving"],
                              index=["fixed", "moving"].index(base.tracking.camera_mode), disabled=not editable,
                              help="'moving' gives a best-effort count only.")
        stride = st.number_input("Frame stride (video)", 1, 30, base.video.frame_stride, disabled=not editable,
                                 help="Analyse every Nth frame. Higher = faster, but may miss fruit and break tracks.")
        demo = st.checkbox("Demo mode (generic COCO model; NOT validated)", value=base.model.demo_mode,
                           disabled=not editable)
    if not species:
        st.error("Select at least one species.")
        return None
    cfg.fruit.species, cfg.fruit.average_weight_grams = species, weights
    cfg.harvest.harvestable_probability = p if use_p else None
    cfg.harvest.visibility_fraction = v if use_v else None
    cfg.model.conf, cfg.model.iou, cfg.counting.min_confidence = conf, iou, min_conf
    cfg.tracking.min_confirm_observations, cfg.tracking.camera_mode = int(confirm), camera
    cfg.video.frame_stride, cfg.model.demo_mode = int(stride), demo
    if demo and cfg.model.path == "models/best.pt":
        cfg.model.path = "yolov8n.pt"
        st.warning("Demo mode uses a generic checkpoint (downloaded by Ultralytics if absent). "
                   "Results are labelled DEMO and must not be used for decisions.")
    try:
        return cfg.validate()
    except ConfigError as exc:
        st.error(f"Invalid settings: {exc}")
        return None


# --------------------------------------------------------------------------- analysis
def run_analysis(cfg: AppConfig, upload) -> None:
    ext = Path(upload.name).suffix.lower()
    is_image, is_video = ext in cfg.dashboard.image_extensions, ext in cfg.dashboard.video_extensions
    if not (is_image or is_video):
        st.error(f"Unsupported file type '{ext}'. Allowed: {cfg.dashboard.image_extensions + cfg.dashboard.video_extensions}")
        return
    if upload.size > cfg.dashboard.max_upload_mb * 1024 * 1024:
        st.error(f"File is {upload.size / 1e6:.0f} MB; the limit is {cfg.dashboard.max_upload_mb} MB.")
        return
    tmpdir = Path(tempfile.mkdtemp(prefix="fruit_"))
    try:
        safe_path = tmpdir / f"upload_{uuid.uuid4().hex}{ext}"          # never use the user's filename on disk
        safe_path.write_bytes(upload.getbuffer())
        manager = load_manager(cfg.model.path, cfg.model.device, cfg.model.demo_mode, tuple(cfg.fruit.species),
                               tuple(cfg.model.demo_class_names), cfg.model.imgsz)
        pipe = build_pipeline(cfg, manager)
        if is_image:
            with st.spinner("Detecting fruit..."):
                res = pipe.analyze_image(safe_path, input_name=upload.name)
        else:
            bar = st.progress(0.0, text="Processing video...")

            def cb(done: int, total: int | None) -> None:
                bar.progress(min(1.0, done / total) if total else 0.0, text=f"Frame {done}" + (f"/{total}" if total else ""))
            res = pipe.analyze_video(safe_path, input_name=upload.name, progress_callback=cb,
                                     max_frames=cfg.dashboard.max_video_frames)
            bar.empty()
        st.session_state["result"], st.session_state["pipe"] = res, pipe
        st.success("Analysis complete. See the Overview and Detection review tabs.")
    except FruitCounterError as exc:
        st.error(str(exc))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)          # uploaded media is deleted immediately


def input_tab(cfg: AppConfig | None, base: AppConfig) -> None:
    st.subheader("Input")
    exts = [e.lstrip(".") for e in base.dashboard.image_extensions + base.dashboard.video_extensions]
    up = st.file_uploader(f"Upload an image or video (max {base.dashboard.max_upload_mb} MB)", type=exts)
    st.caption("Uploads are written to a temporary folder under a random name and deleted right after "
               "processing. Annotated outputs and reports are kept in `outputs/` until you delete them.")
    if st.button("Run analysis", type="primary", disabled=up is None or cfg is None):
        run_analysis(cfg, up)


# --------------------------------------------------------------------------- result tabs
def overview_tab(res: AnalysisResult) -> None:
    c, h = res.counts, res.harvest
    if res.report["demo_mode"]:
        st.error("DEMO MODE: generic model, not validated for orchard fruit. Not for decisions.")
    cols = st.columns(4)
    cols[0].metric("Estimated unique visible fruit", c.estimated_unique_visible_fruit)
    cols[1].metric("Raw detections (all frames)" if res.mode == "video" else "Raw detections", c.raw_detections)
    cols[2].metric("Harvestable (visible, est.)",
                   "n/a" if h.visible_harvestable_count is None else f"{h.visible_harvestable_count:.1f}")
    cols[3].metric("Harvest weight (visible, est.)",
                   "n/a" if h.visible_harvest_weight_kg is None else f"{h.visible_harvest_weight_kg:.2f} kg")
    cols = st.columns(4)
    cols[0].metric("Confirmed tracks", "-" if c.confirmed_tracks is None else c.confirmed_tracks)
    cols[1].metric("Estimated total tree fruit (extrapolated)",
                   "n/a" if h.estimated_total_tree_fruit is None else f"{h.estimated_total_tree_fruit:.0f}")
    cols[2].metric("Tree harvest weight (extrapolated)",
                   "n/a" if h.tree_harvest_weight_kg is None else f"{h.tree_harvest_weight_kg:.2f} kg")
    cols[3].metric("Processing time", f"{res.report['processing_duration_seconds']:.1f} s")
    if h.uncertainty:
        u = h.uncertainty
        st.info(f"Empirical {u.coverage:.0%} interval: {u.lower:.2f}-{u.upper:.2f} {u.unit} ({u.n_samples} validation pairs).")
    else:
        st.caption(h.uncertainty_note)
    st.caption(f"Basis: {c.estimate_basis}")
    st.markdown("**Maturity breakdown**")
    st.write(c.by_maturity or "none")
    if c.manual_additions or c.manual_rejections:
        st.info(f"Manual review applied: +{c.manual_additions} added, -{c.manual_rejections} rejected.")
    for w in res.report["warnings"]:
        st.warning(w)
    with st.expander("Definitions and limitations"):
        for k, v in res.report["definitions"].items():
            st.markdown(f"- **{k}**: {v}")
        for lim in res.report["limitations"]:
            st.markdown(f"- {lim}")


def review_tab(res: AnalysisResult, cfg: AppConfig) -> None:
    st.subheader("Detection review")
    if res.mode == "image" and "annotated_image" in res.context:
        st.image(cv2.cvtColor(res.context["annotated_image"], cv2.COLOR_BGR2RGB), width="stretch")
        table = detections_review_df(res.counted_detections, cfg.counting.low_confidence_flag)
        key_col = "detection_id"
    else:
        p = res.annotated_path
        if p and p.exists() and p.stat().st_size < MAX_INLINE_VIDEO_MB * 1024 * 1024:
            st.video(p.read_bytes())
            st.caption("If the video does not play, the codec is not browser-compatible: enable "
                       "`video.transcode_h264` (needs ffmpeg) or download it from the Export tab. "
                       "The video shows model output before manual review.")
        else:
            st.info("Annotated video is too large to preview; download it from the Export tab.")
        table = tracks_review_df(res.tracks, cfg.counting.low_confidence_flag, cfg.tracking.min_confirm_observations)
        key_col = "track_id"
    st.markdown("Tick **reject** for false positives. Rows flagged *suspicious* are heuristics, not verdicts.")
    edited = st.data_editor(table.head(cfg.dashboard.max_table_rows), hide_index=True, key=f"review_{res.analysis_id}",
                            disabled=[c for c in table.columns if c != "reject"], width="stretch")
    adds = {}
    for s in cfg.fruit.species:
        adds[s] = st.number_input(f"Fruit missed by the detector - {s} (manual count)", 0, 100000, 0,
                                  key=f"add_{s}_{res.analysis_id}")
    if st.button("Apply review"):
        rejected = set(edited.loc[edited["reject"], key_col].tolist())
        corr = ReviewCorrections(
            rejected_detection_ids={str(x) for x in rejected} if key_col == "detection_id" else set(),
            rejected_track_ids={int(x) for x in rejected} if key_col == "track_id" else set(),
            added_by_species={k: v for k, v in adds.items() if v})
        st.session_state["result"] = st.session_state["pipe"].recompute(res, corr, save=False)
        st.rerun()


def analytics_tab(res: AnalysisResult, cfg: AppConfig) -> None:
    st.subheader("Analytics")
    c = res.counts

    def bar(data: dict, title: str, xlabel: str) -> None:
        if not data:
            st.caption(f"{title}: no data")
            return
        df = pd.DataFrame({xlabel: list(data), "count": list(data.values())})
        st.plotly_chart(px.bar(df, x=xlabel, y="count", title=title), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        bar(c.by_species, "Estimated unique visible fruit by species", "species")
        bar(c.by_condition, "Condition distribution", "condition")
    with c2:
        bar(c.by_maturity, "Maturity distribution", "maturity")
        if cfg.counting.zones:
            bar(c.by_zone, "Count per observation zone", "zone")
        else:
            st.caption("Per-tree / per-zone counts are shown only when observation zones are configured "
                       "(`counting.zones`); this system does not identify individual trees.")
    h = res.harvest
    rows = [("visible harvest weight (kg)", h.visible_harvest_weight_kg),
            ("extrapolated tree weight (kg)", h.tree_harvest_weight_kg)]
    wdf = pd.DataFrame([r for r in rows if r[1] is not None], columns=["quantity", "kg"])
    if len(wdf):
        st.plotly_chart(px.bar(wdf, x="quantity", y="kg", title="Estimated harvest weight"), width="stretch")
    else:
        st.caption("Harvest weight not estimated: configure average weight and harvestable probability.")
    if res.mode == "video" and c.track_diagnostics:
        st.markdown("**Tracking diagnostics** (fragmentation indicators)")
        st.json(c.track_diagnostics)


def export_tab(res: AnalysisResult) -> None:
    st.subheader("Export")
    st.download_button("Report (JSON)", report_to_json_bytes(res.report), f"{res.analysis_id}_report.json")
    st.download_button("Summary (CSV)", summary_csv_bytes(res.report), f"{res.analysis_id}_summary.csv")
    if res.tracks:
        st.download_button("Per-track (CSV)", df_to_csv_bytes(tracks_to_df(res.tracks)), f"{res.analysis_id}_tracks.csv")
    if res.counted_detections:
        st.download_button("Per-detection (CSV)", df_to_csv_bytes(detections_to_df(res.counted_detections)),
                           f"{res.analysis_id}_detections.csv")
    if res.annotated_path and res.annotated_path.exists():
        st.download_button("Annotated output", res.annotated_path.read_bytes(), res.annotated_path.name)
    st.divider()
    if st.button("Delete saved outputs for this analysis"):
        for p in [res.annotated_path, *res.report_paths.values()]:
            if p:
                Path(p).unlink(missing_ok=True)
        st.success("Saved files deleted from outputs/.")


# --------------------------------------------------------------------------- main
def main() -> None:
    st.title("Fruit Counting & Harvest Estimator")
    st.caption("Estimates, not ground truth. Counts cover visible fruit only; harvest figures depend on the "
               "calibration assumptions you configure.")
    base = get_base_config()
    tabs = st.tabs(["Overview", "Input", "Detection review", "Analytics", "Calibration", "Export"])
    with tabs[4]:
        cfg = calibration_tab(base)      # executed first so Input can use the settings
    with tabs[1]:
        input_tab(cfg, base)
    res: AnalysisResult | None = st.session_state.get("result")
    for i, fn in ((0, lambda: overview_tab(res)), (2, lambda: review_tab(res, cfg or base)),
                  (3, lambda: analytics_tab(res, cfg or base)), (5, lambda: export_tab(res))):
        with tabs[i]:
            if res is None:
                st.info("Run an analysis in the Input tab first.")
            else:
                fn()


main()

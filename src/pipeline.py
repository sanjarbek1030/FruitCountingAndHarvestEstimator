"""High-level orchestration: detector -> validation -> counting -> harvest estimate -> report.

Business logic lives here (not in Streamlit). The detector is duck-typed so tests can inject fakes.
"""
from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .config import AppConfig
from .counting import count_image, count_video, filter_by_confidence, suppress_duplicates, validate_detections
from .errors import InputError
from .maturity import MaturityClassifier, NullMaturityClassifier, annotate_maturity
from .reporting import build_report, save_reports
from .schemas import CountResult, Detection, HarvestEstimate, ReviewCorrections, TrackSummary
from .video_processor import VideoProcessor, VideoRunResult
from .visualization import draw_detections, draw_summary_panel, save_image
from .yield_estimation import estimate_harvest

logger = logging.getLogger(__name__)


def new_analysis_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def read_image(path: str | Path) -> np.ndarray:
    """Decode an image (unicode-safe). Raises :class:`InputError` when missing or undecodable."""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"Image not found: '{p.name}'")
    try:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise InputError(f"Could not read '{p.name}': {exc}") from exc
    if img is None:
        raise InputError(f"'{p.name}' is not a decodable image.")
    return img


@dataclass
class AnalysisResult:
    analysis_id: str
    input_name: str
    mode: str
    width: int
    height: int
    counts: CountResult
    harvest: HarvestEstimate
    counted_detections: list[Detection]          # image: counted boxes; video: stored detections (may be empty)
    tracks: list[TrackSummary]
    report: dict[str, Any]
    annotated_path: Optional[Path] = None
    report_paths: dict[str, Path] = field(default_factory=dict)
    image_bgr: Optional[np.ndarray] = None
    # Inputs needed to recompute after manual review
    source_detections: list[Detection] = field(default_factory=list)
    source_tracks: list[TrackSummary] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


class AnalysisPipeline:
    """Runs analyses for images and videos using an injected detector."""

    def __init__(self, cfg: AppConfig, detector: Any, maturity: Optional[MaturityClassifier] = None,
                 model_info: Optional[dict[str, Any]] = None, demo_mode: Optional[bool] = None,
                 model_warnings: Optional[list[str]] = None) -> None:
        self.cfg = cfg
        self.detector = detector
        self.maturity = maturity or NullMaturityClassifier()
        mgr = getattr(detector, "manager", None)
        self.model_info = model_info if model_info is not None else dict(getattr(mgr, "info", {}) or
                                                                         {"checkpoint_name": "unknown"})
        self.demo_mode = demo_mode if demo_mode is not None else bool(getattr(mgr, "demo_mode", False))
        self.model_warnings = model_warnings if model_warnings is not None else list(getattr(mgr, "warnings", []))

    # ------------------------------------------------------------------ image
    def analyze_image(self, image_path: str | Path, input_name: Optional[str] = None, save: bool = True,
                      corrections: Optional[ReviewCorrections] = None) -> AnalysisResult:
        frame = read_image(image_path)
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        raw = self.detector.detect(frame, 0, 0.0)
        # classify maturity only for boxes that will actually be counted
        pre, _ = validate_detections(raw, w, h)
        pre, _ = filter_by_confidence(pre, self.cfg.counting.min_confidence)
        pre = suppress_duplicates(pre, self.cfg.counting.duplicate_iou_threshold)
        classified = {d.detection_id: d for d in
                      annotate_maturity(frame, pre, self.maturity, self.cfg.maturity)}
        source = [classified.get(d.detection_id, d) for d in raw]
        elapsed = time.perf_counter() - t0
        res = AnalysisResult(
            analysis_id=new_analysis_id(), input_name=input_name or Path(image_path).name, mode="image",
            width=w, height=h, counts=None, harvest=None, counted_detections=[], tracks=[],  # type: ignore[arg-type]
            report={}, image_bgr=frame, source_detections=source,
            context={"processing_seconds": elapsed, "video": None, "extra_warnings": []})
        return self._finalize(res, corrections, save)

    # ------------------------------------------------------------------ video
    def analyze_video(self, video_path: str | Path, output_path: Optional[str | Path] = None,
                      input_name: Optional[str] = None, save: bool = True,
                      progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
                      should_stop: Optional[Callable[[], bool]] = None, max_frames: Optional[int] = None,
                      corrections: Optional[ReviewCorrections] = None) -> AnalysisResult:
        aid = new_analysis_id()
        if output_path is None:
            output_path = Path(self.cfg.output.annotated_dir) / f"{aid}_annotated.mp4"
        proc = VideoProcessor(self.cfg, self.detector, self.maturity, self.demo_mode)
        run: VideoRunResult = proc.process(video_path, output_path, progress_callback, should_stop, max_frames)
        m = run.metadata
        video_info = {
            "width": m.width, "height": m.height, "fps": m.fps, "fps_source": m.fps_source,
            "frame_count_reported": m.frame_count, "duration_seconds_reported": m.duration_seconds,
            "frames_read": run.frames_read, "frames_processed": run.frames_processed,
            "frames_skipped": run.frames_skipped, "inference_errors": run.inference_errors,
            "throughput_fps": round(run.throughput_fps, 2), "terminated_early": run.terminated_early,
            "timestamp_basis": "frame_index / fps (constant frame rate assumed)"}
        res = AnalysisResult(
            analysis_id=aid, input_name=input_name or Path(video_path).name, mode="video", width=m.width,
            height=m.height, counts=None, harvest=None, counted_detections=run.detections,  # type: ignore[arg-type]
            tracks=[], report={}, annotated_path=run.output_path, source_tracks=run.summaries,
            context={"processing_seconds": run.processing_seconds, "video": video_info,
                     "extra_warnings": run.warnings, "raw_frame_detections": run.raw_frame_detections,
                     "track_diagnostics": run.track_diagnostics})
        return self._finalize(res, corrections, save)

    # ------------------------------------------------------------------ recompute after review
    def recompute(self, result: AnalysisResult, corrections: Optional[ReviewCorrections],
                  save: bool = True) -> AnalysisResult:
        """Re-run counting/harvest/report with manual corrections. Model output is never modified."""
        return self._finalize(dataclasses.replace(result), corrections, save)

    # ------------------------------------------------------------------ shared
    def _finalize(self, res: AnalysisResult, corrections: Optional[ReviewCorrections], save: bool
                  ) -> AnalysisResult:
        cfg = self.cfg
        if res.mode == "image":
            counts, counted = count_image(res.source_detections, res.width, res.height, cfg.counting,
                                          corrections=corrections)
            res.counted_detections, res.tracks = counted, []
            if res.image_bgr is not None:
                shown = draw_detections(res.image_bgr, counted, cfg.annotation)
                lines = [f"Visible fruit (counted): {counts.estimated_unique_visible_fruit}"] + \
                        [f"  {k}: {v}" for k, v in sorted(counts.by_species.items())]
                shown = draw_summary_panel(shown, lines, cfg.annotation,
                                           banner="DEMO MODE - generic model, not validated"
                                           if self.demo_mode else None)
                if save:
                    res.annotated_path = save_image(
                        Path(cfg.output.annotated_dir) / f"{res.analysis_id}_annotated.jpg", shown)
                res.context["annotated_image"] = shown
        else:
            counts, tracks = count_video(res.source_tracks, res.context["raw_frame_detections"], res.width,
                                         res.height, cfg.tracking.camera_mode, cfg.counting,
                                         res.context.get("track_diagnostics"), corrections)
            res.tracks = tracks
        harvest = estimate_harvest(counts.by_species, cfg.harvest, cfg.fruit)
        res.counts, res.harvest = counts, harvest
        res.report = build_report(
            res.analysis_id, res.input_name, res.mode, cfg, counts, harvest, self.model_info, self.demo_mode,
            res.context["processing_seconds"], res.context.get("video"),
            [*self.model_warnings, *res.context.get("extra_warnings", [])])
        if save:
            res.report_paths = save_reports(res.report, res.counted_detections, res.tracks, cfg.output.report_dir)
        return res


def build_pipeline(cfg: AppConfig, manager: Any = None) -> AnalysisPipeline:
    """Create the real pipeline: loads the model once (or reuses a supplied ModelManager)."""
    from .detection import FruitDetector
    from .maturity import build_classifier
    from .model_manager import ModelManager
    if manager is None:
        manager = ModelManager(cfg.model, cfg.fruit.species).load()
    detector = FruitDetector(manager, cfg.model, cfg.tracking)
    return AnalysisPipeline(cfg, detector, build_classifier(cfg.maturity, manager.device))

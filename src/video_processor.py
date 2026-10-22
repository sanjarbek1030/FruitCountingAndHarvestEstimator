"""Sequential, memory-bounded video processing with tracking and annotated output."""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .config import AppConfig
from .counting import filter_by_confidence, validate_detections
from .errors import ModelError, VideoError
from .maturity import MaturityClassifier, NullMaturityClassifier, annotate_maturity
from .schemas import Detection, TrackSummary
from .tracking import IoUTracker, TrackRegistry
from .visualization import draw_detections, draw_summary_panel

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    fps_source: str                 # "metadata" | "fallback"
    frame_count: Optional[int]      # None when the container does not report it
    duration_seconds: Optional[float]


@dataclass
class VideoRunResult:
    metadata: VideoMetadata
    frames_read: int
    frames_processed: int
    frames_skipped: int
    inference_errors: int
    processing_seconds: float
    throughput_fps: float
    summaries: list[TrackSummary]
    track_diagnostics: dict[str, Any]
    raw_frame_detections: int
    rejected_detections: int
    detections: list[Detection]
    output_path: Optional[Path]
    terminated_early: bool
    warnings: list[str] = field(default_factory=list)


def read_metadata(cap: cv2.VideoCapture, first_frame: np.ndarray, fallback_fps: float
                  ) -> tuple[VideoMetadata, list[str]]:
    """Derive metadata; the decoded first frame is authoritative for dimensions."""
    warnings: list[str] = []
    h, w = first_frame.shape[:2]
    prop_w, prop_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if (prop_w, prop_h) != (w, h):
        warnings.append(f"Container reported {prop_w}x{prop_h} but frames are {w}x{h}; using decoded size.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0 or fps > 1000:
        warnings.append(f"FPS metadata missing/invalid ({fps!r}); using fallback {fallback_fps} fps. "
                        "Timestamps are therefore approximate.")
        fps, src = fallback_fps, "fallback"
    else:
        src = "metadata"
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    count = n if n > 0 else None
    return VideoMetadata(w, h, fps, src, count, (count / fps) if count else None), warnings


def transcode_h264(path: Path) -> bool:
    """Re-encode to H.264 with ffmpeg (browser-playable). Returns False if ffmpeg is unavailable/fails."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; leaving %s as written.", path.name)
        return False
    tmp = path.with_suffix(".h264.mp4")
    try:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
                       check=True, timeout=3600)
        tmp.replace(path)
        return True
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffmpeg transcode failed: %s", exc)
        tmp.unlink(missing_ok=True)
        return False


class VideoProcessor:
    """Runs detector -> (tracker) -> registry -> annotation for one video file.

    ``detector`` must provide ``detect(frame, frame_index, ts)``, ``track(frame, frame_index, ts)`` and
    ``reset_tracker()`` (see :class:`src.detection.FruitDetector`). Timestamps are ``frame_index / fps``
    (documented assumption: constant frame rate).
    """

    def __init__(self, cfg: AppConfig, detector: Any, maturity: Optional[MaturityClassifier] = None,
                 demo_mode: bool = False) -> None:
        self.cfg = cfg
        self.detector = detector
        self.maturity = maturity or NullMaturityClassifier()
        self.demo_mode = demo_mode

    def process(self, input_path: str | Path, output_path: Optional[str | Path] = None,
                progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
                should_stop: Optional[Callable[[], bool]] = None,
                max_frames: Optional[int] = None) -> VideoRunResult:
        vcfg, tcfg, ccfg = self.cfg.video, self.cfg.tracking, self.cfg.counting
        src = Path(input_path)
        if not src.is_file():
            raise VideoError(f"Video file not found: '{src.name}'")
        cap = cv2.VideoCapture(str(src))
        writer: Optional[cv2.VideoWriter] = None
        out_path = Path(output_path) if output_path else None
        started = time.perf_counter()
        try:
            if not cap.isOpened():
                raise VideoError(f"OpenCV cannot open '{src.name}' (unsupported codec or corrupt file).")
            ok, first = cap.read()
            if not ok or first is None:
                raise VideoError(f"Could not decode the first frame of '{src.name}'.")
            meta, warnings = read_metadata(cap, first, vcfg.fallback_fps)
            w, h = meta.width, meta.height

            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*vcfg.codec), meta.fps, (w, h))
                if not writer.isOpened():
                    raise VideoError(f"Cannot open video writer (codec '{vcfg.codec}'). Try another video.codec.")

            if hasattr(self.detector, "reset_tracker"):
                self.detector.reset_tracker()
            iou_tracker = IoUTracker(tcfg.iou_match_threshold, tcfg.max_lost_frames) \
                if tcfg.backend == "iou" else None
            registry = TrackRegistry(tcfg.min_confirm_observations)
            stored: list[Detection] = []
            limit = max_frames or vcfg.max_frames
            frames_read = frames_processed = frames_skipped = errors = consecutive = rejected = 0
            current: list[Detection] = []
            terminated_early = False
            pending: Optional[np.ndarray] = first

            while True:
                if pending is not None:
                    frame, pending = pending, None
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                if (limit and frames_read >= limit) or (should_stop and should_stop()):
                    terminated_early = True
                    warnings.append("Processing stopped before the end of the video (limit or user cancel); "
                                    "results are PARTIAL.")
                    break
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                idx = frames_read
                frames_read += 1
                ts = idx / meta.fps

                if idx % vcfg.frame_stride == 0:
                    try:
                        raw = (self.detector.detect(frame, idx, ts) if iou_tracker
                               else self.detector.track(frame, idx, ts))
                        consecutive = 0
                    except ModelError as exc:
                        errors += 1
                        consecutive += 1
                        logger.error("Frame %d inference error: %s", idx, exc)
                        if consecutive >= vcfg.max_consecutive_errors:
                            raise VideoError(f"Aborting after {consecutive} consecutive inference errors.") from exc
                        raw = None
                    if raw is not None:
                        valid, rej_geo = validate_detections(raw, w, h)
                        valid, rej_conf = filter_by_confidence(valid, ccfg.min_confidence)
                        rejected += len(rej_geo) + len(rej_conf)
                        if iou_tracker:
                            valid = iou_tracker.update(valid)
                        skip = [registry.maturity_observation_count(d.track_id) >= tcfg.max_maturity_observations_per_track
                                for d in valid]
                        valid = annotate_maturity(frame, valid, self.maturity, self.cfg.maturity, skip)
                        registry.update(valid)
                        if vcfg.store_detections:
                            stored.extend(valid)
                        current = valid
                        frames_processed += 1
                else:
                    frames_skipped += 1

                if writer is not None:
                    hud = [f"Frame {idx}" + (f"/{meta.frame_count}" if meta.frame_count else "")
                           + f"  t={ts:7.2f}s",
                           f"Detections in this frame: {len(current)}",
                           f"Confirmed tracks so far: {registry.confirmed_count()} (running, not final)",
                           f"Camera: {tcfg.camera_mode} | stride {vcfg.frame_stride}"]
                    shown = draw_detections(frame, current, self.cfg.annotation)
                    writer.write(draw_summary_panel(
                        shown, hud, self.cfg.annotation,
                        banner="DEMO MODE - generic model, not validated" if self.demo_mode else None))
                if progress_callback and frames_read % vcfg.progress_every_n_frames == 0:
                    progress_callback(frames_read, meta.frame_count)

            if (not terminated_early and meta.frame_count and frames_read < 0.98 * meta.frame_count):
                terminated_early = True
                warnings.append(f"Stream ended after {frames_read} of {meta.frame_count} reported frames "
                                "(corrupt data or inaccurate metadata). Results may be incomplete.")
            if errors:
                warnings.append(f"{errors} frame(s) failed inference and were skipped.")
            if vcfg.frame_stride > 1:
                warnings.append(f"frame_stride={vcfg.frame_stride}: only 1 in {vcfg.frame_stride} frames was "
                                "analysed. Fruit visible only in skipped frames may be missed and tracking "
                                "continuity is reduced. Boxes shown on skipped frames are carried over.")
            if progress_callback:
                progress_callback(frames_read, meta.frame_count)
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if out_path is not None and vcfg.transcode_h264:
            transcode_h264(out_path)
        elapsed = time.perf_counter() - started
        diag = registry.diagnostics()
        logger.info("Processed %d frames (%d analysed) in %.1fs", frames_read, frames_processed, elapsed)
        return VideoRunResult(
            metadata=meta, frames_read=frames_read, frames_processed=frames_processed,
            frames_skipped=frames_skipped, inference_errors=errors, processing_seconds=elapsed,
            throughput_fps=frames_read / elapsed if elapsed > 0 else 0.0, summaries=registry.summaries(),
            track_diagnostics=diag, raw_frame_detections=registry.raw_detections + rejected,
            rejected_detections=rejected, detections=stored, output_path=out_path,
            terminated_early=terminated_early, warnings=warnings)

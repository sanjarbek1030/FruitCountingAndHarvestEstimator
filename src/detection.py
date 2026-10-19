"""Detector wrapper: converts Ultralytics results into :class:`Detection` objects.

Coordinates: Ultralytics returns ``boxes.xyxy`` already mapped back to the ORIGINAL input frame
(it letterboxes internally and rescales). For tiled inference this module adds the tile offset, so all
boxes are in original-image pixels.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .config import ModelConfig, TrackingConfig
from .errors import ModelError
from .geometry import nms_xyxy
from .model_manager import ModelManager
from .schemas import Detection

logger = logging.getLogger(__name__)


def detections_from_arrays(xyxy: np.ndarray, confs: np.ndarray, class_ids: np.ndarray,
                           track_ids: Optional[np.ndarray], names: dict[int, str],
                           frame_index: int, timestamp: float, allowed_class_names: set[str],
                           offset: tuple[float, float] = (0.0, 0.0)) -> list[Detection]:
    """Pure conversion of raw arrays to detections (unit-testable without a model)."""
    out: list[Detection] = []
    ox, oy = offset
    for i in range(len(xyxy)):
        cid = int(class_ids[i])
        cname = names.get(cid, str(cid))
        if cname.lower() not in allowed_class_names:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        tid = None
        if track_ids is not None and i < len(track_ids) and not np.isnan(track_ids[i]):
            tid = int(track_ids[i])
        out.append(Detection(
            detection_id="", frame_index=frame_index, timestamp_seconds=timestamp,
            bbox_x1=x1 + ox, bbox_y1=y1 + oy, bbox_x2=x2 + ox, bbox_y2=y2 + oy,
            class_id=cid, class_name=cname.lower(), confidence=float(confs[i]), track_id=tid))
    return out


def assign_detection_ids(dets: Iterable[Detection], frame_index: int) -> list[Detection]:
    """Give each detection a stable id ``f{frame}_d{n}``."""
    import dataclasses
    return [dataclasses.replace(d, detection_id=f"f{frame_index:06d}_d{i:03d}") for i, d in enumerate(dets)]


def tile_origins(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Tile rectangles ``(x0, y0, x1, y1)`` covering the image with the given fractional overlap."""
    step = max(1, int(tile * (1.0 - overlap)))

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        s = list(range(0, size - tile + 1, step))
        if s[-1] != size - tile:
            s.append(size - tile)
        return s
    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in starts(height) for x in starts(width)]


class FruitDetector:
    """Runs detection (and optionally Ultralytics tracking) with a model loaded once."""

    def __init__(self, manager: ModelManager, model_cfg: ModelConfig, tracking_cfg: TrackingConfig) -> None:
        self.manager = manager
        self.cfg = model_cfg
        self.tracking_cfg = tracking_cfg

    def _predict_kwargs(self) -> dict[str, Any]:
        return dict(imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou, max_det=self.cfg.max_det,
                    device=self.manager.device, verbose=False)

    def _parse(self, result: Any, frame_index: int, ts: float, offset=(0.0, 0.0),
               with_ids: bool = False) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        # track ids exist only for tracking calls; plain detection never carries them
        ids = boxes.id.cpu().numpy() if with_ids and getattr(boxes, "id", None) is not None else None
        return detections_from_arrays(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), ids,
            self.manager.class_names, frame_index, ts, self.manager.allowed_class_names, offset)

    def detect(self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0) -> list[Detection]:
        """Detect fruit in one BGR frame (no track ids)."""
        try:
            if self.cfg.tiling_enabled:
                dets = self._detect_tiled(frame, frame_index, timestamp)
            else:
                res = self.manager.model.predict(frame, **self._predict_kwargs())
                dets = self._parse(res[0], frame_index, timestamp)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Inference failed on frame {frame_index}: {exc}") from exc
        return assign_detection_ids(dets, frame_index)

    def _detect_tiled(self, frame: np.ndarray, frame_index: int, ts: float) -> list[Detection]:
        h, w = frame.shape[:2]
        kwargs = self._predict_kwargs()
        dets: list[Detection] = []
        if self.cfg.tiling_include_full_image:
            dets += self._parse(self.manager.model.predict(frame, **kwargs)[0], frame_index, ts)
        for x0, y0, x1, y1 in tile_origins(w, h, self.cfg.tile_size, self.cfg.tile_overlap):
            res = self.manager.model.predict(frame[y0:y1, x0:x1], **kwargs)
            dets += self._parse(res[0], frame_index, ts, offset=(float(x0), float(y0)))
        if not dets:
            return []
        # Class-aware merge. Fruit cut by a tile border may survive as a separate box; see README limitations.
        merged: list[Detection] = []
        for cname in {d.class_name for d in dets}:
            group = [d for d in dets if d.class_name == cname]
            boxes = np.array([[d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2] for d in group])
            scores = np.array([d.confidence for d in group])
            merged += [group[i] for i in nms_xyxy(boxes, scores, self.cfg.iou)]
        return merged

    def tracker_name(self) -> str:
        return self.tracking_cfg.tracker_config_path or f"{self.tracking_cfg.tracker_type}.yaml"

    def track(self, frame: np.ndarray, frame_index: int, timestamp: float) -> list[Detection]:
        """Detect + track with Ultralytics ByteTrack/BoT-SORT (``persist=True`` keeps state across calls)."""
        try:
            res = self.manager.model.track(frame, persist=True, tracker=self.tracker_name(),
                                           **self._predict_kwargs())
            dets = self._parse(res[0], frame_index, timestamp, with_ids=True)
        except Exception as exc:
            raise ModelError(f"Tracking inference failed on frame {frame_index}: {exc}") from exc
        return assign_detection_ids(dets, frame_index)

    def reset_tracker(self) -> None:
        """Drop tracker state between videos. Ultralytics keeps it on ``predictor.trackers``."""
        predictor = getattr(self.manager.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            for t in predictor.trackers:
                if hasattr(t, "reset"):
                    t.reset()

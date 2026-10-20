"""Optional maturity/condition classification on fruit crops.

If no classifier is available, detections keep ``maturity_label=None`` (reported as "unavailable").
Nothing here defaults to "ripe". Colour heuristics are intentionally NOT used: maturity cues differ by
species and variety.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from .config import MaturityConfig
from .errors import ModelError
from .schemas import Detection, NOT_ASSESSED, UNCERTAIN

logger = logging.getLogger(__name__)


@dataclass
class MaturityPrediction:
    label: str
    confidence: float


class MaturityClassifier(Protocol):
    available: bool

    def classify(self, crops: Sequence[np.ndarray]) -> list[Optional[MaturityPrediction]]: ...


class NullMaturityClassifier:
    """Used when maturity analysis is disabled/unavailable: never fabricates predictions."""
    available = False

    def classify(self, crops: Sequence[np.ndarray]) -> list[Optional[MaturityPrediction]]:
        return [None] * len(crops)


class UltralyticsCropClassifier:
    """Ultralytics *classification* checkpoint applied to detection crops."""
    available = True

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        p = Path(model_path)
        if not p.is_file():
            raise ModelError(f"Maturity model not found: '{p}'. Train a classifier or set maturity.enabled: false.")
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(p))
        except ImportError as exc:
            raise ModelError("The 'ultralytics' package is not installed.") from exc
        except Exception as exc:
            raise ModelError(f"Failed to load maturity model '{p.name}': {exc}") from exc
        self._device = device

    def classify(self, crops: Sequence[np.ndarray]) -> list[Optional[MaturityPrediction]]:
        out: list[Optional[MaturityPrediction]] = []
        for crop in crops:
            try:
                r = self._model.predict(crop, device=self._device, verbose=False)[0]
                probs = r.probs
                out.append(MaturityPrediction(str(r.names[int(probs.top1)]), float(probs.top1conf)))
            except Exception as exc:
                logger.warning("Maturity inference failed on a crop: %s", exc)
                out.append(None)
        return out


def build_classifier(cfg: MaturityConfig, device: str = "cpu") -> MaturityClassifier:
    """Return a real classifier when enabled, else the null classifier. Missing model => clear error."""
    if not cfg.enabled:
        return NullMaturityClassifier()
    return UltralyticsCropClassifier(cfg.model_path, device)


def crop_detection(frame: np.ndarray, det: Detection, padding: float) -> np.ndarray:
    """Crop a detection from the frame with fractional padding, clipped to image bounds."""
    h, w = frame.shape[:2]
    px, py = det.width * padding, det.height * padding
    x1, y1 = max(0, int(det.bbox_x1 - px)), max(0, int(det.bbox_y1 - py))
    x2, y2 = min(w, int(det.bbox_x2 + px)), min(h, int(det.bbox_y2 + py))
    return frame[y1:y2, x1:x2]


def apply_prediction(det: Detection, pred: Optional[MaturityPrediction], cfg: MaturityConfig) -> Detection:
    """Map a raw prediction to detection fields, applying the confidence threshold."""
    if pred is None:
        return det
    if pred.confidence < cfg.confidence_threshold:
        return dataclasses.replace(det, maturity_label=UNCERTAIN, maturity_confidence=pred.confidence)
    if pred.label in cfg.condition_labels:
        return dataclasses.replace(det, maturity_label=NOT_ASSESSED, maturity_confidence=pred.confidence,
                                   condition_label=pred.label)
    if pred.label in cfg.maturity_labels:
        return dataclasses.replace(det, maturity_label=pred.label, maturity_confidence=pred.confidence)
    logger.warning("Classifier label %r is in neither maturity_labels nor condition_labels", pred.label)
    return dataclasses.replace(det, maturity_label=UNCERTAIN, maturity_confidence=pred.confidence)


def annotate_maturity(frame: np.ndarray, detections: Sequence[Detection], classifier: MaturityClassifier,
                      cfg: MaturityConfig, skip: Optional[Sequence[bool]] = None) -> list[Detection]:
    """Classify crops for ``detections`` (optionally skipping some) and return updated copies."""
    if not classifier.available:
        return list(detections)
    skip = skip or [False] * len(detections)
    idx, crops = [], []
    out = list(detections)
    for i, d in enumerate(detections):
        if skip[i]:
            continue
        crop = crop_detection(frame, d, cfg.crop_padding)
        if min(crop.shape[:2]) < cfg.min_crop_px:
            out[i] = dataclasses.replace(d, maturity_label=UNCERTAIN, maturity_confidence=None)
            continue
        idx.append(i)
        crops.append(crop)
    for i, pred in zip(idx, classifier.classify(crops)):
        out[i] = apply_prediction(detections[i], pred, cfg)
    return out

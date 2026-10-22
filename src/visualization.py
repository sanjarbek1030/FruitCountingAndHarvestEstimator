"""Drawing utilities. All coordinates are in original-image pixels."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import AnnotationConfig
from .errors import InputError
from .schemas import Detection


def _bgr(rgb: Sequence[int]) -> tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def color_for(det: Detection, cfg: AnnotationConfig) -> tuple[int, int, int]:
    """Colour by maturity label, then condition label, then 'unavailable' / default (configurable)."""
    for key in (det.maturity_label, det.condition_label):
        if key and key in cfg.colors:
            return _bgr(cfg.colors[key])
    return _bgr(cfg.colors.get("unavailable", cfg.colors["default"]))


def draw_detections(frame: np.ndarray, detections: Sequence[Detection], cfg: AnnotationConfig) -> np.ndarray:
    """Return a copy of ``frame`` with boxes and (size-gated) labels drawn."""
    img = frame.copy()
    for d in detections:
        p1, p2 = (int(d.bbox_x1), int(d.bbox_y1)), (int(d.bbox_x2), int(d.bbox_y2))
        col = color_for(d, cfg)
        cv2.rectangle(img, p1, p2, col, cfg.box_thickness)
        if min(d.width, d.height) < cfg.min_label_box_px:
            continue    # avoid clutter on small boxes
        parts = [f"{d.class_name} {d.confidence:.2f}"]
        if cfg.show_track_ids and d.track_id is not None:
            parts.append(f"#{d.track_id}")
        if d.maturity_label and d.maturity_label not in ("not_assessed",):
            parts.append(d.maturity_label)
        if d.condition_label:
            parts.append(d.condition_label)
        text = " ".join(parts)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 1)
        y0 = max(p1[1], th + base + 2)
        cv2.rectangle(img, (p1[0], y0 - th - base - 2), (p1[0] + tw + 4, y0), col, -1)
        cv2.putText(img, text, (p1[0] + 2, y0 - base - 1), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_summary_panel(frame: np.ndarray, lines: Sequence[str], cfg: AnnotationConfig,
                       banner: Optional[str] = None) -> np.ndarray:
    """Draw a translucent text panel (top-left) and an optional red banner (e.g. DEMO MODE)."""
    img = frame.copy()
    scale = max(0.4, cfg.font_scale)
    line_h = int(22 * scale / 0.5)
    pad = 6
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for t in lines] or [0]
    box_w, box_h = max(widths) + 2 * pad, line_h * len(lines) + pad
    if lines:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (min(box_w, img.shape[1]), min(box_h, img.shape[0])), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (pad, (i + 1) * line_h), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
    if banner:
        h = img.shape[0]
        cv2.rectangle(img, (0, h - line_h - pad), (img.shape[1], h), (0, 0, 200), -1)
        cv2.putText(img, banner, (pad, h - pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Write an image (unicode-safe). Raises :class:`InputError` on failure."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise InputError(f"Could not encode image as '{ext}'")
    buf.tofile(str(p))
    return p

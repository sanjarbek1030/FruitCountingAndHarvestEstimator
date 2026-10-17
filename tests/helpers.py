"""Synthetic helpers for deterministic tests (no model, no downloads)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.schemas import Detection


def det(x1=10, y1=10, x2=50, y2=50, conf=0.9, cls="apple", frame=0, tid: Optional[int] = None,
        did: Optional[str] = None, maturity=None, condition=None, fps=30.0) -> Detection:
    return Detection(detection_id=did or f"f{frame:06d}_{x1}_{y1}", frame_index=frame,
                     timestamp_seconds=frame / fps, bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                     class_id=0, class_name=cls, confidence=conf, track_id=tid,
                     maturity_label=maturity, condition_label=condition)


class FakeDetector:
    """Duck-typed detector replaying scripted detections per frame index."""

    def __init__(self, per_frame: dict[int, list[Detection]], fail_frames: set[int] | None = None):
        self.per_frame = per_frame
        self.fail_frames = fail_frames or set()
        self.manager = None

    def _get(self, idx):
        from src.errors import ModelError
        if idx in self.fail_frames:
            raise ModelError("boom")
        import dataclasses
        return [dataclasses.replace(d, frame_index=idx, timestamp_seconds=idx / 30.0,
                                    detection_id=f"f{idx:06d}_d{i:03d}")
                for i, d in enumerate(self.per_frame.get(idx, []))]

    def detect(self, frame, frame_index=0, timestamp=0.0):
        import dataclasses
        return [dataclasses.replace(d, track_id=None) for d in self._get(frame_index)]

    def track(self, frame, frame_index, timestamp):
        return self._get(frame_index)

    def reset_tracker(self):
        pass


def write_video(path, n_frames=12, size=(64, 48), fps=10.0):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened(), "mp4v writer unavailable in this environment"
    for i in range(n_frames):
        w.write(np.full((size[1], size[0], 3), i * 5, dtype=np.uint8))
    w.release()
    return path

"""Tracking support: a built-in IoU tracker and a registry that summarises observations per track.

The registry is tracker-agnostic: it accepts detections that already carry ``track_id`` (from
Ultralytics ByteTrack/BoT-SORT) or from :class:`IoUTracker`.

Documented policy for the built-in IoUTracker (fixed camera only, no motion model):

* A detection is matched to an existing same-class track by greedy highest-IoU assignment, if
  IoU >= ``iou_match_threshold``.
* An unmatched track is kept as *lost* for up to ``max_lost_frames`` update calls and can be
  re-associated if a detection overlaps its last known box with IoU >= threshold.
* After that the track is terminated. A later detection at the same place gets a NEW id
  (fragmentation). This is visible in the diagnostics instead of being hidden.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .geometry import iou_xyxy
from .schemas import Detection, TrackSummary


@dataclass
class _LiveTrack:
    track_id: int
    class_name: str
    box: tuple[float, float, float, float]
    lost: int = 0


class IoUTracker:
    """Minimal greedy IoU tracker. Assigns ``track_id`` to detections."""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self._tracks: dict[int, _LiveTrack] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_track_ids(self) -> list[int]:
        return [t.track_id for t in self._tracks.values() if t.lost == 0]

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        """Associate detections of ONE processed frame; returns copies with ``track_id`` set."""
        dets = list(detections)
        pairs: list[tuple[float, int, int]] = []
        track_ids = list(self._tracks)
        for di, d in enumerate(dets):
            box = (d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2)
            for tid in track_ids:
                t = self._tracks[tid]
                if t.class_name != d.class_name:
                    continue
                v = iou_xyxy(box, t.box)
                if v >= self.iou_threshold:
                    pairs.append((v, di, tid))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        det_to_track: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, di, tid in pairs:
            if di in det_to_track or tid in used_tracks:
                continue
            det_to_track[di] = tid
            used_tracks.add(tid)

        for tid in track_ids:
            if tid not in used_tracks:
                self._tracks[tid].lost += 1
        for tid in [t for t, tr in self._tracks.items() if tr.lost > self.max_lost_frames]:
            del self._tracks[tid]

        out: list[Detection] = []
        for di, d in enumerate(dets):
            box = (d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2)
            if di in det_to_track:
                tr = self._tracks[det_to_track[di]]
                tr.box, tr.lost = box, 0
                tid = tr.track_id
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _LiveTrack(tid, d.class_name, box)
            out.append(dataclasses.replace(d, track_id=tid))
        return out


@dataclass
class _TrackAccumulator:
    class_counts: Counter
    first_frame: int
    last_frame: int
    first_ts: float
    last_ts: float
    frames: list[int]
    conf_sum: float = 0.0
    cx_sum: float = 0.0
    cy_sum: float = 0.0
    mat_labels: Counter = None  # type: ignore[assignment]
    mat_conf_sum: float = 0.0
    mat_n: int = 0
    cond_labels: Counter = None  # type: ignore[assignment]
    zone_labels: Counter = None  # type: ignore[assignment]


class TrackRegistry:
    """Collects per-track observation statistics across processed frames."""

    def __init__(self, min_confirm_observations: int = 3) -> None:
        if min_confirm_observations < 1:
            raise ValueError("min_confirm_observations must be >= 1")
        self.min_confirm_observations = min_confirm_observations
        self._acc: dict[int, _TrackAccumulator] = {}
        self.raw_detections = 0
        self.untracked_detections = 0

    def update(self, detections: Sequence[Detection]) -> None:
        """Register the (validated) detections of one processed frame."""
        for d in detections:
            self.raw_detections += 1
            if d.track_id is None:
                self.untracked_detections += 1   # cannot be confirmed without an id
                continue
            a = self._acc.get(d.track_id)
            if a is None:
                a = _TrackAccumulator(Counter(), d.frame_index, d.frame_index, d.timestamp_seconds,
                                      d.timestamp_seconds, [], mat_labels=Counter(),
                                      cond_labels=Counter(), zone_labels=Counter())
                self._acc[d.track_id] = a
            if d.frame_index not in a.frames[-1:]:
                a.frames.append(d.frame_index)
            a.class_counts[d.class_name] += 1
            a.last_frame, a.last_ts = d.frame_index, d.timestamp_seconds
            a.conf_sum += d.confidence
            a.cx_sum += d.center_x
            a.cy_sum += d.center_y
            if d.maturity_label is not None:
                a.mat_labels[d.maturity_label] += 1
                if d.maturity_confidence is not None:
                    a.mat_conf_sum += d.maturity_confidence
                    a.mat_n += 1
            if d.condition_label is not None:
                a.cond_labels[d.condition_label] += 1
            if d.observation_zone is not None:
                a.zone_labels[d.observation_zone] += 1

    def maturity_observation_count(self, track_id: Optional[int]) -> int:
        """How many maturity predictions this track has already received (cost control)."""
        if track_id is None or track_id not in self._acc:
            return 0
        return sum(self._acc[track_id].mat_labels.values())

    def confirmed_count(self) -> int:
        """Cheap running count of confirmed tracks (used by the video HUD)."""
        return sum(1 for a in self._acc.values()
                   if sum(a.class_counts.values()) >= self.min_confirm_observations)

    def summaries(self) -> list[TrackSummary]:
        out: list[TrackSummary] = []
        for tid, a in sorted(self._acc.items()):
            n = sum(a.class_counts.values())
            out.append(TrackSummary(
                track_id=tid, class_name=a.class_counts.most_common(1)[0][0],
                first_frame=a.first_frame, last_frame=a.last_frame,
                first_timestamp_seconds=a.first_ts, last_timestamp_seconds=a.last_ts,
                n_observations=n, frames_observed=list(a.frames),
                mean_confidence=a.conf_sum / n, mean_center_x=a.cx_sum / n, mean_center_y=a.cy_sum / n,
                maturity_label=a.mat_labels.most_common(1)[0][0] if a.mat_labels else None,
                maturity_confidence=(a.mat_conf_sum / a.mat_n) if a.mat_n else None,
                condition_label=a.cond_labels.most_common(1)[0][0] if a.cond_labels else None,
                observation_zone=a.zone_labels.most_common(1)[0][0] if a.zone_labels else None,
                confirmed=n >= self.min_confirm_observations))
        return out

    def diagnostics(self) -> dict:
        """Numbers that expose tracking quality/fragmentation (reported, never hidden)."""
        lengths = [sum(a.class_counts.values()) for a in self._acc.values()]
        confirmed = [n for n in lengths if n >= self.min_confirm_observations]
        return {
            "total_tracks": len(lengths),
            "confirmed_tracks": len(confirmed),
            "unconfirmed_short_tracks": len(lengths) - len(confirmed),
            "untracked_detections": self.untracked_detections,
            "median_observations_per_track": float(np.median(lengths)) if lengths else 0.0,
            "mean_observations_per_confirmed_track": float(np.mean(confirmed)) if confirmed else 0.0,
            "min_confirm_observations": self.min_confirm_observations,
        }

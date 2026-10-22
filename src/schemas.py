"""Explicit data structures shared by all modules.

Terminology (used identically in code, reports and the dashboard):

* ``raw_detections``            - every box the detector emitted, in every processed frame.
* ``confirmed_tracks``          - tracks observed in >= ``min_confirm_observations`` processed frames.
* ``estimated_unique_visible_fruit`` - the counting policy's best estimate of distinct fruit that
                                  were *visible* in the footage. NOT ground truth.
* ``estimated_total_tree_fruit``     - only available with a configured visibility fraction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

UNAVAILABLE = "unavailable"   # no maturity model was run
UNCERTAIN = "uncertain"       # model ran, confidence below threshold
NOT_ASSESSED = "not_assessed"  # a condition label was assigned instead of a maturity label


@dataclass
class Detection:
    """One detected fruit instance in one frame (or one image)."""

    detection_id: str
    frame_index: int
    timestamp_seconds: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    maturity_label: Optional[str] = None
    maturity_confidence: Optional[float] = None
    condition_label: Optional[str] = None
    observation_zone: Optional[str] = None

    @property
    def center_x(self) -> float:
        return (self.bbox_x1 + self.bbox_x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox_y1 + self.bbox_y2) / 2.0

    @property
    def width(self) -> float:
        return self.bbox_x2 - self.bbox_x1

    @property
    def height(self) -> float:
        return self.bbox_y2 - self.bbox_y1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["center_x"] = self.center_x
        d["center_y"] = self.center_y
        return d


DETECTION_COLUMNS = [
    "detection_id", "frame_index", "timestamp_seconds",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "center_x", "center_y",
    "class_id", "class_name", "confidence", "track_id",
    "maturity_label", "maturity_confidence", "condition_label", "observation_zone",
]


@dataclass
class RejectedDetection:
    """A detection discarded during validation, with the reason."""

    detection_id: str
    reason: str


@dataclass
class TrackSummary:
    """Aggregated observation statistics for one track id (debugging + per-fruit export)."""

    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    first_timestamp_seconds: float
    last_timestamp_seconds: float
    n_observations: int
    frames_observed: list[int]
    mean_confidence: float
    mean_center_x: float
    mean_center_y: float
    maturity_label: Optional[str] = None
    maturity_confidence: Optional[float] = None
    condition_label: Optional[str] = None
    observation_zone: Optional[str] = None
    confirmed: bool = False

    def to_dict(self, include_frames: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_frames:
            d.pop("frames_observed")
        return d


TRACK_COLUMNS = [
    "track_id", "class_name", "confirmed", "first_frame", "last_frame",
    "first_timestamp_seconds", "last_timestamp_seconds", "n_observations",
    "mean_confidence", "mean_center_x", "mean_center_y",
    "maturity_label", "maturity_confidence", "condition_label", "observation_zone",
]


@dataclass
class ReviewCorrections:
    """Manual review input. Corrections are recorded separately from model output."""

    rejected_detection_ids: set[str] = field(default_factory=set)
    rejected_track_ids: set[int] = field(default_factory=set)
    added_by_species: dict[str, int] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.rejected_detection_ids or self.rejected_track_ids
                    or any(v for v in self.added_by_species.values()))


@dataclass
class CountResult:
    """Output of the counting module for an image or a video."""

    mode: str                                   # "image" | "video"
    raw_detections: int                         # all detector boxes seen (all processed frames)
    rejected_detections: int                    # failed validation / confidence filter
    confirmed_tracks: Optional[int]             # video only
    estimated_unique_visible_fruit: int
    by_species: dict[str, int]
    by_maturity: dict[str, int]
    by_condition: dict[str, int]
    by_zone: dict[str, int]
    uncertain_classifications: int
    confidence: Optional[dict[str, float]]      # mean/median/min/max of counted observations
    estimate_basis: str
    manual_additions: int = 0
    manual_rejections: int = 0
    track_diagnostics: Optional[dict[str, Any]] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UncertaintyInterval:
    """Empirical prediction interval. Only produced from validation data."""

    lower: float
    upper: float
    coverage: float
    method: str
    n_samples: int
    unit: str = "kg"


@dataclass
class HarvestEstimate:
    """Harvest estimation output. ``None`` means "cannot be estimated from the configuration"."""

    observed_unique_visible_fruit: int
    harvestable_probability_used: Optional[dict[str, float]]
    average_weight_grams_used: dict[str, float]
    # --- visible fruit only (no extrapolation) ---
    visible_harvestable_count: Optional[float]          # unrounded
    visible_harvestable_count_rounded: Optional[int]    # display only
    visible_harvest_weight_kg: Optional[float]
    # --- extrapolated (needs a disclosed visibility fraction) ---
    visibility_fraction: Optional[float]
    estimated_total_tree_fruit: Optional[float]
    tree_harvestable_count: Optional[float]
    tree_harvest_weight_kg: Optional[float]
    uncertainty: Optional[UncertaintyInterval]
    uncertainty_note: str
    method: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

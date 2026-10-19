"""Typed, validated configuration loaded from ``config.yaml``.

Unknown keys are rejected so that typos do not silently fall back to defaults.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_type_hints

import yaml

from .errors import ConfigError

VALID_DEVICES_HINT = "'auto', 'cpu', 'cuda', 'cuda:0', 'mps'"


def _is_prob(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 1.0


def _is_pos(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


@dataclass
class ModelConfig:
    path: str = "models/best.pt"
    device: str = "auto"
    imgsz: int = 960
    conf: float = 0.25          # detector confidence threshold passed to inference
    iou: float = 0.5            # NMS IoU threshold passed to inference
    max_det: int = 300
    demo_mode: bool = False     # allow a generic COCO model; results are labelled DEMO
    demo_class_names: list[str] = field(default_factory=lambda: ["apple", "orange"])
    tiling_enabled: bool = False
    tile_size: int = 640
    tile_overlap: float = 0.25
    tiling_include_full_image: bool = True

    def validate(self) -> None:
        if not self.path or not isinstance(self.path, str):
            raise ConfigError("model.path must be a non-empty string")
        if not (isinstance(self.device, str) and (
                self.device in ("auto", "cpu", "mps") or self.device.startswith("cuda"))):
            raise ConfigError(f"model.device must be one of {VALID_DEVICES_HINT}, got {self.device!r}")
        if not isinstance(self.imgsz, int) or self.imgsz < 32:
            raise ConfigError("model.imgsz must be an integer >= 32")
        if not _is_prob(self.conf):
            raise ConfigError("model.conf must be within [0, 1]")
        if not _is_prob(self.iou):
            raise ConfigError("model.iou must be within [0, 1]")
        if not isinstance(self.max_det, int) or self.max_det < 1:
            raise ConfigError("model.max_det must be a positive integer")
        if not isinstance(self.tile_size, int) or self.tile_size < 64:
            raise ConfigError("model.tile_size must be an integer >= 64")
        if not (isinstance(self.tile_overlap, (int, float)) and 0.0 <= self.tile_overlap < 1.0):
            raise ConfigError("model.tile_overlap must be within [0, 1)")


@dataclass
class TrackingConfig:
    backend: str = "ultralytics"        # "ultralytics" (ByteTrack/BoT-SORT) or "iou" (built-in)
    tracker_type: str = "bytetrack"     # "bytetrack" | "botsort"
    tracker_config_path: Optional[str] = None
    camera_mode: str = "fixed"          # "fixed" | "moving"
    min_confirm_observations: int = 3
    iou_match_threshold: float = 0.3    # built-in "iou" backend only
    max_lost_frames: int = 15           # built-in "iou" backend only (in processed frames)
    max_maturity_observations_per_track: int = 5

    def validate(self) -> None:
        if self.backend not in ("ultralytics", "iou"):
            raise ConfigError("tracking.backend must be 'ultralytics' or 'iou'")
        if self.tracker_type not in ("bytetrack", "botsort"):
            raise ConfigError("tracking.tracker_type must be 'bytetrack' or 'botsort'")
        if self.camera_mode not in ("fixed", "moving"):
            raise ConfigError("tracking.camera_mode must be 'fixed' or 'moving'")
        if not isinstance(self.min_confirm_observations, int) or self.min_confirm_observations < 1:
            raise ConfigError("tracking.min_confirm_observations must be an integer >= 1")
        if not _is_prob(self.iou_match_threshold) or self.iou_match_threshold == 0:
            raise ConfigError("tracking.iou_match_threshold must be within (0, 1]")
        if not isinstance(self.max_lost_frames, int) or self.max_lost_frames < 0:
            raise ConfigError("tracking.max_lost_frames must be an integer >= 0")
        if not isinstance(self.max_maturity_observations_per_track, int) or \
                self.max_maturity_observations_per_track < 1:
            raise ConfigError("tracking.max_maturity_observations_per_track must be >= 1")


@dataclass
class CountingConfig:
    min_confidence: float = 0.25        # applied after inference (can be stricter than model.conf)
    duplicate_iou_threshold: float = 0.85   # single-image near-identical box suppression
    count_only_in_zones: bool = False
    # Observation zones: list of {"name": str, "box": [x1, y1, x2, y2]} in NORMALISED [0,1] coordinates.
    zones: list[dict[str, Any]] = field(default_factory=list)
    low_confidence_flag: float = 0.40   # dashboard marks detections below this as "suspicious"

    def validate(self) -> None:
        if not _is_prob(self.min_confidence):
            raise ConfigError("counting.min_confidence must be within [0, 1]")
        if not _is_prob(self.duplicate_iou_threshold) or self.duplicate_iou_threshold == 0:
            raise ConfigError("counting.duplicate_iou_threshold must be within (0, 1]")
        if not _is_prob(self.low_confidence_flag):
            raise ConfigError("counting.low_confidence_flag must be within [0, 1]")
        names = set()
        for z in self.zones:
            if not isinstance(z, dict) or "name" not in z or "box" not in z:
                raise ConfigError("each counting.zones entry needs 'name' and 'box'")
            box = z["box"]
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(_is_prob(v) for v in box) or box[0] >= box[2] or box[1] >= box[3]):
                raise ConfigError(f"zone {z.get('name')!r}: box must be [x1, y1, x2, y2] in [0,1] with x1<x2, y1<y2")
            if z["name"] in names:
                raise ConfigError(f"duplicate zone name {z['name']!r}")
            names.add(z["name"])
        if self.count_only_in_zones and not self.zones:
            raise ConfigError("counting.count_only_in_zones requires at least one zone")


@dataclass
class VideoConfig:
    input_path: str = "input_orchard.mp4"
    output_path: str = "output_annotated.mp4"
    frame_stride: int = 1               # process every Nth frame (see README: reduces tracking quality)
    fallback_fps: float = 30.0          # used when FPS metadata is missing/invalid (recorded in report)
    codec: str = "mp4v"
    max_consecutive_errors: int = 10
    max_frames: Optional[int] = None    # safety cap; None = whole video
    store_detections: bool = False      # keep every per-frame detection in memory for CSV export
    transcode_h264: bool = False        # re-encode with ffmpeg (if installed) for browser playback
    progress_every_n_frames: int = 25

    def validate(self) -> None:
        if not isinstance(self.frame_stride, int) or self.frame_stride < 1:
            raise ConfigError("video.frame_stride must be an integer >= 1")
        if not _is_pos(self.fallback_fps):
            raise ConfigError("video.fallback_fps must be positive")
        if not isinstance(self.codec, str) or len(self.codec) != 4:
            raise ConfigError("video.codec must be a 4-character FourCC string")
        if not isinstance(self.max_consecutive_errors, int) or self.max_consecutive_errors < 1:
            raise ConfigError("video.max_consecutive_errors must be >= 1")
        if self.max_frames is not None and (not isinstance(self.max_frames, int) or self.max_frames < 1):
            raise ConfigError("video.max_frames must be null or a positive integer")
        if not isinstance(self.progress_every_n_frames, int) or self.progress_every_n_frames < 1:
            raise ConfigError("video.progress_every_n_frames must be >= 1")


@dataclass
class OutputConfig:
    report_dir: str = "outputs/reports"
    annotated_dir: str = "outputs/annotated"

    def validate(self) -> None:
        if not self.report_dir or not self.annotated_dir:
            raise ConfigError("output.report_dir and output.annotated_dir must be non-empty")


@dataclass
class FruitConfig:
    species: list[str] = field(default_factory=lambda: ["apple"])
    # Mean fresh weight per fruit in grams, per species. MUST come from physical measurements.
    average_weight_grams: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.species or not all(isinstance(s, str) and s.strip() for s in self.species):
            raise ConfigError("fruit.species must be a non-empty list of names")
        for name, w in self.average_weight_grams.items():
            if not _is_pos(w):
                raise ConfigError(f"fruit.average_weight_grams[{name!r}] must be a positive number, got {w!r}")


@dataclass
class HarvestConfig:
    # Calibrated probability that a visible fruit is eventually harvested. None => not estimated.
    harvestable_probability: Optional[float] = None
    harvestable_probability_by_species: dict[str, float] = field(default_factory=dict)
    # Fraction of the tree's fruit that is visible in the footage, in (0, 1]. None => no extrapolation.
    visibility_fraction: Optional[float] = None
    # CSV with columns predicted_weight_kg, actual_weight_kg from field validation.
    validation_pairs_path: Optional[str] = None
    uncertainty_basis: str = "visible"      # "visible" | "tree"
    interval_coverage: float = 0.90
    min_validation_samples: int = 10

    def validate(self) -> None:
        if self.harvestable_probability is not None and not _is_prob(self.harvestable_probability):
            raise ConfigError("harvest.harvestable_probability must be within [0, 1] or null")
        for k, p in self.harvestable_probability_by_species.items():
            if not _is_prob(p):
                raise ConfigError(f"harvest.harvestable_probability_by_species[{k!r}] must be within [0, 1]")
        v = self.visibility_fraction
        if v is not None and not (_is_pos(v) and v <= 1.0):
            raise ConfigError("harvest.visibility_fraction must be > 0 and <= 1, or null")
        if self.uncertainty_basis not in ("visible", "tree"):
            raise ConfigError("harvest.uncertainty_basis must be 'visible' or 'tree'")
        if not (isinstance(self.interval_coverage, float) and 0.5 <= self.interval_coverage < 1.0):
            raise ConfigError("harvest.interval_coverage must be a float within [0.5, 1)")
        if not isinstance(self.min_validation_samples, int) or self.min_validation_samples < 2:
            raise ConfigError("harvest.min_validation_samples must be an integer >= 2")


@dataclass
class MaturityConfig:
    enabled: bool = False
    model_path: str = "models/maturity.pt"      # Ultralytics *classification* checkpoint
    maturity_labels: list[str] = field(default_factory=lambda: ["ripe", "unripe"])
    condition_labels: list[str] = field(default_factory=lambda: ["damaged"])
    confidence_threshold: float = 0.60
    crop_padding: float = 0.10
    min_crop_px: int = 16

    def validate(self) -> None:
        if not _is_prob(self.confidence_threshold):
            raise ConfigError("maturity.confidence_threshold must be within [0, 1]")
        if not (isinstance(self.crop_padding, (int, float)) and 0 <= self.crop_padding <= 1):
            raise ConfigError("maturity.crop_padding must be within [0, 1]")
        if not isinstance(self.min_crop_px, int) or self.min_crop_px < 1:
            raise ConfigError("maturity.min_crop_px must be >= 1")
        if set(self.maturity_labels) & set(self.condition_labels):
            raise ConfigError("maturity_labels and condition_labels must not overlap")


def _default_colors() -> dict[str, list[int]]:
    # RGB triplets. Keys are maturity/condition labels plus fallbacks.
    return {
        "ripe": [220, 40, 40], "unripe": [60, 180, 75], "damaged": [150, 90, 30],
        "uncertain": [240, 180, 0], "unavailable": [60, 120, 230], "default": [60, 120, 230],
    }


@dataclass
class AnnotationConfig:
    colors: dict[str, list[int]] = field(default_factory=_default_colors)
    min_label_box_px: int = 28      # smaller boxes get no text label (avoids clutter)
    show_track_ids: bool = True
    font_scale: float = 0.5
    box_thickness: int = 2

    def validate(self) -> None:
        for k, c in self.colors.items():
            if not (isinstance(c, (list, tuple)) and len(c) == 3
                    and all(isinstance(v, int) and 0 <= v <= 255 for v in c)):
                raise ConfigError(f"annotation.colors[{k!r}] must be [R, G, B] integers in 0..255")
        if "default" not in self.colors:
            raise ConfigError("annotation.colors must contain a 'default' entry")
        if not _is_pos(self.font_scale) or self.box_thickness < 1:
            raise ConfigError("annotation.font_scale must be > 0 and box_thickness >= 1")


@dataclass
class DashboardConfig:
    max_upload_mb: int = 200
    image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    video_extensions: list[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    max_table_rows: int = 2000
    max_video_frames: int = 20000
    admin_password_env: str = "FRUIT_ADMIN_PASSWORD"

    def validate(self) -> None:
        if not isinstance(self.max_upload_mb, int) or self.max_upload_mb < 1:
            raise ConfigError("dashboard.max_upload_mb must be a positive integer")
        for ext in self.image_extensions + self.video_extensions:
            if not ext.startswith("."):
                raise ConfigError(f"extension {ext!r} must start with '.'")


@dataclass
class LoggingConfig:
    level: str = "INFO"

    def validate(self) -> None:
        if str(self.level).upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigError("logging.level must be DEBUG, INFO, WARNING, ERROR or CRITICAL")


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    fruit: FruitConfig = field(default_factory=FruitConfig)
    harvest: HarvestConfig = field(default_factory=HarvestConfig)
    maturity: MaturityConfig = field(default_factory=MaturityConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> "AppConfig":
        """Validate every section; raises :class:`ConfigError`. Returns self for chaining."""
        for f in fields(self):
            getattr(self, f.name).validate()
        return self

    def copy(self) -> "AppConfig":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _from_dict(cls: type, data: Any, where: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config section '{where}' must be a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{where}': {unknown}. Valid keys: {sorted(known)}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            t = hints[f.name]
            kwargs[f.name] = _from_dict(t, data[f.name], f"{where}.{f.name}") if is_dataclass(t) else data[f.name]
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> AppConfig:
    """Build and validate an :class:`AppConfig` from a plain dictionary."""
    cfg = _from_dict(AppConfig, data, "config")
    return cfg.validate()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file.

    Raises:
        ConfigError: if the file is missing, unreadable, not valid YAML, or contains invalid values.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Configuration file not found: '{p.name}' (looked in '{p.parent}'). "
                          "Copy config.yaml from the repository or pass --config.")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration file '{p.name}' is not valid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration file '{p.name}': {exc}") from exc
    return config_from_dict(data or {})

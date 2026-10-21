"""Model management: validation, device resolution, one-time loading of the detector.

Uses the real Ultralytics API (``from ultralytics import YOLO``; ``YOLO(path)``; ``model.names``).
No generic model is silently accepted: real mode requires a checkpoint whose class names contain the
configured species; demo mode must be switched on explicitly and is labelled DEMO downstream.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from .config import ModelConfig
from .errors import ModelError

logger = logging.getLogger(__name__)

NO_DETECTOR_HELP = (
    "No trained fruit detector is available. Train or fine-tune one (see README, 'Model training') and set "
    "model.path in config.yaml, e.g. models/best.pt. For a clearly labelled demonstration with a generic "
    "COCO checkpoint (apple/orange classes only), set model.demo_mode: true.")


def resolve_device(requested: str, torch_module: Any = None) -> tuple[str, str]:
    """Return ``(device, note)``. Falls back to CPU with an explanatory note when a GPU is unavailable."""
    if requested == "cpu":
        return "cpu", "CPU requested."
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError:
            return "cpu", "PyTorch not importable; using CPU."
    cuda_ok = bool(torch_module.cuda.is_available())
    if requested == "auto":
        if cuda_ok:
            return "cuda:0", f"CUDA available ({torch_module.cuda.get_device_name(0)})."
        return "cpu", "No CUDA device detected; using CPU (inference will be slower)."
    if requested.startswith("cuda"):
        if cuda_ok:
            return requested, "CUDA requested and available."
        return "cpu", ("CUDA requested but torch.cuda.is_available() is False (CPU-only PyTorch build, missing "
                       "driver, or no GPU). Falling back to CPU.")
    if requested == "mps":
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "MPS available."
        return "cpu", "MPS requested but unavailable. Falling back to CPU."
    return "cpu", f"Unknown device {requested!r}; using CPU."


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Size and a short SHA-256 prefix to identify the checkpoint in reports (name only, not full path)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"checkpoint_name": path.name, "checkpoint_size_bytes": path.stat().st_size,
            "checkpoint_sha256_prefix": h.hexdigest()[:12]}


class ModelManager:
    """Loads the detector once and exposes it plus metadata. Never reloads per frame."""

    def __init__(self, cfg: ModelConfig, species: list[str]) -> None:
        self.cfg = cfg
        self.species = [s.lower() for s in species]
        self._model: Any = None
        self.device = "cpu"
        self.device_note = ""
        self.class_names: dict[int, str] = {}
        self.allowed_class_names: set[str] = set()
        self.info: dict[str, Any] = {}
        self.warnings: list[str] = []

    @property
    def demo_mode(self) -> bool:
        return self.cfg.demo_mode

    @property
    def model(self) -> Any:
        if self._model is None:
            raise ModelError("Model not loaded. Call ModelManager.load() first.")
        return self._model

    def load(self) -> "ModelManager":
        """Validate the checkpoint, resolve the device and load the model (idempotent)."""
        if self._model is not None:
            return self
        path = Path(self.cfg.path)
        is_bare_name = path.parent == Path(".")
        fingerprint: dict[str, Any] = {"checkpoint_name": path.name}

        if path.is_file():
            try:
                with path.open("rb") as fh:
                    fh.read(1)
                fingerprint = file_fingerprint(path)
            except OSError as exc:
                raise ModelError(f"Checkpoint '{path.name}' exists but is not readable: {exc}") from exc
        elif self.cfg.demo_mode and is_bare_name:
            self.warnings.append(f"DEMO MODE: '{path.name}' not found locally; Ultralytics will download the "
                                 "official generic checkpoint (needs network).")
        else:
            raise ModelError(f"Detector checkpoint not found: '{path}'. " + NO_DETECTOR_HELP)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelError("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt") from exc

        self.device, self.device_note = resolve_device(self.cfg.device)
        logger.info("Device: %s (%s)", self.device, self.device_note)
        try:
            model = YOLO(str(path))
        except Exception as exc:  # Ultralytics raises various types for corrupt checkpoints
            raise ModelError(f"Failed to load checkpoint '{path.name}': {exc}") from exc

        names = getattr(model, "names", None)
        if not isinstance(names, dict) or not names:
            raise ModelError("Loaded model exposes no class names; is it a detection model?")
        self.class_names = {int(k): str(v) for k, v in names.items()}
        self._check_classes()
        self._model = model

        try:
            import ultralytics
            version = getattr(ultralytics, "__version__", "unknown")
        except Exception:  # pragma: no cover
            version = "unknown"
        self.info = {**fingerprint, "ultralytics_version": version, "device": self.device,
                     "device_note": self.device_note, "class_names": list(self.class_names.values()),
                     "demo_mode": self.demo_mode, "counted_classes": sorted(self.allowed_class_names)}
        logger.info("Model loaded: %s | classes=%s | imgsz=%s conf=%s iou=%s", path.name,
                    list(self.class_names.values()), self.cfg.imgsz, self.cfg.conf, self.cfg.iou)
        return self

    def _check_classes(self) -> None:
        model_names = {n.lower() for n in self.class_names.values()}
        wanted = set(self.species)
        if self.cfg.demo_mode:
            wanted &= {c.lower() for c in self.cfg.demo_class_names}
            if not wanted & model_names:
                raise ModelError("Demo mode: none of the configured species are demo classes present in the "
                                 f"model (model classes: {sorted(model_names)}).")
            self.warnings.append("DEMO MODE: generic COCO-style model; not validated for orchard fruit. "
                                 "Do not use these counts for decisions.")
        else:
            if not wanted & model_names:
                raise ModelError(f"The checkpoint has no class matching fruit.species {sorted(wanted)}. "
                                 f"Model classes: {sorted(model_names)}. " + NO_DETECTOR_HELP)
            missing = wanted - model_names
            if missing:
                self.warnings.append(f"Configured species not in model classes (will never be detected): "
                                     f"{sorted(missing)}.")
        self.allowed_class_names = wanted & model_names

"""Harvest estimation. Pure functions; no model or I/O dependencies except CSV validation loading.

Equations (units: grams -> kilograms):

    visible_harvestable_count = sum_s( observed_unique_s * harvestable_probability_s )
    visible_harvest_weight_kg = sum_s( visible_harvestable_count_s * average_weight_grams_s / 1000 )
    estimated_total_tree_fruit = observed_unique / visibility_fraction     (only if configured)

The unrounded values are kept internally and in reports; ``*_rounded`` is for display only.
Nothing here is ground truth. Without calibration inputs the corresponding outputs are ``None``.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import FruitConfig, HarvestConfig
from .errors import YieldEstimationError
from .schemas import HarvestEstimate, UncertaintyInterval

logger = logging.getLogger(__name__)


def validate_average_weight_grams(value: float, species: str = "") -> float:
    """Average fruit weight must be a finite positive number (grams)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise YieldEstimationError(f"average fruit weight{' for ' + species if species else ''} must be > 0 g, got {value!r}")
    return float(value)


def validate_probability(value: float, name: str = "harvestable_probability") -> float:
    """Probabilities must be finite and within [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0.0 <= value <= 1.0:
        raise YieldEstimationError(f"{name} must be within [0, 1], got {value!r}")
    return float(value)


def validate_visibility_fraction(value: float) -> float:
    """Visibility fraction must be > 0 and <= 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0.0 < value <= 1.0:
        raise YieldEstimationError(f"visibility_fraction must be > 0 and <= 1, got {value!r}")
    return float(value)


def harvest_weight_kg(harvestable_count: float, average_weight_grams: float) -> float:
    """``count * grams / 1000`` with validation."""
    if harvestable_count < 0:
        raise YieldEstimationError("harvestable_count cannot be negative")
    return harvestable_count * validate_average_weight_grams(average_weight_grams) / 1000.0


def load_validation_pairs(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load field-validation pairs (``predicted_weight_kg``, ``actual_weight_kg``) from CSV."""
    p = Path(path)
    if not p.is_file():
        raise YieldEstimationError(f"Validation pairs file not found: '{p.name}'")
    df = pd.read_csv(p)
    missing = {"predicted_weight_kg", "actual_weight_kg"} - set(df.columns)
    if missing:
        raise YieldEstimationError(f"Validation CSV is missing columns: {sorted(missing)}")
    df = df[["predicted_weight_kg", "actual_weight_kg"]].apply(pd.to_numeric, errors="coerce").dropna()
    return df["predicted_weight_kg"].to_numpy(float), df["actual_weight_kg"].to_numpy(float)


def empirical_prediction_interval(point_estimate: float, predicted: Sequence[float], actual: Sequence[float],
                                  coverage: float = 0.90, min_samples: int = 10,
                                  unit: str = "kg") -> Optional[UncertaintyInterval]:
    """Empirical prediction interval from past (predicted, actual) pairs.

    Uses quantiles of the ratio ``actual / predicted`` observed during field validation and
    applies them to the new point estimate. Returns ``None`` when fewer than ``min_samples`` valid
    pairs exist: no interval is invented. Assumes the new scene resembles the validation scenes
    (same orchard, camera, season); it is not a guarantee for other conditions.
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        raise YieldEstimationError("predicted and actual must have the same length")
    mask = np.isfinite(p) & np.isfinite(a) & (p > 0) & (a >= 0)
    if int(mask.sum()) < min_samples:
        return None
    ratios = a[mask] / p[mask]
    lo, hi = np.quantile(ratios, [(1 - coverage) / 2, 1 - (1 - coverage) / 2])
    return UncertaintyInterval(lower=float(point_estimate * lo), upper=float(point_estimate * hi),
                               coverage=coverage, method="empirical_ratio_quantiles",
                               n_samples=int(mask.sum()), unit=unit)


def estimate_harvest(unique_visible_by_species: Mapping[str, int], harvest_cfg: HarvestConfig,
                     fruit_cfg: FruitConfig) -> HarvestEstimate:
    """Compute the harvest estimate from per-species unique visible counts and configuration."""
    warnings: list[str] = []
    counts = {s: int(n) for s, n in unique_visible_by_species.items() if n}
    total_visible = sum(counts.values())
    if any(n < 0 for n in counts.values()):
        raise YieldEstimationError("counts cannot be negative")

    # --- validated parameters ------------------------------------------------------------
    weights: dict[str, float] = {}
    for s in counts:
        w = fruit_cfg.average_weight_grams.get(s)
        if w is None:
            warnings.append(f"No average fruit weight configured for '{s}': weight not estimated for this species.")
        else:
            weights[s] = validate_average_weight_grams(w, s)

    probs: dict[str, float] = {}
    for s in counts:
        p = harvest_cfg.harvestable_probability_by_species.get(s, harvest_cfg.harvestable_probability)
        if p is not None:
            probs[s] = validate_probability(p, f"harvestable_probability[{s}]")
    missing_prob = [s for s in counts if s not in probs]
    if missing_prob:
        warnings.append("No calibrated harvestable probability for: " + ", ".join(sorted(missing_prob))
                        + ". Harvestable count and weight are NOT estimated for them (no arbitrary default is applied).")

    vis = harvest_cfg.visibility_fraction
    if vis is not None:
        vis = validate_visibility_fraction(vis)
    else:
        warnings.append("Visibility fraction not configured: total tree yield cannot be reliably inferred. "
                        "Only observed (visible) quantities are reported.")

    # --- visible quantities ---------------------------------------------------------------
    computable = [s for s in counts if s in probs]
    visible_hc: Optional[float] = None
    visible_w: Optional[float] = None
    if computable and not missing_prob:
        visible_hc = sum(counts[s] * probs[s] for s in computable)
        if all(s in weights for s in computable):
            visible_w = sum(harvest_weight_kg(counts[s] * probs[s], weights[s]) for s in computable)
    elif total_visible == 0:
        visible_hc, visible_w = 0.0, (0.0 if harvest_cfg.harvestable_probability is not None else None)
        if visible_w is None and fruit_cfg.average_weight_grams:
            visible_w = None

    # --- extrapolated quantities ------------------------------------------------------------
    tree_total = tree_hc = tree_w = None
    if vis is not None:
        tree_total = total_visible / vis
        if visible_hc is not None:
            tree_hc = visible_hc / vis
        if visible_w is not None:
            tree_w = visible_w / vis
        warnings.append(f"Visibility fraction {vis:g} is a configured assumption, not a measurement by this system.")

    # --- uncertainty -------------------------------------------------------------------------
    unc: Optional[UncertaintyInterval] = None
    note = "No uncertainty interval: no field-validation data configured (harvest.validation_pairs_path)."
    if harvest_cfg.validation_pairs_path:
        basis_value = tree_w if harvest_cfg.uncertainty_basis == "tree" else visible_w
        if basis_value is None:
            note = "Uncertainty interval not computed: the selected basis quantity is unavailable."
        else:
            pred, act = load_validation_pairs(harvest_cfg.validation_pairs_path)
            unc = empirical_prediction_interval(basis_value, pred, act, harvest_cfg.interval_coverage,
                                                harvest_cfg.min_validation_samples)
            if unc is None:
                note = (f"Uncertainty interval not computed: fewer than {harvest_cfg.min_validation_samples} "
                        "valid validation pairs.")
            else:
                note = (f"Empirical {harvest_cfg.interval_coverage:.0%} prediction interval for "
                        f"{harvest_cfg.uncertainty_basis}-basis harvest weight from {unc.n_samples} field-validation "
                        "pairs; valid only for conditions similar to the validation data.")

    method = ("observed_unique_visible * harvestable_probability * average_weight_grams / 1000"
              + (" ; tree totals divided by visibility_fraction" if vis is not None else ""))
    return HarvestEstimate(
        observed_unique_visible_fruit=total_visible,
        harvestable_probability_used=probs or None,
        average_weight_grams_used=weights,
        visible_harvestable_count=visible_hc,
        visible_harvestable_count_rounded=None if visible_hc is None else int(round(visible_hc)),
        visible_harvest_weight_kg=visible_w, visibility_fraction=vis,
        estimated_total_tree_fruit=tree_total, tree_harvestable_count=tree_hc, tree_harvest_weight_kg=tree_w,
        uncertainty=unc, uncertainty_note=note, method=method, warnings=warnings)

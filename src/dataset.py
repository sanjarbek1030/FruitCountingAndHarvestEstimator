"""Group-aware dataset splitting to avoid leakage between near-identical frames/images."""
from __future__ import annotations

import random
from typing import Mapping, Sequence

from .errors import FruitCounterError


def split_groups(groups: Sequence[str], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                 seed: int = 42) -> dict[str, str]:
    """Assign whole groups (tree / recording session / orchard block) to train/val/test.

    Every image of a group lands in the same split, so adjacent video frames or views of one tree never
    straddle splits. With very few groups, splits can be empty: collect more groups rather than
    splitting inside a group.
    """
    if abs(sum(ratios) - 1.0) > 1e-6 or any(r < 0 for r in ratios):
        raise FruitCounterError("ratios must be non-negative and sum to 1")
    uniq = sorted(set(groups))
    if len(uniq) < 3:
        raise FruitCounterError(f"Need >= 3 distinct groups for train/val/test, got {len(uniq)}")
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise FruitCounterError("Not enough groups for the requested ratios")
    out = {}
    for i, g in enumerate(uniq):
        out[g] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def assign_images(image_to_group: Mapping[str, str], **kw) -> dict[str, str]:
    """Map image name -> split using :func:`split_groups` on its group."""
    g2s = split_groups(list(image_to_group.values()), **kw)
    return {img: g2s[g] for img, g in image_to_group.items()}

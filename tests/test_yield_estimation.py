import math

import pytest

from src.config import FruitConfig, HarvestConfig
from src.errors import YieldEstimationError
from src.yield_estimation import (empirical_prediction_interval, estimate_harvest, harvest_weight_kg,
                                  validate_average_weight_grams, validate_probability,
                                  validate_visibility_fraction)

FRUIT = FruitConfig(species=["apple"], average_weight_grams={"apple": 180})


def test_weight_calculation_units():
    # 100 harvestable * 180 g / 1000 = 18 kg
    assert harvest_weight_kg(100, 180) == pytest.approx(18.0)


def test_end_to_end_estimate():
    e = estimate_harvest({"apple": 100}, HarvestConfig(harvestable_probability=0.8), FRUIT)
    assert e.visible_harvestable_count == pytest.approx(80.0)
    assert e.visible_harvest_weight_kg == pytest.approx(14.4)
    assert e.estimated_total_tree_fruit is None and e.tree_harvest_weight_kg is None
    assert any("cannot be reliably inferred" in w for w in e.warnings)


def test_unrounded_kept_rounded_for_display():
    e = estimate_harvest({"apple": 7}, HarvestConfig(harvestable_probability=0.5), FRUIT)
    assert e.visible_harvestable_count_rounded == round(3.5)
    assert e.visible_harvestable_count == pytest.approx(3.5)


@pytest.mark.parametrize("bad", [0, -1, -180.0, float("nan"), float("inf"), True])
def test_bad_weights_rejected(bad):
    with pytest.raises(YieldEstimationError):
        validate_average_weight_grams(bad)


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), None])
def test_bad_probabilities_rejected(bad):
    with pytest.raises(YieldEstimationError):
        validate_probability(bad)


@pytest.mark.parametrize("bad", [0, -0.5, 1.2, float("nan")])
def test_bad_visibility_rejected(bad):
    with pytest.raises(YieldEstimationError):
        validate_visibility_fraction(bad)


def test_visibility_correction_is_disclosed():
    e = estimate_harvest({"apple": 50}, HarvestConfig(harvestable_probability=1.0, visibility_fraction=0.5), FRUIT)
    assert e.estimated_total_tree_fruit == pytest.approx(100.0)
    assert e.tree_harvest_weight_kg == pytest.approx(18.0)
    assert any("configured assumption" in w for w in e.warnings)


def test_no_probability_means_no_harvest_numbers():
    e = estimate_harvest({"apple": 10}, HarvestConfig(), FRUIT)
    assert e.visible_harvestable_count is None and e.visible_harvest_weight_kg is None
    assert e.observed_unique_visible_fruit == 10


def test_missing_species_weight_gives_count_but_no_weight():
    e = estimate_harvest({"pear": 10}, HarvestConfig(harvestable_probability=0.5), FRUIT)
    assert e.visible_harvestable_count == 5.0 and e.visible_harvest_weight_kg is None
    assert any("No average fruit weight" in w for w in e.warnings)


def test_per_species_probability_and_weight():
    fruit = FruitConfig(species=["apple", "pear"], average_weight_grams={"apple": 200, "pear": 100})
    h = HarvestConfig(harvestable_probability=0.5, harvestable_probability_by_species={"pear": 1.0})
    e = estimate_harvest({"apple": 10, "pear": 10}, h, fruit)
    assert e.visible_harvest_weight_kg == pytest.approx(10 * 0.5 * 0.2 + 10 * 1.0 * 0.1)


def test_uncalibrated_uncertainty_is_not_invented():
    e = estimate_harvest({"apple": 10}, HarvestConfig(harvestable_probability=0.8), FRUIT)
    assert e.uncertainty is None and "No uncertainty interval" in e.uncertainty_note


def test_interval_requires_enough_samples():
    assert empirical_prediction_interval(10.0, [1, 2, 3], [1, 2, 3], min_samples=10) is None


def test_interval_from_validation_pairs():
    pred = [10.0] * 20
    actual = [9.0, 11.0] * 10
    u = empirical_prediction_interval(100.0, pred, actual, coverage=0.9, min_samples=10)
    assert u.lower == pytest.approx(90.0) and u.upper == pytest.approx(110.0) and u.n_samples == 20


def test_validation_csv_roundtrip(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("predicted_weight_kg,actual_weight_kg\n" + "\n".join(f"10,{9 + (i % 3)}" for i in range(12)))
    h = HarvestConfig(harvestable_probability=1.0, validation_pairs_path=str(p), min_validation_samples=10)
    e = estimate_harvest({"apple": 100}, h, FruitConfig(species=["apple"], average_weight_grams={"apple": 100}))
    assert e.uncertainty is not None and e.uncertainty.lower <= e.visible_harvest_weight_kg <= e.uncertainty.upper + 1e-9

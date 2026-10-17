import pytest

from src.config import config_from_dict, load_config
from src.errors import ConfigError


def test_missing_config_file_gives_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_repository_config_is_valid():
    cfg = load_config("config.yaml")
    assert cfg.model.path == "models/best.pt"
    assert cfg.harvest.harvestable_probability is None   # no invented default


def test_invalid_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(p)


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        config_from_dict({"model": {"confidence": 0.3}})


@pytest.mark.parametrize("bad", [0, -5, -0.1])
def test_nonpositive_weight_rejected(bad):
    with pytest.raises(ConfigError, match="positive"):
        config_from_dict({"fruit": {"species": ["apple"], "average_weight_grams": {"apple": bad}}})


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_harvestable_probability_rejected(bad):
    with pytest.raises(ConfigError):
        config_from_dict({"harvest": {"harvestable_probability": bad}})


@pytest.mark.parametrize("bad", [0, 0.0, -0.2, 1.01])
def test_invalid_visibility_fraction_rejected(bad):
    with pytest.raises(ConfigError, match="visibility_fraction"):
        config_from_dict({"harvest": {"visibility_fraction": bad}})


def test_visibility_fraction_one_is_allowed():
    assert config_from_dict({"harvest": {"visibility_fraction": 1.0}}).harvest.visibility_fraction == 1.0


def test_zone_validation():
    with pytest.raises(ConfigError, match="zone"):
        config_from_dict({"counting": {"zones": [{"name": "a", "box": [0.5, 0, 0.2, 1]}]}})
    with pytest.raises(ConfigError, match="requires"):
        config_from_dict({"counting": {"count_only_in_zones": True}})


def test_bad_device_and_conf():
    with pytest.raises(ConfigError):
        config_from_dict({"model": {"device": "tpu"}})
    with pytest.raises(ConfigError):
        config_from_dict({"model": {"conf": 2}})

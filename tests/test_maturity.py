import numpy as np

from src.config import MaturityConfig
from src.maturity import (MaturityPrediction, NullMaturityClassifier, annotate_maturity, apply_prediction,
                          build_classifier)
from tests.helpers import det

CFG = MaturityConfig(enabled=True, confidence_threshold=0.6)


class Fake:
    available = True
    def __init__(self, preds): self.preds = preds
    def classify(self, crops): return self.preds[:len(crops)]


def test_null_classifier_never_fabricates():
    frame = np.zeros((100, 100, 3), np.uint8)
    out = annotate_maturity(frame, [det()], NullMaturityClassifier(), CFG)
    assert out[0].maturity_label is None and out[0].maturity_confidence is None
    assert build_classifier(MaturityConfig(enabled=False)).available is False


def test_low_confidence_becomes_uncertain():
    d = apply_prediction(det(), MaturityPrediction("ripe", 0.4), CFG)
    assert d.maturity_label == "uncertain" and d.maturity_confidence == 0.4


def test_condition_label_routed():
    d = apply_prediction(det(), MaturityPrediction("damaged", 0.9), CFG)
    assert d.condition_label == "damaged" and d.maturity_label == "not_assessed"


def test_confident_maturity_label():
    d = apply_prediction(det(), MaturityPrediction("ripe", 0.95), CFG)
    assert d.maturity_label == "ripe"


def test_tiny_crop_is_uncertain_not_ripe():
    frame = np.zeros((100, 100, 3), np.uint8)
    out = annotate_maturity(frame, [det(10, 10, 14, 14)], Fake([MaturityPrediction("ripe", 0.99)]), CFG)
    assert out[0].maturity_label == "uncertain"


def test_skip_flag_leaves_detection_untouched():
    frame = np.zeros((100, 100, 3), np.uint8)
    out = annotate_maturity(frame, [det()], Fake([MaturityPrediction("ripe", 0.99)]), CFG, skip=[True])
    assert out[0].maturity_label is None


def test_missing_maturity_model_gives_clear_error():
    import pytest
    from src.errors import ModelError
    with pytest.raises(ModelError, match="not found"):
        build_classifier(MaturityConfig(enabled=True, model_path="nope.pt"))

"""Smoke test of the Streamlit UI using Streamlit's AppTest (no model, injected result)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.config import AppConfig, FruitConfig, HarvestConfig  # noqa: E402
from src.pipeline import AnalysisPipeline  # noqa: E402
from tests.helpers import FakeDetector, det  # noqa: E402


def test_dashboard_renders_empty_then_with_result(tmp_path):
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert any("Run an analysis" in i.value for i in at.info)

    cfg = AppConfig()
    cfg.fruit = FruitConfig(species=["apple"], average_weight_grams={"apple": 180})
    cfg.harvest = HarvestConfig(harvestable_probability=0.8)
    cfg.output.annotated_dir, cfg.output.report_dir = str(tmp_path / "a"), str(tmp_path / "r")
    img = tmp_path / "i.png"
    cv2.imwrite(str(img), np.full((100, 100, 3), 100, np.uint8))
    pipe = AnalysisPipeline(cfg, FakeDetector({0: [det(0, 0, 30, 30), det(50, 50, 90, 90)]}),
                            model_info={"checkpoint_name": "fake"}, demo_mode=False)
    at.session_state["result"] = pipe.analyze_image(img)
    at.session_state["pipe"] = pipe
    at.run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Estimated unique visible fruit"] == "2"

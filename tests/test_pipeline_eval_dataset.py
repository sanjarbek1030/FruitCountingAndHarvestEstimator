import cv2
import numpy as np
import pytest

from src.config import AppConfig, FruitConfig, HarvestConfig
from src.dataset import assign_images, split_groups
from src.errors import FruitCounterError, InputError
from src.evaluation import count_errors, evaluate_csv
from src.pipeline import AnalysisPipeline, read_image
from src.review import flag_detection
from src.schemas import ReviewCorrections
from tests.helpers import FakeDetector, det


def _cfg(tmp_path):
    cfg = AppConfig()
    cfg.fruit = FruitConfig(species=["apple"], average_weight_grams={"apple": 200})
    cfg.harvest = HarvestConfig(harvestable_probability=0.5)
    cfg.output.annotated_dir = str(tmp_path / "ann"); cfg.output.report_dir = str(tmp_path / "rep")
    return cfg.validate()


def _img(tmp_path):
    p = tmp_path / "t.png"
    cv2.imwrite(str(p), np.full((100, 100, 3), 120, np.uint8))
    return p


def test_image_pipeline_and_review(tmp_path):
    ds = [det(0, 0, 20, 20, did="f000000_d000"), det(30, 0, 50, 20, did="f000000_d001"),
          det(60, 60, 90, 90, did="f000000_d002", conf=0.1)]
    pipe = AnalysisPipeline(_cfg(tmp_path), FakeDetector({0: ds}), model_info={"checkpoint_name": "fake"},
                            demo_mode=False)
    res = pipe.analyze_image(_img(tmp_path))
    assert res.counts.raw_detections == 3 and res.counts.estimated_unique_visible_fruit == 2
    assert res.harvest.visible_harvest_weight_kg == pytest.approx(2 * 0.5 * 0.2)
    assert res.annotated_path.exists() and res.report_paths["json"].exists()
    res2 = pipe.recompute(res, ReviewCorrections({"f000000_d000"}, set(), {}))
    assert res2.counts.estimated_unique_visible_fruit == 1 and res2.counts.manual_rejections == 1
    assert res.counts.estimated_unique_visible_fruit == 2            # original untouched


def test_unreadable_image_error(tmp_path):
    p = tmp_path / "x.jpg"; p.write_bytes(b"not an image")
    with pytest.raises(InputError, match="not a decodable"):
        read_image(p)
    with pytest.raises(InputError, match="not found"):
        read_image(tmp_path / "missing.jpg")


def test_count_errors():
    r = count_errors([10, 12, 8], [10, 10, 10])
    assert r["mae"] == pytest.approx(4 / 3) and r["rmse"] == pytest.approx(np.sqrt(8 / 3)) and r["bias"] == pytest.approx(0)


def test_evaluate_csv_grouped(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("pred,truth,light\n10,10,sun\n8,10,shade\n12,10,shade\n")
    df = evaluate_csv(p, "pred", "truth", "light")
    assert set(df["group"]) == {"ALL", "sun", "shade"}
    with pytest.raises(FruitCounterError):
        evaluate_csv(p, "nope", "truth")


def test_group_split_has_no_leakage():
    groups = {f"img{i}": f"tree{i // 5}" for i in range(50)}          # 10 trees x 5 images
    split = assign_images(groups, seed=1)
    by_tree = {}
    for img, s in split.items():
        by_tree.setdefault(groups[img], set()).add(s)
    assert all(len(v) == 1 for v in by_tree.values())
    assert {"train", "val", "test"} == set(split.values())
    with pytest.raises(FruitCounterError):
        split_groups(["a", "b"])


def test_flag_detection():
    assert "low_confidence" in flag_detection(det(conf=0.2), 0.4)
    assert "very_small_box" in flag_detection(det(0, 0, 5, 5), 0.4)
    assert flag_detection(det(), 0.4) == []

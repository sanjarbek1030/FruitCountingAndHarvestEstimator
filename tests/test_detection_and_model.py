import sys
import types

import numpy as np
import pytest

from src.config import ModelConfig, TrackingConfig
from src.detection import FruitDetector, assign_detection_ids, detections_from_arrays, tile_origins
from src.errors import ModelError
from src.model_manager import ModelManager, resolve_device

NAMES = {0: "apple", 1: "leaf"}


def test_arrays_to_detections_filters_classes_and_ids():
    xyxy = np.array([[10, 10, 50, 50], [0, 0, 5, 5], [20, 20, 40, 40]], float)
    out = detections_from_arrays(xyxy, np.array([.9, .8, .7]), np.array([0, 1, 0]),
                                 np.array([3, 4, np.nan]), NAMES, 2, 0.5, {"apple"}, offset=(100, 200))
    assert len(out) == 2                       # 'leaf' dropped
    assert out[0].track_id == 3 and out[1].track_id is None
    assert out[0].bbox_x1 == 110 and out[0].bbox_y1 == 210   # offset applied
    assert [d.detection_id for d in assign_detection_ids(out, 2)] == ["f000002_d000", "f000002_d001"]


def test_tile_origins_cover_image():
    tiles = tile_origins(1000, 700, 400, 0.25)
    assert max(t[2] for t in tiles) == 1000 and max(t[3] for t in tiles) == 700
    assert all(t[2] - t[0] <= 400 and t[3] - t[1] <= 400 for t in tiles)
    assert tile_origins(300, 200, 400, 0.25) == [(0, 0, 300, 200)]


def test_resolve_device_fallbacks():
    cuda_off = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    dev, note = resolve_device("cuda:0", cuda_off)
    assert dev == "cpu" and "Falling back to CPU" in note
    assert resolve_device("auto", cuda_off)[0] == "cpu"
    cuda_on = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True,
                                                               get_device_name=lambda i: "GPU"))
    assert resolve_device("auto", cuda_on)[0] == "cuda:0"


def test_missing_checkpoint_gives_actionable_error(tmp_path):
    mm = ModelManager(ModelConfig(path=str(tmp_path / "models" / "best.pt")), ["apple"])
    with pytest.raises(ModelError, match="No trained fruit detector"):
        mm.load()


def _install_fake_ultralytics(monkeypatch, names):
    class FakeYOLO:
        def __init__(self, path): self.names = names; self.path = path
    mod = types.ModuleType("ultralytics"); mod.YOLO = FakeYOLO; mod.__version__ = "0.0-test"
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def test_model_without_matching_species_is_rejected(tmp_path, monkeypatch):
    ck = tmp_path / "best.pt"; ck.write_bytes(b"x")
    _install_fake_ultralytics(monkeypatch, {0: "person", 1: "car"})
    with pytest.raises(ModelError, match="no class matching"):
        ModelManager(ModelConfig(path=str(ck), device="cpu"), ["apple"]).load()


def test_real_mode_loads_once_and_reports_info(tmp_path, monkeypatch):
    ck = tmp_path / "best.pt"; ck.write_bytes(b"weights")
    _install_fake_ultralytics(monkeypatch, {0: "apple"})
    mm = ModelManager(ModelConfig(path=str(ck), device="cpu"), ["apple"]).load()
    first = mm.model
    mm.load()
    assert mm.model is first and mm.info["checkpoint_name"] == "best.pt" and not mm.demo_mode
    assert mm.info["checkpoint_sha256_prefix"] and mm.allowed_class_names == {"apple"}


def test_demo_mode_is_explicit_and_warns(tmp_path, monkeypatch):
    ck = tmp_path / "yolo.pt"; ck.write_bytes(b"w")
    _install_fake_ultralytics(monkeypatch, {0: "person", 47: "apple"})
    mm = ModelManager(ModelConfig(path=str(ck), device="cpu", demo_mode=True), ["apple", "pear"]).load()
    assert mm.demo_mode and any("DEMO MODE" in w for w in mm.warnings) and mm.allowed_class_names == {"apple"}


def test_detector_parses_ultralytics_style_results():
    class T:
        def __init__(s, a): s.a = np.asarray(a)
        def cpu(s): return s
        def numpy(s): return s.a
    class B:
        xyxy = T([[10, 10, 50, 50]]); conf = T([0.8]); cls = T([0.0]); id = T([5.0])
        def __len__(s): return 1
    class Model:
        names = NAMES
        def predict(self, frame, **kw): return [types.SimpleNamespace(boxes=B())]
        def track(self, frame, **kw):
            assert kw["persist"] is True and kw["tracker"] == "bytetrack.yaml"
            return [types.SimpleNamespace(boxes=B())]
    mgr = types.SimpleNamespace(model=Model(), class_names=NAMES, allowed_class_names={"apple"}, device="cpu")
    det_ = FruitDetector(mgr, ModelConfig(), TrackingConfig())
    frame = np.zeros((100, 100, 3), np.uint8)
    d = det_.detect(frame, 4, 0.1)[0]
    assert d.class_name == "apple" and d.track_id is None and d.detection_id == "f000004_d000"
    assert det_.track(frame, 5, 0.2)[0].track_id == 5


def test_inference_failure_is_wrapped():
    class Model:
        def predict(self, *a, **k): raise RuntimeError("cuda oom")
    mgr = types.SimpleNamespace(model=Model(), class_names=NAMES, allowed_class_names={"apple"}, device="cpu")
    with pytest.raises(ModelError, match="Inference failed"):
        FruitDetector(mgr, ModelConfig(), TrackingConfig()).detect(np.zeros((10, 10, 3), np.uint8))

import cv2
import pytest

from src.config import AppConfig
from src.errors import VideoError
from src.pipeline import AnalysisPipeline
from src.video_processor import VideoProcessor, read_metadata
from tests.helpers import FakeDetector, det, write_video


def make_cfg(**video):
    cfg = AppConfig()
    cfg.tracking.min_confirm_observations = 3
    for k, v in video.items():
        setattr(cfg.video, k, v)
    return cfg.validate()


def test_missing_file(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        VideoProcessor(make_cfg(), FakeDetector({})).process(tmp_path / "x.mp4")


def test_invalid_video_raises_and_releases(tmp_path, monkeypatch):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video" * 50)
    released = {"cap": 0, "writer": 0}
    real_cap = cv2.VideoCapture

    class SpyCap:
        def __init__(self, p):
            self._c = real_cap(p)
        def __getattr__(self, n):
            return getattr(self._c, n)
        def release(self):
            released["cap"] += 1
            self._c.release()
    monkeypatch.setattr(cv2, "VideoCapture", SpyCap)
    with pytest.raises(VideoError):
        VideoProcessor(make_cfg(), FakeDetector({})).process(bad, tmp_path / "o.mp4")
    assert released["cap"] == 1             # resources released even on failure


def test_exception_mid_stream_releases_resources(tmp_path, monkeypatch):
    vid = write_video(tmp_path / "v.mp4", 10)
    released = {"cap": 0, "writer": 0}
    rc, rw = cv2.VideoCapture, cv2.VideoWriter

    class Cap:
        def __init__(s, p): s._c = rc(p)
        def __getattr__(s, n): return getattr(s._c, n)
        def release(s): released["cap"] += 1; s._c.release()

    class Wr:
        def __init__(s, *a): s._w = rw(*a)
        def __getattr__(s, n): return getattr(s._w, n)
        def release(s): released["writer"] += 1; s._w.release()
    monkeypatch.setattr(cv2, "VideoCapture", Cap)
    monkeypatch.setattr(cv2, "VideoWriter", Wr)
    det_ = FakeDetector({}, fail_frames=set(range(10)))
    with pytest.raises(VideoError, match="consecutive"):
        VideoProcessor(make_cfg(max_consecutive_errors=3), det_).process(vid, tmp_path / "o.mp4")
    assert released == {"cap": 1, "writer": 1}


def test_video_counts_tracks_not_frame_sum(tmp_path):
    vid = write_video(tmp_path / "v.mp4", 12)
    # one fruit visible in 12 frames (id 1), one visible in only 2 frames (id 2, unconfirmed)
    frames = {i: [det(10, 10, 30, 30, tid=1)] for i in range(12)}
    frames[0].append(det(40, 20, 60, 40, tid=2)); frames[1].append(det(40, 20, 60, 40, tid=2))
    cfg = make_cfg(); cfg.output.annotated_dir = str(tmp_path / "ann"); cfg.output.report_dir = str(tmp_path / "rep")
    pipe = AnalysisPipeline(cfg, FakeDetector(frames), model_info={"checkpoint_name": "fake"}, demo_mode=False)
    res = pipe.analyze_video(vid, tmp_path / "out.mp4")
    assert res.counts.raw_detections == 14                      # per-frame sum
    assert res.counts.confirmed_tracks == 1                     # distinct fruit
    assert res.counts.estimated_unique_visible_fruit == 1
    assert res.annotated_path.exists() and res.annotated_path.stat().st_size > 0
    assert res.report["video"]["frames_read"] == 12 and res.report["video"]["fps_source"] == "metadata"
    assert res.report_paths["tracks_csv"].exists()


def test_frame_stride_warning_and_skipped_frames(tmp_path):
    vid = write_video(tmp_path / "v.mp4", 12)
    frames = {i: [det(10, 10, 30, 30, tid=1)] for i in range(12)}
    run = VideoProcessor(make_cfg(frame_stride=3), FakeDetector(frames)).process(vid, tmp_path / "o.mp4")
    assert run.frames_processed == 4 and run.frames_skipped == 8
    assert any("frame_stride=3" in w for w in run.warnings)


def test_fps_fallback_recorded():
    class Cap:
        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 0.0, cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48,
                    cv2.CAP_PROP_FRAME_COUNT: 0}.get(prop, 0)
    import numpy as np
    meta, warns = read_metadata(Cap(), np.zeros((48, 64, 3), np.uint8), 25.0)
    assert meta.fps == 25.0 and meta.fps_source == "fallback" and meta.frame_count is None
    assert any("fallback" in w for w in warns)


def test_cancel_marks_partial(tmp_path):
    vid = write_video(tmp_path / "v.mp4", 12)
    calls = {"n": 0}
    def stop():
        calls["n"] += 1
        return calls["n"] > 4
    run = VideoProcessor(make_cfg(), FakeDetector({})).process(vid, None, should_stop=stop)
    assert run.terminated_early and run.frames_read == 4 and any("PARTIAL" in w for w in run.warnings)

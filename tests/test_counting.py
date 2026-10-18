import math

from src.config import CountingConfig
from src.counting import (count_image, count_video, filter_by_confidence, suppress_duplicates,
                          validate_detections)
from src.schemas import ReviewCorrections, TrackSummary
from tests.helpers import det

CFG = CountingConfig(min_confidence=0.25)


def test_empty_list_returns_zero():
    r, kept = count_image([], 100, 100, CFG)
    assert r.estimated_unique_visible_fruit == 0 and kept == [] and r.confidence is None


def test_single_detection_returns_one():
    r, _ = count_image([det()], 100, 100, CFG)
    assert r.estimated_unique_visible_fruit == 1 and r.by_species == {"apple": 1}


def test_several_independent_detections():
    ds = [det(0, 0, 20, 20, did="a"), det(30, 0, 50, 20, did="b"), det(60, 60, 90, 90, did="c")]
    r, _ = count_image(ds, 100, 100, CFG)
    assert r.estimated_unique_visible_fruit == 3


def test_touching_fruit_are_not_merged():
    # adjacent boxes with close centres but tiny IoU must stay separate
    ds = [det(0, 0, 40, 40, did="a"), det(38, 0, 78, 40, did="b")]
    r, _ = count_image(ds, 100, 100, CFG)
    assert r.estimated_unique_visible_fruit == 2


def test_near_identical_boxes_deduplicated():
    ds = [det(10, 10, 50, 50, conf=0.9, did="a"), det(11, 11, 50, 50, conf=0.8, did="b")]
    r, kept = count_image(ds, 100, 100, CFG)
    assert r.estimated_unique_visible_fruit == 1 and kept[0].detection_id == "a"


def test_invalid_boxes_rejected():
    bad = [det(50, 50, 10, 10, did="inv"), det(float("nan"), 0, 5, 5, did="nan"),
           det(200, 200, 300, 300, did="outside"), det(0, 0, 10, 10, conf=1.7, did="conf"),
           det(0, 0, 1, 1, did="tiny")]
    valid, rej = validate_detections(bad, 100, 100)
    assert valid == [] and {r.detection_id for r in rej} == {"inv", "nan", "outside", "conf", "tiny"}


def test_partially_outside_box_is_clipped_not_rejected():
    valid, rej = validate_detections([det(-10, 10, 40, 50)], 100, 100)
    assert len(valid) == 1 and valid[0].bbox_x1 == 0 and not rej


def test_confidence_filtering():
    kept, rej = filter_by_confidence([det(conf=0.2, did="lo"), det(conf=0.3, did="hi")], 0.25)
    assert [d.detection_id for d in kept] == ["hi"] and rej[0].detection_id == "lo"
    r, _ = count_image([det(conf=0.2)], 100, 100, CFG)
    assert r.estimated_unique_visible_fruit == 0 and r.rejected_detections == 1


def test_unavailable_maturity_is_not_fabricated():
    r, _ = count_image([det(did="a"), det(60, 60, 90, 90, did="b")], 100, 100, CFG)
    assert r.by_maturity == {"unavailable": 2} and "ripe" not in r.by_maturity
    assert any("Maturity classification unavailable" in w for w in r.warnings)


def test_maturity_and_uncertain_counts():
    ds = [det(0, 0, 20, 20, did="a", maturity="ripe"), det(30, 0, 50, 20, did="b", maturity="uncertain"),
          det(60, 60, 90, 90, did="c", maturity="not_assessed", condition="damaged")]
    r, _ = count_image(ds, 100, 100, CFG)
    assert r.uncertain_classifications == 1 and r.by_condition["damaged"] == 1 and r.by_maturity["ripe"] == 1


def test_manual_review_tracked_separately():
    ds = [det(0, 0, 20, 20, did="a"), det(30, 0, 50, 20, did="b")]
    r, _ = count_image(ds, 100, 100, CFG, corrections=ReviewCorrections({"a"}, set(), {"apple": 2}))
    assert r.estimated_unique_visible_fruit == 3 and r.manual_additions == 2 and r.manual_rejections == 1


def test_zones():
    cfg = CountingConfig(zones=[{"name": "left", "box": [0, 0, 0.5, 1]}], count_only_in_zones=True)
    ds = [det(0, 0, 20, 20, did="a"), det(70, 0, 90, 20, did="b")]
    r, _ = count_image(ds, 100, 100, cfg)
    assert r.estimated_unique_visible_fruit == 1 and r.by_zone == {"left": 1}


def test_suppress_duplicates_is_class_aware():
    kept = suppress_duplicates([det(cls="apple", did="a"), det(cls="pear", did="b")], 0.8)
    assert len(kept) == 2


def _track(tid, n, min_confirm=3, conf=0.8):
    return TrackSummary(track_id=tid, class_name="apple", first_frame=0, last_frame=n, first_timestamp_seconds=0,
                        last_timestamp_seconds=1, n_observations=n, frames_observed=list(range(n)),
                        mean_confidence=conf, mean_center_x=10, mean_center_y=10, confirmed=n >= min_confirm)


def test_video_count_uses_tracks_not_frame_sum():
    tracks = [_track(1, 100), _track(2, 50), _track(3, 1)]     # 151 raw detections, 2 confirmed fruit
    r, _ = count_video(tracks, 151, 100, 100, "fixed", CFG)
    assert r.raw_detections == 151 and r.confirmed_tracks == 2 and r.estimated_unique_visible_fruit == 2


def test_moving_camera_is_labelled_best_effort():
    r, _ = count_video([_track(1, 10)], 10, 100, 100, "moving", CFG)
    assert "best_effort" in r.estimate_basis and any("Moving camera" in w for w in r.warnings)


def test_video_manual_rejection():
    r, _ = count_video([_track(1, 10), _track(2, 10)], 20, 100, 100, "fixed", CFG,
                       corrections=ReviewCorrections(rejected_track_ids={2}))
    assert r.estimated_unique_visible_fruit == 1 and r.manual_rejections == 1

from src.counting import count_video
from src.config import CountingConfig
from src.tracking import IoUTracker, TrackRegistry
from tests.helpers import det


def run(tracker, frames):
    out = []
    for i, ds in enumerate(frames):
        out.append(tracker.update([d for d in ds]))
    return out


def test_same_fruit_in_consecutive_frames_keeps_one_id_and_counts_once():
    tr, reg = IoUTracker(0.3, 5), TrackRegistry(min_confirm_observations=3)
    for f in range(10):
        reg.update(tr.update([det(10 + f, 10, 50 + f, 50, frame=f, did=f"d{f}")]))
    s = reg.summaries()
    assert len(s) == 1 and s[0].n_observations == 10 and s[0].confirmed
    assert reg.raw_detections == 10          # raw is 10, unique is 1
    r, _ = count_video(s, reg.raw_detections, 200, 200, "fixed", CountingConfig())
    assert r.estimated_unique_visible_fruit == 1


def test_different_fruit_remain_separate():
    tr, reg = IoUTracker(0.3, 5), TrackRegistry(2)
    for f in range(4):
        reg.update(tr.update([det(10, 10, 40, 40, frame=f, did=f"a{f}"), det(100, 100, 130, 130, frame=f, did=f"b{f}")]))
    assert len(reg.summaries()) == 2 and reg.diagnostics()["confirmed_tracks"] == 2


def test_reassociation_after_short_loss():
    tr = IoUTracker(0.3, max_lost_frames=3)
    a = tr.update([det(10, 10, 50, 50, frame=0)])[0].track_id
    tr.update([]); tr.update([])                      # lost for 2 frames (<= max)
    b = tr.update([det(11, 11, 51, 51, frame=3)])[0].track_id
    assert a == b


def test_track_terminated_after_long_loss_gets_new_id():
    tr = IoUTracker(0.3, max_lost_frames=2)
    a = tr.update([det(10, 10, 50, 50, frame=0)])[0].track_id
    for _ in range(4):
        tr.update([])
    b = tr.update([det(10, 10, 50, 50, frame=5)])[0].track_id
    assert a != b                                     # documented fragmentation


def test_overlapping_detections_get_distinct_ids():
    tr = IoUTracker(0.5, 3)
    out = tr.update([det(10, 10, 50, 50, did="a"), det(40, 10, 80, 50, did="b")])
    assert out[0].track_id != out[1].track_id


def test_missing_track_ids_are_handled_and_not_confirmed():
    reg = TrackRegistry(1)
    reg.update([det(tid=None), det(60, 60, 90, 90, tid=None)])
    assert reg.summaries() == [] and reg.untracked_detections == 2 and reg.raw_detections == 2
    r, _ = count_video(reg.summaries(), reg.raw_detections, 100, 100, "fixed", CountingConfig(),
                       reg.diagnostics())
    assert r.estimated_unique_visible_fruit == 0 and any("no track id" in w for w in r.warnings)


def test_unconfirmed_short_track_not_counted():
    reg = TrackRegistry(3)
    reg.update([det(frame=0, tid=7)]); reg.update([det(frame=1, tid=7)])
    s = reg.summaries()
    assert not s[0].confirmed and reg.confirmed_count() == 0


def test_registry_records_first_last_and_frames():
    reg = TrackRegistry(1)
    for f in (3, 4, 9):
        reg.update([det(frame=f, tid=1)])
    t = reg.summaries()[0]
    assert (t.first_frame, t.last_frame, t.frames_observed) == (3, 9, [3, 4, 9])


def test_maturity_vote_per_track():
    reg = TrackRegistry(1)
    for f, m in enumerate(["ripe", "ripe", "unripe"]):
        reg.update([det(frame=f, tid=1, maturity=m)])
    assert reg.summaries()[0].maturity_label == "ripe" and reg.maturity_observation_count(1) == 3

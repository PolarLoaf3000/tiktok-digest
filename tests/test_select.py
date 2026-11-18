"""Selection rules, without HTTP and without ffmpeg.

These tests are the contract for a digest: the right number of intervals,
no overlaps, a source that is too short, and a fragment count the timeline
cannot hold as separate cuts.
"""

from __future__ import annotations

import random

import pytest

from app.errors import ClipError
from app.select import OUTPUT_FPS, select_intervals


def test_intervals_match_count_and_duration() -> None:
    intervals = select_intervals(60.0, 15.0, 5, rng=random.Random(0))
    assert len(intervals) == 5
    assert _total(intervals) == pytest.approx(15.0, abs=1e-9)
    _assert_inside(intervals, 60.0)


def test_intervals_do_not_overlap() -> None:
    generator = random.Random(1)
    for _ in range(40):
        intervals = select_intervals(120.0, 30.0, 8, rng=generator)
        _assert_no_overlap(intervals)
        _assert_inside(intervals, 120.0)
        assert _total(intervals) == pytest.approx(30.0, abs=1e-9)


def test_same_seed_is_stable_and_other_seeds_differ() -> None:
    first = select_intervals(60.0, 12.0, 4, rng=random.Random(7))
    second = select_intervals(60.0, 12.0, 4, rng=random.Random(7))
    other = select_intervals(60.0, 12.0, 4, rng=random.Random(8))
    assert first == second
    assert first != other


def test_exact_fit_covers_the_source_without_overlap() -> None:
    intervals = select_intervals(10.0, 10.0, 4, rng=random.Random(3))
    ordered = sorted(intervals)
    assert ordered[0][0] == pytest.approx(0.0, abs=1e-9)
    assert ordered[-1][1] == pytest.approx(10.0, abs=1e-9)
    for (_, end), (next_start, _) in zip(ordered, ordered[1:]):
        assert end == pytest.approx(next_start, abs=1e-9)
    assert _total(intervals) == pytest.approx(10.0, abs=1e-9)


def test_one_frame_fragments_stay_separate() -> None:
    intervals = select_intervals(10.0, 1.0, OUTPUT_FPS, rng=random.Random(1))
    assert len(intervals) == OUTPUT_FPS
    _assert_no_overlap(intervals)
    assert _total(intervals) == pytest.approx(1.0, abs=1e-9)


def test_source_too_short() -> None:
    with pytest.raises(ClipError, match="shorter"):
        select_intervals(5.0, 10.0, 3, rng=random.Random(0))


def test_source_one_frame_short() -> None:
    source = (5 * OUTPUT_FPS - 1) / OUTPUT_FPS
    with pytest.raises(ClipError, match="shorter"):
        select_intervals(source, 5.0, 2, rng=random.Random(0))


def test_too_few_non_overlapping_clips() -> None:
    # The source is far longer than the digest. The failure is the fragment
    # count: 31 cuts cannot each own a frame inside a 1s, 30 fps timeline
    # without overlapping or disappearing.
    with pytest.raises(ClipError, match="non-overlapping") as caught:
        select_intervals(120.0, 1.0, OUTPUT_FPS + 1, rng=random.Random(0))
    assert "shorter" not in str(caught.value)


def test_fragment_count_must_be_positive() -> None:
    with pytest.raises(ClipError, match="positive integer"):
        select_intervals(10.0, 5.0, 0)


def test_durations_must_be_positive() -> None:
    with pytest.raises(ClipError, match="positive"):
        select_intervals(10.0, 0.0, 2)
    with pytest.raises(ClipError, match="positive"):
        select_intervals(0.0, 5.0, 2)


def test_absurd_durations_are_rejected() -> None:
    # 1e307 and 1e308 are finite floats. Scaling either into frames overflows
    # the float range; that used to raise OverflowError and surface as a 500.
    for duration in (1e307, 1e308):
        with pytest.raises(ClipError, match="too large"):
            select_intervals(10.0, duration, 1)
        with pytest.raises(ClipError, match="too large"):
            select_intervals(duration, 5.0, 1)


def test_ordinary_long_duration_still_selects() -> None:
    # An hour is a duration a person can type. It must not take the overflow
    # path that rejects 1e307.
    intervals = select_intervals(4000.0, 3600.0, 4, rng=random.Random(2))
    assert len(intervals) == 4
    assert _total(intervals) == pytest.approx(3600.0, abs=1e-6)
    _assert_inside(intervals, 4000.0)
    _assert_no_overlap(intervals)


def _total(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _assert_inside(intervals: list[tuple[float, float]], source: float) -> None:
    for start, end in intervals:
        assert 0.0 <= start < end
        assert end <= source + 1e-9


def _assert_no_overlap(intervals: list[tuple[float, float]]) -> None:
    # Compare in frames. Second values are k/30, and a float comparison of
    # those endpoints can call a touching pair an overlap.
    spans = sorted(_frames(interval) for interval in intervals)
    for (start, end), (next_start, next_end) in zip(spans, spans[1:]):
        assert start < end
        assert end <= next_start
        assert next_start < next_end


def _frames(interval: tuple[float, float]) -> tuple[int, int]:
    start, end = interval
    return round(start * OUTPUT_FPS), round(end * OUTPUT_FPS)

import pytest

from worker.pipeline.perception.pts_resample import (
    CADENCE_NS,
    PtsGapTooLargeError,
    resample_pts,
)


def test_resample_jitter_duplicates_and_gaps() -> None:
    rows = list(
        resample_pts(
            [(10, "a"), (10 + CADENCE_NS - 1, "extra"), (10 + 3 * CADENCE_NS, "b"), (9, "old")]
        )
    )
    assert [(item.pts_ns, item.value, item.valid) for item in rows] == [
        (10, "a", 1),
        (10 + CADENCE_NS, None, 0),
        (10 + 2 * CADENCE_NS, None, 0),
        (10 + 3 * CADENCE_NS, "b", 1),
    ]


def test_resample_epoch_is_reset_by_caller() -> None:
    assert [item.valid for item in resample_pts([(100, "a")])] == [1]
    assert [item.pts_ns for item in resample_pts([(5, "b")])] == [5]


def test_resample_rejects_unbounded_gap_expansion() -> None:
    with pytest.raises(PtsGapTooLargeError):
        list(resample_pts([(0, "a"), (3 * CADENCE_NS, "b")], max_gap_rows=1))

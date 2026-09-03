from worker.pipeline.perception.pts_resample import CADENCE_NS, resample_pts


def test_resample_jitter_duplicates_and_gaps() -> None:
    rows = list(
        resample_pts(
            [(10, "a"), (10 + CADENCE_NS - 1, "extra"), (10 + 3 * CADENCE_NS, "b"), (9, "old")]
        )
    )
    assert [(item.pts_ns, item.value, item.valid) for item in rows] == [
        (10, "a", 1), (10 + CADENCE_NS, None, 0), (10 + 2 * CADENCE_NS, None, 0),
        (10 + 3 * CADENCE_NS, "b", 1),
    ]


def test_resample_epoch_is_reset_by_caller() -> None:
    assert [item.valid for item in resample_pts([(100, "a")])] == [1]
    assert [item.pts_ns for item in resample_pts([(5, "b")])] == [5]

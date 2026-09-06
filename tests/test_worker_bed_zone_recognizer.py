"""Strict bed-segmentation conversion for the on-demand overlay route."""

from __future__ import annotations

from typing import Final
from uuid import UUID

import numpy as np
import pytest

from contracts.runner import BedRunnerResult, Image, PoseRunnerResult, RunnerResult, bed_result
from worker.pipeline.output.mjpeg_server import BedZoneNotFoundError
from worker.runtime.nvidia_bed_zone_recognizer import (
    NvidiaBedZoneRecognizer,
    bed_zone_response,
)

_IMAGE: Final[Image] = np.zeros((480, 640, 3), dtype=np.uint8)


class _RunnerServingClient:
    def __init__(self, runner: object) -> None:
        self.runner = runner

    def create(self, task: str, **options: object) -> object:
        assert task == "bed"
        assert options == {"device": "cpu", "confidence": 0.05, "max_points": 16}
        return self.runner


def _recognizer(runner: object) -> NvidiaBedZoneRecognizer:
    return NvidiaBedZoneRecognizer(
        _RunnerServingClient(runner),  # type: ignore[arg-type]
        timeout_s=1.0,
    )


def test_bed_zone_recognizer_returns_all_valid_beds_in_confidence_order() -> None:
    def runner(_image: Image) -> RunnerResult:
        return bed_result(
            [
                (0, 0, 10, 10, 0.4, [[0, 0], [10, 0], [10, 10], [0, 10]]),
                (5, 5, 50, 50, 0.9, [[5, 5], [50, 5], [50, 50], [5, 50]]),
                (1, 1, 2, 2, 0.6, [[1, 1], [2, 1], [2, 2], [1, 2]]),
                (20, 20, 30, 30, 0.7, [[20, 20], [30, 20], [30, 30], [20, 30]]),
            ]
        )

    payload = _recognizer(runner)(_IMAGE)

    assert [region.polygon for region in payload.regions] == [
        ((5, 5), (50, 5), (50, 50), (5, 50)),
        ((20, 20), (30, 20), (30, 30), (20, 30)),
        ((1, 1), (2, 1), (2, 2), (1, 2)),
        ((0, 0), (10, 0), (10, 10), (0, 10)),
    ]
    assert all(region.origin == "model" for region in payload.regions)
    assert len({UUID(region.id) for region in payload.regions}) == 4
    assert payload.image_width == 640
    assert payload.image_height == 480


def test_bed_zone_recognizer_refuses_box_when_polygon_is_empty() -> None:
    def runner(_image: Image) -> RunnerResult:
        return bed_result([(10, 20, 110, 220, 0.8, [])])

    with pytest.raises(BedZoneNotFoundError):
        _recognizer(runner)(_IMAGE)


def test_bed_zone_response_excludes_invalid_candidate_without_hiding_valid_ones() -> None:
    result = bed_result(
        [
            (10, 20, 110, 220, 0.99),
            (5, 5, 50, 50, 0.7, [[5, 5], [50, 5], [50, 50], [5, 50]]),
        ]
    )

    payload = bed_zone_response(_IMAGE, result)

    assert [region.polygon for region in payload.regions] == [((5, 5), (50, 5), (50, 50), (5, 50))]


@pytest.mark.parametrize(
    "box",
    [
        (10, 20, 110, 220, 0.8),
        (10, 20, 110, 220, 0.8, []),
        (10, 20, 110, 220, 0.8, [[1, 1], [1, 1], [2, 2]]),
        (10, 20, 110, 220, 0.8, [[1, 1], [2, 2], [3, 3]]),
        (10, 20, 110, 220, 0.8, [[1, 1], [2, 1], [float("nan"), 2]]),
        (10, 20, 110, 220, 0.8, [[1, 1], [2, 1], [float("inf"), 2]]),
        (10, 20, 110, 220, 0.8, [[1, 1], [2, 1], ["3", 2]]),
        (10, 20, 110, 220, 0.8, [[1, 1], [2, 1], [640, 2]]),
        (10, 20, 110, 220, 0.8, [[1, 1], [2, 1], [3]]),
    ],
    ids=[
        "missing",
        "empty",
        "duplicate-degenerate",
        "zero-area",
        "nan",
        "infinity",
        "nonnumeric",
        "out-of-bounds",
        "malformed-point",
    ],
)
def test_bed_zone_response_refuses_absent_degenerate_or_malformed_polygon(
    box: object,
) -> None:
    result = BedRunnerResult(kind="bed", boxes=[box])  # type: ignore[list-item]

    with pytest.raises(BedZoneNotFoundError):
        bed_zone_response(_IMAGE, result)


def test_nvidia_recognizer_uses_strict_shared_response_conversion() -> None:
    def runner(_image: Image) -> RunnerResult:
        return bed_result(
            [
                (10, 20, 110, 220, 0.99),
                (2, 2, 40, 40, 0.6, [[2, 2], [40, 2], [40, 40], [2, 40]]),
            ]
        )

    recognizer = NvidiaBedZoneRecognizer(
        _RunnerServingClient(runner),  # type: ignore[arg-type]
        timeout_s=1.0,
    )

    assert recognizer(_IMAGE).regions[0].polygon == (
        (2, 2),
        (40, 2),
        (40, 40),
        (2, 40),
    )


def test_bed_zone_recognizer_accepts_a_run_method_runner_not_just_callable() -> None:
    """A runner exposing ``.run`` instead of ``__call__`` must still work."""

    class _RunOnlyRunner:
        def run(self, _image: Image) -> RunnerResult:
            return bed_result([(2, 2, 4, 4, 0.5, [[2, 2], [4, 2], [4, 4], [2, 4]])])

    payload = _recognizer(_RunOnlyRunner())(_IMAGE)

    assert payload.regions[0].polygon == ((2, 2), (4, 2), (4, 4), (2, 4))


def test_bed_zone_recognizer_applies_each_requested_confidence_to_cached_runner_output() -> None:
    def runner(_image: Image) -> RunnerResult:
        return bed_result(
            [
                (0, 0, 10, 10, 0.1, [[0, 0], [10, 0], [10, 10]]),
                (0, 0, 20, 20, 0.8, [[0, 0], [20, 0], [20, 20]]),
            ]
        )

    recognizer = _recognizer(runner)

    assert len(recognizer(_IMAGE, 0.05).regions) == 2
    assert len(recognizer(_IMAGE, 0.5).regions) == 1
    with pytest.raises(BedZoneNotFoundError):
        recognizer(_IMAGE, 0.9)


def test_bed_zone_response_caps_regions_and_rejects_overlong_polygon() -> None:
    boxes = [
        (
            0,
            0,
            20,
            20,
            0.9 - index / 100,
            [[0, 0], [20, 0], [20, 20], [0, 20]],
        )
        for index in range(9)
    ]
    boxes.insert(
        0,
        (
            0,
            0,
            20,
            20,
            0.99,
            [[index, 0] for index in range(17)],
        ),
    )

    payload = bed_zone_response(_IMAGE, bed_result(boxes))

    assert len(payload.regions) == 8


def test_bed_zone_recognizer_raises_not_found_when_no_beds_detected() -> None:
    with pytest.raises(BedZoneNotFoundError):
        _recognizer(lambda _image: bed_result([]))(_IMAGE)


def test_bed_zone_recognizer_raises_not_found_on_unexpected_result_kind() -> None:
    def runner(_image: Image) -> RunnerResult:
        # A misconfigured serving client could hand back some other task's
        # result kind here; this must still fail closed via the same
        # structured error, not crash the HTTP thread with an AttributeError
        # from treating it as a `BedRunnerResult`.
        return PoseRunnerResult(kind="pose", poses=(), boxes=())

    with pytest.raises(BedZoneNotFoundError):
        _recognizer(runner)(_IMAGE)

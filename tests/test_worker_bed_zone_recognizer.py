"""Strict bed-segmentation conversion for the on-demand overlay route."""

from __future__ import annotations

from typing import Final

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
        assert options == {"device": "cpu"}
        return self.runner


def _recognizer(runner: object) -> NvidiaBedZoneRecognizer:
    return NvidiaBedZoneRecognizer(
        _RunnerServingClient(runner),  # type: ignore[arg-type]
        timeout_s=1.0,
    )


def test_bed_zone_recognizer_picks_highest_confidence_box_and_its_polygon() -> None:
    def runner(_image: Image) -> RunnerResult:
        return bed_result(
            [
                (0, 0, 10, 10, 0.4, [[0, 0], [10, 0], [10, 10], [0, 10]]),
                (5, 5, 50, 50, 0.9, [[5, 5], [50, 5], [50, 50], [5, 50]]),
                (1, 1, 2, 2, 0.6, [[1, 1], [2, 1], [2, 2], [1, 2]]),
            ]
        )

    payload = _recognizer(runner)(_IMAGE)

    assert payload.as_dict() == {
        "polygon": [[5, 5], [50, 5], [50, 50], [5, 50]],
        "image_width": 640,
        "image_height": 480,
    }


def test_bed_zone_recognizer_refuses_box_when_polygon_is_empty() -> None:
    def runner(_image: Image) -> RunnerResult:
        return bed_result([(10, 20, 110, 220, 0.8, [])])

    with pytest.raises(BedZoneNotFoundError):
        _recognizer(runner)(_IMAGE)


def test_bed_zone_response_prefers_valid_segment_over_higher_score_box_only() -> None:
    result = bed_result(
        [
            (10, 20, 110, 220, 0.99),
            (5, 5, 50, 50, 0.7, [[5, 5], [50, 5], [50, 50], [5, 50]]),
        ]
    )

    payload = bed_zone_response(_IMAGE, result)

    assert payload.polygon == ((5, 5), (50, 5), (50, 50), (5, 50))


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

    assert recognizer(_IMAGE).polygon == ((2, 2), (40, 2), (40, 40), (2, 40))


def test_bed_zone_recognizer_accepts_a_run_method_runner_not_just_callable() -> None:
    """A runner exposing ``.run`` instead of ``__call__`` must still work."""

    class _RunOnlyRunner:
        def run(self, _image: Image) -> RunnerResult:
            return bed_result([(2, 2, 4, 4, 0.5, [[2, 2], [4, 2], [4, 4], [2, 4]])])

    payload = _recognizer(_RunOnlyRunner())(_IMAGE)

    assert payload.polygon == ((2, 2), (4, 2), (4, 4), (2, 4))


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

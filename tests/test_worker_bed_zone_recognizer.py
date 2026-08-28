"""``WorkerRuntime._bed_zone_recognizer`` -- the on-demand, one-shot bed
segmentation pass backing the ``POST /overlay/{camera_id}/bed-zone/recognize``
route (see ``worker/pipeline/output/_mjpeg_http.py``) -- must pick the
highest-confidence bed, prefer its simplified polygon when the runner
produced one, fall back to a rectangle when it did not, and raise
``BedZoneNotFoundError`` when the runner found nothing usable.

These tests exercise the method directly against a minimally constructed
``WorkerRuntime`` (no ``.run()``, no real model loading) with a fake
``shared_yolo.bed.runner``, mirroring the ``NamedExtractor._call`` resolution
idiom (``runner if callable(runner) else runner.run``) it is built to match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest

from contracts.runner import Image, PoseRunnerResult, RunnerResult, bed_result
from worker.pipeline.analytics import NamedExtractor
from worker.pipeline.output.mjpeg_server import BedZoneNotFoundError
from worker.runtime.config import WorkerConfig
from worker.runtime.lease import GpuLease
from worker.runtime.model_composition import SharedYoloExtractors
from worker.runtime.worker import WorkerRuntime

_IMAGE: Final[Image] = np.zeros((480, 640, 3), dtype=np.uint8)


class _FakeServingClient:
    def create(self, task: str, **_options: object) -> object:
        raise AssertionError(f"model composition must not run in this test (task={task!r})")


def _config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": "camera-a",
                    "facility_id": "facility-a",
                    "rtsp_url": "rtsp://example.test/camera-a",
                }
            ],
        }
    )


def _runtime(tmp_path: Path) -> WorkerRuntime:
    return WorkerRuntime(
        _config(),
        env={"ML_WORKER_PROFILE": "cpu"},
        serving_client=_FakeServingClient(),
        acquire_lease=lambda: GpuLease.acquire(tmp_path),
        state_dir=tmp_path,
    )


def _bed_extractor(runner: object) -> NamedExtractor:
    return NamedExtractor(
        module_name="bed",
        runner=runner,  # type: ignore[arg-type]
        _call=lambda image: runner(image),  # type: ignore[operator]
        _clock=lambda: 0.0,
    )


def _with_bed_runner(runtime: WorkerRuntime, runner: object) -> None:
    placeholder = _bed_extractor(runner)
    runtime.shared_yolo = SharedYoloExtractors(
        pose=placeholder, person=placeholder, bed=_bed_extractor(runner)
    )


def test_bed_zone_recognizer_picks_highest_confidence_box_and_its_polygon(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)

    def runner(_image: Image) -> RunnerResult:
        return bed_result(
            [
                (0, 0, 10, 10, 0.4, [[0, 0], [10, 0], [10, 10], [0, 10]]),
                (5, 5, 50, 50, 0.9, [[5, 5], [50, 5], [50, 50], [5, 50]]),
                (1, 1, 2, 2, 0.6, [[1, 1], [2, 1], [2, 2], [1, 2]]),
            ]
        )

    _with_bed_runner(runtime, runner)

    payload = runtime._bed_zone_recognizer(_IMAGE)  # noqa: SLF001

    assert payload.as_dict() == {
        "polygon": [[5, 5], [50, 5], [50, 50], [5, 50]],
        "image_width": 640,
        "image_height": 480,
    }


def test_bed_zone_recognizer_falls_back_to_rectangle_when_polygon_is_empty(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)

    def runner(_image: Image) -> RunnerResult:
        # `_simplify_polygon` can legitimately produce an empty polygon for a
        # degenerate mask even though the box itself is a real detection.
        return bed_result([(10, 20, 110, 220, 0.8, [])])

    _with_bed_runner(runtime, runner)

    payload = runtime._bed_zone_recognizer(_IMAGE)  # noqa: SLF001

    assert payload.as_dict() == {
        "polygon": [[10, 20], [110, 20], [110, 220], [10, 220]],
        "image_width": 640,
        "image_height": 480,
    }


def test_bed_zone_recognizer_accepts_a_run_method_runner_not_just_callable(
    tmp_path: Path,
) -> None:
    """Mirrors ``NamedExtractor``'s own ``_runner_call`` resolution: a runner
    that is not directly callable but exposes ``.run`` must still work."""
    runtime = _runtime(tmp_path)

    class _RunOnlyRunner:
        def run(self, _image: Image) -> RunnerResult:
            return bed_result([(2, 2, 4, 4, 0.5, [[2, 2], [4, 2], [4, 4], [2, 4]])])

    _with_bed_runner(runtime, _RunOnlyRunner())

    payload = runtime._bed_zone_recognizer(_IMAGE)  # noqa: SLF001

    assert payload.polygon == ((2, 2), (4, 2), (4, 4), (2, 4))


def test_bed_zone_recognizer_raises_not_found_when_no_beds_detected(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    _with_bed_runner(runtime, lambda _image: bed_result([]))

    with pytest.raises(BedZoneNotFoundError):
        runtime._bed_zone_recognizer(_IMAGE)  # noqa: SLF001


def test_bed_zone_recognizer_raises_not_found_on_unexpected_result_kind(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)

    def runner(_image: Image) -> RunnerResult:
        # A misconfigured serving client could hand back some other task's
        # result kind here; this must still fail closed via the same
        # structured error, not crash the HTTP thread with an AttributeError
        # from treating it as a `BedRunnerResult`.
        return PoseRunnerResult(kind="pose", poses=(), boxes=())

    _with_bed_runner(runtime, runner)

    with pytest.raises(BedZoneNotFoundError):
        runtime._bed_zone_recognizer(_IMAGE)  # noqa: SLF001


def test_bed_zone_recognizer_raises_when_models_not_initialized(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    assert runtime.shared_yolo is None

    with pytest.raises(RuntimeError, match="before models were initialized"):
        runtime._bed_zone_recognizer(_IMAGE)  # noqa: SLF001

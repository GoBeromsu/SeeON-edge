"""Bounded CPU-only, on-demand bed segmentation for the NVIDIA media plane."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from threading import Lock
from typing import cast

from contracts.runner import BedRunnerResult, Image, RunnerProtocol
from worker.interfaces.serving import ServingClient
from worker.pipeline.output.live_view_api import BedZoneRecognizeResponse
from worker.pipeline.output.mjpeg_server import BedZoneNotFoundError

DEFAULT_BED_ZONE_RECOGNITION_TIMEOUT_S = 5.0


class BedZoneRecognizerUnavailableError(RuntimeError):
    """The CPU bed model could not be constructed for an on-demand request."""


class BedZoneRecognitionTimeoutError(RuntimeError):
    """An on-demand bed recognition did not complete before its HTTP deadline."""


class NvidiaBedZoneRecognizer:
    """Lazily provision and run one CPU bed segmentation outside the media plane."""

    def __init__(self, serving_client: ServingClient, *, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError("bed-zone recognition timeout must be positive")
        self._serving_client = serving_client
        self._timeout_s = timeout_s
        self._runner: RunnerProtocol | None = None
        self._runner_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bed-zone-http")

    def __call__(self, image: Image) -> BedZoneRecognizeResponse:
        future: Future[BedZoneRecognizeResponse] = self._executor.submit(self._recognize, image)
        try:
            return future.result(timeout=self._timeout_s)
        except TimeoutError as exc:
            raise BedZoneRecognitionTimeoutError("bed-zone recognition timed out") from exc

    def _recognize(self, image: Image) -> BedZoneRecognizeResponse:
        runner = self._get_runner()
        call = runner if callable(runner) else runner.run
        result = call(image)
        if not isinstance(result, BedRunnerResult):
            raise BedZoneNotFoundError("bed runner returned an unexpected result")
        return _response_from_result(image, result)

    def _get_runner(self) -> RunnerProtocol:
        with self._runner_lock:
            if self._runner is not None:
                return self._runner
            try:
                runner = self._serving_client.create("bed", device="cpu")
            except Exception as exc:  # noqa: BLE001 - HTTP seam exposes a typed failure
                raise BedZoneRecognizerUnavailableError(
                    "CPU bed-zone recognizer could not be constructed"
                ) from exc
            if not callable(runner) and not callable(getattr(runner, "run", None)):
                raise BedZoneRecognizerUnavailableError(
                    "CPU bed-zone recognizer did not provide a runnable model"
                )
            self._runner = cast("RunnerProtocol", runner)
            return self._runner


def _response_from_result(image: Image, result: BedRunnerResult) -> BedZoneRecognizeResponse:
    height, width = int(image.shape[0]), int(image.shape[1])
    best_box: Sequence[float | Sequence[Sequence[int]]] | None = None
    best_score = -1.0
    for box in result.boxes:
        if not isinstance(box[4], (int, float)):
            continue
        score = float(box[4])
        if score > best_score:
            best_score = score
            best_box = box
    if best_box is None:
        raise BedZoneNotFoundError("no bed detected in the current frame")
    coordinates = cast("Sequence[float]", best_box[:5])
    polygon_field = best_box[5] if len(best_box) > 5 else ()
    polygon = (
        [[int(point[0]), int(point[1])] for point in polygon_field if isinstance(point, Sequence)]
        if isinstance(polygon_field, Sequence)
        else []
    )
    if not polygon:
        x1, y1, x2, y2 = (
            int(coordinates[0]),
            int(coordinates[1]),
            int(coordinates[2]),
            int(coordinates[3]),
        )
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return BedZoneRecognizeResponse(
        polygon=tuple((point[0], point[1]) for point in polygon),
        image_width=width,
        image_height=height,
    )


__all__ = [
    "DEFAULT_BED_ZONE_RECOGNITION_TIMEOUT_S",
    "BedZoneRecognitionTimeoutError",
    "BedZoneRecognizerUnavailableError",
    "NvidiaBedZoneRecognizer",
]

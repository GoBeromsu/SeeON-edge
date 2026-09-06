"""Bounded CPU-only, on-demand bed segmentation for the NVIDIA media plane."""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from numbers import Real
from threading import Lock
from uuid import uuid4

from contracts.runner import BedRunnerResult, Image, RunnerProtocol
from worker.interfaces.serving import ServingClient
from worker.pipeline.output.live_view_api import (
    MAX_BED_ZONE_CONFIDENCE,
    MIN_BED_ZONE_CONFIDENCE,
    BedZoneRecognizeRegion,
    BedZoneRecognizeResponse,
)
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

    def __call__(self, image: Image, confidence: float = 0.25) -> BedZoneRecognizeResponse:
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not math.isfinite(float(confidence))
            or not MIN_BED_ZONE_CONFIDENCE <= float(confidence) <= MAX_BED_ZONE_CONFIDENCE
        ):
            raise ValueError("bed-zone confidence must be between 0.05 and 0.95")
        future: Future[BedZoneRecognizeResponse] = self._executor.submit(
            self._recognize, image, float(confidence)
        )
        try:
            return future.result(timeout=self._timeout_s)
        except TimeoutError as exc:
            raise BedZoneRecognitionTimeoutError("bed-zone recognition timed out") from exc

    def _recognize(self, image: Image, confidence: float) -> BedZoneRecognizeResponse:
        runner = self._get_runner()
        call = runner if callable(runner) else runner.run
        result = call(image)
        if not isinstance(result, BedRunnerResult):
            raise BedZoneNotFoundError("bed runner returned an unexpected result")
        return bed_zone_response(image, result, confidence=confidence)

    def _get_runner(self) -> RunnerProtocol:
        with self._runner_lock:
            if self._runner is not None:
                return self._runner
            try:
                runner = self._serving_client.create(
                    "bed", device="cpu", confidence=0.05, max_points=16
                )
            except Exception as exc:  # noqa: BLE001 - HTTP seam exposes a typed failure
                raise BedZoneRecognizerUnavailableError(
                    "CPU bed-zone recognizer could not be constructed"
                ) from exc
            self._runner = runner
            return self._runner


def bed_zone_response(
    image: Image,
    result: BedRunnerResult,
    *,
    confidence: float = 0.25,
) -> BedZoneRecognizeResponse:
    """Build a response from all qualifying, valid bed segmentations."""
    height, width = int(image.shape[0]), int(image.shape[1])
    candidates: list[tuple[float, tuple[tuple[int, int], ...]]] = []
    for box in result.boxes:
        if not isinstance(box, Sequence) or len(box) < 6:
            continue
        score_field = box[4]
        if (
            isinstance(score_field, bool)
            or not isinstance(score_field, Real)
            or not math.isfinite(float(score_field))
        ):
            continue
        score = float(score_field)
        if score < confidence:
            continue
        polygon = _valid_polygon(box[5], width=width, height=height)
        if polygon is None:
            continue
        candidates.append((score, polygon))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if not candidates:
        raise BedZoneNotFoundError("no bed detected in the current frame")
    return BedZoneRecognizeResponse(
        regions=tuple(
            BedZoneRecognizeRegion(id=str(uuid4()), polygon=polygon)
            for _, polygon in candidates[:8]
        ),
        image_width=width,
        image_height=height,
    )


def _valid_polygon(
    polygon_field: object,
    *,
    width: int,
    height: int,
) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(polygon_field, Sequence) or len(polygon_field) > 16:
        return None

    polygon: list[tuple[int, int]] = []
    for point in polygon_field:
        if not isinstance(point, Sequence) or len(point) != 2:
            return None
        x, y = point
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, Real)
            or not isinstance(y, Real)
        ):
            return None
        x_float, y_float = float(x), float(y)
        if (
            not math.isfinite(x_float)
            or not math.isfinite(y_float)
            or not 0 <= x_float < width
            or not 0 <= y_float < height
        ):
            return None
        polygon.append((int(x_float), int(y_float)))

    if len(set(polygon)) < 3:
        return None
    twice_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True)
    )
    if twice_area == 0:
        return None
    return tuple(polygon)


__all__ = [
    "DEFAULT_BED_ZONE_RECOGNITION_TIMEOUT_S",
    "BedZoneRecognitionTimeoutError",
    "BedZoneRecognizerUnavailableError",
    "NvidiaBedZoneRecognizer",
    "bed_zone_response",
]

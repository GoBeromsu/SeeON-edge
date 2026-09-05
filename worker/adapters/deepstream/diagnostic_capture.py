"""One-shot raw-frame capture for an explicitly run DeepStream diagnostic."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from worker.adapters.deepstream.tensor_rows import host_array_from_tensor

_PIXEL_FORMAT: Final = "RGB"
_CHANNELS: Final = 3


@dataclass(frozen=True, slots=True)
class DiagnosticCaptureResult:
    """Owned pixels and identity read from the same DeepStream buffer callback."""

    pixels: NDArray[np.uint8]
    pixel_format: str
    source_id: int
    pad_index: int
    frame_number: int
    batch_id: int
    buffer_pts: int


class DiagnosticCaptureOperator:
    """Vendor-neutral one-shot handler wrapped as an SDK BufferOperator at runtime."""

    def __init__(self, *, width: int, height: int, pixel_format: str) -> None:
        self._width, self._height, self._pixel_format = _validate_capture_spec(
            width, height, pixel_format
        )
        self._completed = threading.Event()
        self._lock = threading.Lock()
        self._result: DiagnosticCaptureResult | None = None
        self._error: Exception | None = None

    def handle_buffer(self, buffer: Any) -> bool:
        """Copy at most one frame, preserving its metadata-selected batch identity."""
        if self._completed.is_set():
            return True
        with self._lock:
            if self._completed.is_set():
                return True
            try:
                self._result = self._capture(buffer)
            except (
                AttributeError,
                OSError,
                OverflowError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                self._error = error
            finally:
                self._completed.set()
        return True

    def _capture(self, buffer: Any) -> DiagnosticCaptureResult:
        # The SDK's iterator owns one FrameMetadata value and advances it in
        # place. Consume each yielded frame before requesting the next one.
        batch_meta = buffer.batch_meta
        frame_item_iterable = batch_meta.frame_items
        captured: DiagnosticCaptureResult | None = None
        for frame_meta in frame_item_iterable:
            if captured is not None:
                raise ValueError(
                    "diagnostic capture requires exactly one frame in the one-source batch"
                )
            captured = self._capture_frame(buffer, frame_meta)
        if captured is None:
            raise ValueError(
                "diagnostic capture requires exactly one frame in the one-source batch"
            )
        return captured

    def _capture_frame(self, buffer: Any, frame_meta: Any) -> DiagnosticCaptureResult:
        batch_id = int(frame_meta.batch_id)
        tensor = buffer.extract(batch_id)
        if not tensor:
            raise ValueError("raw RGB buffer extraction returned an empty tensor")
        pixels = host_array_from_tensor(tensor)
        self._validate_pixels(pixels)
        return DiagnosticCaptureResult(
            pixels=pixels,
            pixel_format=self._pixel_format,
            source_id=int(frame_meta.source_id),
            pad_index=int(frame_meta.pad_index),
            frame_number=int(frame_meta.frame_number),
            batch_id=batch_id,
            buffer_pts=int(frame_meta.buffer_pts),
        )

    def wait(self, timeout_seconds: float) -> DiagnosticCaptureResult:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self._completed.wait(timeout_seconds):
            raise TimeoutError("timed out waiting for one raw DeepStream frame")
        if self._error is not None:
            raise self._error
        if self._result is None:  # pragma: no cover - guarded by the callback invariant
            raise RuntimeError("diagnostic capture completed without a result")
        return self._result

    def _validate_pixels(self, pixels: NDArray[Any]) -> None:
        expected_shape = (self._height, self._width, _CHANNELS)
        if pixels.dtype != np.uint8:
            raise ValueError(f"raw {self._pixel_format} tensor must have uint8 dtype")
        if pixels.shape != expected_shape:
            raise ValueError(
                f"raw {self._pixel_format} tensor shape must be {expected_shape}, "
                f"got {pixels.shape}"
            )


def capture_one_raw_frame(
    uri: str,
    *,
    width: int,
    height: int,
    pixel_format: str = "RGB",
    timeout_seconds: float = 10.0,
) -> DiagnosticCaptureResult:
    """Run an isolated capture-only pipeline and return its first raw frame.

    The capsfilter explicitly negotiates RGB NVMM before the ``BufferOperator``.
    The operator selects pixels with the same frame metadata's ``batch_id`` and
    copies them before returning from the callback.

    Run only in a disposable process with an external deadline and cleanup.
    ``timeout_seconds`` bounds frame waiting, not native pipeline shutdown.
    """
    width, height, pixel_format = _validate_capture_spec(width, height, pixel_format)
    if not uri:
        raise ValueError("uri must not be empty")

    from pyservicemaker import BufferOperator, Pipeline, Probe

    capture = DiagnosticCaptureOperator(width=width, height=height, pixel_format=pixel_format)

    class _Operator(BufferOperator):
        def __init__(self) -> None:
            super().__init__()

        def handle_buffer(self, buffer: Any) -> bool:
            return capture.handle_buffer(buffer)

    pipeline = Pipeline("diagnostic-raw-capture")
    pipeline.add(
        "nvurisrcbin",
        "diagnostic-source",
        {"uri": uri, "gpu-id": 0, "file-loop": False},
    )
    pipeline.add(
        "nvstreammux",
        "diagnostic-mux",
        {
            "batch-size": 1,
            "width": width,
            "height": height,
            "batched-push-timeout": 33000,
            "buffer-pool-size": 4,
            "drop-pipeline-eos": False,
            "live-source": False,
            "gpu-id": 0,
            "compute-hw": 1,
        },
    )
    pipeline.add(
        "nvvideoconvert",
        "diagnostic-convert",
        {"gpu-id": 0, "compute-hw": 1},
    )
    pipeline.add(
        "capsfilter",
        "diagnostic-caps",
        {
            "caps": (
                f"video/x-raw(memory:NVMM), format={pixel_format}, width={width}, height={height}"
            )
        },
    )
    pipeline.add("fakesink", "diagnostic-sink", {"sync": False})
    pipeline.link(("diagnostic-source", "diagnostic-mux"), ("vsrc_%u", ""))
    pipeline.link("diagnostic-mux", "diagnostic-convert", "diagnostic-caps", "diagnostic-sink")
    pipeline.attach("diagnostic-caps", Probe("diagnostic-raw-probe", _Operator()))

    started = False
    try:
        pipeline.start()
        started = True
        return capture.wait(timeout_seconds)
    finally:
        if started:
            pipeline.stop().wait()


def _validate_capture_spec(width: int, height: int, pixel_format: str) -> tuple[int, int, str]:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    if pixel_format != _PIXEL_FORMAT:
        raise ValueError("pixel_format must be RGB")
    return width, height, pixel_format


__all__ = ["DiagnosticCaptureOperator", "DiagnosticCaptureResult", "capture_one_raw_frame"]

"""One-shot raw-frame capture for an explicitly run DeepStream diagnostic."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from worker.adapters.deepstream.metadata import association_pass
from worker.adapters.deepstream.tensor_rows import host_array_from_tensor, rows_from_tensor
from worker.interfaces.association import AssociationObservation
from worker.types.perception_frame import PerceptionFrameIdentity

_PIXEL_FORMAT: Final = "RGB"
_CHANNELS: Final = 3
TrackedObject = tuple[int, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class DiagnosticInferenceConfig:
    """Explicit nvinfer and NvDCF settings for an isolated diagnostic."""

    infer_config_path: str
    tracker_config_path: str
    tracker_library_path: str
    camera_id: str
    capture_id: str


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
    pose_rows: NDArray[np.float32] | None = None
    tracked_objects: tuple[TrackedObject, ...] | None = None
    association: AssociationObservation | None = None


class DiagnosticCaptureOperator:
    """Bounded frame collector wrapped as an SDK BufferOperator at runtime."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        pixel_format: str,
        capture_inference: bool = False,
        camera_id: str | None = None,
        capture_id: str | None = None,
        warmup_frames: int = 0,
        sample_count: int = 1,
        sample_stride: int = 1,
    ) -> None:
        self._width, self._height, self._pixel_format = _validate_capture_spec(
            width, height, pixel_format
        )
        if capture_inference and (not camera_id or not capture_id):
            raise ValueError("camera_id and capture_id are required for inference capture")
        self._capture_inference = capture_inference
        self._camera_id = camera_id
        self._capture_id = capture_id
        self._warmup_frames = _validate_warmup_frames(warmup_frames)
        self._sample_count = _validate_sample_count(sample_count)
        self._sample_stride = _validate_sample_stride(sample_stride)
        self._stride_remaining = 0
        self._completed = threading.Event()
        self._lock = threading.Lock()
        self._results: list[DiagnosticCaptureResult] = []
        self._error: Exception | None = None
        self._source_pad: tuple[int, int] | None = None
        self._last_frame_number: int | None = None

    def handle_buffer(self, buffer: Any) -> bool:
        """Copy at most one frame, preserving its metadata-selected batch identity."""
        if self._completed.is_set():
            return True
        with self._lock:
            if self._completed.is_set():
                return True
            try:
                if self._warmup_frames:
                    self._consume_single_frame(buffer, capture=False)
                    self._warmup_frames -= 1
                elif self._stride_remaining:
                    self._consume_single_frame(buffer, capture=False)
                    self._stride_remaining -= 1
                else:
                    self._capture_sample(buffer)
            except (
                AttributeError,
                OSError,
                OverflowError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                self._error = error
                self._completed.set()
        return True

    def _capture_sample(self, buffer: Any) -> None:
        result = self._consume_single_frame(buffer, capture=True)
        if result is None:  # pragma: no cover - capture=True invariant
            raise RuntimeError("selected frame produced no capture")
        self._results.append(result)
        if len(self._results) == self._sample_count:
            self._completed.set()
        else:
            self._stride_remaining = self._sample_stride - 1

    def _consume_single_frame(
        self, buffer: Any, *, capture: bool
    ) -> DiagnosticCaptureResult | None:
        batch_meta = buffer.batch_meta
        frame_item_iterable = batch_meta.frame_items
        consumed = False
        result = None
        for frame_meta in frame_item_iterable:
            if consumed:
                raise ValueError(
                    "diagnostic capture requires exactly one frame in the one-source batch"
                )
            consumed = True
            self._validate_sequence(frame_meta)
            if capture:
                result = self._capture_frame(buffer, frame_meta)
        if not consumed:
            raise ValueError(
                "diagnostic capture requires exactly one frame in the one-source batch"
            )
        return result

    def _validate_sequence(self, frame_meta: Any) -> None:
        source_pad = (int(frame_meta.source_id), int(frame_meta.pad_index))
        frame_number = int(frame_meta.frame_number)
        if self._source_pad is not None and source_pad != self._source_pad:
            raise ValueError("diagnostic source or pad changed during capture")
        if self._last_frame_number is not None and frame_number <= self._last_frame_number:
            raise ValueError("diagnostic frame number did not increase")
        self._source_pad = source_pad
        self._last_frame_number = frame_number

    def _capture_frame(self, buffer: Any, frame_meta: Any) -> DiagnosticCaptureResult:
        batch_id = int(frame_meta.batch_id)
        tensor = buffer.extract(batch_id)
        if not tensor:
            raise ValueError("raw RGB buffer extraction returned an empty tensor")
        pixels = host_array_from_tensor(tensor)
        self._validate_pixels(pixels)
        pose_rows, tracked_objects, association = self._capture_inference_metadata(frame_meta)
        return DiagnosticCaptureResult(
            pixels=pixels,
            pixel_format=self._pixel_format,
            source_id=int(frame_meta.source_id),
            pad_index=int(frame_meta.pad_index),
            frame_number=int(frame_meta.frame_number),
            batch_id=batch_id,
            buffer_pts=int(frame_meta.buffer_pts),
            pose_rows=pose_rows,
            tracked_objects=tracked_objects,
            association=association,
        )

    def _capture_inference_metadata(
        self, frame_meta: Any
    ) -> tuple[
        NDArray[np.float32] | None,
        tuple[TrackedObject, ...] | None,
        AssociationObservation | None,
    ]:
        if not self._capture_inference:
            return None, None, None
        tensor_items = frame_meta.tensor_items
        pose_rows: NDArray[np.float32] | None = None
        for item in tensor_items:
            if pose_rows is not None:
                raise ValueError("diagnostic frame has ambiguous tensor output metadata")
            layers = item.as_tensor_output().get_layers()
            layer = layers.get("output0")
            if layer is None or not layer:
                raise ValueError("diagnostic frame has no usable output0 tensor")
            pose_rows = rows_from_tensor(layer)
            if pose_rows.size == 0:
                raise ValueError("diagnostic frame output0 tensor is empty")
        if pose_rows is None:
            raise ValueError("diagnostic frame has no tensor output metadata")

        if self._capture_id is None or self._camera_id is None:
            raise RuntimeError("diagnostic inference identity is not configured")
        identity = PerceptionFrameIdentity(
            worker_boot_id=self._capture_id,
            camera_id=self._camera_id,
            stream_epoch=0,
            seq=int(frame_meta.frame_number),
            source_pts=int(frame_meta.buffer_pts),
        )
        linked = association_pass(
            frame_meta,
            rows=pose_rows,
            identity=identity,
            frame_w=self._width,
            frame_h=self._height,
        )
        tracked_objects = tuple(
            (
                track.track_id,
                track.box[0],
                track.box[1],
                track.box[2] - track.box[0],
                track.box[3] - track.box[1],
                track.confidence,
            )
            for track in linked.observation.tracks
        )
        return pose_rows, tracked_objects, linked.observation

    def wait(self, timeout_seconds: float) -> tuple[DiagnosticCaptureResult, ...]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self._completed.wait(timeout_seconds):
            raise TimeoutError("timed out waiting for diagnostic frames")
        if self._error is not None:
            raise self._error
        if len(self._results) != self._sample_count:
            raise RuntimeError("diagnostic capture completed without all results")
        return tuple(self._results)

    def _validate_pixels(self, pixels: NDArray[Any]) -> None:
        expected_shape = (self._height, self._width, _CHANNELS)
        if pixels.dtype != np.uint8:
            raise ValueError(f"raw {self._pixel_format} tensor must have uint8 dtype")
        if pixels.shape != expected_shape:
            raise ValueError(
                f"raw {self._pixel_format} tensor shape must be {expected_shape}, "
                f"got {pixels.shape}"
            )


def capture_diagnostic_frames(
    uri: str,
    *,
    width: int,
    height: int,
    pixel_format: str = "RGB",
    timeout_seconds: float = 10.0,
    inference: DiagnosticInferenceConfig | None = None,
    warmup_frames: int = 0,
    sample_count: int = 1,
    sample_stride: int = 1,
) -> tuple[DiagnosticCaptureResult, ...]:
    """Run an isolated diagnostic and return scheduled raw frames.

    The capsfilter explicitly negotiates RGB NVMM before the ``BufferOperator``.
    The operator selects pixels with the same frame metadata's ``batch_id`` and
    copies them before returning from the callback. Optional inference adds
    tensor and tracked-object observations from that same frame.
    ``warmup_frames`` skips exactly that many valid single-frame callbacks
    without inspecting their pixels, tensor output, or detections. After the
    first selection, ``sample_stride`` is the number of actual callbacks from
    one selected frame to the next.

    Run only in a disposable process with an external deadline and cleanup.
    ``timeout_seconds`` bounds frame waiting, not native pipeline shutdown.
    """
    width, height, pixel_format = _validate_capture_spec(width, height, pixel_format)
    if not uri:
        raise ValueError("uri must not be empty")
    if inference is not None:
        _validate_inference_config(inference)
    warmup_frames = _validate_warmup_frames(warmup_frames)
    sample_count = _validate_sample_count(sample_count)
    sample_stride = _validate_sample_stride(sample_stride)

    from pyservicemaker import BufferOperator, Pipeline, Probe

    capture = DiagnosticCaptureOperator(
        width=width,
        height=height,
        pixel_format=pixel_format,
        capture_inference=inference is not None,
        camera_id=None if inference is None else inference.camera_id,
        capture_id=None if inference is None else inference.capture_id,
        warmup_frames=warmup_frames,
        sample_count=sample_count,
        sample_stride=sample_stride,
    )

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
    processing_elements = ["diagnostic-mux"]
    if inference is not None:
        pipeline.add(
            "nvinfer",
            "diagnostic-infer",
            {
                "config-file-path": inference.infer_config_path,
                "batch-size": 1,
            },
        )
        pipeline.add(
            "nvtrackerbin",
            "diagnostic-tracker",
            {
                "ll-config-file": inference.tracker_config_path,
                "ll-lib-file": inference.tracker_library_path,
            },
        )
        processing_elements.extend(["diagnostic-infer", "diagnostic-tracker"])
    processing_elements.extend(["diagnostic-convert", "diagnostic-caps", "diagnostic-sink"])
    pipeline.link(*processing_elements)
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


def _validate_inference_config(config: DiagnosticInferenceConfig) -> None:
    if not all(
        (
            config.infer_config_path,
            config.tracker_config_path,
            config.tracker_library_path,
            config.camera_id,
            config.capture_id,
        )
    ):
        raise ValueError("inference paths, camera_id, and capture_id must not be empty")


def _validate_warmup_frames(warmup_frames: int) -> int:
    if isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int) or warmup_frames < 0:
        raise ValueError("warmup_frames must be a non-negative integer")
    return warmup_frames


def _validate_sample_count(sample_count: int) -> int:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 1 <= sample_count <= 60
    ):
        raise ValueError("sample_count must be an integer from 1 through 60")
    return sample_count


def _validate_sample_stride(sample_stride: int) -> int:
    if isinstance(sample_stride, bool) or not isinstance(sample_stride, int) or sample_stride <= 0:
        raise ValueError("sample_stride must be a positive integer")
    return sample_stride


__all__ = [
    "DiagnosticCaptureOperator",
    "DiagnosticCaptureResult",
    "DiagnosticInferenceConfig",
    "capture_diagnostic_frames",
]

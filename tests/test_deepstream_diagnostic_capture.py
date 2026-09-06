from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from worker.adapters.deepstream.diagnostic_capture import DiagnosticCaptureOperator


@dataclass
class _FrameMeta:
    source_id: int = 17
    pad_index: int = 4
    frame_number: int = 91
    batch_id: int = 7
    buffer_pts: int = 123_456
    tensor_items: Any = field(default_factory=tuple)
    object_items: Any = field(default_factory=tuple)


class _BatchMeta:
    def __init__(self, frame_items: list[_FrameMeta]) -> None:
        self.frame_items = iter(frame_items)


class _Tensor:
    def __init__(self, pixels: NDArray[Any], *, valid: bool = True) -> None:
        self._pixels = pixels
        self._valid = valid

    def __bool__(self) -> bool:
        return self._valid

    def __dlpack__(self, stream: Any = None) -> Any:
        return self._pixels.__dlpack__(stream=stream)

    def copy(self) -> NDArray[Any]:
        return self._pixels.copy()


class _TensorOutput:
    def __init__(self, layers: dict[str, _Tensor]) -> None:
        self._layers = layers

    def get_layers(self) -> dict[str, _Tensor]:
        return self._layers


class _TensorMeta:
    def __init__(self, layers: dict[str, _Tensor]) -> None:
        self._output = _TensorOutput(layers)

    def as_tensor_output(self) -> _TensorOutput:
        return self._output


@dataclass
class _Rect:
    left: float
    top: float
    width: float
    height: float


@dataclass
class _ObjectMeta:
    object_id: int
    rect_params: _Rect
    confidence: float


class _Buffer:
    def __init__(self, frame_meta: _FrameMeta, pixels: NDArray[Any]) -> None:
        self.batch_meta = _BatchMeta([frame_meta])
        self._pixels = pixels
        self.extracted_batch_ids: list[int] = []

    def extract(self, batch_id: int) -> _Tensor:
        self.extracted_batch_ids.append(batch_id)
        return _Tensor(self._pixels)


class _BorrowedFrame:
    def __init__(self, owner: object) -> None:
        self._owner = weakref.ref(owner)

    def _value(self, value: int) -> int:
        gc.collect()
        if self._owner() is None:
            raise RuntimeError("borrowed frame outlived batch metadata")
        return value

    source_id = property(lambda self: self._value(17))
    pad_index = property(lambda self: self._value(4))
    frame_number = property(lambda self: self._value(91))
    batch_id = property(lambda self: self._value(7))
    buffer_pts = property(lambda self: self._value(123_456))


class _BorrowingBatchMeta:
    @property
    def frame_items(self) -> Any:
        frame = _BorrowedFrame(self)
        return (item for item in (frame,))


class _FrameLease:
    valid = True


class _LeasedFrame(_FrameMeta):
    def __init__(self, lease: _FrameLease) -> None:
        self._lease = lease

    def __getattribute__(self, name: str) -> Any:
        if name in {"source_id", "pad_index", "frame_number", "batch_id", "buffer_pts"}:
            lease = object.__getattribute__(self, "_lease")
            if not lease.valid:
                raise RuntimeError("frame lease expired when iterator advanced")
        return super().__getattribute__(name)


class _LeasedFrameIterator:
    def __init__(self) -> None:
        self._lease = _FrameLease()
        self._yielded = False

    def __iter__(self) -> _LeasedFrameIterator:
        return self

    def __next__(self) -> _LeasedFrame:
        if self._yielded:
            self._lease.valid = False
            raise StopIteration
        self._yielded = True
        return _LeasedFrame(self._lease)


class _LeasedBatchMeta:
    @property
    def frame_items(self) -> _LeasedFrameIterator:
        return _LeasedFrameIterator()


class _LeasedBuffer:
    def __init__(self, pixels: NDArray[Any]) -> None:
        self._pixels = pixels
        self.extracted_batch_ids: list[int] = []

    @property
    def batch_meta(self) -> _LeasedBatchMeta:
        return _LeasedBatchMeta()

    def extract(self, batch_id: int) -> _Tensor:
        self.extracted_batch_ids.append(batch_id)
        return _Tensor(self._pixels)


class _BorrowingBuffer:
    def __init__(self) -> None:
        self._pixels = np.zeros((2, 4, 3), dtype=np.uint8)
        self.extracted_batch_ids: list[int] = []

    @property
    def batch_meta(self) -> _BorrowingBatchMeta:
        return _BorrowingBatchMeta()

    def extract(self, batch_id: int) -> _Tensor:
        self.extracted_batch_ids.append(batch_id)
        return _Tensor(self._pixels)


def test_capture_selects_pixels_by_frame_metadata_batch_id_and_records_identity() -> None:
    source = np.arange(24, dtype=np.uint8).reshape((2, 4, 3))
    expected = source.copy()
    frame_meta = _FrameMeta()
    buffer = _Buffer(frame_meta, source)
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB")

    assert operator.handle_buffer(buffer) is True
    source.fill(0)
    (result,) = operator.wait(0.1)

    assert buffer.extracted_batch_ids == [frame_meta.batch_id]
    assert not np.shares_memory(result.pixels, source)
    np.testing.assert_array_equal(result.pixels, expected)
    assert result.pose_rows is None
    assert result.tracked_objects is None
    assert result.association is None
    assert (
        result.source_id,
        result.pad_index,
        result.frame_number,
        result.batch_id,
        result.buffer_pts,
    ) == (17, 4, 91, 7, 123_456)


def test_inference_capture_copies_pose_and_objects_from_the_same_frame() -> None:
    pose = np.zeros((2, 57), dtype=np.float32)
    pose[0, :5] = (160, 160, 320, 320, 0.9)
    pose[1, :5] = (0, 0, 10, 10, 0.8)
    frame_meta = _FrameMeta(
        frame_number=314,
        tensor_items=iter([_TensorMeta({"output0": _Tensor(pose)})]),
        object_items=iter(
            [
                _ObjectMeta(42, _Rect(1.0, 1.0, 1.0, 1.0), 0.75),
                _ObjectMeta(43, _Rect(3.0, 1.0, 1.0, 1.0), 0.6),
            ]
        ),
    )
    operator = DiagnosticCaptureOperator(
        width=4,
        height=2,
        pixel_format="RGB",
        capture_inference=True,
        camera_id="camera-diagnostic",
        capture_id="capture-1",
    )

    operator.handle_buffer(_Buffer(frame_meta, np.zeros((2, 4, 3), dtype=np.uint8)))
    (result,) = operator.wait(0.1)

    assert result.frame_number == 314
    assert result.pose_rows is not None
    assert not np.shares_memory(result.pose_rows, pose)
    np.testing.assert_array_equal(result.pose_rows, pose)
    assert result.tracked_objects == (
        (42, 1.0, 1.0, 1.0, 1.0, 0.75),
        (43, 3.0, 1.0, 1.0, 1.0, 0.6),
    )
    assert result.association is not None
    assert result.association.identity.worker_boot_id == "capture-1"
    assert result.association.identity.camera_id == "camera-diagnostic"
    assert result.association.identity.stream_epoch == 0
    assert result.association.identity.seq == 314
    assert result.association.identity.source_pts == 123_456
    assert result.association.tracks[0].pose_row == 0
    assert result.association.tracks[1].pose_row is None
    assert result.association.unmatched_tracks == 1


def test_inference_capture_accepts_a_present_tensor_with_no_tracked_objects() -> None:
    frame_meta = _FrameMeta(
        tensor_items=iter([_TensorMeta({"output0": _Tensor(np.ones((1, 57), dtype=np.float32))})]),
        object_items=iter(()),
    )
    operator = DiagnosticCaptureOperator(
        width=4,
        height=2,
        pixel_format="RGB",
        capture_inference=True,
        camera_id="camera-diagnostic",
        capture_id="capture-2",
    )

    operator.handle_buffer(_Buffer(frame_meta, np.zeros((2, 4, 3), dtype=np.uint8)))

    assert operator.wait(0.1)[0].tracked_objects == ()


@pytest.mark.parametrize(
    "tensor_items",
    [
        (),
        (_TensorMeta({"other": _Tensor(np.ones((1, 57), dtype=np.float32))}),),
        (_TensorMeta({"output0": _Tensor(np.ones((1, 57), dtype=np.float32), valid=False)}),),
        (_TensorMeta({"output0": _Tensor(np.empty((0, 57), dtype=np.float32))}),),
        (
            _TensorMeta({"output0": _Tensor(np.ones((1, 57), dtype=np.float32))}),
            _TensorMeta({"output0": _Tensor(np.ones((1, 57), dtype=np.float32))}),
        ),
    ],
)
def test_inference_capture_rejects_missing_or_ambiguous_output(
    tensor_items: tuple[_TensorMeta, ...],
) -> None:
    frame_meta = _FrameMeta(tensor_items=iter(tensor_items))
    operator = DiagnosticCaptureOperator(
        width=4,
        height=2,
        pixel_format="RGB",
        capture_inference=True,
        camera_id="camera-diagnostic",
        capture_id="capture-3",
    )

    operator.handle_buffer(_Buffer(frame_meta, np.zeros((2, 4, 3), dtype=np.uint8)))

    with pytest.raises(ValueError):
        operator.wait(0.1)


def test_capture_retains_batch_owner_while_reading_borrowed_frame_and_copying() -> None:
    buffer = _BorrowingBuffer()
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB")

    operator.handle_buffer(buffer)

    (result,) = operator.wait(0.1)
    assert result.batch_id == 7
    assert buffer.extracted_batch_ids == [7]


def test_capture_consumes_borrowed_frame_before_iterator_advances() -> None:
    buffer = _LeasedBuffer(np.zeros((2, 4, 3), dtype=np.uint8))
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB")

    operator.handle_buffer(buffer)

    (result,) = operator.wait(0.1)
    assert result.batch_id == 7
    assert buffer.extracted_batch_ids == [7]


def test_capture_copies_before_returning_from_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_active = True

    def copy_while_active(tensor: _Tensor) -> NDArray[Any]:
        assert callback_active
        return tensor.copy()

    monkeypatch.setattr(
        "worker.adapters.deepstream.diagnostic_capture.host_array_from_tensor",
        copy_while_active,
    )
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB")
    buffer = _Buffer(_FrameMeta(), np.zeros((2, 4, 3), dtype=np.uint8))

    operator.handle_buffer(buffer)
    callback_active = False

    assert operator.wait(0.1)[0].pixels.shape == (2, 4, 3)


def test_capture_is_one_shot() -> None:
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB")
    first = _Buffer(_FrameMeta(frame_number=1), np.ones((2, 4, 3), dtype=np.uint8))
    second = _Buffer(_FrameMeta(frame_number=2), np.zeros((2, 4, 3), dtype=np.uint8))

    assert operator.handle_buffer(first) is True
    assert operator.handle_buffer(second) is True

    (result,) = operator.wait(0.1)
    assert result.frame_number == 1
    assert first.extracted_batch_ids == [7]
    assert second.extracted_batch_ids == []


def test_warmup_skips_exact_callbacks_and_captures_selected_frame_identity() -> None:
    operator = DiagnosticCaptureOperator(
        width=4,
        height=2,
        pixel_format="RGB",
        capture_inference=True,
        camera_id="camera-diagnostic",
        capture_id="capture-warmup",
        warmup_frames=2,
    )
    skipped = [
        _Buffer(
            _FrameMeta(frame_number=number, tensor_items=()),
            np.zeros((2, 4, 3), dtype=np.uint8),
        )
        for number in (10, 11)
    ]
    selected = _Buffer(
        _FrameMeta(
            frame_number=12,
            buffer_pts=999,
            tensor_items=iter(
                [_TensorMeta({"output0": _Tensor(np.ones((1, 57), dtype=np.float32))})]
            ),
            object_items=iter([_ObjectMeta(8, _Rect(1.0, 2.0, 3.0, 4.0), 0.5)]),
        ),
        np.ones((2, 4, 3), dtype=np.uint8),
    )

    for buffer in skipped:
        operator.handle_buffer(buffer)
        with pytest.raises(TimeoutError):
            operator.wait(0.001)
        assert buffer.extracted_batch_ids == []

    operator.handle_buffer(selected)
    (result,) = operator.wait(0.1)
    assert (result.frame_number, result.buffer_pts) == (12, 999)
    assert result.tracked_objects == ((8, 1.0, 2.0, 3.0, 4.0, 0.5),)
    assert selected.extracted_batch_ids == [7]


def test_capture_times_out_while_declared_warmup_is_incomplete() -> None:
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB", warmup_frames=2)
    operator.handle_buffer(_Buffer(_FrameMeta(), np.zeros((2, 4, 3), dtype=np.uint8)))

    with pytest.raises(TimeoutError):
        operator.wait(0.001)


def test_missing_tensor_fails_on_selected_frame_after_warmup() -> None:
    operator = DiagnosticCaptureOperator(
        width=4,
        height=2,
        pixel_format="RGB",
        capture_inference=True,
        camera_id="camera-diagnostic",
        capture_id="capture-missing",
        warmup_frames=1,
    )
    operator.handle_buffer(
        _Buffer(
            _FrameMeta(frame_number=1),
            np.zeros((2, 4, 3), dtype=np.uint8),
        )
    )
    operator.handle_buffer(
        _Buffer(
            _FrameMeta(frame_number=2, tensor_items=()),
            np.zeros((2, 4, 3), dtype=np.uint8),
        )
    )

    with pytest.raises(ValueError, match="no tensor output"):
        operator.wait(0.1)


def test_capture_uses_fixed_count_and_stride_and_copies_before_next_frame() -> None:
    operator = DiagnosticCaptureOperator(
        width=4,
        height=2,
        pixel_format="RGB",
        sample_count=3,
        sample_stride=2,
    )
    buffers = [
        _Buffer(
            _FrameMeta(frame_number=number, buffer_pts=number * 100),
            np.full((2, 4, 3), number, dtype=np.uint8),
        )
        for number in range(1, 6)
    ]

    for buffer in buffers:
        operator.handle_buffer(buffer)
        buffer._pixels.fill(0)

    results = operator.wait(0.1)
    assert tuple(result.frame_number for result in results) == (1, 3, 5)
    assert tuple(result.buffer_pts for result in results) == (100, 300, 500)
    assert tuple(int(result.pixels[0, 0, 0]) for result in results) == (1, 3, 5)
    assert [buffer.extracted_batch_ids for buffer in buffers] == [
        [7],
        [],
        [7],
        [],
        [7],
    ]


def test_capture_timeout_does_not_return_partial_samples() -> None:
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB", sample_count=2)
    operator.handle_buffer(_Buffer(_FrameMeta(), np.zeros((2, 4, 3), dtype=np.uint8)))

    with pytest.raises(TimeoutError):
        operator.wait(0.001)


@pytest.mark.parametrize("sample_count", [0, 61, True, 1.5])
def test_capture_rejects_invalid_sample_count(sample_count: Any) -> None:
    with pytest.raises(ValueError, match="sample_count"):
        DiagnosticCaptureOperator(
            width=4,
            height=2,
            pixel_format="RGB",
            sample_count=sample_count,
        )


@pytest.mark.parametrize("sample_stride", [0, -1, True, 1.5])
def test_capture_rejects_invalid_sample_stride(sample_stride: Any) -> None:
    with pytest.raises(ValueError, match="sample_stride"):
        DiagnosticCaptureOperator(
            width=4,
            height=2,
            pixel_format="RGB",
            sample_stride=sample_stride,
        )


@pytest.mark.parametrize(
    "frames",
    [
        (_FrameMeta(frame_number=2), _FrameMeta(frame_number=2)),
        (_FrameMeta(frame_number=2), _FrameMeta(frame_number=1)),
        (_FrameMeta(frame_number=2), _FrameMeta(frame_number=3, source_id=18)),
        (_FrameMeta(frame_number=2), _FrameMeta(frame_number=3, pad_index=5)),
    ],
)
def test_capture_rejects_generation_or_nonincreasing_frame_changes(
    frames: tuple[_FrameMeta, _FrameMeta],
) -> None:
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB", sample_count=2)
    for frame in frames:
        operator.handle_buffer(_Buffer(frame, np.zeros((2, 4, 3), dtype=np.uint8)))

    with pytest.raises(ValueError):
        operator.wait(0.1)


@pytest.mark.parametrize("warmup_frames", [-1, True, 1.5, "1"])
def test_capture_rejects_invalid_warmup_frames(warmup_frames: Any) -> None:
    with pytest.raises(ValueError, match="warmup_frames"):
        DiagnosticCaptureOperator(
            width=4,
            height=2,
            pixel_format="RGB",
            warmup_frames=warmup_frames,
        )


@pytest.mark.parametrize(
    ("width", "height", "pixel_format"),
    [
        (0, 2, "RGB"),
        (4, -1, "RGB"),
        (4, 2, "RGBA"),
        (4, 2, "NV12"),
        (True, 2, "RGB"),
    ],
)
def test_capture_rejects_invalid_dimensions_or_format(
    width: int, height: int, pixel_format: str
) -> None:
    with pytest.raises(ValueError):
        DiagnosticCaptureOperator(width=width, height=height, pixel_format=pixel_format)


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((2, 4, 4), dtype=np.uint8),
        np.zeros((2, 4, 3), dtype=np.float32),
    ],
)
def test_capture_rejects_pixels_that_do_not_match_negotiated_rgb_caps(
    pixels: NDArray[Any],
) -> None:
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB")

    operator.handle_buffer(_Buffer(_FrameMeta(), pixels))

    with pytest.raises(ValueError):
        operator.wait(0.1)


def test_capture_rejects_a_batch_without_exactly_one_source_frame() -> None:
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB")
    buffer = _Buffer(_FrameMeta(), np.zeros((2, 4, 3), dtype=np.uint8))
    buffer.batch_meta = _BatchMeta([_FrameMeta(), _FrameMeta(batch_id=8)])

    operator.handle_buffer(buffer)

    with pytest.raises(ValueError, match="exactly one frame"):
        operator.wait(0.1)
    assert buffer.extracted_batch_ids == [7]


class _EmptyTensorBuffer(_Buffer):
    def extract(self, batch_id: int) -> _Tensor:
        self.extracted_batch_ids.append(batch_id)
        return _Tensor(self._pixels, valid=False)


def test_capture_rejects_an_empty_extracted_tensor_before_host_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "worker.adapters.deepstream.diagnostic_capture.host_array_from_tensor",
        lambda tensor: pytest.fail("empty tensor must not reach the host copier"),
    )
    operator = DiagnosticCaptureOperator(width=4, height=2, pixel_format="RGB")
    buffer = _EmptyTensorBuffer(_FrameMeta(), np.zeros((2, 4, 3), dtype=np.uint8))

    operator.handle_buffer(buffer)

    with pytest.raises(ValueError, match="empty tensor"):
        operator.wait(0.1)

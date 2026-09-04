"""The Flow preview retriever publishes bounded latest JPEG frames."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from worker.adapters.deepstream.service_maker import (
    DeepStreamMediaPlane,
    DeepStreamMediaPlaneConfig,
    _FlowHandle,
)
from worker.adapters.deepstream.tensor_rows import host_array_from_tensor
from worker.native.deepstream.metadata import LatestMetadataSlot


class _Pipeline:
    def stop(self) -> None:
        pass

    def __getitem__(self, name: str) -> object:
        del name
        return type("Element", (), {"set": lambda self, properties: None})()


class _Flow:
    def __init__(self) -> None:
        self.retriever: object | None = None

    def batch_capture(self, uris: list[str], **kwargs: object) -> _Flow:
        del uris, kwargs
        return self

    def infer(self, config: str) -> _Flow:
        del config
        return self

    def track(self, **kwargs: object) -> _Flow:
        del kwargs
        return self

    def attach(self, what: object) -> _Flow:
        del what
        return self

    def fork(self) -> _Flow:
        return self

    def retrieve(self, retriever: object) -> _Flow:
        self.retriever = retriever
        return self

    def render(self, **kwargs: object) -> _Flow:
        del kwargs
        return self


def _armed(plane: DeepStreamMediaPlane) -> DeepStreamMediaPlane:
    """The retriever is demand-driven, so a preview must be asked for first."""
    for camera_id in plane.camera_ids_for_preview():
        plane.request_preview(camera_id)
    return plane


def _plane(*, stride: int = 1) -> tuple[DeepStreamMediaPlane, _Flow]:
    """The preview retriever is opt-in in production; these tests enable it."""
    flow = _Flow()
    plane = DeepStreamMediaPlane(
        DeepStreamMediaPlaneConfig(
            "infer", "tracker", "lib", Path("/tmp"), 5, 8, 6, stride, preview_retriever_enabled=True
        ),
        metadata_slot=LatestMetadataSlot(),
        flow_factory=lambda _: _FlowHandle(
            flow=flow,
            pipeline=_Pipeline(),
            record_config=lambda **kwargs: kwargs,
            render_mode_discard="discard",
            make_probe=lambda name, probe: (name, probe),
            make_retriever=lambda retriever: retriever,
        ),
    )
    plane.add_source("camera-a", "rtsp://a")
    plane.add_source("camera-b", "rtsp://b")
    plane._build_flow()  # noqa: SLF001 - inspect the fake Flow graph directly
    return plane, flow


class _Buffer:
    def __init__(self, *pixels: np.ndarray) -> None:
        self._pixels = pixels
        self.batch_size = len(pixels)

    def extract(self, batch_id: int) -> np.ndarray:
        return self._pixels[batch_id]


class _HostTensor:
    def __init__(self, pixels: np.ndarray) -> None:
        self._pixels = pixels

    def __dlpack__(self, stream: object) -> object:
        del stream
        return self._pixels.__dlpack__()


class _NoCuda:
    def cudaMemcpy(self, destination: int, source: int, size: int, kind: int) -> int:
        del destination, source, size, kind
        raise AssertionError("host tensors must not use cudart")


def _pixels(value: int) -> np.ndarray:
    return np.full((6, 8, 4), value, dtype=np.uint8)


def test_consumed_batched_buffer_publishes_a_jpeg_for_its_camera() -> None:
    plane, flow = _plane()
    assert flow.retriever is not None
    assert _armed(plane) and flow.retriever.consume(_Buffer(_pixels(10), _pixels(20))) == 0
    first = plane.snapshot("camera-a")
    second = plane.snapshot("camera-b")
    assert first.startswith(b"\xff\xd8")
    assert second.startswith(b"\xff\xd8")
    assert first != second


def test_retriever_honours_per_camera_stride() -> None:
    plane, flow = _plane(stride=3)
    assert flow.retriever is not None
    for _ in range(2):
        _armed(plane) and flow.retriever.consume(_Buffer(_pixels(1)))
    assert "camera-a" not in plane._latest_jpegs  # noqa: SLF001
    _armed(plane) and flow.retriever.consume(_Buffer(_pixels(1)))
    assert plane.snapshot("camera-a").startswith(b"\xff\xd8")


def test_encode_failure_is_logged_and_does_not_escape(monkeypatch, caplog) -> None:
    plane, flow = _plane()
    assert flow.retriever is not None
    monkeypatch.setattr(
        "worker.adapters.deepstream.service_maker._encode_preview_jpeg",
        lambda pixels: (_ for _ in ()).throw(ValueError("bad pixels")),
    )
    with caplog.at_level(logging.WARNING):
        assert _armed(plane) and flow.retriever.consume(_Buffer(_pixels(1))) == 0
    assert "dropping DeepStream preview frame" in caplog.text


def test_latest_slot_replaces_the_previous_frame() -> None:
    plane, flow = _plane()
    assert flow.retriever is not None
    _armed(plane) and flow.retriever.consume(_Buffer(_pixels(1)))
    first = plane.snapshot("camera-a")
    _armed(plane) and flow.retriever.consume(_Buffer(_pixels(200)))
    assert plane.snapshot("camera-a") != first


def test_host_tensor_is_copied_without_cudart(monkeypatch) -> None:
    plane, flow = _plane()
    assert flow.retriever is not None
    monkeypatch.setattr(
        "worker.adapters.deepstream.service_maker.host_array_from_tensor",
        lambda tensor: host_array_from_tensor(tensor, cudart=_NoCuda()),
    )
    _armed(plane) and flow.retriever.consume(_Buffer(_HostTensor(_pixels(1))))
    assert plane.snapshot("camera-a").startswith(b"\xff\xd8")


def _capsule_for(buffer: object, shape: tuple[int, ...], strides: tuple[int, ...]) -> object:
    """Build a DLPack capsule over an existing host buffer, as a vendor tensor would."""
    import ctypes

    import numpy as np

    from worker.adapters.deepstream import tensor_rows

    array = buffer
    assert isinstance(array, np.ndarray)
    ndim = len(shape)
    shape_array = (ctypes.c_int64 * ndim)(*shape)
    stride_array = (ctypes.c_int64 * ndim)(*strides)
    managed = tensor_rows._DLManagedTensor()  # noqa: SLF001 - building a vendor-shaped capsule
    managed.dl_tensor.data = ctypes.c_void_p(array.ctypes.data)
    managed.dl_tensor.device = tensor_rows._DLDevice(2, 0)  # noqa: SLF001
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = tensor_rows._DLDataType(1, 8, 1)  # noqa: SLF001 - uint8
    managed.dl_tensor.shape = shape_array
    managed.dl_tensor.strides = stride_array
    managed.dl_tensor.byte_offset = 0
    _CAPSULE_KEEPALIVE.extend([managed, shape_array, stride_array, array])
    factory = ctypes.pythonapi.PyCapsule_New
    factory.restype = ctypes.py_object
    factory.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return factory(ctypes.byref(managed), b"dltensor", None)


#: DLPack capsules borrow memory; the test keeps every backing object alive.
_CAPSULE_KEEPALIVE: list[object] = []


def test_a_row_padded_device_tensor_is_copied_without_its_padding() -> None:
    """The defect this guards: a DeepStream frame surface is row-padded.

    A 640x360 RGB frame reports a 2048-element row stride for 1920 used
    elements, so a copy sized to the logical extent walks off the rows and the
    image shears. The copy must span the padded extent and slice each logical
    row back out.
    """
    import ctypes

    import numpy as np

    from worker.adapters.deepstream.tensor_rows import host_array_from_tensor

    height, width, channels, row_stride = 4, 5, 3, 20
    used = width * channels
    device = np.zeros(height * row_stride, dtype=np.uint8)
    for row in range(height):
        device[row * row_stride : row * row_stride + used] = row + 1
        device[row * row_stride + used :][: row_stride - used] = 0xEE  # padding

    class _PaddedTensor:
        """A device-resident DLPack tensor whose rows carry trailing padding."""

        def __dlpack_device__(self) -> tuple[int, int]:
            return (2, 0)  # kDLCUDA

        def __dlpack__(self, stream: object = None) -> object:
            del stream
            return _capsule_for(device, (height, width, channels), (row_stride, channels, 1))

    copies: list[int] = []

    class _FakeCudart:
        def cudaMemcpy(self, dst: int, src: int, count: int, kind: int) -> int:
            copies.append(count)
            ctypes.memmove(dst, src, count)
            return 0

    copied = host_array_from_tensor(_PaddedTensor(), cudart=_FakeCudart())

    assert copied.shape == (height, width, channels)
    assert copies == [height * row_stride], "the copy must span the padded extent"
    for row in range(height):
        assert (copied[row] == row + 1).all(), "each row must be its own logical data"
    assert 0xEE not in copied, "padding must not survive into the image"

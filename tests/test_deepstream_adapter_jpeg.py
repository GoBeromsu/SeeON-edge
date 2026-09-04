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


def _plane(*, stride: int = 1) -> tuple[DeepStreamMediaPlane, _Flow]:
    flow = _Flow()
    plane = DeepStreamMediaPlane(
        DeepStreamMediaPlaneConfig("infer", "tracker", "lib", Path("/tmp"), 5, 8, 6, stride),
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
    assert flow.retriever.consume(_Buffer(_pixels(10), _pixels(20))) == 0
    first = plane.snapshot("camera-a")
    second = plane.snapshot("camera-b")
    assert first.startswith(b"\xff\xd8")
    assert second.startswith(b"\xff\xd8")
    assert first != second


def test_retriever_honours_per_camera_stride() -> None:
    plane, flow = _plane(stride=3)
    assert flow.retriever is not None
    for _ in range(2):
        flow.retriever.consume(_Buffer(_pixels(1)))
    assert "camera-a" not in plane._latest_jpegs  # noqa: SLF001
    flow.retriever.consume(_Buffer(_pixels(1)))
    assert plane.snapshot("camera-a").startswith(b"\xff\xd8")


def test_encode_failure_is_logged_and_does_not_escape(monkeypatch, caplog) -> None:
    _, flow = _plane()
    assert flow.retriever is not None
    monkeypatch.setattr(
        "worker.adapters.deepstream.service_maker._encode_preview_jpeg",
        lambda pixels: (_ for _ in ()).throw(ValueError("bad pixels")),
    )
    with caplog.at_level(logging.WARNING):
        assert flow.retriever.consume(_Buffer(_pixels(1))) == 0
    assert "dropping DeepStream preview frame" in caplog.text


def test_latest_slot_replaces_the_previous_frame() -> None:
    plane, flow = _plane()
    assert flow.retriever is not None
    flow.retriever.consume(_Buffer(_pixels(1)))
    first = plane.snapshot("camera-a")
    flow.retriever.consume(_Buffer(_pixels(200)))
    assert plane.snapshot("camera-a") != first


def test_host_tensor_is_copied_without_cudart(monkeypatch) -> None:
    plane, flow = _plane()
    assert flow.retriever is not None
    monkeypatch.setattr(
        "worker.adapters.deepstream.service_maker.host_array_from_tensor",
        lambda tensor: host_array_from_tensor(tensor, cudart=_NoCuda()),
    )
    flow.retriever.consume(_Buffer(_HostTensor(_pixels(1))))
    assert plane.snapshot("camera-a").startswith(b"\xff\xd8")

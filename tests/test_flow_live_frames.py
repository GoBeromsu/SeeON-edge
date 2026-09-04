"""Flow OSD JPEG publication into the shared live-frame store."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker.adapters.deepstream.service_maker import _FlowHandle
from worker.interfaces.media_plane import SnapshotUnavailable
from worker.pipeline.output.live_view import LatestFrameStore
from worker.runtime.flow.media_plane import FlowMediaPlane, FlowMediaPlaneConfig


def _config() -> FlowMediaPlaneConfig:
    return FlowMediaPlaneConfig("infer", "tracker", "library", Path("/tmp"), 5, 640, 360)


class _Pipeline:
    def __getitem__(self, name: str) -> _Pipeline:
        del name
        return self

    def set(self, properties: dict[str, object]) -> None:
        del properties

    def stop(self) -> None:
        return None


class _Flow:
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

    def render(self, **kwargs: object) -> _Flow:
        del kwargs
        return self


def _flow_factory(_: object) -> _FlowHandle:
    return _FlowHandle(
        flow=_Flow(),
        pipeline=_Pipeline(),
        record_config=lambda **kwargs: kwargs,
        render_mode_discard="discard",
        make_probe=lambda name, probe: (name, probe),
    )


def test_adapter_published_osd_jpeg_reaches_the_shared_live_store() -> None:
    store = LatestFrameStore()
    store.register_camera("camera")
    plane = FlowMediaPlane(_config(), live_frames=store, flow_factory=_flow_factory)
    plane.add_source("camera", "rtsp://one")

    plane.plane.publish_jpeg("camera", b"\xff\xd8osd\xff\xd9")

    frame = store.get_latest("camera")
    assert frame is not None
    assert frame.jpeg == b"\xff\xd8osd\xff\xd9"


def test_snapshot_publishes_one_encoder_result_and_returns_it() -> None:
    store = LatestFrameStore()
    store.register_camera("camera")
    calls: list[str] = []
    plane = FlowMediaPlane(
        _config(),
        live_frames=store,
        flow_factory=_flow_factory,
        snapshot_encoder=lambda camera_id: calls.append(camera_id) or b"\xff\xd8osd\xff\xd9",
    )
    plane.add_source("camera", "rtsp://one")

    assert plane.snapshot("camera") == b"\xff\xd8osd\xff\xd9"
    assert calls == ["camera"]
    frame = store.get_latest("camera")
    assert frame is not None
    assert frame.jpeg == b"\xff\xd8osd\xff\xd9"


def test_snapshot_without_a_frame_is_typed_unavailable() -> None:
    plane = FlowMediaPlane(_config(), live_frames=LatestFrameStore(), flow_factory=_flow_factory)
    plane.add_source("camera", "rtsp://one")

    with pytest.raises(SnapshotUnavailable):
        plane.snapshot("camera")

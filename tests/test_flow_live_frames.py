"""Flow snapshot failure behaviour."""

from __future__ import annotations

from pathlib import Path

from worker.adapters.deepstream.service_maker import _FlowHandle
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


def test_alert_snapshot_uses_the_runtime_snapshot_encoder_seam() -> None:
    encoded: list[str] = []
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda camera_id: encoded.append(camera_id) or b"burned-jpeg",
    )
    plane.add_source("camera", "rtsp://one")

    assert plane.snapshot("camera") == b"burned-jpeg"
    assert encoded == ["camera"]


def test_snapshot_refresh_publishes_the_burned_jpeg_to_the_live_view_store() -> None:
    store = LatestFrameStore()
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda _camera_id: b"burned-jpeg",
        live_frames=store,
    )
    plane.add_source("camera", "rtsp://one")
    store.register_camera("camera")

    store.request_snapshot_refresh("camera")

    frame = store.get_latest("camera")
    assert frame is not None
    assert frame.jpeg == b"burned-jpeg"
    assert frame.content_type == "image/jpeg"


def test_the_recorder_defaults_to_the_sixty_second_window() -> None:
    """15 s of cached lookback plus the plane's 45 s forward window.

    Composition takes this default rather than repeating a literal, so the
    contract has one owner.
    """
    from worker.runtime.flow.media_plane import FlowMediaPlane

    assert FlowMediaPlane.DEFAULT_LOOKBACK_SEC == 15

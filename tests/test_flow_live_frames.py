"""Flow snapshot failure behaviour."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from worker.adapters.deepstream.service_maker import _FlowHandle
from worker.pipeline.output.live_view import LatestFrameStore
from worker.runtime.flow.media_plane import (
    BedZoneGeometry,
    FlowMediaPlane,
    FlowMediaPlaneConfig,
)


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


def _jpeg(*, width: int = 200, height: int = 100) -> bytes:
    encoded, output = cv2.imencode(
        ".jpg",
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    assert encoded
    return output.tobytes()


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


def test_clean_snapshot_disables_sdk_objects_without_changing_evidence_default() -> None:
    encoded: list[str] = []
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda camera_id: encoded.append(camera_id) or b"burned-jpeg",
    )
    plane.add_source("camera", "rtsp://one")
    object_flags: list[bool] = []
    adapter_snapshot = plane.plane.snapshot

    def capture_mode(camera_id: str, *, draw_objects: bool = True) -> bytes:
        object_flags.append(draw_objects)
        return adapter_snapshot(camera_id, draw_objects=draw_objects)

    plane.plane.snapshot = capture_mode  # type: ignore[method-assign]

    assert plane.clean_snapshot("camera") == b"burned-jpeg"
    assert plane.snapshot("camera") == b"burned-jpeg"
    assert object_flags == [False, True]


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


def test_store_passes_each_camera_mode_on_connect_refresh_and_disconnect() -> None:
    store = LatestFrameStore()
    calls: list[tuple[str, int, str, bool]] = []
    store.register_camera("camera")
    store.set_mode("camera", "bedexit")
    store.set_demand_listener(
        lambda camera_id, viewers, mode, requested: calls.append(
            (camera_id, viewers, mode, requested)
        )
    )

    store.mark_viewer_connected("camera")
    store.request_snapshot_refresh("camera")
    store.mark_viewer_disconnected("camera")

    assert calls == [
        ("camera", 1, "bedexit", False),
        ("camera", 1, "bedexit", True),
        ("camera", 0, "bedexit", False),
    ]


def test_preview_routes_clean_and_sdk_object_modes_without_changing_evidence_default() -> None:
    store = LatestFrameStore()
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda _camera_id: _jpeg(),
        live_frames=store,
    )
    plane.add_source("camera", "rtsp://one")
    store.register_camera("camera")
    object_flags: list[bool] = []
    adapter_snapshot = plane.plane.snapshot

    def capture_mode(camera_id: str, *, draw_objects: bool = True) -> bytes:
        object_flags.append(draw_objects)
        return adapter_snapshot(camera_id, draw_objects=draw_objects)

    plane.plane.snapshot = capture_mode  # type: ignore[method-assign]

    store.request_snapshot_refresh("camera")
    store.set_mode("camera", "fall")
    assert plane.snapshot("camera") == _jpeg()

    assert object_flags == [False, True, True]


def test_bedexit_draws_scaled_saved_polygon_and_never_invents_missing_geometry() -> None:
    source = _jpeg()
    store = LatestFrameStore()
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda _camera_id: source,
        live_frames=store,
        bed_zone_geometry={
            "with-bed": BedZoneGeometry(
                polygon=((10, 10), (40, 10), (40, 30), (10, 30)),
                image_width=100,
                image_height=50,
            )
        },
    )
    plane.add_source("with-bed", "rtsp://one")
    plane.add_source("without-bed", "rtsp://two")
    store.register_camera("with-bed")
    store.register_camera("without-bed")

    store.set_mode("with-bed", "bedexit")
    with_bed = store.get_latest("with-bed")
    assert with_bed is not None
    rendered = cv2.imdecode(np.frombuffer(with_bed.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert rendered is not None
    # Saved (10, 10) in a 100x50 source scales to (20, 20) in this 200x100 JPEG.
    assert int(rendered[20, 20, 1]) > 80
    assert int(rendered[5, 5].max()) < 20

    store.set_mode("without-bed", "bedexit")
    without_bed = store.get_latest("without-bed")
    assert without_bed is not None
    assert without_bed.jpeg == source


def test_mode_change_clears_stale_frame_when_selected_refresh_fails() -> None:
    store = LatestFrameStore()
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda _camera_id: _jpeg(),
        live_frames=store,
    )
    plane.add_source("camera", "rtsp://one")
    store.register_camera("camera")
    store.publish_jpeg("camera", _jpeg(), frame_index=1)

    def fail_snapshot(camera_id: str, *, draw_objects: bool = True) -> bytes:
        del camera_id, draw_objects
        raise OSError("capture failed")

    plane.plane.snapshot = fail_snapshot  # type: ignore[method-assign]
    store.set_mode("camera", "fall")

    assert store.get_latest("camera") is None


def test_in_flight_old_mode_cannot_publish_after_mode_change() -> None:
    store = LatestFrameStore()
    capture_started = threading.Event()
    release_capture = threading.Event()
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda _camera_id: _jpeg(),
        live_frames=store,
    )
    plane.add_source("camera", "rtsp://one")
    store.register_camera("camera")
    adapter_snapshot = plane.plane.snapshot

    def blocked_snapshot(camera_id: str, *, draw_objects: bool = True) -> bytes:
        if draw_objects:
            capture_started.set()
            assert release_capture.wait(timeout=1)
        return adapter_snapshot(camera_id, draw_objects=draw_objects)

    plane.plane.snapshot = blocked_snapshot  # type: ignore[method-assign]
    refresh = threading.Thread(target=store.set_mode, args=("camera", "fall"))
    refresh.start()
    assert capture_started.wait(timeout=1)

    store.set_mode("camera", "none")
    none_frame = store.get_latest("camera")
    assert none_frame is not None
    release_capture.set()
    refresh.join(timeout=1)

    assert not refresh.is_alive()
    assert store.get_latest("camera") is none_frame


def test_modes_and_refreshes_are_isolated_per_camera() -> None:
    store = LatestFrameStore()
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda camera_id: _jpeg(width=200 if camera_id == "a" else 100),
        live_frames=store,
    )
    for camera_id in ("a", "b"):
        plane.add_source(camera_id, f"rtsp://{camera_id}")
        store.register_camera(camera_id)
    calls: list[tuple[str, bool]] = []
    adapter_snapshot = plane.plane.snapshot

    def capture_mode(camera_id: str, *, draw_objects: bool = True) -> bytes:
        calls.append((camera_id, draw_objects))
        return adapter_snapshot(camera_id, draw_objects=draw_objects)

    plane.plane.snapshot = capture_mode  # type: ignore[method-assign]

    store.set_mode("a", "fall")
    store.request_snapshot_refresh("b")

    assert store.get_mode("a") == "fall"
    assert store.get_mode("b") == "none"
    assert calls == [("a", True), ("b", False)]
    assert store.get_latest("a") is not None
    assert store.get_latest("b") is not None


def test_the_recorder_defaults_to_the_sixty_second_window() -> None:
    """15 s of cached lookback plus the plane's 45 s forward window.

    Composition takes this default rather than repeating a literal, so the
    contract has one owner.
    """
    from worker.runtime.flow.media_plane import FlowMediaPlane

    assert FlowMediaPlane.DEFAULT_LOOKBACK_SEC == 15

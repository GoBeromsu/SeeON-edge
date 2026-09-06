"""Flow live-preview snapshot and overlay behaviour."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from worker.adapters.deepstream.service_maker import _FlowHandle
from worker.pipeline.output.live_view import LatestFrameStore
from worker.pipeline.output.preview_renderer import BedZoneGeometry, PreviewTrack
from worker.runtime.flow.media_plane import FlowMediaPlane, FlowMediaPlaneConfig
from worker.types.perception_frame import PersonBox
from worker.types.preview import OverlaySelection


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


class _Renderer:
    def __init__(self) -> None:
        self.calls: list[
            tuple[bytes, OverlaySelection, tuple[PreviewTrack, ...], object, object]
        ] = []

    def render(
        self,
        jpeg: bytes,
        selection: OverlaySelection,
        tracks: tuple[PreviewTrack, ...],
        bed_geometry: object,
        fall_states: object,
    ) -> bytes:
        self.calls.append((jpeg, selection, tuple(tracks), bed_geometry, fall_states))
        return b"preview-jpeg"


def _live_plane(
    store: LatestFrameStore,
    renderer: _Renderer,
    *,
    snapshot: bytes = b"sdk-jpeg",
    bed_geometry: dict[str, BedZoneGeometry] | None = None,
) -> FlowMediaPlane:
    return FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda _camera_id: snapshot,
        live_frames=store,
        renderer=renderer,  # type: ignore[arg-type]
        fall_states=lambda _camera_id: {},
        bed_zone_geometry=bed_geometry,
    )


def test_alert_snapshot_uses_runtime_encoder_and_keeps_sdk_object_default() -> None:
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda _camera_id: b"burned-jpeg",
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


def test_preview_always_requests_clean_sdk_snapshot() -> None:
    store = LatestFrameStore()
    renderer = _Renderer()
    plane = _live_plane(store, renderer)
    plane.add_source("camera", "rtsp://one")
    store.register_camera("camera")
    object_flags: list[bool] = []
    adapter_snapshot = plane.plane.snapshot

    def capture_mode(camera_id: str, *, draw_objects: bool = True) -> bytes:
        object_flags.append(draw_objects)
        return adapter_snapshot(camera_id, draw_objects=draw_objects)

    plane.plane.snapshot = capture_mode  # type: ignore[method-assign]
    store.request_snapshot_refresh("camera")

    assert object_flags == [False]
    assert store.get_latest("camera") is not None
    assert store.get_latest("camera").jpeg == b"preview-jpeg"  # type: ignore[union-attr]


def test_latest_metadata_boxes_and_association_ids_are_passed_to_renderer() -> None:
    store = LatestFrameStore()
    renderer = _Renderer()
    plane = _live_plane(store, renderer)
    plane.add_source("camera", "rtsp://one")
    store.register_camera("camera")
    box_a = PersonBox(10, 20, 30, 40, 0.8)
    box_b = PersonBox(50, 60, 70, 80, 0.7)
    plane.metadata = SimpleNamespace(  # type: ignore[assignment]
        peek=lambda _camera_id: SimpleNamespace(
            source_width=100,
            source_height=90,
            frame=SimpleNamespace(
                person_box=SimpleNamespace(boxes=(box_a, box_b)),
                association=SimpleNamespace(selected_cue_indexes=(1,), track_ids=(22,)),
            ),
        )
    )

    store.request_snapshot_refresh("camera")

    tracks = renderer.calls[0][2]
    assert tracks == (
        PreviewTrack(box_a, 100, 90, None),
        PreviewTrack(box_b, 100, 90, 22),
    )


def test_absent_metadata_passes_only_persisted_bed_geometry() -> None:
    store = LatestFrameStore()
    renderer = _Renderer()
    bed = BedZoneGeometry(((1, 1), (9, 1), (9, 9)), 10, 10)
    plane = _live_plane(store, renderer, bed_geometry={"camera": bed})
    plane.add_source("camera", "rtsp://one")
    store.register_camera("camera")

    store.request_snapshot_refresh("camera")

    assert renderer.calls[0][2] == ()
    assert renderer.calls[0][3] is bed


def test_in_flight_stale_selection_cannot_publish() -> None:
    store = LatestFrameStore()
    capture_started = threading.Event()
    release_capture = threading.Event()

    class _BlockingRenderer(_Renderer):
        def render(self, *args: object, **kwargs: object) -> bytes:
            if not capture_started.is_set():
                capture_started.set()
                assert release_capture.wait(timeout=1)
            return super().render(*args, **kwargs)  # type: ignore[arg-type]

    renderer = _BlockingRenderer()
    plane = _live_plane(store, renderer)
    plane.add_source("camera", "rtsp://one")
    store.register_camera("camera")

    refresh = threading.Thread(
        target=store.set_selection,
        args=("camera", OverlaySelection(person=False, bed=True)),
    )
    refresh.start()
    assert capture_started.wait(timeout=1)

    store.set_selection("camera", OverlaySelection(person=True, bed=False))
    current = store.get_latest("camera")
    assert current is not None
    release_capture.set()
    refresh.join(timeout=1)

    assert not refresh.is_alive()
    assert store.get_latest("camera") is current


def test_missing_preview_dependencies_refuse_start() -> None:
    plane = FlowMediaPlane(
        _config(),
        flow_factory=_flow_factory,
        snapshot_encoder=lambda _camera_id: b"jpeg",
        live_frames=LatestFrameStore(),
    )

    try:
        plane.start()
    except RuntimeError as error:
        assert "renderer and fall-state provider" in str(error)
    else:
        raise AssertionError("FlowMediaPlane.start() accepted incomplete preview wiring")


def test_the_recorder_defaults_to_the_sixty_second_window() -> None:
    assert FlowMediaPlane.DEFAULT_LOOKBACK_SEC == 15

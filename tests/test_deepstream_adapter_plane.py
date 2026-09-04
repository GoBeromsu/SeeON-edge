"""DeepStreamMediaPlane against a fake pyservicemaker seam.

The contract is what the G8a spike measured on a live camera
(docs/research/pyservicemaker-p1b-spike.md): ``Pipeline.start_recording``
returns a session id and delivers ``RecordingInfo`` to its callback when the
clip seals; ``Pipeline.stop_recording`` is defective, so the plane refuses
early stops with a typed error; a second start while one is in flight is
absorbed into the same session; a Flow fixes its sources when built.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from worker.adapters.deepstream.service_maker import (
    DeepStreamMediaPlane,
    DeepStreamMediaPlaneConfig,
    _FlowHandle,
)
from worker.interfaces.media_plane import (
    EarlyStopUnsupported,
    MediaPlane,
    RecordingInfo,
    RecordingRefused,
    SnapshotUnavailable,
    SourceRosterFixed,
)
from worker.native.deepstream.metadata import LatestMetadataSlot


class _Element:
    def __init__(self) -> None:
        self.properties: dict[str, object] = {}

    def set(self, properties: dict[str, object]) -> None:
        self.properties.update(properties)


class _Pipeline:
    def __init__(self) -> None:
        self.elements: dict[str, _Element] = {}
        self.started: list[tuple[str, int, int]] = []
        self.callbacks: dict[str, Callable[[object], None]] = {}
        self.stopped = False

    def __getitem__(self, name: str) -> _Element:
        return self.elements.setdefault(name, _Element())

    def start_recording(
        self, source: str, lookback: int, duration: int, callback: Callable[[object], None]
    ) -> int:
        self.started.append((source, lookback, duration))
        self.callbacks[source] = callback
        return 4

    def stop_recording(self, source: str) -> bool:
        raise AssertionError("the binding's stop_recording is defective and must never be called")

    def stop(self) -> None:
        self.stopped = True


class _Flow:
    def __init__(self) -> None:
        self.built = False
        self.ran = False

    def batch_capture(self, uris: list[str], **kwargs: object) -> _Flow:
        del uris, kwargs
        self.built = True
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

    def __call__(self) -> None:
        self.ran = True


def _plane(pipeline: _Pipeline | None = None) -> tuple[DeepStreamMediaPlane, _Pipeline]:
    pipeline = pipeline or _Pipeline()
    config = DeepStreamMediaPlaneConfig("infer", "tracker", "lib", Path("/tmp"), 5, 640, 360)
    plane = DeepStreamMediaPlane(
        config,
        metadata_slot=LatestMetadataSlot(),
        flow_factory=lambda _: _FlowHandle(
            flow=_Flow(),
            pipeline=pipeline,
            record_config=lambda **kwargs: kwargs,
            render_mode_discard="discard",
            make_probe=lambda name, probe: (name, probe),
        ),
        worker_boot_id="boot",
        child_instance_id="child",
    )
    return plane, pipeline


def _info(session: int) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session, dirpath="/tmp", filename="clip.mp4", duration=20, width=640, height=360
    )


def test_plane_satisfies_the_vendor_neutral_protocol() -> None:
    plane, _ = _plane()
    assert isinstance(plane, MediaPlane)


def test_source_failure_rebuilds_the_binding_on_a_new_generation() -> None:
    plane, _ = _plane()
    first = plane.add_source("camera", "rtsp://one")
    replacement = plane.source_failure("camera", "timeout")
    assert replacement.camera_id == first.camera_id == "camera"
    assert replacement.source_generation == first.source_generation + 1


def test_recording_is_refused_before_the_source_has_published_a_frame() -> None:
    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")
    with pytest.raises(RecordingRefused, match="has not published"):
        plane.start_recording("camera", lookback_sec=1, duration_sec=2, on_sealed=lambda _: None)


def test_start_uses_the_sdk_primitive_and_an_inflight_start_is_absorbed() -> None:
    plane, pipeline = _plane()
    plane.add_source("camera", "rtsp://one")
    plane._live.add("camera")  # noqa: SLF001 - a frame has been published
    sealed: list[object] = []
    session = plane.start_recording(
        "camera", lookback_sec=15, duration_sec=45, on_sealed=sealed.append
    )
    again = plane.start_recording("camera", lookback_sec=1, duration_sec=2, on_sealed=sealed.append)
    assert again == session
    assert pipeline.started == [("batch_capture-source-0_0", 15, 45)]


def test_early_stop_is_a_typed_refusal_not_a_silent_no_op() -> None:
    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")
    plane._live.add("camera")  # noqa: SLF001
    session = plane.start_recording(
        "camera", lookback_sec=1, duration_sec=2, on_sealed=lambda _: None
    )
    with pytest.raises(EarlyStopUnsupported, match="seals at its start duration"):
        plane.stop_recording("camera", session)
    with pytest.raises(RecordingRefused, match="unknown recording session"):
        plane.stop_recording("camera", session + 1)


def test_sealed_callback_fires_exactly_once_per_session() -> None:
    plane, pipeline = _plane()
    plane.add_source("camera", "rtsp://one")
    plane._live.add("camera")  # noqa: SLF001
    sealed: list[RecordingInfo] = []
    session = plane.start_recording(
        "camera", lookback_sec=1, duration_sec=2, on_sealed=sealed.append
    )
    callback = pipeline.callbacks["batch_capture-source-0_0"]
    callback(_info(session))
    callback(_info(session))
    assert [item.session_id for item in sealed] == [session]
    assert sealed[0].path == "/tmp/clip.mp4"
    assert sealed[0].duration_ms == 20


def test_sealed_callback_is_retryable_when_the_handoff_raises() -> None:
    plane, pipeline = _plane()
    plane.add_source("camera", "rtsp://one")
    plane._live.add("camera")  # noqa: SLF001
    sealed: list[RecordingInfo] = []
    failures = [True]

    def handoff(info: RecordingInfo) -> None:
        if failures[0]:
            failures[0] = False
            raise RuntimeError("publication failed")
        sealed.append(info)

    session = plane.start_recording("camera", lookback_sec=1, duration_sec=2, on_sealed=handoff)
    callback = pipeline.callbacks["batch_capture-source-0_0"]
    with pytest.raises(RuntimeError, match="publication failed"):
        callback(_info(session))

    callback(_info(session))
    assert [item.session_id for item in sealed] == [session]


def test_sources_are_fixed_once_the_flow_runs() -> None:
    plane, pipeline = _plane()
    plane.add_source("camera", "rtsp://one")
    plane.start()
    with pytest.raises(SourceRosterFixed, match="restart the worker"):
        plane.add_source("late", "rtsp://two")
    with pytest.raises(SourceRosterFixed):
        plane.remove_source("camera")
    plane.stop()
    assert pipeline.stopped


def test_source_properties_follow_the_measured_rtsp_shape() -> None:
    plane, pipeline = _plane()
    plane.add_source("camera", "rtsp://one")
    plane.start()
    props = pipeline["batch_capture-source-0_0"].properties
    assert props["select-rtp-protocol"] == 4
    assert props["init-rtsp-reconnect-interval"] == 5
    assert props["rtsp-reconnect-interval"] == 5
    plane.stop()


def test_smart_record_owns_no_encode_sessions() -> None:
    plane, _ = _plane()
    assert plane.status().nvenc_sessions_active == 0


def test_latest_osd_jpeg_slot_serves_snapshot_without_an_encoder() -> None:
    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")
    plane.publish_jpeg("camera", b"\xff\xd8osd\xff\xd9")
    assert plane.snapshot("camera") == b"\xff\xd8osd\xff\xd9"


def test_snapshot_encodes_once_and_publishes_the_latest_osd_jpeg() -> None:
    calls: list[str] = []
    published: list[tuple[str, bytes]] = []
    config = DeepStreamMediaPlaneConfig("infer", "tracker", "lib", Path("/tmp"), 5, 640, 360)
    plane = DeepStreamMediaPlane(
        config,
        metadata_slot=LatestMetadataSlot(),
        flow_factory=lambda _: _FlowHandle(
            flow=_Flow(),
            pipeline=_Pipeline(),
            record_config=lambda **kwargs: kwargs,
            render_mode_discard="discard",
            make_probe=lambda name, probe: (name, probe),
        ),
        snapshot_encoder=lambda camera_id: calls.append(camera_id) or b"\xff\xd8osd\xff\xd9",
        jpeg_publisher=lambda camera_id, jpeg: published.append((camera_id, jpeg)),
    )
    plane.add_source("camera", "rtsp://one")
    assert plane.snapshot("camera") == b"\xff\xd8osd\xff\xd9"
    assert calls == ["camera"]
    assert published == [("camera", b"\xff\xd8osd\xff\xd9")]


def test_snapshot_without_a_frame_is_a_typed_unavailable() -> None:
    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")
    with pytest.raises(SnapshotUnavailable, match="has not produced"):
        plane.snapshot("camera")


def test_a_source_failure_rotates_the_stream_identity_and_keeps_the_camera_id() -> None:
    """P1b-AC2: the sensor id is the canonical camera id across a forced rebuild.

    The Flow's sources are fixed once it runs, so a failure rotates the stream
    identity rather than rebuilding the element: frames after the outage cannot
    be mistaken for the old stream, and stale preview state is dropped.
    """
    plane, _ = _plane()
    first = plane.add_source("room-208", "rtsp://one")
    plane._live.add("room-208")  # noqa: SLF001 - a frame has been published

    replacement = plane.source_failure("room-208", "timeout")

    assert replacement.camera_id == first.camera_id == "room-208"
    assert replacement.stream_epoch > first.stream_epoch or (
        replacement.source_generation > first.source_generation
    )
    assert "room-208" not in plane._live  # noqa: SLF001 - liveness must be re-proven
    with pytest.raises(SnapshotUnavailable):
        plane.snapshot("room-208")

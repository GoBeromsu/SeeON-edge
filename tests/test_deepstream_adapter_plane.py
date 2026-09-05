"""DeepStreamMediaPlane against a fake pyservicemaker seam.

The contract is what the G8a spike measured on a live camera
(docs/research/pyservicemaker-p1b-spike.md): ``Pipeline.start_recording``
returns a session id and delivers ``RecordingInfo`` to its callback when the
clip seals; ``Pipeline.stop_recording`` is defective, so the plane refuses
early stops with a typed error; a second start while one is in flight is
absorbed into the same session; a Flow fixes its sources when built.
"""

from __future__ import annotations

import logging
import threading
import time
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
from worker.runtime.flow.metadata_slot import LatestMetadataSlot

_JPEG = b"\xff\xd8complete-jpeg\xff\xd9"


class _Element:
    def __init__(self) -> None:
        self.properties: dict[str, object] = {}

    def set(self, properties: dict[str, object]) -> None:
        self.properties.update(properties)


class _Pipeline:
    def __init__(self) -> None:
        self.elements: dict[str, _Element] = {}
        self.added: list[tuple[str, str, dict[str, object]]] = []
        self.links: list[tuple[object, ...]] = []
        self.attached: dict[str, object] = {}
        self.started: list[tuple[str, int, int]] = []
        self.callbacks: dict[str, Callable[[object], None]] = {}
        self.stopped = False

    def __getitem__(self, name: str) -> _Element:
        return self.elements.setdefault(name, _Element())

    def add(
        self, type_name: str, name: str, properties: dict[str, object] | None = None
    ) -> _Pipeline:
        self.added.append((type_name, name, properties or {}))
        self[name].set(properties or {})
        return self

    def link(self, *elements: object) -> None:
        self.links.append(elements)

    def attach(self, name: str, receiver: object, *, tips: str) -> None:
        assert tips == "new-sample"
        self.attached[name] = receiver

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
        self.render_calls: list[dict[str, object]] = []
        self._streams = [SimpleNamespace(originator="fork-tee-0")]

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

    def fork(self) -> _Flow:
        return self

    def render(self, **kwargs: object) -> _Flow:
        self.render_calls.append(kwargs)
        return self

    def __call__(self) -> None:
        self.ran = True


def _plane(
    pipeline: _Pipeline | None = None, *, snapshot_branch_enabled: bool = False
) -> tuple[DeepStreamMediaPlane, _Pipeline]:
    pipeline = pipeline or _Pipeline()
    config = DeepStreamMediaPlaneConfig(
        "infer",
        "tracker",
        "lib",
        Path("/tmp/seeon-adapter-tests"),
        5,
        640,
        360,
        snapshot_branch_enabled=snapshot_branch_enabled,
    )
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
        # Field names as the SDK's RecordingInfo actually spells them; the fake
        # previously used dirpath/filename, which is why a real seal aborted the
        # process from inside the completion callback.
        session_id=session,
        file_directory="/tmp",
        file_name="clip.mp4",
        duration=20,
        width=640,
        height=360,
    )


def _admit_frame(plane: DeepStreamMediaPlane, *, pad_index: int = 0) -> None:
    plane.publish_frame(
        SimpleNamespace(
            pad_index=pad_index,
            buffer_pts=1,
            tensor_items=[],
            object_items=[],
        )
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


def test_a_failed_handoff_is_logged_and_never_raises_into_the_sdk_callback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The completion callback runs on the pipeline thread.

    A live run proved the cost of raising here: a thumbnail failure propagated
    out of the callback and terminated the worker, taking every camera down.
    The media and its contributor sidecar are already on disk, so the honest
    behaviour is to log, keep the media, and let the startup replay finish the
    publication.
    """
    plane, pipeline = _plane()
    plane.add_source("camera", "rtsp://one")
    plane._live.add("camera")  # noqa: SLF001

    def handoff(info: RecordingInfo) -> None:
        raise RuntimeError("publication failed")

    session = plane.start_recording("camera", lookback_sec=1, duration_sec=2, on_sealed=handoff)
    callback = pipeline.callbacks["batch_capture-source-0_0"]

    with caplog.at_level(logging.ERROR):
        callback(_info(session))

    assert "clip publication failed" in caplog.text
    assert "/tmp/clip.mp4" in caplog.text, "the operator must be told where the media is"


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


def test_snapshot_without_an_active_osd_branch_is_typed_unavailable() -> None:
    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")
    with pytest.raises(SnapshotUnavailable, match="has not started"):
        plane.snapshot("camera")


def test_default_steady_state_builds_only_the_discard_sink() -> None:
    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")
    plane._build_flow()  # noqa: SLF001 - assert the fake Flow's terminal sink

    flow = plane._flow  # noqa: SLF001 - the configured graph is adapter behaviour
    assert isinstance(flow, _Flow)
    assert flow.render_calls == [{"mode": "discard", "enable_osd": False, "sync": False}]


def test_snapshot_encoder_runs_only_for_the_alert_that_requested_a_snapshot() -> None:
    encoded: list[str] = []
    plane, pipeline = _plane()
    plane._snapshot_encoder = lambda camera_id: (  # noqa: SLF001 - adapter seam
        encoded.append(camera_id) or _JPEG
    )
    plane.add_source("camera", "rtsp://one")

    assert encoded == []
    assert plane.snapshot("camera", draw_objects=False) == _JPEG
    assert encoded == ["camera"]
    assert "snapshot-osd" not in pipeline.elements


def test_enabled_snapshot_branch_uses_the_fork_as_the_discard_terminal() -> None:
    plane, pipeline = _plane(snapshot_branch_enabled=True)
    plane.add_source("camera", "rtsp://one")
    plane._build_flow()  # noqa: SLF001 - topology is adapter behaviour

    assert pipeline.links == [
        (
            "fork-tee-0",
            "snapshot-tee-queue",
            "snapshot-valve",
            "snapshot-tiler",
            "snapshot-convert",
            "snapshot-osd",
            "snapshot-post-osd-convert",
            "snapshot-caps",
            "snapshot-encoder",
            "snapshot-sink",
        ),
    ]
    assert pipeline["snapshot-valve"].properties == {"drop": True, "drop-mode": 2}
    assert pipeline["snapshot-tiler"].properties == {
        "rows": 1,
        "columns": 1,
        "width": 640,
        "height": 360,
        "show-source": 0,
    }
    assert pipeline["snapshot-osd"].properties["display-bbox"] == 1
    assert pipeline["snapshot-osd"].properties["display-text"] == 1
    assert pipeline["snapshot-sink"].properties["next-file"] == 0
    flow = plane._flow  # noqa: SLF001 - the configured graph is adapter behaviour
    assert isinstance(flow, _Flow)
    assert flow.render_calls == [{"mode": "discard", "enable_osd": False, "sync": False}]


def test_snapshot_refuses_a_source_that_has_not_published_a_frame() -> None:
    plane, pipeline = _plane(snapshot_branch_enabled=True)
    plane.add_source("camera", "rtsp://one")
    plane.start()

    with pytest.raises(SnapshotUnavailable, match="has not published a frame"):
        plane.snapshot("camera")

    assert pipeline["snapshot-valve"].properties["drop"] is True
    assert tuple(plane._snapshot_dir.glob("snapshot-*.jpg")) == ()  # noqa: SLF001
    plane.stop()


def test_enabled_snapshot_branch_closes_its_valve_after_one_jpeg() -> None:
    plane, pipeline = _plane(snapshot_branch_enabled=True)
    plane.add_source("camera", "rtsp://one")
    _admit_frame(plane)
    plane.start()
    result: list[bytes] = []
    request = threading.Thread(target=lambda: result.append(plane.snapshot("camera")))
    request.start()
    for _ in range(100):
        if pipeline["snapshot-valve"].properties.get("drop") is False:
            break
        time.sleep(0.01)
    (plane._snapshot_dir / "snapshot-0000000000.jpg").write_bytes(  # noqa: SLF001
        b"\xff\xd8burned\xff\xd9"
    )
    request.join(timeout=1)

    assert result == [b"\xff\xd8burned\xff\xd9"]
    for _ in range(100):
        if pipeline["snapshot-valve"].properties.get("drop") is True:
            break
        time.sleep(0.01)
    assert pipeline["snapshot-valve"].properties["drop"] is True
    plane.stop()


def test_snapshot_waits_for_a_complete_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane, pipeline = _plane(snapshot_branch_enabled=True)
    plane.add_source("camera", "rtsp://one")
    _admit_frame(plane)
    plane.start()
    valve_opened = threading.Event()
    incomplete_observed = threading.Event()
    valve = pipeline["snapshot-valve"]
    original_set = valve.set

    def observe_valve(properties: dict[str, object]) -> None:
        original_set(properties)
        if properties.get("drop") is False:
            valve_opened.set()

    monkeypatch.setattr(valve, "set", observe_valve)
    original_read_bytes = Path.read_bytes

    def observe_read(path: Path) -> bytes:
        data = original_read_bytes(path)
        if data == b"\xff\xd8incomplete":
            incomplete_observed.set()
        return data

    monkeypatch.setattr(Path, "read_bytes", observe_read)
    result: list[bytes] = []
    errors: list[BaseException] = []

    def capture() -> None:
        try:
            result.append(plane.snapshot("camera"))
        except BaseException as error:  # noqa: BLE001 - surfaced to the test thread
            errors.append(error)

    request = threading.Thread(target=capture)
    request.start()
    assert valve_opened.wait(timeout=1)
    path = plane._snapshot_dir / "snapshot-0000000000.jpg"  # noqa: SLF001
    path.write_bytes(b"\xff\xd8incomplete")
    assert incomplete_observed.wait(timeout=1)
    assert request.is_alive()

    path.write_bytes(_JPEG)
    request.join(timeout=1)

    assert not request.is_alive()
    assert errors == []
    assert result == [_JPEG]
    assert pipeline["snapshot-valve"].properties["drop"] is True
    plane.stop()


def test_snapshot_configures_requested_osd_mode_and_resets_to_evidence_default() -> None:
    plane, pipeline = _plane(snapshot_branch_enabled=True)
    plane.add_source("camera", "rtsp://one")
    _admit_frame(plane)
    plane.start()

    def capture(*, draw_objects: bool | None) -> bytes:
        result: list[bytes] = []
        request = threading.Thread(
            target=lambda: result.append(
                plane.snapshot("camera")
                if draw_objects is None
                else plane.snapshot("camera", draw_objects=draw_objects)
            )
        )
        request.start()
        for _ in range(100):
            if pipeline["snapshot-valve"].properties.get("drop") is False:
                break
            time.sleep(0.01)
        expected = 1 if draw_objects is None else int(draw_objects)
        assert pipeline["snapshot-osd"].properties["display-bbox"] == expected
        assert pipeline["snapshot-osd"].properties["display-text"] == expected
        (plane._snapshot_dir / "snapshot-0000000000.jpg").write_bytes(  # noqa: SLF001
            b"\xff\xd8capture\xff\xd9"
        )
        request.join(timeout=1)
        assert not request.is_alive()
        assert result == [b"\xff\xd8capture\xff\xd9"]
        return result[0]

    capture(draw_objects=False)
    assert pipeline["snapshot-valve"].properties["drop"] is True
    assert pipeline["snapshot-osd"].properties["display-bbox"] == 1
    assert pipeline["snapshot-osd"].properties["display-text"] == 1

    capture(draw_objects=None)
    assert pipeline["snapshot-osd"].properties["display-bbox"] == 1
    assert pipeline["snapshot-osd"].properties["display-text"] == 1
    plane.stop()


def test_snapshots_serialize_the_shared_bridge_across_cameras() -> None:
    plane, pipeline = _plane(snapshot_branch_enabled=True)
    plane.add_source("first", "rtsp://one")
    plane.add_source("second", "rtsp://two")
    _admit_frame(plane, pad_index=0)
    _admit_frame(plane, pad_index=1)
    plane.start()
    results: dict[str, bytes] = {}
    errors: list[BaseException] = []

    def capture(camera_id: str) -> None:
        try:
            results[camera_id] = plane.snapshot(camera_id)
        except BaseException as error:  # noqa: BLE001 - surfaced to the test thread
            errors.append(error)

    first = threading.Thread(target=capture, args=("first",))
    first.start()
    for _ in range(100):
        if pipeline["snapshot-valve"].properties.get("drop") is False:
            break
        time.sleep(0.01)
    assert pipeline["snapshot-tiler"].properties["show-source"] == 0

    second = threading.Thread(target=capture, args=("second",))
    second.start()
    second.join(timeout=0.05)
    assert second.is_alive()
    assert pipeline["snapshot-tiler"].properties["show-source"] == 0

    (plane._snapshot_dir / "snapshot-0000000000.jpg").write_bytes(  # noqa: SLF001
        b"\xff\xd8first\xff\xd9"
    )
    first.join(timeout=1)
    for _ in range(100):
        if (
            pipeline["snapshot-valve"].properties.get("drop") is False
            and pipeline["snapshot-tiler"].properties.get("show-source") == 1
        ):
            break
        time.sleep(0.01)
    (plane._snapshot_dir / "snapshot-0000000000.jpg").write_bytes(  # noqa: SLF001
        b"\xff\xd8second\xff\xd9"
    )
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results == {
        "first": b"\xff\xd8first\xff\xd9",
        "second": b"\xff\xd8second\xff\xd9",
    }
    assert pipeline["snapshot-valve"].properties["drop"] is True
    plane.stop()


def test_snapshot_read_error_closes_resets_and_cleans_the_shared_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane, pipeline = _plane(snapshot_branch_enabled=True)
    plane.add_source("camera", "rtsp://one")
    _admit_frame(plane)
    plane.start()

    def produce() -> None:
        for _ in range(100):
            if pipeline["snapshot-valve"].properties.get("drop") is False:
                break
            time.sleep(0.01)
        (plane._snapshot_dir / "snapshot-0000000000.jpg").write_bytes(  # noqa: SLF001
            b"\xff\xd8capture\xff\xd9"
        )

    producer = threading.Thread(target=produce)
    producer.start()

    def fail_read(_path: Path) -> bytes:
        raise OSError("read")

    monkeypatch.setattr(Path, "read_bytes", fail_read)

    with pytest.raises(OSError, match="read"):
        plane.snapshot("camera", draw_objects=False)

    producer.join(timeout=1)
    assert pipeline["snapshot-valve"].properties["drop"] is True
    assert pipeline["snapshot-osd"].properties["display-bbox"] == 1
    assert pipeline["snapshot-osd"].properties["display-text"] == 1
    assert tuple(plane._snapshot_dir.glob("snapshot-*.jpg")) == ()  # noqa: SLF001
    plane.stop()


def test_snapshot_selects_the_requested_camera_before_opening_the_shared_valve() -> None:
    plane, pipeline = _plane(snapshot_branch_enabled=True)
    plane.add_source("first", "rtsp://one")
    plane.add_source("second", "rtsp://two")
    _admit_frame(plane, pad_index=0)
    _admit_frame(plane, pad_index=1)
    plane.start()
    result: list[bytes] = []
    request = threading.Thread(target=lambda: result.append(plane.snapshot("second")))
    request.start()
    for _ in range(100):
        if pipeline["snapshot-valve"].properties.get("drop") is False:
            break
        time.sleep(0.01)
    assert pipeline["snapshot-tiler"].properties["show-source"] == 1
    (plane._snapshot_dir / "snapshot-0000000000.jpg").write_bytes(  # noqa: SLF001
        b"\xff\xd8burned\xff\xd9"
    )
    request.join(timeout=1)

    assert result == [b"\xff\xd8burned\xff\xd9"]
    plane.stop()


def test_enabled_snapshot_branch_times_out_and_recloses_its_valve() -> None:
    plane, pipeline = _plane(snapshot_branch_enabled=True)
    plane.add_source("camera", "rtsp://one")
    _admit_frame(plane)
    plane.start()

    with pytest.raises(SnapshotUnavailable, match="timed out"):
        plane.snapshot("camera")

    assert pipeline["snapshot-valve"].properties["drop"] is True
    plane.stop()


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


def test_an_unmapped_mux_pad_is_dropped_and_never_raises_into_the_probe() -> None:
    """An exception in an SDK probe callback aborts the whole process.

    A 13-camera run died exactly this way when a frame arrived on a pad the
    source table did not know. Such a frame must be counted out and dropped.
    """
    from types import SimpleNamespace

    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")

    plane.publish_frame(
        SimpleNamespace(pad_index=99, buffer_pts=1, tensor_items=[], object_items=[])
    )

    assert plane.published_frames("camera") == 0
    assert plane.camera_id_for_pad(99) is None
    assert plane.camera_id_for_pad(0) == "camera"


def test_a_frame_without_pose_tensor_is_counted_and_named_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A dead pose path must not look like an empty room.

    A mis-bound output layer produced no pose rows for an entire bring-up while
    the pipeline ran at full frame rate, so the plane counts such frames and
    names the camera once instead of staying silent.
    """
    from types import SimpleNamespace

    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")

    with caplog.at_level(logging.WARNING):
        for objects in (3, 2):
            plane.publish_frame(
                SimpleNamespace(
                    pad_index=0,
                    buffer_pts=1,
                    tensor_items=[],
                    object_items=[
                        SimpleNamespace(
                            object_id=index,
                            confidence=0.9,
                            rect_params=SimpleNamespace(left=0.0, top=0.0, width=10.0, height=10.0),
                        )
                        for index in range(objects)
                    ],
                    num_obj_meta=objects,
                )
            )

    observed, without_tensor = plane.perception_counters("camera")
    assert without_tensor == 2
    assert observed == 5
    warnings = [
        record for record in caplog.records if "no pose tensor metadata" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "camera" in warnings[0].getMessage()


def test_the_probe_stops_converting_once_the_plane_is_stopping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Teardown empties the source table while the SDK still delivers buffers.

    A 13-camera teardown logged 'dropping frames from unmapped mux pad N; known
    pads are []' and then core-dumped, so the probe must go quiet the moment
    stopping begins rather than convert against a table being emptied.
    """
    from types import SimpleNamespace

    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")
    plane.start()
    plane.stop()

    with caplog.at_level(logging.WARNING):
        plane.publish_frame(
            SimpleNamespace(pad_index=0, buffer_pts=1, tensor_items=[], object_items=[])
        )

    assert plane.published_frames("camera") == 0
    assert not [record for record in caplog.records if "unmapped mux pad" in record.getMessage()]


def test_an_isolated_conversion_failure_is_dropped_without_tripping_the_plane() -> None:
    """An exception escaping a probe callback aborts the process inside the SDK.

    A 13-camera run died during an RTSP reconnect, so a frame whose conversion
    raises must be counted and dropped instead of taking the worker down.
    """
    from types import SimpleNamespace

    plane, _ = _plane()
    plane.add_source("camera", "rtsp://one")

    class _Exploding:
        pad_index = 0

        def __getattr__(self, name: str) -> object:
            raise RuntimeError(f"vendor metadata blew up reading {name}")

    plane.publish_frame(_Exploding())

    assert plane.published_frames("camera") == 0
    assert plane.status().fatal_error is None
    plane.publish_frame(
        SimpleNamespace(pad_index=0, buffer_pts=1, tensor_items=[], object_items=[])
    )
    assert plane.published_frames("camera") == 1
    assert plane.status().fatal_error is None


def test_sustained_conversion_failure_trips_only_the_affected_camera() -> None:
    """A broken SDK conversion contract must reach the lifecycle supervisor."""
    plane, _ = _plane()
    plane.add_source("broken", "rtsp://one")
    plane.add_source("healthy", "rtsp://two")

    class _Exploding:
        pad_index = 0

        def __getattr__(self, name: str) -> object:
            raise RuntimeError(f"vendor metadata blew up reading {name}")

    for _ in range(3):
        plane.publish_frame(_Exploding())
    plane.publish_frame(
        SimpleNamespace(pad_index=1, buffer_pts=1, tensor_items=[], object_items=[])
    )

    assert "camera_id=broken" in (plane.status().fatal_error or "")
    assert plane.published_frames("broken") == 0
    assert plane.published_frames("healthy") == 1


def test_perception_counters_are_camera_local() -> None:
    plane, _ = _plane()
    plane.add_source("one", "rtsp://one")
    plane.add_source("two", "rtsp://two")

    plane.publish_frame(
        SimpleNamespace(
            pad_index=0,
            buffer_pts=1,
            tensor_items=[],
            object_items=[
                SimpleNamespace(
                    object_id=1,
                    confidence=0.9,
                    rect_params=SimpleNamespace(left=0.0, top=0.0, width=10.0, height=10.0),
                )
            ],
        )
    )
    plane.publish_frame(
        SimpleNamespace(
            pad_index=1,
            buffer_pts=1,
            tensor_items=[],
            object_items=[
                SimpleNamespace(
                    object_id=index,
                    confidence=0.9,
                    rect_params=SimpleNamespace(left=0.0, top=0.0, width=10.0, height=10.0),
                )
                for index in range(2)
            ],
        )
    )

    assert plane.perception_counters("one") == (1, 1)
    assert plane.perception_counters("two") == (2, 1)

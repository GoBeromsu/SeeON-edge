from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

import numpy as np
import pytest

from contracts.frame import Frame
from worker.interfaces.bus import FrameBus, FrameSubscription
from worker.pipeline.ingest.lifecycle import (
    CameraIngestLoop,
    CameraIngestPorts,
    CameraIngestSpec,
    CapturePolicy,
    IngestEvent,
    IngestSupervisor,
)
from worker.pipeline.ingest.registry import ResolvedSource, SourceRecord, SourceRegistry
from worker.types import FramePacket


@dataclass(frozen=True, slots=True)
class _DecodeConfig:
    camera_id: str
    url: str


@final
class _Session:
    def __init__(self, outcomes: list[FramePacket | Exception | None]) -> None:
        self._outcomes = outcomes
        self.close_count = 0

    def read(self) -> FramePacket | None:
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self.close_count += 1


@final
class _Adapter:
    def __init__(self, outcomes: list[_Session | Exception]) -> None:
        self._outcomes = outcomes
        self.configs: list[_DecodeConfig] = []

    def open(self, config: _DecodeConfig) -> _Session:
        self.configs.append(config)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@final
class _Subscription:
    def take(self, *, timeout_sec: float | None = None) -> FramePacket | None:
        del timeout_sec
        return None

    def close(self) -> None:
        return None


@final
class _PublishError(RuntimeError):
    pass


@final
class _FakeBus:
    def __init__(self, *, fail_camera_id: str | None = None) -> None:
        self.fail_camera_id = fail_camera_id
        self.packets: list[FramePacket] = []
        self.thread_names: list[str] = []
        self.on_publish: Callable[[FramePacket], None] | None = None
        self._lock = threading.Lock()

    def subscribe(
        self,
        name: str,
        *,
        capacity: int,
        latest_only: bool = False,
    ) -> FrameSubscription:
        assert name and capacity > 0
        assert isinstance(latest_only, bool)
        return _Subscription()

    def publish(self, packet: FramePacket) -> None:
        with self._lock:
            self.thread_names.append(threading.current_thread().name)
            should_fail = packet.camera_id == self.fail_camera_id
            if not should_fail:
                self.packets.append(packet)
        if self.on_publish is not None:
            self.on_publish(packet)
        if should_fail:
            raise _PublishError("downstream processing failed")


@final
class _Reporter:
    def __init__(self) -> None:
        self.states: dict[str, str] = {}
        self.categories: dict[str, str] = {}
        self.events: list[IngestEvent] = []
        self.ready_calls: list[str] = []
        self._lock = threading.Lock()

    def mark_starting(self, camera_id: str) -> None:
        with self._lock:
            self.states[camera_id] = "starting"

    def mark_ready(self, camera_id: str) -> None:
        with self._lock:
            self.states[camera_id] = "ready"
            self.ready_calls.append(camera_id)

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        with self._lock:
            self.states[camera_id] = "degraded"
            self.categories[camera_id] = category

    def emit(self, event: IngestEvent) -> None:
        with self._lock:
            self.events.append(event)


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((2, 3, 3), seq, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 3, 2, 0.25)


def _assert_published_packets(actual: list[FramePacket], source: list[FramePacket]) -> None:
    assert len(actual) == len(source)
    for published, decoded in zip(actual, source, strict=True):
        assert published is not decoded
        assert published.frame is decoded.frame
        assert published.lease is decoded.lease
        assert (published.camera_id, published.seq, published.pts) == (
            decoded.camera_id,
            decoded.seq,
            decoded.pts,
        )
        assert published.worker_boot_id
        assert published.stream_epoch > 0


def _registry(*camera_ids: str) -> SourceRegistry:
    records = {
        camera_id: SourceRecord(
            camera_id,
            Path(f"rtsp:/{camera_id}"),
            0.0,
            "",
            kind="live",
            trusted_live=True,
        )
        for camera_id in camera_ids
    }
    return SourceRegistry(records=records)


def _decode_config(camera_id: str, source: ResolvedSource) -> _DecodeConfig:
    return _DecodeConfig(camera_id, f"rtsp://example.test/{source.record.source_id}")


def _spec(camera_id: str, *, source_id: str | None = None) -> CameraIngestSpec[_DecodeConfig]:
    return CameraIngestSpec(
        camera_id=camera_id,
        source_id=camera_id if source_id is None else source_id,
        make_decode_config=_decode_config,
        policy=CapturePolicy(max_failures=1, max_total_reconnects=0, target_fps=1000.0),
    )


def _loop(
    camera_id: str,
    adapter: _Adapter,
    bus: _FakeBus,
    reporter: _Reporter,
    registry: SourceRegistry,
    *,
    source_id: str | None = None,
) -> CameraIngestLoop[_DecodeConfig]:
    ports = CameraIngestPorts(registry=registry, decoder=adapter, bus=bus, reporter=reporter)
    return CameraIngestLoop(_spec(camera_id, source_id=source_id), ports)


def test_registered_loop_publishes_the_exact_packet_and_closes_its_session() -> None:
    # Given
    packets = [_packet("camera-a", 7), _packet("camera-a", 8)]
    session = _Session([*packets])
    adapter = _Adapter([session])
    bus = _FakeBus()
    reporter = _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, _registry("camera-a"))
    bus.on_publish = lambda _packet: loop.stop() if len(bus.packets) == 2 else None

    # When
    loop.run()

    # Then
    assert isinstance(bus, FrameBus)
    assert adapter.configs == [_DecodeConfig("camera-a", "rtsp://example.test/camera-a")]
    _assert_published_packets(bus.packets, packets)
    assert [(packet.seq, packet.pts) for packet in bus.packets] == [(7, 1.4), (8, 1.6)]
    assert reporter.states == {"camera-a": "ready"}
    assert session.close_count == 1


def test_healthy_stream_calls_mark_ready_for_every_packet_not_only_once() -> None:
    # Given: a healthy stream that yields three packets without ever reconnecting.
    packets = [_packet("camera-a", 1), _packet("camera-a", 2), _packet("camera-a", 3)]
    session = _Session([*packets])
    adapter = _Adapter([session])
    bus = _FakeBus()
    reporter = _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, _registry("camera-a"))
    bus.on_publish = lambda _packet: loop.stop() if len(bus.packets) == 3 else None

    # When
    loop.run()

    # Then: mark_ready fires per packet, not once on the READY transition, so the
    # reporter's own heartbeat_interval_sec throttle -- not this loop -- controls
    # how often the relay actually sees a ping.
    assert reporter.ready_calls == ["camera-a", "camera-a", "camera-a"]
    assert len(reporter.ready_calls) == len(packets)


def test_registered_loop_reports_reconnect_and_recovery_without_replacing_packets() -> None:
    # Given
    first = _Session([None])
    recovered_packet = _packet("camera-a", 9)
    second = _Session([recovered_packet])
    adapter = _Adapter([first, second])
    bus = _FakeBus()
    reporter = _Reporter()
    spec = _spec("camera-a")
    spec = CameraIngestSpec(
        spec.camera_id,
        spec.source_id,
        spec.make_decode_config,
        CapturePolicy(max_failures=1, max_total_reconnects=1, target_fps=1000.0),
    )
    loop = CameraIngestLoop(
        spec,
        CameraIngestPorts(_registry("camera-a"), adapter, bus, reporter),
    )
    bus.on_publish = lambda _packet: loop.stop()

    # When
    loop.run()

    # Then
    _assert_published_packets(bus.packets, [recovered_packet])
    assert [event.event_type for event in reporter.events] == [
        "camera.offline",
        "camera.recovered",
    ]
    assert reporter.states == {"camera-a": "ready"}
    assert first.close_count == second.close_count == 1


def test_reconnecting_warning_log_identifies_the_camera(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #113/#115: the retry/reconnect warning must name the camera.

    Before #113, ``_record_reconnecting`` only updated the in-memory status
    store (``mark_degraded``/``emit``) -- there was no console-visible log
    line at all for a camera's reconnect loop, so an operator tailing the
    worker log had no trace of which camera was retrying or why.

    #113's fix passed camera_id/reason via ``extra=`` only, which QA found
    never actually reaches the console: worker/__main__.py's format string
    (``%(asctime)s - %(name)s - %(levelname)s - %(message)s``) doesn't
    render extra fields, so ``record.getMessage()`` -- what a real console
    handler substitutes for ``%(message)s`` -- must itself contain the
    camera_id, not just the LogRecord's extra attribute."""
    # Given
    first = _Session([None])
    recovered_packet = _packet("camera-a", 9)
    second = _Session([recovered_packet])
    adapter = _Adapter([first, second])
    bus = _FakeBus()
    reporter = _Reporter()
    spec = _spec("camera-a")
    spec = CameraIngestSpec(
        spec.camera_id,
        spec.source_id,
        spec.make_decode_config,
        CapturePolicy(max_failures=1, max_total_reconnects=1, target_fps=1000.0),
    )
    loop = CameraIngestLoop(
        spec,
        CameraIngestPorts(_registry("camera-a"), adapter, bus, reporter),
    )
    bus.on_publish = lambda _packet: loop.stop()

    # When
    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()

    # Then
    reconnect_records = [
        record for record in caplog.records if "reconnecting" in record.message
    ]
    assert reconnect_records, "expected a reconnecting warning to be logged"
    assert all(record.camera_id == "camera-a" for record in reconnect_records)  # type: ignore[attr-defined]
    # The console formatter only renders %(message)s, not extra fields, so
    # the camera_id must be baked into the message text itself.
    assert all("camera_id=camera-a" in record.message for record in reconnect_records)


def test_source_construction_failure_marks_only_that_camera_offline() -> None:
    # Given
    adapter = _Adapter([])
    reporter = _Reporter()
    loop = _loop("camera-a", adapter, _FakeBus(), reporter, _registry(), source_id="missing")

    # When
    loop.run()

    # Then
    assert adapter.configs == []
    assert reporter.states == {"camera-a": "degraded"}
    assert reporter.categories == {"camera-a": "SourceRegistryError"}
    assert [event.event_type for event in reporter.events] == ["camera.offline"]


def test_two_camera_supervisor_isolates_read_failure_and_masks_credentials() -> None:
    # Given
    raw_url = "rtsp://operator:s3cr3t@example.test/live?token=plain"
    bad_session = _Session([RuntimeError(f"read failed for {raw_url}")])
    good_packets = [_packet("camera-good", 1), _packet("camera-good", 2)]
    good_session = _Session([*good_packets])
    bus = _FakeBus()
    reporter = _Reporter()
    registry = _registry("camera-bad", "camera-good")
    bad_loop = _loop("camera-bad", _Adapter([bad_session]), bus, reporter, registry)
    good_loop = _loop("camera-good", _Adapter([good_session]), bus, reporter, registry)
    bus.on_publish = lambda packet: good_loop.stop() if packet.seq == 2 else None

    # When
    IngestSupervisor((bad_loop, good_loop)).run()

    # Then
    _assert_published_packets(bus.packets, good_packets)
    assert set(bus.thread_names) == {"worker-ingest-camera-good"}
    assert reporter.states == {"camera-bad": "degraded", "camera-good": "ready"}
    offline_cameras = [
        event.camera_id for event in reporter.events if event.event_type == "camera.offline"
    ]
    assert offline_cameras == ["camera-bad"]
    rendered_events = " ".join(f"{event.category} {event.detail}" for event in reporter.events)
    assert all(secret not in rendered_events for secret in (raw_url, "operator", "s3cr3t", "plain"))
    assert bad_session.close_count == good_session.close_count == 1


def test_publish_processing_failure_is_not_classified_as_camera_offline() -> None:
    # Given
    session = _Session([_packet("camera-a", 1)])
    bus = _FakeBus(fail_camera_id="camera-a")
    reporter = _Reporter()
    loop = _loop("camera-a", _Adapter([session]), bus, reporter, _registry("camera-a"))
    bus.on_publish = lambda _packet: loop.stop()

    # When
    loop.run()

    # Then
    assert reporter.states == {"camera-a": "ready"}
    assert [event.event_type for event in reporter.events] == ["frame.processing_error"]
    assert "camera.offline" not in {event.event_type for event in reporter.events}
    assert session.close_count == 1


@final
class _BlockingLoop:
    """A fake ingest loop that only returns once `stop()` is called."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._stop_event = threading.Event()
        self.stop_count = 0

    def run(self) -> None:
        self._stop_event.wait()

    def stop(self) -> None:
        self.stop_count += 1
        self._stop_event.set()


def test_restart_check_returning_true_stops_the_supervisor_cleanly() -> None:
    # Given: a camera loop that would otherwise block ingesting forever.
    loop = _BlockingLoop("camera-a")
    calls: list[None] = []

    def restart_check() -> bool:
        calls.append(None)
        return True

    supervisor = IngestSupervisor(
        (loop,), restart_check=restart_check, restart_poll_interval_sec=0.01
    )

    # When: the supervisor starts and the restart watcher observes a True directive.
    supervisor.start()
    supervisor.join(timeout_sec=2.0)

    # Then: the stop event breaks both the ingest loop and the restart watch loop.
    assert loop.stop_count >= 1
    assert len(calls) >= 1


def test_omitting_restart_check_preserves_default_clean_completion() -> None:
    # Given: a camera that reports two packets then stops itself, matching pre-restart-check
    # supervisor behavior exactly.
    packets = [_packet("camera-a", 1), _packet("camera-a", 2)]
    session = _Session([*packets])
    adapter = _Adapter([session])
    bus = _FakeBus()
    reporter = _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, _registry("camera-a"))
    bus.on_publish = lambda _packet: loop.stop() if len(bus.packets) == 2 else None

    # When: restart_check is omitted entirely (the default).
    supervisor = IngestSupervisor((loop,))
    supervisor.run()

    # Then: ingestion completes exactly as it did before restart_check existed.
    _assert_published_packets(bus.packets, packets)
    assert reporter.states == {"camera-a": "ready"}
    assert session.close_count == 1


def test_completion_check_returning_true_stops_the_supervisor_cleanly() -> None:
    # Given: a camera loop that would otherwise block ingesting forever, and a
    # completion directive (the --max-frames-per-camera analog of restart_check).
    loop = _BlockingLoop("camera-a")
    calls: list[None] = []

    def completion_check() -> bool:
        calls.append(None)
        return True

    supervisor = IngestSupervisor(
        (loop,), completion_check=completion_check, completion_poll_interval_sec=0.01
    )

    # When: the supervisor starts and the completion watcher observes a True directive.
    supervisor.start()
    supervisor.join(timeout_sec=2.0)

    # Then: the stop event breaks both the ingest loop and the completion watch loop.
    assert loop.stop_count >= 1
    assert len(calls) >= 1


def test_completion_check_waits_until_every_camera_reports_done() -> None:
    # Given: two cameras with independently-tracked completion state, mirroring
    # --max-frames-per-camera's "every camera must reach its cap" contract --
    # not just the first one to finish.
    loop_a = _BlockingLoop("camera-a")
    loop_b = _BlockingLoop("camera-b")
    done = {"camera-a": False, "camera-b": False}

    supervisor = IngestSupervisor(
        (loop_a, loop_b),
        completion_check=lambda: all(done.values()),
        completion_poll_interval_sec=0.01,
    )

    # When: only camera-a finishes first, giving the watcher time to poll at least once.
    supervisor.start()
    done["camera-a"] = True
    threading.Event().wait(timeout=0.05)

    # Then: the supervisor must not stop while camera-b is still outstanding.
    assert loop_a.stop_count == 0
    assert loop_b.stop_count == 0

    # When: camera-b finishes too.
    done["camera-b"] = True
    supervisor.join(timeout_sec=2.0)

    # Then: both loops are stopped only once every camera is done.
    assert loop_a.stop_count >= 1
    assert loop_b.stop_count >= 1


def test_omitting_completion_check_preserves_default_clean_completion() -> None:
    # Given: a camera that reports two packets then stops itself, matching
    # pre-completion-check supervisor behavior exactly (no watcher spawned).
    packets = [_packet("camera-a", 1), _packet("camera-a", 2)]
    session = _Session([*packets])
    adapter = _Adapter([session])
    bus = _FakeBus()
    reporter = _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, _registry("camera-a"))
    bus.on_publish = lambda _packet: loop.stop() if len(bus.packets) == 2 else None

    # When: completion_check is omitted entirely (the default).
    supervisor = IngestSupervisor((loop,))
    supervisor.run()

    # Then: ingestion completes exactly as it did before completion_check existed.
    _assert_published_packets(bus.packets, packets)
    assert reporter.states == {"camera-a": "ready"}
    assert session.close_count == 1


def test_restart_check_always_false_is_consulted_but_never_stops_early() -> None:
    # Given: a camera that reports two packets, waiting briefly before stopping itself so the
    # restart watcher has time to poll at least once beforehand.
    packets = [_packet("camera-a", 1), _packet("camera-a", 2)]
    session = _Session([*packets])
    adapter = _Adapter([session])
    bus = _FakeBus()
    reporter = _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, _registry("camera-a"))
    calls: list[None] = []

    def never_restart() -> bool:
        calls.append(None)
        return False

    def on_publish(_packet: FramePacket) -> None:
        if len(bus.packets) == 2:
            threading.Event().wait(timeout=0.05)
            loop.stop()

    bus.on_publish = on_publish
    supervisor = IngestSupervisor(
        (loop,), restart_check=never_restart, restart_poll_interval_sec=0.01
    )

    # When
    supervisor.run()
    supervisor.stop()

    # Then: the always-false directive is consulted at least once, and ingestion still
    # completes in full instead of terminating early.
    _assert_published_packets(bus.packets, packets)
    assert reporter.states == {"camera-a": "ready"}
    assert len(calls) >= 1


def test_stop_never_races_a_partially_started_thread_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stop()`` must never observe a Thread ``start()`` has constructed but
    not yet actually started.

    Regression test for the CI flake that hit main right after PR #60
    merged: ``IngestSupervisor.start()`` used to publish ``self._threads``
    before every ``Thread`` in it had its ``.start()`` called, so a ``stop()``
    racing with an in-flight ``start()`` could ``Thread.join()`` an
    unstarted thread and raise
    ``RuntimeError: cannot join thread before it is started``.

    Deterministic by construction -- no sleeps as synchronization. A pair of
    ``threading.Event`` barriers pins the exact interleaving: camera-a's
    thread is allowed to actually start and begin running, camera-b's real
    ``Thread.start()`` is held until this test explicitly releases it, and
    ``stop()`` is driven from a third thread in that exact window. Pre-fix,
    ``self._threads`` already holds camera-b's constructed-but-not-started
    ``Thread`` at that instant, so ``stop()``'s ``join()`` raises. Post-fix,
    ``self._threads`` is still empty at that instant (not published until
    every thread's real ``.start()`` has returned), so ``join()`` is a no-op
    and ``stop()`` returns cleanly.
    """
    camera_a_running = threading.Event()
    release_camera_b_start = threading.Event()
    real_start = threading.Thread.start

    def patched_start(self: threading.Thread) -> None:
        if self.name == "worker-ingest-camera-b":
            assert release_camera_b_start.wait(
                timeout=5.0
            ), "test never released camera-b's Thread.start()"
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", patched_start)

    loop_a = _BlockingLoop("camera-a")
    loop_b = _BlockingLoop("camera-b")
    original_run_a = loop_a.run

    def running_run_a() -> None:
        camera_a_running.set()
        original_run_a()

    loop_a.run = running_run_a  # type: ignore[method-assign]

    supervisor = IngestSupervisor((loop_a, loop_b))
    starter = threading.Thread(target=supervisor.start)

    errors: list[BaseException] = []

    def guarded_stop() -> None:
        try:
            supervisor.stop(join_timeout_sec=2.0)
        except BaseException as exc:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - capture for assertion
            errors.append(exc)

    # When: start() is mid-flight (camera-a's thread is up, camera-b's real
    # start() is held), stop() races in from a separate thread.
    starter.start()
    try:
        assert camera_a_running.wait(timeout=5.0), "camera-a thread never started running"
        stopper = threading.Thread(target=guarded_stop)
        stopper.start()
        stopper.join(timeout=5.0)
        assert not stopper.is_alive(), "stop() never returned"
    finally:
        release_camera_b_start.set()
        starter.join(timeout=5.0)

    # Then: start() finished, stop() raised nothing, and both loops still
    # end up cleanly stopped once camera-b's thread is released to run.
    assert not starter.is_alive(), "start() never returned"
    assert not errors, f"stop() raised: {errors!r}"
    supervisor.join(timeout_sec=2.0)
    assert loop_a.stop_count >= 1
    assert loop_b.stop_count >= 1


# --- issue #150: zero-camera roster ------------------------------------------


def test_zero_loops_join_returns_immediately() -> None:
    """Pins the exact failure mode issue #150 fixes: with no ingest loops at
    all, ``join()`` has nothing to wait on and returns right away -- a caller
    relying on it to keep a process alive would exit immediately after boot.
    ``wait_until_stopped()`` (below) is the replacement for that case."""
    supervisor = IngestSupervisor(())

    supervisor.start()
    supervisor.join(timeout_sec=5.0)  # must not hang even with no timeout


def test_zero_loops_wait_until_stopped_blocks_until_an_external_stop() -> None:
    """With zero cameras, ``wait_until_stopped()`` must actually block --
    unlike ``join()`` above -- until something calls ``stop()``."""
    supervisor = IngestSupervisor(())
    supervisor.start()
    unblocked = threading.Event()

    def waiter() -> None:
        supervisor.wait_until_stopped()
        unblocked.set()

    thread = threading.Thread(target=waiter)
    thread.start()
    try:
        # Given: nothing has stopped the supervisor yet.
        assert not unblocked.wait(timeout=0.2)

        # When: an external stop() runs (mirrors WorkerRuntime.stop() on SIGTERM).
        supervisor.stop()

        # Then: the wait unblocks.
        assert unblocked.wait(timeout=2.0)
    finally:
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def test_zero_loops_wait_until_stopped_unblocks_when_restart_check_fires() -> None:
    """With zero cameras, the restart watcher is what is supposed to end the
    wait once a config pull reports a fresh directive (e.g. the first camera
    got registered) -- proving the same mechanism issue #150 relies on to let
    a zero-camera worker exit and restart into the real pipeline."""
    calls: list[None] = []

    def restart_check() -> bool:
        calls.append(None)
        return True

    supervisor = IngestSupervisor(
        (), restart_check=restart_check, restart_poll_interval_sec=0.01
    )
    supervisor.start()

    supervisor.wait_until_stopped()

    assert len(calls) >= 1

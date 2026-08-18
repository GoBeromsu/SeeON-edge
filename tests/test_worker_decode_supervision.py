"""Per-camera decode supervision (plan todo 6).

Todo 4 moved video decode into a per-camera ffmpeg child behind
``NvdecPacketTeeSession``.  That created two failure modes the pre-existing
ingest machinery did not cover:

* **transient decode silence** -- the child is alive but emits nothing;
  ``session.read()`` returns ``None`` forever, so ``RTSPSource``'s
  count-based ``max_failures`` reconnect only fires after an unbounded
  amount of wall time (each ``None`` costs at most one read timeout, and a
  child that returns ``None`` promptly burns the budget in milliseconds
  while a child that blocks burns minutes).
* **dead child** -- ``NvdecUnavailableError`` (a ``RuntimeError``) escapes
  ``session.read()``, past ``RTSPSource.__iter__`` (which only guards
  ``open``), into ``CameraIngestLoop.run()``.  Before this todo that marked
  the camera DEGRADED and *returned*: the loop thread ended and the camera
  never came back for the life of the process.

This module pins the supervision contract that closes both:
respawn-with-bounded-exponential-backoff inside ``CameraIngestLoop``,
persistent DEGRADED while the decode path is down, recovery on the first
packet from a respawned source, and -- critically -- that none of it ever
reaches ``FaultHandler``: a dead or hung *camera* must never arm the 30s
accelerator deadline, which stays reserved for a genuinely hung forward
pass.

Every timing assertion here runs on an injected fake clock; nothing sleeps.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import final

import numpy as np
import pytest

import worker.runtime.ingest_composition as ingest_composition_module
from contracts.frame import Frame
from worker.interfaces.bus import FrameSubscription
from worker.pipeline.bus import BoundedFrameBus
from worker.pipeline.ingest.lifecycle import (
    DECODE_RESPAWN_BACKOFF_CAP_SEC,
    DECODE_RESPAWN_INITIAL_BACKOFF_SEC,
    DECODE_SILENCE_DEADLINE_SEC,
    CameraIngestLoop,
    CameraIngestPorts,
    CameraIngestSpec,
    CapturePolicy,
    DecodeSupervisionPolicy,
    IngestEvent,
    IngestSupervisor,
)
from worker.pipeline.ingest.registry import ResolvedSource, SourceRecord, SourceRegistry
from worker.runtime.config import WorkerConfig
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.worker import WorkerRuntime
from worker.types import FramePacket


@dataclass(frozen=True, slots=True)
class _DecodeConfig:
    camera_id: str
    url: str


@final
class _FakeClock:
    """Monotonic clock advanced only by explicit test action or by waits."""

    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    def wait(self, delay_sec: float) -> bool:
        """Stand-in for ``threading.Event.wait``: advances, never blocks."""
        self.waits.append(delay_sec)
        self.now += delay_sec
        return False


@final
class _Session:
    """A decode session scripted with per-read outcomes.

    ``None`` models decode silence (child alive, no frame); an ``Exception``
    models the loud dead-child family from todo 4; a float advances the fake
    clock before the outcome is produced, so silence can be given a duration
    without any real waiting.
    """

    def __init__(
        self,
        outcomes: list[FramePacket | Exception | None | float],
        clock: _FakeClock,
    ) -> None:
        self._outcomes = list(outcomes)
        self._clock = clock
        self.close_count = 0

    def read(self) -> FramePacket | None:
        while self._outcomes:
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, float):
                self._clock.now += outcome
                continue
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return None

    def close(self) -> None:
        self.close_count += 1


@final
class _Adapter:
    """Decode adapter whose ``open`` hands out scripted sessions in order."""

    def __init__(self, outcomes: list[_Session | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.open_count = 0

    def open(self, config: _DecodeConfig) -> _Session:
        del config
        self.open_count += 1
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if outcome is None:
            raise AssertionError("adapter opened more sessions than the test scripted")
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
class _FakeBus:
    def __init__(self) -> None:
        self.packets: list[FramePacket] = []
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
            self.packets.append(packet)
        if self.on_publish is not None:
            self.on_publish(packet)


@final
class _Reporter:
    def __init__(self) -> None:
        self.states: dict[str, str] = {}
        self.categories: dict[str, str] = {}
        self.events: list[IngestEvent] = []
        self.state_history: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def mark_starting(self, camera_id: str) -> None:
        with self._lock:
            self.states[camera_id] = "starting"
            self.state_history.append((camera_id, "starting"))

    def mark_ready(self, camera_id: str) -> None:
        with self._lock:
            self.states[camera_id] = "ready"
            self.state_history.append((camera_id, "ready"))

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        with self._lock:
            self.states[camera_id] = "degraded"
            self.categories[camera_id] = category
            self.state_history.append((camera_id, "degraded"))

    def emit(self, event: IngestEvent) -> None:
        with self._lock:
            self.events.append(event)


@final
class _FaultHandlerSpy:
    """Stands in for ``FaultHandler``; supervision must never touch it."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def handle(self, exc: object, record: object) -> None:  # pragma: no cover - must not run
        self.calls.append((exc, record))


class _DeadDecoderError(RuntimeError):
    """Stands in for ``NvdecUnavailableError`` without importing the adapter."""


def _packet(camera_id: str, seq: int) -> FramePacket:
    image = np.full((2, 3, 3), seq % 256, dtype=np.uint8)
    frame = Frame(index=seq, time_sec=seq / 5.0, image=image)
    return FramePacket(camera_id, frame, seq / 5.0, seq, 3, 2, 0.25)


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


def _loop(
    camera_id: str,
    adapter: _Adapter,
    bus: _FakeBus,
    reporter: _Reporter,
    clock: _FakeClock,
    *,
    max_respawns: int = 3,
    max_failures: int = 1,
) -> CameraIngestLoop[_DecodeConfig]:
    """A loop whose RTSP-level reconnect is neutralized.

    ``max_total_reconnects=0`` makes ``RTSPSource`` give up its own reopen
    immediately, so every observation below is attributable to the decode
    supervision layer under test rather than to the pre-existing RTSP
    reconnect loop.  Silence tests additionally raise ``max_failures`` so the
    count-based RTSP trigger cannot fire before the time-based decode
    deadline -- proving the new gate is what acted, not the old one.
    """
    spec = CameraIngestSpec(
        camera_id=camera_id,
        source_id=camera_id,
        make_decode_config=_decode_config,
        policy=CapturePolicy(
            max_failures=max_failures, max_total_reconnects=0, target_fps=1000.0
        ),
        decode_supervision=DecodeSupervisionPolicy(max_respawns=max_respawns),
    )
    ports = CameraIngestPorts(
        registry=_registry(camera_id),
        decoder=adapter,
        bus=bus,
        reporter=reporter,
    )
    return CameraIngestLoop(spec, ports, clock=clock, respawn_wait=clock.wait)


# --- silence -> respawn -------------------------------------------------------


def test_silent_camera_respawns_its_decode_path_and_recovers() -> None:
    """T seconds without a frame kills and respawns the decode path.

    The first session stays alive but silent for longer than
    ``DECODE_SILENCE_DEADLINE_SEC``; the supervision layer must close it,
    open a fresh one, and the camera must return to READY on the first
    packet the replacement produces.
    """
    # Given: a session silent past the deadline, then a healthy replacement.
    clock = _FakeClock()
    silent = _Session([None, DECODE_SILENCE_DEADLINE_SEC + 0.1, None], clock)
    recovered = _packet("camera-a", 1)
    healthy = _Session([recovered], clock)
    adapter = _Adapter([silent, healthy])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock, max_failures=100)
    bus.on_publish = lambda _packet: loop.stop()

    # When
    loop.run()

    # Then: the silent session was closed, a replacement was opened, and the
    # camera reports READY again off the replacement's first packet.
    assert silent.close_count == 1
    assert adapter.open_count == 2
    assert [packet.seq for packet in bus.packets] == [1]
    assert reporter.states == {"camera-a": "ready"}
    assert ("camera-a", "degraded") in reporter.state_history
    assert reporter.state_history[-1] == ("camera-a", "ready")


def test_silence_below_the_deadline_never_respawns() -> None:
    """Silence shorter than T is normal decode jitter, not a stall."""
    # Given: a gap just under the deadline, then a frame.
    clock = _FakeClock()
    session = _Session(
        [None, DECODE_SILENCE_DEADLINE_SEC - 0.1, _packet("camera-a", 1)], clock
    )
    adapter = _Adapter([session])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock, max_failures=100)
    bus.on_publish = lambda _packet: loop.stop()

    # When
    loop.run()

    # Then
    assert adapter.open_count == 1
    assert session.close_count == 1
    assert reporter.states == {"camera-a": "ready"}
    assert not any(state == "degraded" for _camera, state in reporter.state_history)


def test_dead_decode_child_respawns_instead_of_ending_the_camera() -> None:
    """The loud dead-child error is recoverable, not terminal.

    Before this todo the ``RuntimeError`` family raised by the decoder
    subprocess wrapper ended ``CameraIngestLoop.run()`` outright, so the
    camera's ingest thread exited and never restarted.
    """
    # Given
    clock = _FakeClock()
    dead = _Session([_DeadDecoderError("ffmpeg decoder exited")], clock)
    recovered = _packet("camera-a", 4)
    healthy = _Session([recovered], clock)
    adapter = _Adapter([dead, healthy])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()

    # When
    loop.run()

    # Then
    assert adapter.open_count == 2
    assert [packet.seq for packet in bus.packets] == [4]
    assert reporter.states == {"camera-a": "ready"}
    offline = [event for event in reporter.events if event.event_type == "camera.offline"]
    recoveries = [event for event in reporter.events if event.event_type == "camera.recovered"]
    assert len(offline) == 1
    assert len(recoveries) == 1


def test_decode_respawn_logs_the_camera_id_in_the_rendered_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The console formatter renders only ``%(message)s`` (AGENTS.md "침묵하는
    extra= 로그"), so camera_id must be in the message text itself."""
    # Given
    clock = _FakeClock()
    dead = _Session([_DeadDecoderError("ffmpeg decoder exited")], clock)
    healthy = _Session([_packet("camera-a", 1)], clock)
    adapter = _Adapter([dead, healthy])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()

    # When
    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()

    # Then
    respawn_records = [
        record for record in caplog.records if "decode respawn" in record.getMessage()
    ]
    assert respawn_records, "expected a decode respawn warning"
    assert all("camera_id=camera-a" in record.getMessage() for record in respawn_records)


def test_current_respawn_logs_keep_status_extra_and_existing_message_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Baseline: wrappers/status/extra stay put; detail is still absent."""
    # Given
    clock = _FakeClock()
    dead = _Session([_DeadDecoderError("ffmpeg decoder exited")], clock)
    healthy = _Session([_packet("camera-a", 1)], clock)
    adapter = _Adapter([dead, healthy])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()

    # When
    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()

    # Then: status category remains the exception type, extra fields stay mapped.
    assert reporter.categories["camera-a"] == "_DeadDecoderError"
    offline = [event for event in reporter.events if event.event_type == "camera.offline"]
    assert offline and offline[0].category == "_DeadDecoderError"
    respawn_records = [
        record for record in caplog.records if "camera decode respawn:" in record.getMessage()
    ]
    assert len(respawn_records) == 1
    message = respawn_records[0].getMessage()
    assert "camera_id=camera-a" in message
    assert "attempt=1/3" in message
    assert "reason=_DeadDecoderError" in message
    assert "backoff=0.5s" in message
    assert "detail=unavailable" in message
    assert "ffmpeg decoder exited" not in message


def test_current_exhausted_logs_keep_status_extra_and_existing_message_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Baseline: budget-exhausted status/extra stay put; detail is still absent."""
    # Given
    clock = _FakeClock()
    sessions = [
        _Session([_DeadDecoderError("ffmpeg decoder exited")], clock) for _ in range(4)
    ]
    adapter = _Adapter(list(sessions))
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock, max_respawns=3)

    # When
    with caplog.at_level("ERROR", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()

    # Then
    assert reporter.states == {"camera-a": "degraded"}
    assert reporter.categories["camera-a"] == "_DeadDecoderError"
    exhausted_records = [
        record
        for record in caplog.records
        if "respawn budget exhausted" in record.getMessage()
    ]
    assert len(exhausted_records) == 1
    message = exhausted_records[0].getMessage()
    assert "camera_id=camera-a" in message
    assert "attempts=3" in message
    assert "reason=_DeadDecoderError" in message
    assert "DEGRADED" in message
    assert "detail=unavailable" in message
    assert "ffmpeg decoder exited" not in message
    assert reporter.events[0].category == "_DeadDecoderError"


def test_respawn_and_exhausted_messages_render_safe_chained_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Visible detail comes from the first safe_log_detail in the cause chain."""
    from worker.adapters.decode.nvdec_cuvid.errors import NvdecUnavailableError

    class _NonStringSafe(RuntimeError):
        safe_log_detail: int = 123

    class _BlankSafe(RuntimeError):
        safe_log_detail: str = "   "

    root = NvdecUnavailableError("cuvid decode failed: codec not supported", returncode=69)
    mid = _BlankSafe("ignored")
    mid.__cause__ = root
    wrapper = RuntimeError("packet-preserving NVDEC decode failed (NvdecUnavailableError)")
    wrapper.__cause__ = mid
    unsafe = _NonStringSafe("outer")
    unsafe.__cause__ = wrapper

    clock = _FakeClock()
    dead = _Session([unsafe], clock)
    healthy = _Session([_packet("camera-a", 1)], clock)
    adapter = _Adapter([dead, healthy])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()

    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()

    respawn_records = [
        record for record in caplog.records if "camera decode respawn:" in record.getMessage()
    ]
    assert respawn_records
    message = respawn_records[0].getMessage()
    assert "camera_id=camera-a" in message
    assert "attempt=1/3" in message
    assert "reason=_NonStringSafe" in message
    assert "detail=NvdecUnavailableError: cuvid decode failed: codec not supported" in message
    assert reporter.categories["camera-a"] == "_NonStringSafe"

    clock = _FakeClock()
    adapter = _Adapter([_Session([unsafe], clock) for _ in range(4)])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock, max_respawns=3)
    with caplog.at_level("ERROR", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()
    exhausted = [
        record for record in caplog.records if "respawn budget exhausted" in record.getMessage()
    ]
    assert exhausted
    exhausted_message = exhausted[-1].getMessage()
    assert "camera_id=camera-a" in exhausted_message
    assert "attempts=3" in exhausted_message
    assert "reason=_NonStringSafe" in exhausted_message
    assert "DEGRADED" in exhausted_message
    assert (
        "detail=NvdecUnavailableError: cuvid decode failed: codec not supported"
        in exhausted_message
    )
    assert reporter.categories["camera-a"] == "_NonStringSafe"


def test_cycle_and_arbitrary_exception_strings_never_become_detail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Four-link cycle-safe walk must ignore raw exception text."""
    first = RuntimeError("rtsp://admin:secret@camera/token=abc")
    second = RuntimeError("outer")
    first.__cause__ = second
    second.__cause__ = first

    clock = _FakeClock()
    dead = _Session([first], clock)
    healthy = _Session([_packet("camera-a", 1)], clock)
    adapter = _Adapter([dead, healthy])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()

    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()

    respawn_records = [
        record for record in caplog.records if "camera decode respawn:" in record.getMessage()
    ]
    assert respawn_records
    message = respawn_records[0].getMessage()
    assert "detail=unavailable" in message
    assert "secret" not in message
    assert "abc" not in message
    assert reporter.categories["camera-a"] == "RuntimeError"


def test_raising_safe_log_detail_falls_back_to_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising safe_log_detail property must not abort the walk."""
    from worker.adapters.decode.nvdec_cuvid.errors import NvdecUnavailableError

    class _RaisingSafe(RuntimeError):
        @property
        def safe_log_detail(self) -> str:
            raise RuntimeError("safe_log_detail boom")

    root = NvdecUnavailableError("cuvid decode failed: codec not supported")
    wrapper = _RaisingSafe("outer")
    wrapper.__cause__ = root

    clock = _FakeClock()
    dead = _Session([wrapper], clock)
    healthy = _Session([_packet("camera-a", 1)], clock)
    adapter = _Adapter([dead, healthy])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()

    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()

    respawn_records = [
        record for record in caplog.records if "camera decode respawn:" in record.getMessage()
    ]
    assert respawn_records
    message = respawn_records[0].getMessage()
    assert (
        "detail=NvdecUnavailableError: cuvid decode failed: codec not supported"
        in message
    )
    assert "safe_log_detail boom" not in message
    assert reporter.categories["camera-a"] == "_RaisingSafe"

    only_raiser = _RaisingSafe("solo")
    clock = _FakeClock()
    adapter = _Adapter(
        [_Session([only_raiser], clock), _Session([_packet("camera-a", 2)], clock)]
    )
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()
    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()
    fallback = [
        record for record in caplog.records if "camera decode respawn:" in record.getMessage()
    ][-1].getMessage()
    assert "detail=unavailable" in fallback


def test_oserror_from_safe_log_detail_does_not_abort_supervision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """OSError from safe_log_detail must stay inside the logging walk."""
    from worker.adapters.decode.nvdec_cuvid.errors import NvdecUnavailableError

    class _OSErrorSafe(RuntimeError):
        @property
        def safe_log_detail(self) -> str:
            raise OSError("safe_log_detail oserror")

    root = NvdecUnavailableError("cuvid decode failed: codec not supported")
    wrapper = _OSErrorSafe("outer")
    wrapper.__cause__ = root
    clock = _FakeClock()
    adapter = _Adapter(
        [_Session([wrapper], clock), _Session([_packet("camera-a", 1)], clock)]
    )
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()
    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()
    message = [
        record.getMessage()
        for record in caplog.records
        if "camera decode respawn:" in record.getMessage()
    ][0]
    assert "detail=NvdecUnavailableError: cuvid decode failed: codec not supported" in message
    assert "safe_log_detail oserror" not in message
    assert reporter.categories["camera-a"] == "_OSErrorSafe"


def test_custom_exception_from_safe_log_detail_does_not_abort_supervision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A custom Exception subclass from safe_log_detail must not escape."""

    class _DetailBoom(Exception):
        pass

    class _CustomSafe(RuntimeError):
        @property
        def safe_log_detail(self) -> str:
            raise _DetailBoom("safe_log_detail custom")

    only_raiser = _CustomSafe("solo")
    clock = _FakeClock()
    adapter = _Adapter(
        [_Session([only_raiser], clock), _Session([_packet("camera-a", 2)], clock)]
    )
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()
    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()
    message = [
        record.getMessage()
        for record in caplog.records
        if "camera decode respawn:" in record.getMessage()
    ][-1]
    assert "detail=unavailable" in message
    assert "safe_log_detail custom" not in message
    assert reporter.categories["camera-a"] == "_CustomSafe"


def test_safe_detail_is_found_four_cause_links_below_the_wrapper(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Inspect the exception four __cause__ links below the logged wrapper."""
    from worker.adapters.decode.nvdec_cuvid.errors import NvdecUnavailableError

    root = NvdecUnavailableError("cuvid four links down")
    link3 = RuntimeError("link-3")
    link3.__cause__ = root
    link2 = RuntimeError("link-2")
    link2.__cause__ = link3
    link1 = RuntimeError("link-1")
    link1.__cause__ = link2
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = link1

    clock = _FakeClock()
    dead = _Session([wrapper], clock)
    healthy = _Session([_packet("camera-a", 1)], clock)
    adapter = _Adapter([dead, healthy])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()

    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()

    message = [
        record.getMessage()
        for record in caplog.records
        if "camera decode respawn:" in record.getMessage()
    ][0]
    assert "detail=NvdecUnavailableError: cuvid four links down" in message
    assert reporter.categories["camera-a"] == "RuntimeError"

    beyond = RuntimeError("beyond")
    beyond.__cause__ = wrapper
    clock = _FakeClock()
    adapter = _Adapter(
        [_Session([beyond], clock), _Session([_packet("camera-a", 3)], clock)]
    )
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)
    bus.on_publish = lambda _packet: loop.stop()
    with caplog.at_level("WARNING", logger="worker.pipeline.ingest.lifecycle"):
        loop.run()
    beyond_message = [
        record.getMessage()
        for record in caplog.records
        if "camera decode respawn:" in record.getMessage()
    ][-1]
    assert "detail=unavailable" in beyond_message


# --- bounded backoff, permanent death ----------------------------------------


def test_respawn_backoff_grows_exponentially_and_is_capped() -> None:
    """Backoff doubles per consecutive respawn and never exceeds the cap."""
    # Given: a camera whose decode path is dead on every attempt.
    clock = _FakeClock()
    sessions = [
        _Session([_DeadDecoderError("ffmpeg decoder exited")], clock) for _ in range(9)
    ]
    adapter = _Adapter(list(sessions))
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock, max_respawns=8)

    # When
    loop.run()

    # Then: the observed waits are the doubling sequence, clamped at the cap.
    expected = [
        min(DECODE_RESPAWN_INITIAL_BACKOFF_SEC * (2.0**attempt), DECODE_RESPAWN_BACKOFF_CAP_SEC)
        for attempt in range(8)
    ]
    assert clock.waits == expected
    assert max(clock.waits) <= DECODE_RESPAWN_BACKOFF_CAP_SEC
    assert clock.waits[-1] == DECODE_RESPAWN_BACKOFF_CAP_SEC


def test_permanently_dead_camera_stops_respawning_and_stays_degraded() -> None:
    """Bounded retries: a permanently dead camera must not loop forever.

    QA failure scenario -- respawn fails three times, the camera stays
    DEGRADED, backoff stayed capped, and the loop returns instead of
    spinning (which is what keeps the process alive for its siblings).
    """
    # Given
    clock = _FakeClock()
    sessions = [
        _Session([_DeadDecoderError("ffmpeg decoder exited")], clock) for _ in range(4)
    ]
    adapter = _Adapter(list(sessions))
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock, max_respawns=3)

    # When
    loop.run()

    # Then: exactly 1 initial open + 3 respawns, then it gives up DEGRADED.
    assert adapter.open_count == 4
    assert len(clock.waits) == 3
    assert max(clock.waits) <= DECODE_RESPAWN_BACKOFF_CAP_SEC
    assert reporter.states == {"camera-a": "degraded"}
    assert reporter.categories["camera-a"] == "_DeadDecoderError"
    assert [event.event_type for event in reporter.events] == ["camera.offline"]


def test_a_dead_camera_never_stops_its_siblings() -> None:
    """Multi-camera isolation: one permanently dead decode path must leave
    every other camera streaming, on its own thread, untouched."""
    # Given: camera-bad is dead on every attempt; camera-good is healthy.
    clock = _FakeClock()
    bad_adapter = _Adapter(
        [_Session([_DeadDecoderError("ffmpeg decoder exited")], clock) for _ in range(4)]
    )
    good_packets = [_packet("camera-good", 1), _packet("camera-good", 2)]
    good_adapter = _Adapter([_Session(list(good_packets), clock)])
    bus, reporter = _FakeBus(), _Reporter()
    bad_loop = _loop("camera-bad", bad_adapter, bus, reporter, clock, max_respawns=3)
    good_loop = _loop("camera-good", good_adapter, bus, reporter, clock)
    bus.on_publish = lambda packet: good_loop.stop() if packet.seq == 2 else None

    # When
    IngestSupervisor((bad_loop, good_loop)).run()

    # Then
    assert [packet.seq for packet in bus.packets] == [1, 2]
    assert reporter.states == {"camera-bad": "degraded", "camera-good": "ready"}
    assert good_adapter.open_count == 1
    assert bad_adapter.open_count == 4


# --- watchdog responsibility split -------------------------------------------


def test_decode_supervision_never_reaches_the_fault_handler() -> None:
    """A dead or hung *camera* must never arm the accelerator fault boundary.

    The 30s ``InferenceWatchdog`` deadline is reserved for a genuinely hung
    forward pass; camera-gap monitoring lives in this supervision layer.  The
    spy stands in for the composition root's ``FaultHandler``: silence,
    dead-child respawns, and permanent give-up must all leave it untouched.
    """
    # Given: a camera that is silent, then dead, then permanently dead.
    clock = _FakeClock()
    handler = _FaultHandlerSpy()
    sessions = [
        _Session([None, DECODE_SILENCE_DEADLINE_SEC + 0.1, None], clock),
        _Session([_DeadDecoderError("ffmpeg decoder exited")], clock),
        _Session([_DeadDecoderError("ffmpeg decoder exited")], clock),
        _Session([_DeadDecoderError("ffmpeg decoder exited")], clock),
    ]
    adapter = _Adapter(list(sessions))
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop(
        "camera-a", adapter, bus, reporter, clock, max_respawns=3, max_failures=100
    )

    # When
    loop.run()

    # Then
    assert handler.calls == []
    assert reporter.states == {"camera-a": "degraded"}


def test_fatal_accelerator_errors_still_escape_decode_supervision() -> None:
    """Supervision must not swallow a real accelerator fault.

    ``FatalAcceleratorError`` is the one failure the ingest loop deliberately
    re-raises so ``_FaultAwareLoop`` can drive ``FaultHandler``; the respawn
    loop must keep that path intact rather than treating the accelerator
    fault as a respawnable decode stall.
    """
    from worker.adapters.model.errors import FatalAcceleratorError

    # Given
    clock = _FakeClock()
    fatal = FatalAcceleratorError("CUDA context is unusable", camera_id="camera-a")
    adapter = _Adapter([_Session([fatal], clock)])
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock)

    # When / Then
    with pytest.raises(FatalAcceleratorError):
        loop.run()
    assert adapter.open_count == 1
    assert clock.waits == []


def test_supervision_stops_promptly_on_shutdown() -> None:
    """``stop()`` during backoff must end the loop instead of finishing the
    remaining respawn budget -- otherwise shutdown waits out the cap."""
    # Given: a permanently dead camera whose loop is stopped mid-backoff.
    clock = _FakeClock()
    adapter = _Adapter(
        [_Session([_DeadDecoderError("ffmpeg decoder exited")], clock) for _ in range(4)]
    )
    bus, reporter = _FakeBus(), _Reporter()
    loop = _loop("camera-a", adapter, bus, reporter, clock, max_respawns=3)

    def stop_during_backoff(delay_sec: float) -> bool:
        clock.waits.append(delay_sec)
        clock.now += delay_sec
        loop.stop()
        return True

    loop = _loop("camera-a", adapter, bus, reporter, clock, max_respawns=3)
    object.__setattr__(loop, "_respawn_wait", stop_during_backoff)

    # When
    loop.run()

    # Then: one dead session, one backoff, then shutdown -- no further opens.
    assert adapter.open_count == 1
    assert len(clock.waits) == 1
    assert reporter.states == {"camera-a": "degraded"}


# --- decode-backend observability wiring -------------------------------------
#
# `WorkerDiagnostics.record_decode_backend()` is provided by the runtime
# decode-selection lane; the composition root is what has to actually call it.
# Without a call from the real camera-loop composition point the field ships
# dead -- boot succeeds, the snapshot stays empty, and the feature exists only
# in its own unit test (worker/runtime/AGENTS.md "no stub wiring at this
# composition root", root anti-pattern #81).


@final
class _FakeServingClient:
    def create(self, task: str, **_options: object) -> object:
        raise AssertionError(f"decode supervision tests must not create a model: {task}")


@final
class _NullReporter:
    def mark_starting(self, camera_id: str) -> None:
        del camera_id

    def mark_ready(self, camera_id: str) -> None:
        del camera_id

    def mark_degraded(self, camera_id: str, *, category: str) -> None:
        del camera_id, category

    def emit(self, event: IngestEvent) -> None:
        del event


def _worker_config(camera_id: str) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "version": 7,
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "cameras": [
                {
                    "camera_id": camera_id,
                    "facility_id": "facility-a",
                    "rtsp_url": f"rtsp://8.8.8.8/{camera_id}",
                    "heartbeat_interval_sec": 30.0,
                }
            ],
        }
    )


def _boot_context_for(profile_name: str) -> BootContext:
    spec = PROFILE_REGISTRY[profile_name]
    return BootContext(profile=spec, device=spec.device, decode=spec.decode, encode=spec.encode)


@final
class _StubCpuAdapter:
    """Named stand-in so the recorded adapter class is unambiguous."""

    def open(self, config: object) -> object:  # pragma: no cover - never opened here
        raise AssertionError("composition-only test must not open a decode session")


def test_camera_loop_composition_populates_the_decode_backend_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composing a camera's ingest loop records the requested/resolved/actual trio.

    Proves the composition root really calls ``record_decode_backend`` during
    the same step that builds the adapter, and that the recorded class is the
    adapter the loop actually holds rather than a value re-derived from the
    profile token (which would agree by construction and could never catch a
    selection/observability disagreement).
    """
    # Given: a runtime whose boot profile resolved to the cpu decode token.
    monkeypatch.setattr(
        ingest_composition_module, "CpuAvAdapter", lambda: _StubCpuAdapter()
    )
    config = _worker_config("camera-a")
    runtime = WorkerRuntime(config, serving_client=_FakeServingClient())
    runtime._boot = _boot_context_for("cpu")  # noqa: SLF001 - post-model-init state

    # When: the real per-camera loop composition point runs.
    loop = runtime._default_loop_factory(  # noqa: SLF001
        config.cameras[0], BoundedFrameBus(), _NullReporter()
    )

    # Then: the local snapshot carries this camera's trio, and "actual" is the
    # class of the adapter the composed loop decodes with.
    recorded = runtime.diagnostics.decode_backend_snapshot()["camera-a"]
    assert recorded.requested_profile_decode == runtime._boot.decode  # noqa: SLF001
    assert recorded.resolved_backend == "opencv"
    assert recorded.actual_adapter_class == "_StubCpuAdapter"
    assert recorded.actual_adapter_class == type(loop.decode_adapter).__name__


def test_decode_backend_snapshot_is_recorded_per_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each composed camera gets its own entry -- no shared/global record."""
    # Given
    monkeypatch.setattr(
        ingest_composition_module, "CpuAvAdapter", lambda: _StubCpuAdapter()
    )
    runtime = WorkerRuntime(_worker_config("camera-a"), serving_client=_FakeServingClient())
    runtime._boot = _boot_context_for("cpu")  # noqa: SLF001

    # When: two cameras are composed through the same factory.
    for camera_id in ("camera-a", "camera-b"):
        camera = _worker_config(camera_id).cameras[0]
        _ = runtime._default_loop_factory(  # noqa: SLF001
            camera, BoundedFrameBus(), _NullReporter()
        )

    # Then
    snapshot = runtime.diagnostics.decode_backend_snapshot()
    assert set(snapshot) == {"camera-a", "camera-b"}
    assert all(entry.resolved_backend == "opencv" for entry in snapshot.values())

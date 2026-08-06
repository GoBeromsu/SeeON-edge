from __future__ import annotations

import logging
import time
from typing import final

import pytest

from contracts.decode_diagnostics import DecodeSelection
from contracts.observation import BedRegionCacheState
from worker.pipeline.perception.scene_state import BedRegionCacheCounters
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.runtime.telemetry.runtime_status_sender import (
    RuntimeStatusSender,
    RuntimeStatusSenderConfig,
)
from worker.runtime.telemetry.wire import (
    ClipRecorderStatus,
    RelayRuntimeStatusPayload,
)

# test_sender_uses_latest_snapshot_and_bearer_auth (edge): payload shape, generation
# bookkeeping, and bearer auth are superseded by tests/test_worker_telemetry_status_sender.py
# ::test_sender_preserves_generation_and_monotonic_sequence and
# ::test_relay_transport_uses_bounded_status_endpoint_and_parses_receipt (auth header lives
# at the RelayRuntimeStatusTransport boundary now). Neither covers start()/stop() actually
# delivering a queued snapshot through the background thread -- ported below.
# test_sender_posts_separate_facility_payloads_with_only_bound_cameras (edge): per-facility
# partitioning is superseded by
# tests/test_worker_telemetry_status_sender.py::test_sender_partitions_cameras_by_facility
# (exercised synchronously via publish_once()); the thread-lifecycle test below already drives
# the same _run() loop for the single-facility case, so re-porting the multi-facility split
# would only duplicate that loop, not add coverage.


@final
class _RecordingTransport:
    __slots__ = ("payloads",)

    def __init__(self) -> None:
        self.payloads: list[RelayRuntimeStatusPayload] = []

    def send(self, payload: RelayRuntimeStatusPayload) -> int | None:
        self.payloads.append(payload)
        return 7


@final
class _FlakyTransport:
    __slots__ = ("attempts",)

    def __init__(self) -> None:
        self.attempts = 0

    def send(self, _payload: RelayRuntimeStatusPayload) -> int | None:
        self.attempts += 1
        return None if self.attempts == 1 else 3


@final
class _UnusedTransport:
    def send(self, _payload: RelayRuntimeStatusPayload) -> int | None:
        raise AssertionError("transport.send must not be called before start()")


def _diagnostics() -> WorkerDiagnostics:
    diagnostics = WorkerDiagnostics()
    diagnostics.update_decode(
        "camera-a",
        DecodeSelection(
            requested="auto",
            selected="nvdec",
            fallback_count=1,
            last_reason="spawn_failed",
            updated_at_sec=1.0,
        ),
    )
    return diagnostics


def test_sender_start_delivers_queued_snapshot_via_background_thread() -> None:
    transport = _RecordingTransport()
    sender = RuntimeStatusSender(
        _diagnostics(),
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(publish_interval_sec=0.01),
    )

    sender.start()
    try:
        _wait_until(lambda: bool(transport.payloads))
    finally:
        sender.stop()

    assert transport.payloads[0]["cameras"] == [
        {
            "camera_id": "camera-a",
            "decode": {
                "requested": "auto",
                "selected": "nvdec",
                "fallback_count": 1,
                "last_reason": "spawn_failed",
                "updated_at_sec": 1.0,
            },
        }
    ]
    assert sender.generation == 7
    assert sender.is_alive is False


def test_sender_retries_with_backoff_and_recovers() -> None:
    transport = _FlakyTransport()
    sender = RuntimeStatusSender(
        _diagnostics(),
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(
            publish_interval_sec=1.0,
            initial_backoff_sec=0.01,
            max_backoff_sec=0.02,
        ),
    )

    sender.start()
    _wait_until(lambda: transport.attempts >= 2)
    sender.stop()

    assert sender.generation == 3
    assert sender.is_alive is False


def test_sender_publish_never_blocks_when_latest_slot_is_full() -> None:
    sender = RuntimeStatusSender(_diagnostics(), "facility-a", _UnusedTransport())

    assert sender.publish() is True
    start = time.monotonic()
    assert sender.publish() is True

    assert time.monotonic() - start < 0.1


def test_before_publish_hook_refreshes_diagnostics_on_every_tick() -> None:
    """#165: nothing previously re-read live clip-recorder counters into
    ``WorkerDiagnostics`` after recorder start, so every runtime-status
    payload's ``clip_recorder`` stayed frozen at its startup values for the
    rest of the process. ``before_publish`` is the seam that fixes that --
    this pins that it actually runs on every publish (including background
    ticks), not just once at construction.
    """
    diagnostics = _diagnostics()
    calls = {"count": 0}

    def before_publish() -> None:
        calls["count"] += 1
        diagnostics.set_clip_recorder_status(
            ClipRecorderStatus(available=True, finalized_clips=calls["count"])
        )

    transport = _RecordingTransport()
    sender = RuntimeStatusSender(
        diagnostics,
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(publish_interval_sec=0.01),
        before_publish=before_publish,
    )

    sender.start()
    try:
        _wait_until(lambda: len(transport.payloads) >= 2)
    finally:
        sender.stop()

    # Called on every tick, not cached from the first call.
    assert calls["count"] >= 2
    # The most recently delivered payload reflects the latest live value,
    # proving the hook re-runs (rather than a value snapshotted once).
    assert transport.payloads[-1]["clip_recorder"]["finalized_clips"] == calls["count"]


def test_sender_logs_a_local_diagnostics_snapshot_on_its_own_tick(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """#207: the sender's existing background tick is reused to finally call
    ``WorkerDiagnostics.log_snapshot()``, which had no production caller.
    """
    diagnostics = _diagnostics()
    counters = BedRegionCacheCounters(fresh=2)
    diagnostics.record_bed_region("camera-a", BedRegionCacheState.FRESH, counters.snapshot())
    transport = _RecordingTransport()
    sender = RuntimeStatusSender(
        diagnostics,
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(publish_interval_sec=0.01),
    )

    with caplog.at_level(logging.INFO):
        sender.start()
        try:
            _wait_until(lambda: bool(transport.payloads))
        finally:
            sender.stop()

    telemetry_records = [
        record for record in caplog.records if record.getMessage() == "worker.runtime.telemetry"
    ]
    assert telemetry_records
    assert vars(telemetry_records[-1]).get("camera_id") == "camera-a"
    assert vars(telemetry_records[-1]).get("bed_region", {}).get("freshness") == "fresh"


@final
class _LogSnapshotAlwaysFailsDiagnostics:
    """Delegates the relay-facing methods to a real ``WorkerDiagnostics`` but
    makes ``log_snapshot()`` raise, to pin that a local-logging defect can
    never take down relay delivery (issue #207).
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: WorkerDiagnostics) -> None:
        self._inner = inner

    def to_payload(
        self, facility_id: str, generation: int | None, seq: int
    ) -> RelayRuntimeStatusPayload:
        return self._inner.to_payload(facility_id, generation, seq)

    def to_payloads(
        self, camera_facilities: object, generation: int | None, seq: int
    ) -> list[RelayRuntimeStatusPayload]:
        return self._inner.to_payloads(camera_facilities, generation, seq)  # type: ignore[arg-type]

    def log_snapshot(self) -> None:
        raise RuntimeError("boom")


def test_sender_survives_a_log_snapshot_failure_and_keeps_delivering(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = _RecordingTransport()
    sender = RuntimeStatusSender(
        _LogSnapshotAlwaysFailsDiagnostics(_diagnostics()),  # type: ignore[arg-type]
        "facility-a",
        transport,
        RuntimeStatusSenderConfig(publish_interval_sec=0.01),
    )

    with caplog.at_level(logging.WARNING):
        sender.start()
        try:
            _wait_until(lambda: bool(transport.payloads))
        finally:
            sender.stop()

    # Relay delivery happened despite log_snapshot() always raising.
    assert transport.payloads
    assert any("log_snapshot" in record.getMessage() for record in caplog.records)


def _wait_until(predicate, timeout_sec: float = 0.5) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for runtime status sender")

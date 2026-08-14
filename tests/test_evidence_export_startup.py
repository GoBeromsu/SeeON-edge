from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import worker.pipeline.output.evidence.clip_recorder as recorder_module
from contracts.frame import Frame
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder, ClipRecorderConfig
from worker.pipeline.output.evidence.evidence_runtime import EvidenceExportRuntime
from worker.types import BusinessEvent, FramePacket


def _packet() -> FramePacket:
    frame = Frame(1, 1.0, np.zeros((8, 8, 3), dtype=np.uint8))
    return FramePacket("camera-1", frame, 1.0, 1, 8, 8, 0.1)


class FakeLock:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("lock-close")


class FakeSender:
    def run_once(self):
        raise AssertionError("sender must not start inside the recorder lock hook")


def test_startup_initializes_evidence_under_one_lock_before_sweep_rotate_accept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the recorder owns the only clip-store lifetime lock.
    events: list[str] = []
    runtime = EvidenceExportRuntime(
        store_dir=tmp_path,
        database_path=tmp_path / "evidence.sqlite3",
        sender=FakeSender(),
        on_open=lambda: events.append("outbox-open-migrate"),
        on_reconciled=lambda: events.append("reconcile"),
        on_holds_loaded=lambda: events.append("holds-load"),
    )
    monkeypatch.setattr(
        recorder_module.ClipStoreLock,
        "acquire",
        lambda _path: events.append("lock") or FakeLock(events),
    )
    original_default_services = recorder_module._default_services  # noqa: SLF001
    monkeypatch.setattr(
        recorder_module,
        "_default_services",
        lambda config, repository: (
            events.append("accept") or original_default_services(config, repository)
        ),
    )
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=tmp_path),
        is_clip_held=runtime.is_clip_held,
        startup_hook=runtime.initialize_under_lock,
    )
    monkeypatch.setattr(
        recorder,
        "_sweep_stale_staging",
        lambda: events.append("sweep"),
    )
    monkeypatch.setattr(
        recorder,
        "_rotate",
        lambda *, force=False: events.append(f"rotate:{force}"),
    )

    # When: production startup enters the recorder boundary.
    recorder.start()
    try:
        pass
    finally:
        recorder.stop()

    # Then: reconciliation and hold state are ready before destructive maintenance/admission.
    assert events[:7] == [
        "lock",
        "outbox-open-migrate",
        "reconcile",
        "holds-load",
        "sweep",
        "rotate:True",
        "accept",
    ]
    assert events.count("lock") == 1
    assert events[-1] == "lock-close"


def test_evidence_startup_failure_releases_lock_and_never_accepts_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: durable reconciliation fails after the one lifetime lock is acquired.
    events: list[str] = []
    monkeypatch.setattr(
        recorder_module.ClipStoreLock,
        "acquire",
        lambda _path: events.append("lock") or FakeLock(events),
    )
    recorder = ClipRecorder(
        ClipRecorderConfig(store_dir=tmp_path),
        startup_hook=lambda: (_ for _ in ()).throw(RuntimeError("reconcile failed")),
    )
    monkeypatch.setattr(
        recorder,
        "_sweep_stale_staging",
        lambda: events.append("unexpected-sweep"),
    )

    # When: startup cannot establish the durable evidence truth.
    with pytest.raises(RuntimeError, match="reconcile failed"):
        recorder.start()

    # Then: the recorder remains fail-closed and releases the same lock.
    packet = _packet()
    try:
        assert (
            recorder.on_event(
                packet,
                BusinessEvent(
                    "fall",
                    "fall.detected",
                    "event-1",
                    "camera-1",
                    "facility-1",
                    1.0,
                    0.9,
                ),
            )
            is None
        )
    finally:
        packet.release()
    assert events == ["lock", "lock-close"]


def test_event_delivery_runtime_exists_when_clip_policy_is_off(tmp_path: Path) -> None:
    runtime = EvidenceExportRuntime.from_config(
        store_dir=tmp_path,
        relay_url="http://relay.test",
        relay_token="relay-token",
        probe_camera_id="camera-1",
        clip_export_enabled=lambda: False,
    )

    assert runtime is not None


def test_event_delivery_runtime_rejects_missing_relay_credentials(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relay URL, token"):
        EvidenceExportRuntime.from_config(
            store_dir=tmp_path,
            relay_url="",
            relay_token=None,
            probe_camera_id="camera-1",
            clip_export_enabled=lambda: False,
        )


def test_runtime_builds_stager_on_the_initialized_outbox(tmp_path: Path) -> None:
    runtime = EvidenceExportRuntime(
        store_dir=tmp_path,
        database_path=tmp_path / "evidence.sqlite3",
        sender=FakeSender(),
    )
    stager = runtime.stager(
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id="resident-1",
        config_version=7,
    )

    assert stager.database_path == runtime.database_path
    assert stager.camera_id == "camera-1"
    assert stager.facility_id == "facility-1"
    assert stager.resident_id == "resident-1"
    assert stager.config_version == 7

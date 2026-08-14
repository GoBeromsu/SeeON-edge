from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from contracts.frame import Frame
from worker.pipeline.output.evidence.clip_identity import ClipReservation
from worker.pipeline.output.evidence.clip_publication import ClipPublicationMetadata
from worker.pipeline.output.evidence.clip_recorder import (
    ClipRecorder,
    ClipRecorderConfig,
    ClipRecorderServices,
)
from worker.pipeline.output.evidence.clip_recording import (
    ClipOutcome,
    ClipReasonCode,
    ClipUnavailable,
)
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    EvidenceReasonCode,
)
from worker.types import BusinessEvent, FramePacket


class _DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def _frame(index: int, time_sec: float) -> Frame:
    return Frame(index, time_sec, np.zeros((8, 8, 3), dtype=np.uint8))


@dataclass(frozen=True, slots=True)
class _Publisher:
    published: list[ClipId] = field(default_factory=list)
    metadata: list[ClipPublicationMetadata] = field(default_factory=list)

    def publish_ready(
        self,
        reservation: ClipReservation,
        artifact_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> None:
        del artifact_path
        self.metadata.append(metadata)
        self._publish(reservation)

    def publish_unavailable(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
        reason_code: EvidenceReasonCode,
    ) -> None:
        del reason_code
        self.metadata.append(metadata)
        self._publish(reservation)

    def _publish(self, reservation: ClipReservation) -> None:
        reservation.final_dir.mkdir(parents=True)
        (reservation.final_dir / "manifest.json").write_text(
            json.dumps({"clip_id": reservation.clip_id, "finalized": True}),
            encoding="utf-8",
        )
        shutil.rmtree(reservation.staging_dir)
        self.published.append(reservation.clip_id)


@dataclass(frozen=True, slots=True)
class _Coordinator:
    outcome: ClipOutcome | None = None
    write_started: threading.Event | None = None
    allow_write: threading.Event | None = None
    seal_started: threading.Event | None = None
    allow_seal: threading.Event | None = None
    close_all_calls: list[None] = field(default_factory=list)

    def set_camera_fps(self, camera_id: str, fps: float) -> None:
        del camera_id, fps

    def write(self, packet: FramePacket) -> bool:
        del packet
        if self.write_started is not None and self.allow_write is not None:
            self.write_started.set()
            assert self.allow_write.wait(timeout=1.0)
        return True

    def seal(self, camera_id: str) -> bool:
        del camera_id
        if self.seal_started is not None and self.allow_seal is not None:
            self.seal_started.set()
            assert self.allow_seal.wait(timeout=1.0)
        return True

    def finalize(
        self,
        *,
        camera_id: str,
        clip_id: str,
        event_time_sec: float,
        event: BusinessEvent,
        output_dir: Path | None = None,
    ) -> ClipOutcome:
        del camera_id, event_time_sec, event, output_dir
        if self.outcome is None:
            raise OSError("staging unavailable")
        return self.outcome

    def close(self, camera_id: str) -> None:
        del camera_id

    def close_all(self) -> None:
        self.close_all_calls.append(None)


def _recorder(
    tmp_path: Path,
    coordinator: _Coordinator,
    publisher: _Publisher,
    *,
    max_queue_size: int = 16,
    post_event_seconds: float = 0.0,
) -> ClipRecorder:
    return ClipRecorder(
        ClipRecorderConfig(
            store_dir=tmp_path,
            max_queue_size=max_queue_size,
            post_event_seconds=post_event_seconds,
        ),
        ClipRecorderServices(coordinator, publisher, "libx264"),
        disk_usage_provider=lambda _path: _DiskUsage(100, 10, 90),
        is_clip_held=lambda _clip_id: False,
    )


def test_recorder_coalesces_event_ids_and_preserves_reference_order(tmp_path: Path) -> None:
    publisher = _Publisher()
    recorder = _recorder(
        tmp_path,
        _Coordinator(ClipUnavailable("", ClipReasonCode.ENCODER_FAILED)),
        publisher,
        post_event_seconds=10.0,
    )
    recorder.start()
    try:
        assert recorder.on_frame("cam-1", _frame(1, 1.0))
        first = recorder.on_event("cam-1", "event-1", "fall.detected")
        duplicate = recorder.on_event("cam-1", "event-1", "fall.detected")
        second = recorder.on_event("cam-1", "event-2", "fall.detected")
        assert first is not None
        assert duplicate == first
        assert second == first
        assert recorder.flush()
    finally:
        recorder.stop()

    assert publisher.published == [ClipId(first)]
    assert publisher.metadata[0].event_refs == ("event-1", "event-2")


def test_event_admitted_before_finalization_stays_on_active_clip(tmp_path: Path) -> None:
    publisher = _Publisher()
    recorder = _recorder(
        tmp_path,
        _Coordinator(ClipUnavailable("", ClipReasonCode.ENCODER_FAILED)),
        publisher,
        post_event_seconds=1.0,
    )
    recorder.start()
    try:
        first = recorder.on_event("cam-1", "event-1", "fall.detected")
        second = recorder.on_event("cam-1", "event-2", "fall.detected")
        assert first is not None
        assert second == first
        assert recorder.on_frame("cam-1", _frame(1, 1.0))
        assert recorder.flush()
    finally:
        recorder.stop()

    assert publisher.metadata[0].event_refs == ("event-1", "event-2")


def test_event_admitted_after_finalization_starts_gets_fresh_clip_id(
    tmp_path: Path,
) -> None:
    seal_started = threading.Event()
    allow_seal = threading.Event()
    publisher = _Publisher()
    coordinator = _Coordinator(
        ClipUnavailable("", ClipReasonCode.ENCODER_FAILED),
        seal_started=seal_started,
        allow_seal=allow_seal,
    )
    recorder = _recorder(tmp_path, coordinator, publisher)
    recorder.start()
    try:
        first = recorder.on_event("cam-1", "event-1", "fall.detected")
        assert first is not None
        assert seal_started.wait(timeout=1.0)

        second = recorder.on_event("cam-1", "event-2", "fall.detected")

        assert second is not None
        assert second != first
        allow_seal.set()
        assert recorder.flush()
    finally:
        allow_seal.set()
        recorder.stop()

    assert [metadata.event_refs for metadata in publisher.metadata] == [
        ("event-1",),
        ("event-2",),
    ]


def test_stop_drains_full_queue_after_blocked_frame_write(tmp_path: Path) -> None:
    write_started = threading.Event()
    allow_write = threading.Event()
    coordinator = _Coordinator(
        ClipUnavailable("", ClipReasonCode.ENCODER_FAILED),
        write_started,
        allow_write,
    )
    publisher = _Publisher()
    recorder = _recorder(tmp_path, coordinator, publisher, max_queue_size=1)
    recorder.start()
    try:
        assert recorder.on_frame("cam-1", _frame(1, 1.0))
        assert write_started.wait(timeout=1.0)
        clip_id = recorder.on_event("cam-1", "event-1", "fall.detected")
        assert clip_id is not None
        recorder.stop(timeout=0.01)
        assert recorder._thread is not None and recorder._thread.is_alive()
        allow_write.set()
        recorder.stop(timeout=1.0)
    finally:
        allow_write.set()
        recorder.stop()

    assert publisher.published == [ClipId(clip_id)]
    assert (tmp_path / "clips" / clip_id / "manifest.json").is_file()
    assert coordinator.close_all_calls


def test_recorder_cleans_failed_reservation_before_retrying_event(tmp_path: Path) -> None:
    coordinator = _Coordinator()
    publisher = _Publisher()
    recorder = _recorder(tmp_path, coordinator, publisher)
    recorder.start()
    try:
        first = recorder.on_event("cam-1", "event-1", "fall.detected")
        assert first is not None
        assert recorder.flush()
        assert not (tmp_path / "clips" / ".staging" / first).exists()
        second = recorder.on_event("cam-1", "event-2", "fall.detected")
        assert second is not None
        assert second != first
    finally:
        recorder.stop()

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
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


@dataclass(frozen=True, slots=True)
class _Coordinator:
    packets: list[FramePacket] = field(default_factory=list)
    camera_fps: dict[str, float] = field(default_factory=dict)
    close_all_calls: list[None] = field(default_factory=list)

    def set_camera_fps(self, camera_id: str, fps: float) -> None:
        self.camera_fps[camera_id] = fps

    def write(self, packet: FramePacket) -> bool:
        self.packets.append(packet)
        return True

    def seal(self, camera_id: str) -> bool:
        del camera_id
        return True

    def finalize(
        self,
        *,
        camera_id: str,
        clip_id: str,
        event_time_sec: float,
        event: BusinessEvent,
        output_dir: Path | None = None,
        trigger_frame_key: object | None = None,
        window_bounds: tuple[float, float] | None = None,
    ) -> ClipOutcome:
        del camera_id, event_time_sec, event, output_dir, trigger_frame_key, window_bounds
        return ClipUnavailable(clip_id, ClipReasonCode.ENCODER_FAILED)

    def close(self, camera_id: str) -> None:
        del camera_id

    def close_all(self) -> None:
        self.close_all_calls.append(None)


@dataclass(frozen=True, slots=True)
class _Publisher:
    published: list[ClipId] = field(default_factory=list)

    def publish_ready(
        self,
        reservation: ClipReservation,
        artifact_path: Path,
        metadata: ClipPublicationMetadata,
    ) -> None:
        del artifact_path, metadata
        self._publish(reservation)

    def publish_unavailable(
        self,
        reservation: ClipReservation,
        metadata: ClipPublicationMetadata,
        reason_code: EvidenceReasonCode,
    ) -> None:
        del metadata, reason_code
        self._publish(reservation)

    def _publish(self, reservation: ClipReservation) -> None:
        reservation.final_dir.mkdir(parents=True)
        (reservation.final_dir / "manifest.json").write_text(
            json.dumps({"clip_id": reservation.clip_id, "finalized": True}),
            encoding="utf-8",
        )
        shutil.rmtree(reservation.staging_dir)
        self.published.append(reservation.clip_id)


def _frame(index: int, time_sec: float) -> Frame:
    return Frame(
        index=index,
        time_sec=time_sec,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def _packet(index: int, time_sec: float, *, camera_id: str = "cam-1") -> FramePacket:
    return FramePacket(camera_id, _frame(index, time_sec), time_sec, index, 8, 8, 0.1)


def _event(identity: str, time_sec: float = 1.0) -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", identity, "cam-1", "facility-1", time_sec, 0.9)


def _recorder(
    tmp_path: Path,
    *,
    max_queue_size: int = 16,
    on_clip_finalized: Callable[[ClipId], None] | None = None,
) -> tuple[ClipRecorder, _Coordinator, _Publisher]:
    coordinator = _Coordinator()
    publisher = _Publisher()
    recorder = ClipRecorder(
        ClipRecorderConfig(
            store_dir=tmp_path,
            post_event_seconds=0.0,
            max_queue_size=max_queue_size,
        ),
        ClipRecorderServices(coordinator, publisher, "libx264"),
        disk_usage_provider=lambda _path: _DiskUsage(100, 10, 90),
        is_clip_held=lambda _clip_id: False,
        on_clip_finalized=on_clip_finalized,
    )
    return recorder, coordinator, publisher


def test_recorder_publishes_event_and_notifies_after_manifest_exists(
    tmp_path: Path,
) -> None:
    notifications: list[ClipId] = []

    def finalized(clip_id: ClipId) -> None:
        assert (tmp_path / "clips" / clip_id / "manifest.json").is_file()
        notifications.append(clip_id)

    recorder, coordinator, publisher = _recorder(
        tmp_path,
        on_clip_finalized=finalized,
    )
    recorder.start()
    try:
        packet = _packet(1, 1.0)
        assert recorder.on_frame(packet)
        clip_id = recorder.on_event(packet, _event("event-1"), detected_at=datetime.now(UTC))
        packet.release()
        assert clip_id is not None
        assert recorder.flush()
    finally:
        recorder.stop()

    assert publisher.published == [ClipId(clip_id)]
    assert notifications == [ClipId(clip_id)]
    assert len(coordinator.packets) == 1
    assert recorder.stats.finalized_clips == 1


def test_stop_drains_an_accepted_event_before_releasing_store_lock(tmp_path: Path) -> None:
    recorder, coordinator, publisher = _recorder(tmp_path)
    recorder.start()

    trigger = _packet(1, 1.0)
    clip_id = recorder.on_event(trigger, _event("event-1"), detected_at=datetime.now(UTC))
    trigger.release()
    recorder.stop()

    assert clip_id is not None
    assert publisher.published == [ClipId(clip_id)]
    assert len(coordinator.close_all_calls) >= 1


def test_bounded_queue_rejects_without_blocking(tmp_path: Path) -> None:
    recorder, _, _ = _recorder(tmp_path, max_queue_size=1)

    first = _packet(1, 1.0)
    second = _packet(2, 2.0)
    assert recorder.on_frame(first)
    assert recorder.on_frame(second) is False
    first.release()
    second.release()
    recorder.stop()
    assert recorder.dropped_frame_count == 1


def test_camera_fps_registration_reaches_coordinator(tmp_path: Path) -> None:
    recorder, coordinator, _ = _recorder(tmp_path)

    recorder.set_camera_fps("cam-1", 15.0)

    assert coordinator.camera_fps == {"cam-1": 15.0}

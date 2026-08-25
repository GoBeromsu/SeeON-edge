from __future__ import annotations

import queue
from pathlib import Path

import numpy as np

from contracts.frame import Frame
from worker.pipeline.output.evidence.clip_admission import ClipAdmission
from worker.pipeline.output.evidence.clip_recorder_models import (
    ClipRecorderConfig,
    ClipRecorderStats,
    EventMessage,
    FrameMessage,
)
from worker.types import BusinessEvent, FramePacket

RUNTIME_MANIFEST_SHA256 = "b" * 64


def _packet(*, pts: float, seq: int, epoch: int, generation: int = 1) -> FramePacket:
    frame = Frame(index=seq, time_sec=pts, image=np.zeros((2, 3, 3), dtype=np.uint8))
    return FramePacket(
        camera_id="camera-1",
        frame=frame,
        pts=pts,
        seq=seq,
        width=3,
        height=2,
        decode_time_ms=0.25,
        worker_boot_id="boot-1",
        stream_epoch=epoch,
        source_generation=generation,
    )


def test_admission_preserves_the_whole_frame_packet(tmp_path: Path) -> None:
    messages: queue.Queue[object] = queue.Queue()
    admission = ClipAdmission(ClipRecorderConfig(store_dir=tmp_path), ClipRecorderStats(), messages)
    packet = _packet(pts=42.25, seq=8, epoch=3)

    assert admission.accept_frame(packet) is True

    message = messages.get_nowait()
    assert isinstance(message, FrameMessage)
    assert message.packet is not packet
    assert message.packet.frame_key == packet.frame_key
    assert message.packet.frame is packet.frame
    assert message.packet.lease is not packet.lease
    assert message.packet.descriptor == packet.descriptor


def test_event_uses_the_trigger_packet_not_the_latest_evidence_packet(tmp_path: Path) -> None:
    messages: queue.Queue[object] = queue.Queue()
    admission = ClipAdmission(ClipRecorderConfig(store_dir=tmp_path), ClipRecorderStats(), messages)
    trigger = _packet(pts=12.5, seq=4, epoch=7)
    delayed_evidence = _packet(pts=99.0, seq=40, epoch=7)
    assert admission.accept_frame(delayed_evidence)
    _ = messages.get_nowait()

    event = BusinessEvent(
        "fall",
        "fall.detected",
        "event-1",
        "camera-1",
        "facility-1",
        12.5,
        0.9,
        audit={"runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256},
    )
    clip_id = admission.accept_event(trigger, event, allow_new_clip=True)

    assert clip_id is not None
    message = messages.get_nowait()
    assert isinstance(message, EventMessage)
    assert message.trigger_packet is not trigger
    assert message.trigger_packet.frame is trigger.frame
    assert message.trigger_packet.lease is not trigger.lease
    assert message.event is event
    assert message.event.audit == {"runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256}
    assert message.event.time_sec == 12.5
    assert message.trigger_packet.frame_key == trigger.frame_key


def test_cross_epoch_and_generation_events_never_union(tmp_path: Path) -> None:
    messages: queue.Queue[object] = queue.Queue()
    admission = ClipAdmission(ClipRecorderConfig(store_dir=tmp_path), ClipRecorderStats(), messages)
    event = BusinessEvent(
        "fall", "fall.detected", "event-epoch", "camera-1", "facility-1", 10.0, 0.9,
        audit={"runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256},
    )
    first = admission.accept_event(
        _packet(pts=10.0, seq=1, epoch=2, generation=3), event, allow_new_clip=True
    )
    second = admission.accept_event(
        _packet(pts=10.1, seq=2, epoch=3, generation=3), event, allow_new_clip=True
    )
    third = admission.accept_event(
        _packet(pts=10.2, seq=3, epoch=3, generation=4), event, allow_new_clip=True
    )

    assert first is not None
    assert len({first, second, third}) == 3

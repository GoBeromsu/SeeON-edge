from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import final

from worker.pipeline.output.evidence.clip_config import (
    MIN_RETENTION_DAYS,
    configured_disk_high_watermark,
    configured_ffmpeg_bin,
    configured_retention_days,
    configured_store_dir,
)
from worker.pipeline.output.evidence.clip_identity import ClipReservation
from worker.types import BusinessEvent, FrameKey, FramePacket


@dataclass(frozen=True, slots=True)
class ClipRecorderConfig:
    store_dir: Path = field(default_factory=configured_store_dir)
    segment_seconds: float = 2.0
    pre_event_seconds: float = 30.0
    post_event_seconds: float = 30.0
    fps: float = 5.0
    retention_days: int = field(default_factory=configured_retention_days)
    disk_high_watermark: float = field(default_factory=configured_disk_high_watermark)
    max_queue_size: int = 128
    finalize_grace_seconds: float = 5.0
    stale_staging_seconds: float = 60.0 * 60.0
    rotate_min_interval_seconds: float = 30.0
    ffmpeg_bin: str = field(default_factory=configured_ffmpeg_bin)
    packet_ring_max_packets: int = 32_768
    packet_ring_max_bytes_per_camera: int = 32 * 1024 * 1024
    packet_ring_global_max_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "retention_days",
            max(MIN_RETENTION_DAYS, self.retention_days),
        )
        if (
            self.packet_ring_max_packets <= 0
            or self.packet_ring_max_bytes_per_camera <= 0
            or self.packet_ring_global_max_bytes <= 0
        ):
            raise ValueError("packet ring limits must be positive")


@final
class ClipRecorderStats:
    __slots__ = (
        "active_clips",
        "attach_missed_events",
        "clip_id_collisions",
        "dropped_events",
        "dropped_frames",
        "encoder",
        "failed_writes",
        "finalized_clips",
        "forced_finalized",
        "held_clips",
        "purge_failures",
        "recording_suspended",
        "stale_staging_cleaned",
        "video_unavailable_clips",
    )

    def __init__(self) -> None:
        self.dropped_frames = 0
        self.dropped_events = 0
        self.attach_missed_events = 0
        self.clip_id_collisions = 0
        self.finalized_clips = 0
        self.failed_writes = 0
        self.video_unavailable_clips = 0
        self.active_clips = 0
        self.forced_finalized = 0
        self.stale_staging_cleaned = 0
        self.encoder: str | None = None
        self.held_clips = 0
        self.purge_failures = 0
        self.recording_suspended = False


@dataclass(frozen=True, slots=True)
class FrameMessage:
    packet: FramePacket


@dataclass(frozen=True, slots=True)
class EventMessage:
    reservation: ClipReservation
    event_ref: str
    event_type: str | None
    event: BusinessEvent
    trigger_packet: FramePacket
    allow_new_clip: bool


@dataclass(frozen=True, slots=True)
class FlushMessage:
    done: threading.Event


RecorderMessage = FrameMessage | EventMessage | FlushMessage


@final
class ActiveClip:
    __slots__ = (
        "cutoff_time_sec",
        "event",
        "event_ref",
        "event_refs",
        "event_time_sec",
        "event_type",
        "last_time_sec",
        "opened_monotonic",
        "reservation",
        "start_time_sec",
        "started_at",
        "trigger_frame_key",
    )

    def __init__(
        self,
        reservation: ClipReservation,
        event_ref: str,
        event_type: str | None,
        event: BusinessEvent,
        event_time_sec: float,
        cutoff_time_sec: float,
        started_at: datetime,
        start_time_sec: float,
        last_time_sec: float,
        event_refs: list[str],
        trigger_frame_key: FrameKey | None = None,
        opened_monotonic: float | None = None,
    ) -> None:
        self.reservation = reservation
        self.event_ref = event_ref
        self.event_type = event_type
        self.event = event
        self.event_time_sec = event_time_sec
        self.cutoff_time_sec = cutoff_time_sec
        self.started_at = started_at
        self.start_time_sec = start_time_sec
        self.last_time_sec = last_time_sec
        self.event_refs = event_refs
        self.trigger_frame_key = trigger_frame_key
        self.opened_monotonic = time.monotonic() if opened_monotonic is None else opened_monotonic


__all__ = [
    "ActiveClip",
    "ClipRecorderConfig",
    "ClipRecorderStats",
    "EventMessage",
    "FlushMessage",
    "FrameMessage",
    "RecorderMessage",
]

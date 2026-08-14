from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import final

from worker.types.source_packet import (
    PacketSelectionError,
    PacketTruncationReason,
    SourcePacket,
    SourceStreamConfiguration,
    StreamEpoch,
)


@dataclass(frozen=True, slots=True)
class PacketRingLimits:
    max_packets: int
    max_bytes: int
    max_duration_seconds: float

    def __post_init__(self) -> None:
        if self.max_packets <= 0 or self.max_bytes <= 0:
            raise ValueError("packet ring count and byte limits must be positive")
        if not math.isfinite(self.max_duration_seconds) or self.max_duration_seconds <= 0:
            raise ValueError("packet ring duration limit must be finite and positive")


@dataclass(slots=True)
class PacketRingMetrics:
    accepted_packets: int = 0
    accepted_bytes: int = 0
    dropped_packets: int = 0
    dropped_bytes: int = 0
    evicted_packets: int = 0
    evicted_bytes: int = 0
    lease_backpressure_drops: int = 0
    active_leases: int = 0


@dataclass(slots=True)
class _Entry:
    packet: SourcePacket
    lease_count: int = 0


@final
class PacketSelection:
    def __init__(
        self,
        owner: SourcePacketRing,
        entries: tuple[_Entry, ...],
        *,
        configuration: SourceStreamConfiguration,
        requested_start: Fraction,
        requested_end: Fraction,
        selected_start: Fraction,
        selected_end: Fraction,
        truncations: tuple[PacketTruncationReason, ...],
    ) -> None:
        self._owner = owner
        self._entries = entries
        self.configuration = configuration
        self.requested_start = requested_start
        self.requested_end = requested_end
        self.selected_start = selected_start
        self.selected_end = selected_end
        self.truncations = truncations
        self._closed = False

    @property
    def packets(self) -> tuple[SourcePacket, ...]:
        if self._closed:
            raise RuntimeError("packet selection lease is closed")
        return tuple(entry.packet for entry in self._entries)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._owner._release(self._entries)  # noqa: SLF001 - paired ownership hook

    def __enter__(self) -> PacketSelection:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@final
class SourcePacketRing:
    def __init__(self, camera_id: str, limits: PacketRingLimits) -> None:
        if not camera_id:
            raise ValueError("packet ring camera id must not be blank")
        self.camera_id = camera_id
        self.limits = limits
        self.metrics = PacketRingMetrics()
        self._entries: deque[_Entry] = deque()
        self._total_bytes = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def packet_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def snapshot(self) -> tuple[SourcePacket, ...]:
        with self._lock:
            return tuple(entry.packet for entry in self._entries)

    def append(self, packet: SourcePacket) -> bool:
        if packet.epoch.camera_id != self.camera_id:
            raise ValueError("packet camera does not match its ring")
        with self._lock:
            if self._closed or packet.size_bytes > self.limits.max_bytes:
                self._drop(packet, lease_pressure=False)
                return False
            entry = _Entry(packet)
            self._entries.append(entry)
            self._total_bytes += packet.size_bytes
            if not self._trim_to_limits():
                removed = self._entries.pop()
                self._total_bytes -= removed.packet.size_bytes
                self._drop(packet, lease_pressure=True)
                return False
            self.metrics.accepted_packets += 1
            self.metrics.accepted_bytes += packet.size_bytes
            return True

    def select(
        self,
        *,
        trigger_epoch: StreamEpoch,
        trigger_pts: Fraction,
        pre_seconds: Fraction,
        post_seconds: Fraction,
    ) -> PacketSelection:
        if pre_seconds < 0 or post_seconds < 0:
            raise ValueError("packet selection windows must not be negative")
        with self._lock:
            if self._closed:
                raise PacketSelectionError(
                    PacketTruncationReason.RING_CLOSED,
                    "packet history is closed",
                )
            if trigger_epoch.camera_id != self.camera_id:
                raise PacketSelectionError(
                    PacketTruncationReason.KEYFRAME_UNAVAILABLE,
                    "trigger camera does not match packet ring",
                )
            epoch_entries = tuple(
                entry for entry in self._entries if entry.packet.epoch == trigger_epoch
            )
            trigger_video = [
                entry
                for entry in epoch_entries
                if entry.packet.stream.media_type == "video"
                and _safe_pts(entry.packet) <= trigger_pts
            ]
            if not trigger_video:
                raise PacketSelectionError(
                    PacketTruncationReason.KEYFRAME_UNAVAILABLE,
                    "no video packet exists at or before the trigger",
                )
            configuration = trigger_video[-1].packet.configuration
            config_entries = tuple(
                entry
                for entry in epoch_entries
                if entry.packet.configuration.configuration_id == configuration.configuration_id
            )
            requested_start = trigger_pts - pre_seconds
            requested_end = trigger_pts + post_seconds
            keyframes = [
                entry
                for entry in config_entries
                if entry.packet.stream.media_type == "video"
                and entry.packet.is_keyframe
                and _safe_pts(entry.packet) <= requested_start
            ]
            truncations: list[PacketTruncationReason] = []
            if keyframes:
                first = keyframes[-1]
            else:
                later_keyframes = [
                    entry
                    for entry in config_entries
                    if entry.packet.stream.media_type == "video"
                    and entry.packet.is_keyframe
                    and _safe_pts(entry.packet) <= trigger_pts
                ]
                if not later_keyframes:
                    raise PacketSelectionError(
                        PacketTruncationReason.KEYFRAME_UNAVAILABLE,
                        "no decodable keyframe exists at or before the trigger",
                    )
                first = later_keyframes[0]
                truncations.append(PacketTruncationReason.HISTORY_UNAVAILABLE)
            selected_start = _safe_pts(first.packet)
            end_candidates = tuple(
                entry
                for entry in config_entries
                if entry.packet.arrival_index >= first.packet.arrival_index
                and _safe_pts(entry.packet) <= requested_end
            )
            if not end_candidates:
                raise PacketSelectionError(
                    PacketTruncationReason.KEYFRAME_UNAVAILABLE,
                    "keyframe selection produced no packets",
                )
            # Keep one contiguous demux-order interval. Filtering every packet by
            # PTS would punch holes around B-frames (a later-presented packet can
            # arrive before an earlier-presented one), causing the muxer to infer
            # different durations for packets adjacent to those holes.
            last_arrival_index = max(entry.packet.arrival_index for entry in end_candidates)
            selected_entries = tuple(
                entry
                for entry in config_entries
                if first.packet.arrival_index <= entry.packet.arrival_index <= last_arrival_index
            )
            if any(entry.packet.discontinuity for entry in selected_entries):
                raise PacketSelectionError(
                    PacketTruncationReason.TIMESTAMP_DISCONTINUITY,
                    "selected packet interval crosses a timestamp discontinuity",
                )
            other_configs = {
                entry.packet.configuration.configuration_id
                for entry in epoch_entries
                if entry.packet.configuration.configuration_id != configuration.configuration_id
                and _safe_pts(entry.packet) <= trigger_pts
            }
            if other_configs:
                truncations.append(PacketTruncationReason.CONFIGURATION_CHANGED)
            latest_video_time = max(
                (
                    _safe_pts(entry.packet)
                    for entry in config_entries
                    if entry.packet.stream.media_type == "video"
                ),
                default=selected_start,
            )
            if latest_video_time < requested_end:
                truncations.append(PacketTruncationReason.FUTURE_UNAVAILABLE)
            ordered_truncations = tuple(dict.fromkeys(truncations))
            for entry in selected_entries:
                entry.lease_count += 1
            self.metrics.active_leases += 1
            return PacketSelection(
                self,
                selected_entries,
                configuration=configuration,
                requested_start=requested_start,
                requested_end=requested_end,
                selected_start=selected_start,
                selected_end=max(_safe_pts(entry.packet) for entry in selected_entries),
                truncations=ordered_truncations,
            )

    def evict_oldest_unleased(self) -> SourcePacket | None:
        """Release one oldest packet for process-wide budget recovery."""
        with self._lock:
            if self._closed or not self._entries or self._entries[0].lease_count:
                return None
            removed = self._entries.popleft().packet
            self._total_bytes -= removed.size_bytes
            self.metrics.evicted_packets += 1
            self.metrics.evicted_bytes += removed.size_bytes
            return removed

    def close(self) -> None:
        with self._lock:
            self._closed = True
            retained = deque(entry for entry in self._entries if entry.lease_count > 0)
            self._entries = retained
            self._total_bytes = sum(entry.packet.size_bytes for entry in retained)

    def _release(self, entries: tuple[_Entry, ...]) -> None:
        with self._lock:
            for entry in entries:
                if entry.lease_count <= 0:
                    raise RuntimeError("packet selection lease was released twice")
                entry.lease_count -= 1
            self.metrics.active_leases -= 1
            if self._closed:
                self._entries.clear()
                self._total_bytes = 0

    def _trim_to_limits(self) -> bool:
        while self._over_limit():
            oldest = self._entries[0]
            if oldest.lease_count:
                return False
            removed = self._entries.popleft()
            self._total_bytes -= removed.packet.size_bytes
            self.metrics.evicted_packets += 1
            self.metrics.evicted_bytes += removed.packet.size_bytes
        return True

    def _over_limit(self) -> bool:
        if len(self._entries) > self.limits.max_packets:
            return True
        if self._total_bytes > self.limits.max_bytes:
            return True
        video_times = [
            _safe_pts(entry.packet)
            for entry in self._entries
            if entry.packet.stream.media_type == "video"
        ]
        return bool(
            video_times
            and video_times[-1] - video_times[0] > Fraction(str(self.limits.max_duration_seconds))
        )

    def _drop(self, packet: SourcePacket, *, lease_pressure: bool) -> None:
        self.metrics.dropped_packets += 1
        self.metrics.dropped_bytes += packet.size_bytes
        if lease_pressure:
            self.metrics.lease_backpressure_drops += 1


def _safe_pts(packet: SourcePacket) -> Fraction:
    try:
        return packet.presentation_time
    except ValueError as exc:
        raise PacketSelectionError(
            PacketTruncationReason.TIMESTAMP_DISCONTINUITY,
            "packet PTS is unavailable",
        ) from exc


__all__ = [
    "PacketRingLimits",
    "PacketRingMetrics",
    "PacketSelection",
    "SourcePacketRing",
]

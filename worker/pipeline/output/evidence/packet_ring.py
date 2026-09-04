from __future__ import annotations

import logging
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

_LOGGER = logging.getLogger(__name__)


def _describe_video_pts(packet: SourcePacket) -> Fraction | None:
    try:
        return _safe_pts(packet)
    except PacketSelectionError:
        return None


def _describe_pts(entry: _Entry) -> float | None:
    """Best-effort PTS for a diagnostic line; never raises."""
    try:
        return round(_safe_pts(entry.packet), 3)
    except PacketSelectionError:
        return None


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
    epoch_rolls: int = 0


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
        self._retired_entries: deque[_Entry] = deque()
        self._total_bytes = 0
        self._active_epoch: StreamEpoch | None = None
        # Newest video PTS admitted for the active epoch, so the alignment
        # barrier answers in O(1) instead of rescanning the ring per packet.
        self._newest_video_pts: Fraction | None = None
        self._max_duration = Fraction(str(limits.max_duration_seconds))
        self._closed = False
        self._lock = threading.RLock()
        # Signalled whenever the active epoch rolls or a packet lands, so an
        # evidence trigger can wait for this ring to catch up with the
        # perception plane instead of being refused for a skew it cannot see.
        self._advanced = threading.Condition(self._lock)

    @property
    def active_epoch(self) -> StreamEpoch | None:
        with self._lock:
            return self._active_epoch

    @property
    def packet_count(self) -> int:
        with self._lock:
            return len(self._entries) + len(self._retired_entries)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def snapshot(self) -> tuple[SourcePacket, ...]:
        with self._lock:
            return tuple(entry.packet for entry in self._entries)

    def roll_epoch(self, epoch: StreamEpoch) -> None:
        if epoch.camera_id != self.camera_id:
            raise ValueError("stream epoch camera does not match its ring")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot roll a closed packet ring")
            current = self._active_epoch
            if (
                current is not None
                and current.worker_boot_id == epoch.worker_boot_id
                and epoch.stream_epoch < current.stream_epoch
            ):
                raise ValueError("packet ring epoch cannot move backwards")
            removed_packets = len(self._entries)
            removed_bytes = sum(entry.packet.size_bytes for entry in self._entries)
            self._retired_entries.extend(entry for entry in self._entries if entry.lease_count > 0)
            self._entries.clear()
            self._total_bytes = sum(entry.packet.size_bytes for entry in self._retired_entries)
            self._active_epoch = epoch
            self._newest_video_pts = None
            self.metrics.epoch_rolls += 1
            self.metrics.evicted_packets += removed_packets
            self.metrics.evicted_bytes += removed_bytes
            self._advanced.notify_all()

    def wait_until_ready(
        self,
        *,
        epoch: StreamEpoch,
        through_pts: Fraction,
        timeout_sec: float,
    ) -> bool:
        """Wait briefly for this ring's video to reach ``through_pts`` on ``epoch``.

        The AU tee and the perception metadata leave the child on separate
        sockets, so the access unit for a trigger can land a moment after the
        frame that raised it. Waiting for that PTS is cheap and bounded.

        The barrier does NOT wait for the ring to adopt ``epoch``. The ring
        rolls only after the recorder has sealed the previous epoch's clip,
        and that seal runs on the same actor thread this method is called
        from; waiting here for a roll that is queued behind the caller can
        never succeed, and every second spent doing so backs up a 128-slot
        recorder queue fed at ~195 messages/s. So a ring on another epoch
        returns ``False`` at once. A ring that never rolled an epoch admits
        everything and returns ``True`` at once for the same reason.

        The caller must NOT relabel the trigger on ``False``: emitting with
        the original identity is what lets the evidence path record an honest
        ``video_unavailable`` cause.
        """

        def state() -> bool | None:
            if self._closed:
                return False
            if self._active_epoch is None:
                return True
            if self._active_epoch != epoch:
                return False
            if self._newest_video_pts is not None and self._newest_video_pts >= through_pts:
                return True
            return None

        with self._advanced:
            self._advanced.wait_for(lambda: state() is not None, timeout=timeout_sec)
            return state() is True

    def append(self, packet: SourcePacket) -> bool:
        if packet.epoch.camera_id != self.camera_id:
            raise ValueError("packet camera does not match its ring")
        with self._lock:
            if (
                self._closed
                or packet.size_bytes > self.limits.max_bytes
                or (self._active_epoch is not None and packet.epoch != self._active_epoch)
            ):
                self._drop(packet, lease_pressure=False)
                return False
            entry = _Entry(packet)
            self._entries.append(entry)
            self._total_bytes += packet.size_bytes
            newest = self._newest_video_pts
            if packet.stream.media_type == "video":
                pts = _describe_video_pts(packet)
                if pts is not None and (newest is None or pts > newest):
                    newest = pts
            if not self._trim_to_limits(newest):
                removed = self._entries.pop()
                self._total_bytes -= removed.packet.size_bytes
                self._drop(packet, lease_pressure=True)
                return False
            if newest is not self._newest_video_pts:
                self._newest_video_pts = newest
                self._advanced.notify_all()
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
                self._log_selection_failure(
                    "trigger camera does not match packet ring",
                    trigger_camera=trigger_epoch.camera_id,
                    ring_camera=self.camera_id,
                )
                raise PacketSelectionError(
                    PacketTruncationReason.KEYFRAME_UNAVAILABLE,
                    "trigger camera does not match packet ring",
                )
            if self._active_epoch is not None and trigger_epoch != self._active_epoch:
                self._log_selection_failure(
                    "trigger stream epoch is no longer active",
                    trigger_epoch=trigger_epoch.stream_epoch,
                    active_epoch=self._active_epoch.stream_epoch,
                    trigger_generation=trigger_epoch.source_generation,
                    active_generation=self._active_epoch.source_generation,
                )
                raise PacketSelectionError(
                    PacketTruncationReason.STREAM_EPOCH_MISMATCH,
                    "trigger stream epoch is no longer active",
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
                self._log_selection_failure(
                    "no video packet exists at or before the trigger",
                    ring_entries=len(self._entries),
                    epoch_entries=len(epoch_entries),
                    epoch_video=sum(
                        1 for e in epoch_entries if e.packet.stream.media_type == "video"
                    ),
                    trigger_pts=round(trigger_pts, 3),
                    oldest_pts=_describe_pts(epoch_entries[0]) if epoch_entries else None,
                    newest_pts=_describe_pts(epoch_entries[-1]) if epoch_entries else None,
                )
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
                    self._log_selection_failure(
                        "no decodable keyframe exists at or before the trigger",
                        ring_entries=len(self._entries),
                        epoch_entries=len(epoch_entries),
                        config_entries=len(config_entries),
                        config_video=sum(
                            1 for e in config_entries if e.packet.stream.media_type == "video"
                        ),
                        config_keyframes=sum(
                            1
                            for e in config_entries
                            if e.packet.stream.media_type == "video" and e.packet.is_keyframe
                        ),
                        epoch_keyframes=sum(
                            1
                            for e in epoch_entries
                            if e.packet.stream.media_type == "video" and e.packet.is_keyframe
                        ),
                        trigger_pts=round(trigger_pts, 3),
                        requested_start=round(requested_start, 3),
                        oldest_pts=_describe_pts(config_entries[0]) if config_entries else None,
                        newest_pts=_describe_pts(config_entries[-1]) if config_entries else None,
                    )
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
                self._log_selection_failure(
                    "keyframe selection produced no packets",
                    config_entries=len(config_entries),
                    selected_start=round(selected_start, 3),
                    requested_end=round(requested_end, 3),
                )
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
            self._retired_entries.extend(entry for entry in self._entries if entry.lease_count > 0)
            self._entries.clear()
            self._total_bytes = sum(entry.packet.size_bytes for entry in self._retired_entries)

    def _release(self, entries: tuple[_Entry, ...]) -> None:
        with self._lock:
            for entry in entries:
                if entry.lease_count <= 0:
                    raise RuntimeError("packet selection lease was released twice")
                entry.lease_count -= 1
            self.metrics.active_leases -= 1
            released_retired_bytes = sum(
                entry.packet.size_bytes for entry in self._retired_entries if entry.lease_count == 0
            )
            self._retired_entries = deque(
                entry for entry in self._retired_entries if entry.lease_count > 0
            )
            self._total_bytes -= released_retired_bytes

    def _log_selection_failure(self, detail: str, **facts: object) -> None:
        """Name which selection predicate rejected the clip window.

        Five distinct conditions all raise ``KEYFRAME_UNAVAILABLE``, and only
        the reason code reaches the manifest, so an operator sees one opaque
        code for five different faults. Rendered into the message string rather
        than ``extra=`` because the worker's ``basicConfig`` format is
        ``%(message)s`` only.
        """
        rendered = " ".join(f"{key}={value}" for key, value in facts.items())
        _LOGGER.warning(
            "packet selection failed: camera_id=%s detail=%s %s",
            self.camera_id,
            detail,
            rendered,
        )

    def _trim_to_limits(self, newest_video_pts: Fraction | None) -> bool:
        while self._over_limit(newest_video_pts):
            if len(self._entries) == 1 and self._retired_entries:
                return False
            oldest = self._entries[0]
            if oldest.lease_count:
                return False
            removed = self._entries.popleft()
            self._total_bytes -= removed.packet.size_bytes
            self.metrics.evicted_packets += 1
            self.metrics.evicted_bytes += removed.packet.size_bytes
        return True

    def _over_limit(self, newest_video_pts: Fraction | None) -> bool:
        if len(self._entries) + len(self._retired_entries) > self.limits.max_packets:
            return True
        if self._total_bytes > self.limits.max_bytes:
            return True
        if newest_video_pts is None:
            return False
        # Oldest video entry is at (or next to) the front; the newest PTS is
        # tracked by append. This used to recompute every entry's PTS on every
        # append -- 2.5 ms per unit at 1000 entries, 64% of a core at fleet
        # rate, which is what made the AU thread fall behind (#429).
        for entry in self._entries:
            if entry.packet.stream.media_type == "video":
                return newest_video_pts - _safe_pts(entry.packet) > self._max_duration
        return False

    def _drop(self, packet: SourcePacket, *, lease_pressure: bool) -> None:
        self.metrics.dropped_packets += 1
        self.metrics.dropped_bytes += packet.size_bytes
        if not lease_pressure:
            return
        self.metrics.lease_backpressure_drops += 1
        # A lease-pressure drop punches a hole into a clip window that is still
        # being recorded: select() has pinned the window, so the ring cannot trim
        # and discards the arriving packet instead. Downstream this surfaces only
        # as "remuxed packet duration changed", because MP4 stretches the
        # pre-gap packet's stts duration to the next survivor. The counter for
        # this existed but was never reported anywhere, which is why the
        # condition stayed invisible while clips silently failed to finalize.
        # Throttled to powers of two, mirroring the demuxer's drop log.
        count = self.metrics.lease_backpressure_drops
        if count & (count - 1) == 0:
            _LOGGER.warning(
                "packet ring dropped an arriving packet under lease backpressure: "
                "camera_id=%s lease_backpressure_drops=%s active_leases=%s; "
                "the in-flight clip window will be discontiguous",
                self.camera_id,
                count,
                self.metrics.active_leases,
                extra={"camera_id": self.camera_id, "lease_backpressure_drops": count},
            )


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

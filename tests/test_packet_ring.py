from __future__ import annotations

from fractions import Fraction

import pytest

from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.packet_ring import (
    PacketRingLimits,
    SourcePacketRing,
)
from worker.types.source_packet import (
    PacketSelectionError,
    PacketTruncationReason,
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
    StreamEpoch,
)


def _configuration(
    *, extradata: bytes = b"avcc-1", audio: bool = True
) -> SourceStreamConfiguration:
    streams = [
        SourceStreamDescriptor(
            index=0,
            media_type="video",
            codec_name="h264",
            codec_tag="avc1",
            time_base=Fraction(1, 90_000),
            extradata=extradata,
            width=1920,
            height=1080,
        )
    ]
    if audio:
        streams.append(
            SourceStreamDescriptor(
                index=1,
                media_type="audio",
                codec_name="aac",
                codec_tag="mp4a",
                time_base=Fraction(1, 48_000),
                extradata=b"aac-1",
                sample_rate=48_000,
                channels=1,
            )
        )
    return SourceStreamConfiguration.from_streams(streams)


def _packet(
    pts_seconds: Fraction,
    *,
    epoch: int = 1,
    configuration: SourceStreamConfiguration | None = None,
    stream_index: int = 0,
    dts_seconds: Fraction | None = None,
    keyframe: bool = False,
    size: int = 10,
    arrival_index: int | None = None,
    discontinuity: str | None = None,
) -> SourcePacket:
    config = configuration or _configuration()
    stream = config.stream(stream_index)
    pts = int(pts_seconds / stream.time_base)
    dts_value = pts_seconds if dts_seconds is None else dts_seconds
    dts = int(dts_value / stream.time_base)
    return SourcePacket(
        epoch=StreamEpoch("boot-1", "camera-1", epoch),
        configuration=config,
        stream_index=stream_index,
        pts=pts,
        dts=dts,
        duration=1,
        is_keyframe=keyframe,
        payload=b"x" * size,
        arrival_index=pts if arrival_index is None else arrival_index,
        discontinuity=discontinuity,
    )


def _append_gop(ring: SourcePacketRing, *, start: int, stop: int, epoch: int = 1) -> None:
    for second in range(start, stop + 1):
        assert ring.append(_packet(Fraction(second), epoch=epoch, keyframe=second % 4 == 0))


def test_selects_one_epoch_and_configuration_from_keyframe_through_post_window() -> None:
    ring = SourcePacketRing("camera-1", PacketRingLimits(64, 4_096, 60.0))
    _append_gop(ring, start=0, stop=12)

    with ring.select(
        trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
        trigger_pts=Fraction(7),
        pre_seconds=Fraction(2),
        post_seconds=Fraction(3),
    ) as selection:
        video_pts = tuple(packet.presentation_time for packet in selection.packets)
        assert video_pts == tuple(Fraction(value) for value in range(4, 11))
        assert selection.selected_start == Fraction(4)
        assert selection.selected_end == Fraction(10)
        assert selection.requested_start == Fraction(5)
        assert selection.requested_end == Fraction(10)
        assert selection.truncations == ()
        assert {packet.configuration.configuration_id for packet in selection.packets} == {
            selection.configuration.configuration_id
        }


def test_b_frames_remain_in_demux_order_with_original_pts_dts_and_vfr_duration() -> None:
    ring = SourcePacketRing("camera-1", PacketRingLimits(32, 4_096, 60.0))
    packets = (
        _packet(Fraction(0), dts_seconds=Fraction(0), keyframe=True, arrival_index=0),
        _packet(Fraction(4), dts_seconds=Fraction(1), arrival_index=1),
        _packet(Fraction(1), dts_seconds=Fraction(2), arrival_index=2),
        _packet(Fraction(7, 2), dts_seconds=Fraction(3), arrival_index=3),
    )
    for packet in packets:
        assert ring.append(packet)

    with ring.select(
        trigger_epoch=packets[0].epoch,
        trigger_pts=Fraction(2),
        pre_seconds=Fraction(2),
        post_seconds=Fraction(3, 2),
    ) as selection:
        # The PTS=4 packet is beyond the requested end but cannot be removed:
        # doing so would punch a hole before reordered packets that present
        # earlier and alter stream-copy packet durations.
        assert tuple(packet.arrival_index for packet in selection.packets) == (0, 1, 2, 3)
        assert tuple(packet.pts for packet in selection.packets) == tuple(
            packet.pts for packet in packets
        )
        assert tuple(packet.dts for packet in selection.packets) == tuple(
            packet.dts for packet in packets
        )
        assert selection.selected_end == Fraction(4)


def test_optional_audio_is_selected_without_requiring_audio() -> None:
    with_audio = _configuration(audio=True)
    ring = SourcePacketRing("camera-1", PacketRingLimits(32, 4_096, 60.0))
    video = _packet(Fraction(0), configuration=with_audio, keyframe=True, arrival_index=0)
    audio = _packet(
        Fraction(1, 2),
        configuration=with_audio,
        stream_index=1,
        arrival_index=1,
    )
    later_video = _packet(Fraction(1), configuration=with_audio, arrival_index=2)
    for packet in (video, audio, later_video):
        assert ring.append(packet)
    with ring.select(
        trigger_epoch=video.epoch,
        trigger_pts=Fraction(1, 2),
        pre_seconds=Fraction(1, 2),
        post_seconds=Fraction(1, 2),
    ) as selection:
        assert tuple(packet.stream_index for packet in selection.packets) == (0, 1, 0)

    video_only = _configuration(audio=False)
    second = SourcePacketRing("camera-1", PacketRingLimits(8, 512, 10.0))
    assert second.append(_packet(Fraction(0), configuration=video_only, keyframe=True))
    assert second.append(_packet(Fraction(1), configuration=video_only))
    with second.select(
        trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
        trigger_pts=Fraction(0),
        pre_seconds=Fraction(0),
        post_seconds=Fraction(1),
    ) as selection:
        assert selection.configuration.audio_streams == ()


def test_reconnect_and_extradata_change_never_mix_packets() -> None:
    ring = SourcePacketRing("camera-1", PacketRingLimits(64, 4_096, 60.0))
    old = _configuration(extradata=b"old")
    new = _configuration(extradata=b"new")
    assert ring.append(_packet(Fraction(0), configuration=old, keyframe=True))
    assert ring.append(_packet(Fraction(1), configuration=old))
    assert ring.append(_packet(Fraction(0), epoch=2, configuration=new, keyframe=True))
    assert ring.append(_packet(Fraction(1), epoch=2, configuration=new))

    with ring.select(
        trigger_epoch=StreamEpoch("boot-1", "camera-1", 2),
        trigger_pts=Fraction(0),
        pre_seconds=Fraction(1),
        post_seconds=Fraction(1),
    ) as selection:
        assert {packet.epoch.stream_epoch for packet in selection.packets} == {2}
        assert {packet.configuration.configuration_id for packet in selection.packets} == {
            new.configuration_id
        }
        assert PacketTruncationReason.HISTORY_UNAVAILABLE in selection.truncations

    changed_same_epoch = SourcePacketRing("camera-1", PacketRingLimits(32, 4_096, 60.0))
    assert changed_same_epoch.append(_packet(Fraction(0), configuration=old, keyframe=True))
    assert changed_same_epoch.append(_packet(Fraction(1), configuration=old))
    assert changed_same_epoch.append(_packet(Fraction(2), configuration=new, keyframe=True))
    assert changed_same_epoch.append(_packet(Fraction(3), configuration=new))
    with changed_same_epoch.select(
        trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
        trigger_pts=Fraction(3),
        pre_seconds=Fraction(3),
        post_seconds=Fraction(0),
    ) as selection:
        assert selection.configuration.configuration_id == new.configuration_id
        assert tuple(packet.presentation_time for packet in selection.packets) == (
            Fraction(2),
            Fraction(3),
        )
        assert PacketTruncationReason.CONFIGURATION_CHANGED in selection.truncations


def test_missing_keyframe_fails_closed_and_missing_future_is_observable() -> None:
    ring = SourcePacketRing("camera-1", PacketRingLimits(16, 1_024, 30.0))
    assert ring.append(_packet(Fraction(1)))
    assert ring.append(_packet(Fraction(2)))
    with pytest.raises(PacketSelectionError) as raised:
        ring.select(
            trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
            trigger_pts=Fraction(2),
            pre_seconds=Fraction(1),
            post_seconds=Fraction(1),
        )
    assert raised.value.reason is PacketTruncationReason.KEYFRAME_UNAVAILABLE

    ready = SourcePacketRing("camera-1", PacketRingLimits(16, 1_024, 30.0))
    assert ready.append(_packet(Fraction(0), keyframe=True))
    assert ready.append(_packet(Fraction(1)))
    with ready.select(
        trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
        trigger_pts=Fraction(1),
        pre_seconds=Fraction(1),
        post_seconds=Fraction(3),
    ) as selection:
        assert selection.truncations == (PacketTruncationReason.FUTURE_UNAVAILABLE,)


def test_discontinuity_after_keyframe_fails_closed_instead_of_coercing_timestamps() -> None:
    ring = SourcePacketRing("camera-1", PacketRingLimits(16, 1_024, 30.0))
    assert ring.append(_packet(Fraction(0), keyframe=True))
    assert ring.append(_packet(Fraction(1), discontinuity="pts-wrap-or-jump"))

    with pytest.raises(PacketSelectionError) as raised:
        ring.select(
            trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
            trigger_pts=Fraction(1),
            pre_seconds=Fraction(1),
            post_seconds=Fraction(0),
        )
    assert raised.value.reason is PacketTruncationReason.TIMESTAMP_DISCONTINUITY


def test_limits_evict_oldest_unleased_packets_and_drop_under_lease_pressure() -> None:
    ring = SourcePacketRing("camera-1", PacketRingLimits(3, 30, 30.0))
    for second in range(3):
        assert ring.append(_packet(Fraction(second), keyframe=second == 0))

    lease = ring.select(
        trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
        trigger_pts=Fraction(1),
        pre_seconds=Fraction(1),
        post_seconds=Fraction(1),
    )
    assert ring.append(_packet(Fraction(3))) is False
    assert ring.metrics.dropped_packets == 1
    assert ring.metrics.lease_backpressure_drops == 1
    assert ring.total_bytes == 30

    lease.close()
    assert ring.append(_packet(Fraction(3))) is True
    assert ring.packet_count == 3
    assert tuple(packet.presentation_time for packet in ring.snapshot()) == (
        Fraction(1),
        Fraction(2),
        Fraction(3),
    )
    assert ring.metrics.evicted_packets == 1


def test_repository_enforces_one_global_byte_cap_and_recovers_by_eviction() -> None:
    repository = PacketRingRepository(
        ("camera-1", "camera-2"),
        per_camera_limits=PacketRingLimits(8, 100, 30.0),
        global_max_bytes=15,
    )
    first = _packet(Fraction(0), keyframe=True, size=10)
    second = SourcePacket(
        epoch=StreamEpoch("boot-1", "camera-2", 1),
        configuration=first.configuration,
        stream_index=0,
        pts=first.pts,
        dts=first.dts,
        duration=first.duration,
        is_keyframe=True,
        payload=b"y" * 10,
        arrival_index=0,
    )

    assert repository.append(first) is True
    assert repository.append(second) is True
    assert repository.total_bytes == 10
    assert repository.ring("camera-1").packet_count == 0
    assert repository.ring("camera-2").packet_count == 1
    assert repository.metrics.global_evicted_packets == 1
    assert repository.metrics.global_evicted_bytes == 10


def test_repository_drops_under_global_lease_pressure_then_recovers() -> None:
    repository = PacketRingRepository(
        ("camera-1", "camera-2"),
        per_camera_limits=PacketRingLimits(8, 100, 30.0),
        global_max_bytes=20,
    )
    first = _packet(Fraction(0), keyframe=True, size=10, arrival_index=0)
    later = _packet(Fraction(1), size=10, arrival_index=1)
    assert repository.append(first)
    assert repository.append(later)
    lease = repository.ring("camera-1").select(
        trigger_epoch=first.epoch,
        trigger_pts=Fraction(1),
        pre_seconds=Fraction(1),
        post_seconds=Fraction(0),
    )
    other = SourcePacket(
        epoch=StreamEpoch("boot-1", "camera-2", 1),
        configuration=first.configuration,
        stream_index=0,
        pts=first.pts,
        dts=first.dts,
        duration=first.duration,
        is_keyframe=True,
        payload=b"y" * 10,
        arrival_index=0,
    )

    assert repository.append(other) is False
    assert repository.metrics.global_limit_drops == 1
    assert repository.total_bytes == 20
    lease.close()
    assert repository.append(other) is True
    assert repository.total_bytes == 20
    assert repository.metrics.global_evicted_packets == 1


def test_close_releases_memory_and_rejects_new_packets_or_leases() -> None:
    ring = SourcePacketRing("camera-1", PacketRingLimits(8, 1_024, 30.0))
    packet = _packet(Fraction(0), keyframe=True)
    assert ring.append(packet)
    ring.close()

    assert ring.total_bytes == 0
    assert ring.packet_count == 0
    assert ring.append(packet) is False
    with pytest.raises(PacketSelectionError) as raised:
        ring.select(
            trigger_epoch=packet.epoch,
            trigger_pts=Fraction(0),
            pre_seconds=Fraction(0),
            post_seconds=Fraction(0),
        )
    assert raised.value.reason is PacketTruncationReason.RING_CLOSED

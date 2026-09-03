"""Pin the explicit ``SourcePacketRing`` epoch boundary added by Wave 2.

The pre-refactor seam tests proved that ``select()`` was the ring's only epoch
boundary: stale packets remained resident and duration trimming went dormant
when PTS restarted. The NVDEC packet tee now calls ``roll_epoch()`` before a
reconnected/configuration-reset stream can append. These tests pin the new
surface without changing CPU/VAAPI preserving-session behavior.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import final

import av
import numpy as np
import pytest

from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.adapters.decode.pyav_preserving import PyAvPreservingAdapter
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


@final
class _RecordingSink:
    def __init__(self) -> None:
        self.packets: list[SourcePacket] = []

    def append(self, packet: SourcePacket) -> bool:
        self.packets.append(packet)
        return True


def _encode(path: Path, *, width: int, height: int, frames: int = 16) -> None:
    container = av.open(str(path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=30)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"g": "4"}
    for index in range(frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, index % 3] = 255
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _demux_into(sink: SourcePacketRing | _RecordingSink, path: Path, epoch: int) -> None:
    session = PyAvPreservingAdapter(sink, decode_backend="cpu").open(
        CpuAvConfig("camera-1", str(path), open_timeout_ms=2_000, read_timeout_ms=2_000)
    )
    session.set_stream_identity("boot-1", epoch)
    assert session.wait_demux_complete(15)
    session.close()


def _configuration(extradata: bytes) -> SourceStreamConfiguration:
    return SourceStreamConfiguration.from_streams(
        [
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
    )


def _packet(
    second: int,
    *,
    epoch: int,
    configuration: SourceStreamConfiguration,
    keyframe: bool,
) -> SourcePacket:
    stream = configuration.stream(0)
    ticks = int(Fraction(second) / stream.time_base)
    return SourcePacket(
        epoch=StreamEpoch("boot-1", "camera-1", epoch),
        configuration=configuration,
        stream_index=0,
        pts=ticks,
        dts=ticks,
        duration=1,
        is_keyframe=keyframe,
        payload=b"x" * 10,
        arrival_index=second,
    )


def test_epoch_reset_with_new_sps_rolls_and_purges_the_ring(tmp_path: Path) -> None:
    """C1: a reconnect with genuinely different SPS/PPS starts clean history."""
    first = tmp_path / "epoch-1.mp4"
    second = tmp_path / "epoch-2.mp4"
    _encode(first, width=64, height=48)
    _encode(second, width=80, height=64)
    observed = _RecordingSink()
    _demux_into(observed, first, 1)
    _demux_into(observed, second, 2)
    assert observed.packets[0].stream.extradata != observed.packets[-1].stream.extradata

    ring = SourcePacketRing("camera-1", PacketRingLimits(1_000, 8 * 1024 * 1024, 60.0))
    first_epoch = StreamEpoch("boot-1", "camera-1", 1)
    second_epoch = StreamEpoch("boot-1", "camera-1", 2)
    ring.roll_epoch(first_epoch)
    _demux_into(ring, first, 1)
    old_payloads = tuple(packet.payload for packet in ring.snapshot())

    ring.roll_epoch(second_epoch)
    _demux_into(ring, second, 2)

    snapshot = ring.snapshot()
    assert snapshot
    assert ring.active_epoch == second_epoch
    assert {packet.epoch for packet in snapshot} == {second_epoch}
    assert tuple(packet.payload for packet in snapshot) != old_payloads
    assert ring.metrics.dropped_packets == 0


def test_select_rejects_a_stale_epoch_after_roll(tmp_path: Path) -> None:
    """C2: old-epoch packets cannot become new evidence after a reconnect."""
    first = tmp_path / "epoch-1.mp4"
    second = tmp_path / "epoch-2.mp4"
    _encode(first, width=64, height=48)
    _encode(second, width=80, height=64)
    ring = SourcePacketRing("camera-1", PacketRingLimits(1_000, 8 * 1024 * 1024, 60.0))
    old_epoch = StreamEpoch("boot-1", "camera-1", 1)
    new_epoch = StreamEpoch("boot-1", "camera-1", 2)
    ring.roll_epoch(old_epoch)
    _demux_into(ring, first, 1)
    old_trigger = max(packet.presentation_time for packet in ring.snapshot())
    ring.roll_epoch(new_epoch)
    _demux_into(ring, second, 2)

    with pytest.raises(PacketSelectionError) as raised:
        ring.select(
            trigger_epoch=old_epoch,
            trigger_pts=old_trigger,
            pre_seconds=Fraction(10),
            post_seconds=Fraction(0),
        )
    assert raised.value.reason is PacketTruncationReason.STREAM_EPOCH_MISMATCH

    current_packets = ring.snapshot()
    with ring.select(
        trigger_epoch=new_epoch,
        trigger_pts=max(packet.presentation_time for packet in current_packets),
        pre_seconds=Fraction(10),
        post_seconds=Fraction(0),
    ) as current:
        assert {packet.epoch for packet in current.packets} == {new_epoch}


def test_epoch_roll_hides_leased_history_but_keeps_it_inside_memory_budget() -> None:
    """A pre-roll selection stays valid without making stale history selectable."""
    configuration = _configuration(b"sps-old")
    ring = SourcePacketRing("camera-1", PacketRingLimits(100, 10_000, 60.0))
    old_epoch = StreamEpoch("boot-1", "camera-1", 1)
    new_epoch = StreamEpoch("boot-1", "camera-1", 2)
    ring.roll_epoch(old_epoch)
    assert ring.append(_packet(0, epoch=1, configuration=configuration, keyframe=True))
    assert ring.append(_packet(1, epoch=1, configuration=configuration, keyframe=False))
    selection = ring.select(
        trigger_epoch=old_epoch,
        trigger_pts=Fraction(1),
        pre_seconds=Fraction(1),
        post_seconds=Fraction(0),
    )

    ring.roll_epoch(new_epoch)

    assert ring.snapshot() == ()
    assert ring.total_bytes == 20
    assert tuple(packet.epoch for packet in selection.packets) == (old_epoch, old_epoch)
    with pytest.raises(PacketSelectionError):
        ring.select(
            trigger_epoch=old_epoch,
            trigger_pts=Fraction(1),
            pre_seconds=Fraction(1),
            post_seconds=Fraction(0),
        )
    selection.close()
    assert ring.total_bytes == 0


def test_arrival_index_restarts_per_epoch_without_resident_collisions(
    tmp_path: Path,
) -> None:
    """C3: session-local indexes still restart, but rolled history cannot collide."""
    first = tmp_path / "epoch-1.mp4"
    second = tmp_path / "epoch-2.mp4"
    _encode(first, width=64, height=48)
    _encode(second, width=80, height=64)
    ring = SourcePacketRing("camera-1", PacketRingLimits(1_000, 8 * 1024 * 1024, 60.0))
    ring.roll_epoch(StreamEpoch("boot-1", "camera-1", 1))
    _demux_into(ring, first, 1)
    assert ring.snapshot()[0].arrival_index == 0

    ring.roll_epoch(StreamEpoch("boot-1", "camera-1", 2))
    _demux_into(ring, second, 2)

    snapshot = ring.snapshot()
    assert snapshot[0].arrival_index == 0
    assert [packet.arrival_index for packet in snapshot] == list(range(len(snapshot)))
    assert {packet.epoch.stream_epoch for packet in snapshot} == {2}


def test_duration_trim_stays_active_after_pts_restart_when_epoch_is_rolled() -> None:
    """C4: purging before a PTS restart keeps duration trimming meaningful."""
    old = _configuration(b"sps-old")
    new = _configuration(b"sps-new")
    ring = SourcePacketRing("camera-1", PacketRingLimits(100, 10_000, 2.0))
    ring.roll_epoch(StreamEpoch("boot-1", "camera-1", 1))
    for second in range(10):
        assert ring.append(_packet(second, epoch=1, configuration=old, keyframe=True))
    assert [float(packet.presentation_time) for packet in ring.snapshot()] == [7.0, 8.0, 9.0]

    ring.roll_epoch(StreamEpoch("boot-1", "camera-1", 2))
    for second in range(6):
        assert ring.append(_packet(second, epoch=2, configuration=new, keyframe=True))

    assert [
        (packet.epoch.stream_epoch, float(packet.presentation_time)) for packet in ring.snapshot()
    ] == [(2, 3.0), (2, 4.0), (2, 5.0)]


def test_append_rejects_stale_epoch_after_roll_and_foreign_camera_always() -> None:
    """C5: once activated, a ring refuses late writes from superseded demuxers."""
    configuration = _configuration(b"sps-1")
    ring = SourcePacketRing("camera-1", PacketRingLimits(100, 10_000, 60.0))
    active = StreamEpoch("boot-1", "camera-1", 5)
    ring.roll_epoch(active)

    assert ring.append(_packet(0, epoch=5, configuration=configuration, keyframe=True))
    assert not ring.append(_packet(1, epoch=2, configuration=configuration, keyframe=True))
    assert [packet.epoch for packet in ring.snapshot()] == [active]
    assert ring.metrics.dropped_packets == 1

    foreign = SourcePacket(
        epoch=StreamEpoch("boot-1", "camera-2", 1),
        configuration=configuration,
        stream_index=0,
        pts=0,
        dts=0,
        duration=1,
        is_keyframe=True,
        payload=b"y" * 10,
        arrival_index=0,
    )
    with pytest.raises(ValueError, match="packet camera does not match its ring"):
        ring.append(foreign)

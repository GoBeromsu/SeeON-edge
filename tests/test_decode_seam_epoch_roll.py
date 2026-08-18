"""Pin what actually happens to ``SourcePacketRing`` across a stream-epoch reset.

Discovered empirically and documented here as CURRENT behavior (plan todo 3
QA scenario: "assert the ring rolls its epoch, or document current behavior
precisely if it does not"). The finding is: **the ring does not roll**.

Three facts these tests pin, all reproduced from real objects:

1. ``SourcePacketRing`` holds no epoch state whatsoever. A new epoch's packets
   are simply appended after the old epoch's; nothing is purged, no metric
   moves, and stale-epoch packets stay selectable forever.
2. Epoch isolation is enforced only at ``select()`` time, by filtering entries
   on ``packet.epoch`` equality -- never at ``append()`` time.
3. ``arrival_index`` restarts at 0 for each session, and the duration trim
   (``_over_limit``) compares raw per-epoch PTS across the whole deque. Because
   a new epoch's PTS restarts near 0, the computed span goes negative right
   after a reset and the duration limit stops evicting until the new epoch
   catches up. Count and byte limits keep working.

Wave 2 (plan todo 4) is required to "roll the packet-ring epoch" on
SPS/PPS/extradata change -- these tests record exactly what "roll" must be
built on top of, so a regression against today's guarantees is visible.

Hermetic and CI-safe: media is synthesized in-process by PyAV (no ffmpeg
binary, no RTSP, no GPU).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import final

import av
import numpy as np

from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.adapters.decode.pyav_preserving import PyAvPreservingAdapter
from worker.pipeline.output.evidence.packet_ring import (
    PacketRingLimits,
    SourcePacketRing,
)
from worker.types.source_packet import (
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


def test_epoch_reset_with_new_sps_does_not_roll_or_purge_the_ring(tmp_path: Path) -> None:
    """SEAM C1 (CURRENT BEHAVIOR, empirically discovered): a real stream reset --
    new session, new epoch, genuinely different SPS/extradata -- leaves every
    old-epoch packet resident in the ring. No purge, no eviction, no metric.

    Wave 2's "roll the packet-ring epoch" therefore ADDS behavior; it must not
    assume the ring already discards prior-epoch packets."""
    # Given
    first = tmp_path / "epoch-1.mp4"
    second = tmp_path / "epoch-2.mp4"
    _encode(first, width=64, height=48)
    _encode(second, width=80, height=64)
    observed = _RecordingSink()
    _demux_into(observed, first, 1)
    _demux_into(observed, second, 2)
    old_extradata = observed.packets[0].stream.extradata
    new_extradata = observed.packets[-1].stream.extradata
    assert old_extradata != new_extradata  # a genuine SPS/PPS change
    ring = SourcePacketRing("camera-1", PacketRingLimits(1_000, 8 * 1024 * 1024, 60.0))

    # When
    _demux_into(ring, first, 1)
    epoch_one = ring.snapshot()
    _demux_into(ring, second, 2)

    # Then
    snapshot = ring.snapshot()
    assert tuple(packet.payload for packet in snapshot[: len(epoch_one)]) == tuple(
        packet.payload for packet in epoch_one
    )
    assert sorted({packet.epoch.stream_epoch for packet in snapshot}) == [1, 2]
    assert len({packet.configuration.configuration_id for packet in snapshot}) == 2
    assert ring.metrics.evicted_packets == 0
    assert ring.metrics.dropped_packets == 0
    assert ring.metrics.accepted_packets == len(snapshot)


def test_select_is_the_only_epoch_boundary_and_stale_epochs_stay_selectable(
    tmp_path: Path,
) -> None:
    """SEAM C2 (CURRENT BEHAVIOR): epoch isolation lives entirely in ``select()``.
    A selection never mixes epochs, but a selection against the OLD epoch still
    succeeds after the reset -- stale packets are readable, not discarded."""
    # Given
    first = tmp_path / "epoch-1.mp4"
    second = tmp_path / "epoch-2.mp4"
    _encode(first, width=64, height=48)
    _encode(second, width=80, height=64)
    ring = SourcePacketRing("camera-1", PacketRingLimits(1_000, 8 * 1024 * 1024, 60.0))
    _demux_into(ring, first, 1)
    _demux_into(ring, second, 2)
    snapshot = ring.snapshot()
    old_packets = tuple(packet for packet in snapshot if packet.epoch.stream_epoch == 1)
    new_packets = tuple(packet for packet in snapshot if packet.epoch.stream_epoch == 2)

    # When
    with ring.select(
        trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
        trigger_pts=max(packet.presentation_time for packet in old_packets),
        pre_seconds=Fraction(10),
        post_seconds=Fraction(0),
    ) as stale:
        stale_payloads = tuple(packet.payload for packet in stale.packets)
        stale_epochs = {packet.epoch.stream_epoch for packet in stale.packets}
    with ring.select(
        trigger_epoch=StreamEpoch("boot-1", "camera-1", 2),
        trigger_pts=max(packet.presentation_time for packet in new_packets),
        pre_seconds=Fraction(10),
        post_seconds=Fraction(0),
    ) as current:
        current_payloads = tuple(packet.payload for packet in current.packets)
        current_epochs = {packet.epoch.stream_epoch for packet in current.packets}

    # Then
    assert stale_epochs == {1}
    assert current_epochs == {2}
    assert stale_payloads == tuple(packet.payload for packet in old_packets)
    assert current_payloads == tuple(packet.payload for packet in new_packets)
    assert set(stale_payloads).isdisjoint(set(current_payloads))


def test_arrival_index_restarts_per_epoch_so_indexes_collide_inside_one_ring(
    tmp_path: Path,
) -> None:
    """SEAM C3 (CURRENT BEHAVIOR): ``arrival_index`` is session-scoped, not
    ring-scoped. Two epochs in one ring carry the same indexes; ``select()`` is
    only correct because it filters by epoch BEFORE comparing arrival indexes.
    Wave 2 must keep that ordering or the interval math silently mixes epochs."""
    # Given
    first = tmp_path / "epoch-1.mp4"
    second = tmp_path / "epoch-2.mp4"
    _encode(first, width=64, height=48)
    _encode(second, width=80, height=64)
    ring = SourcePacketRing("camera-1", PacketRingLimits(1_000, 8 * 1024 * 1024, 60.0))

    # When
    _demux_into(ring, first, 1)
    _demux_into(ring, second, 2)

    # Then
    snapshot = ring.snapshot()
    epoch_one = [packet.arrival_index for packet in snapshot if packet.epoch.stream_epoch == 1]
    epoch_two = [packet.arrival_index for packet in snapshot if packet.epoch.stream_epoch == 2]
    assert epoch_one == list(range(len(epoch_one)))
    assert epoch_two == list(range(len(epoch_two)))
    assert set(epoch_one) & set(epoch_two)


def test_duration_trim_goes_dormant_after_pts_restart_while_count_limit_holds() -> None:
    """SEAM C4 (CURRENT BEHAVIOR, the sharpest edge found): the duration limit is
    computed as ``video_times[-1] - video_times[0]`` over the whole deque. After
    an epoch reset the newest PTS is SMALLER than the oldest retained PTS, the
    span goes negative, and duration-based eviction stops -- old-epoch packets
    are pinned past the configured window. Only the count/byte caps still bound
    the ring. Constructed packets are used here so the PTS restart is exact."""
    # Given
    old = _configuration(b"sps-old")
    new = _configuration(b"sps-new")
    ring = SourcePacketRing("camera-1", PacketRingLimits(100, 10_000, 2.0))
    for second in range(10):
        assert ring.append(_packet(second, epoch=1, configuration=old, keyframe=second % 4 == 0))
    before = ring.snapshot()

    # When
    for second in range(6):
        assert ring.append(_packet(second, epoch=2, configuration=new, keyframe=second % 4 == 0))
    after = ring.snapshot()

    # Then
    assert [float(packet.presentation_time) for packet in before] == [7.0, 8.0, 9.0]
    assert [
        (packet.epoch.stream_epoch, float(packet.presentation_time)) for packet in after
    ] == [
        (1, 7.0),
        (1, 8.0),
        (1, 9.0),
        (2, 0.0),
        (2, 1.0),
        (2, 2.0),
        (2, 3.0),
        (2, 4.0),
        (2, 5.0),
    ]
    assert ring.metrics.evicted_packets == 7  # all seven evictions happened pre-reset

    # And the count cap is the bound that actually flushes the old epoch: FIFO
    # count pressure pushes stale-epoch packets out one at a time, and the
    # duration trim resumes only once the deque is entirely the new epoch.
    capped = SourcePacketRing("camera-1", PacketRingLimits(4, 10_000, 2.0))
    for second in range(6):
        assert capped.append(_packet(second, epoch=1, configuration=old, keyframe=True))
    assert [
        (packet.epoch.stream_epoch, float(packet.presentation_time))
        for packet in capped.snapshot()
    ] == [(1, 3.0), (1, 4.0), (1, 5.0)]
    observed: list[list[tuple[int, float]]] = []
    for second in range(6):
        assert capped.append(_packet(second, epoch=2, configuration=new, keyframe=True))
        observed.append(
            [
                (packet.epoch.stream_epoch, float(packet.presentation_time))
                for packet in capped.snapshot()
            ]
        )
    assert observed == [
        [(1, 3.0), (1, 4.0), (1, 5.0), (2, 0.0)],
        [(1, 4.0), (1, 5.0), (2, 0.0), (2, 1.0)],
        [(1, 5.0), (2, 0.0), (2, 1.0), (2, 2.0)],
        [(2, 1.0), (2, 2.0), (2, 3.0)],
        [(2, 2.0), (2, 3.0), (2, 4.0)],
        [(2, 3.0), (2, 4.0), (2, 5.0)],
    ]
    assert {packet.epoch.stream_epoch for packet in capped.snapshot()} == {2}


def test_append_never_inspects_epoch_only_camera_identity() -> None:
    """SEAM C5 (CURRENT BEHAVIOR): ``append`` validates the camera and the byte
    budget, nothing else. An out-of-order (older) epoch is accepted after a
    newer one; only a foreign camera is refused."""
    # Given
    configuration = _configuration(b"sps-1")
    ring = SourcePacketRing("camera-1", PacketRingLimits(100, 10_000, 60.0))

    # When
    assert ring.append(_packet(0, epoch=5, configuration=configuration, keyframe=True))
    assert ring.append(_packet(1, epoch=2, configuration=configuration, keyframe=True))
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

    # Then
    assert [packet.epoch.stream_epoch for packet in ring.snapshot()] == [5, 2]
    try:
        _ = ring.append(foreign)
    except ValueError as error:
        assert "packet camera does not match its ring" in str(error)
    else:  # pragma: no cover - documents the refusal contract
        raise AssertionError("ring accepted a packet from a foreign camera")

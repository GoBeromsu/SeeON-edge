"""The container-normalization exemption skips three checks but not the fourth.

`_verify_packet_facts` takes a `container_normalized_streams` set naming streams
whose framing the container legitimately rewrote. For those streams it skips the
keyframe-identity and payload-byte checks, because an MP4 muxer converting H.264
from Annex-B to length-prefixed AVCC changes both by design.

The duration check does not sit inside that exemption. It runs unconditionally.

That asymmetry is real. `tests/test_worker_packet_remux_real_ffmpeg.py` documents
in a strict xfail that a production RTSP source delivers Annex-B while the PyAV
mux template capsule carries AVCC, so every live video stream lands in the
exemption; that reproduces exactly, with a 38-byte Annex-B descriptor extradata
against a 46-byte AVCC capsule extradata.

**The asymmetry is not, however, why production fails.** Measurement disproved
that: muxing contiguous packets through the capsule-normalized path preserves
every duration exactly, across CFR, B-frames, VFR and audio-plus-video sources.
The real trigger is a *gap* in the packet sequence. MP4 derives per-sample
durations in its `stts` table from successive timestamps, so a dropped packet
makes the preceding packet's muxed duration expand to the distance to the next
survivor -- doubling for one lost packet, quintupling for three. The live worker
logged 423 `source packet ring dropped packet` warnings in 48 hours.

So the duration check is not misfiring on container metadata. It is correctly
detecting that the clip window has holes. Moving it inside the exemption would
silence a true integrity signal and publish clips with silently missing video.

See `test_a_dropped_packet_is_what_actually_changes_duration` below, which pins
the real mechanism. The remaining tests pin the exemption's shape.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

import worker.adapters.encode.packet_remuxer as packet_remuxer
from worker.types.source_packet import (
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
    StreamEpoch,
)

_VIDEO_STREAM = 0


def _config() -> SourceStreamConfiguration:
    return SourceStreamConfiguration.from_streams(
        [SourceStreamDescriptor(0, "video", "h264", "avc1", Fraction(1, 1000), b"avcc", 16, 16)],
        mux_template=b"detached-template",
    )


def _packet(ms: int, *, keyframe: bool = False) -> SourcePacket:
    return SourcePacket(
        StreamEpoch("boot-1", "camera-1", 3),
        _config(),
        0,
        ms,
        ms,
        40,
        keyframe,
        bytes([ms % 251 + 1]),
        ms,
    )


def _muxed_fact(packet: SourcePacket, *, translation: int = 0):
    return packet_remuxer._MuxedPacketFact(  # noqa: SLF001
        stream_index=packet.stream_index,
        pts=None if packet.pts is None else packet.pts + translation,
        dts=None if packet.dts is None else packet.dts + translation,
        duration=packet.duration,
        time_base=packet.stream.time_base,
        is_keyframe=packet.is_keyframe,
        payload=packet.payload,
    )


def _source_and_muxed(*, translation: int = -10):
    packets = (_packet(1000, keyframe=True), _packet(2000))
    muxed = tuple(_muxed_fact(packet, translation=translation) for packet in packets)
    return packets, muxed


def test_annexb_normalizer_rejects_unbounded_start_code_count() -> None:
    payload = b"\0\0\1e" * 4_097

    with pytest.raises(ValueError, match="NAL unit count"):
        packet_remuxer._annexb_to_length_prefixed(payload, 4)  # noqa: SLF001


def test_normalized_stream_rejects_unexplained_payload_rewrite() -> None:
    """Container normalization is not an exemption from AU byte verification."""
    packets, muxed = _source_and_muxed()
    rewritten = (replace(muxed[0], payload=b"avcc-length-prefixed"), muxed[1])

    with pytest.raises(ValueError, match="payload changed"):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            packets,
            rewritten,
            packets[0].configuration,
            container_normalized_streams={_VIDEO_STREAM},
        )


def test_normalized_stream_rejects_lost_keyframe_identity() -> None:
    """A container rewrite cannot make a known source keyframe optional."""
    packets, muxed = _source_and_muxed()
    reflagged = (replace(muxed[0], is_keyframe=False), muxed[1])

    with pytest.raises(ValueError, match="keyframe identity changed"):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            packets,
            reflagged,
            packets[0].configuration,
            container_normalized_streams={_VIDEO_STREAM},
        )


def test_a_recomputed_duration_no_longer_destroys_the_clip() -> None:
    """The asymmetry, resolved.

    This test used to assert the production failure and said so in its own
    docstring: "Every live clip dies here." It pinned the defect rather than a
    property worth keeping. MP4 derives durations from its stts table, so a
    packet duration can differ from the source's declared value with no data
    lost -- measured here at 2880/2970/3060 ticks against a nominal 3000 --
    and rejecting that wrote off 100% of this deployment's video evidence.

    A refinement must now survive. Stretching across a missing packet must
    still fail, which is pinned separately below.
    """
    packets, muxed = _source_and_muxed()
    recomputed = (replace(muxed[0], duration=muxed[0].duration + 1), muxed[1])

    translations, _ = packet_remuxer._verify_packet_facts(  # noqa: SLF001
        packets,
        recomputed,
        packets[0].configuration,
        container_normalized_streams={_VIDEO_STREAM},
    )

    assert translations


def test_a_normalized_stream_still_enforces_the_timing_invariants() -> None:
    """The exemption must not become a blanket bypass.

    Whatever repair is chosen for the duration check, these must keep failing:
    they are timeline integrity, not container framing.
    """
    packets, muxed = _source_and_muxed()
    kwargs = {"container_normalized_streams": {_VIDEO_STREAM}}

    drifted = (muxed[0], replace(muxed[1], pts=muxed[1].pts + 5, dts=muxed[1].dts + 5))
    with pytest.raises(ValueError, match="drift nonuniformly"):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            packets, drifted, packets[0].configuration, **kwargs
        )

    # Moving dts alone breaks the PTS-DTS composition offset, which is checked
    # before decode order; either way the timeline is still defended.
    skewed = (muxed[0], replace(muxed[1], dts=muxed[0].dts - 1))
    with pytest.raises(ValueError, match="composition offset changed"):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            packets, skewed, packets[0].configuration, **kwargs
        )

    dropped = (muxed[0],)
    with pytest.raises(ValueError, match="packet count changed"):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            packets, dropped, packets[0].configuration, **kwargs
        )


def test_a_dropped_packet_is_what_actually_changes_duration() -> None:
    """A gap in the packet sequence is the real production trigger.

    Reproduced against the live mechanism: MP4 derives per-sample durations from
    successive timestamps, so when the source packet ring drops a packet, the
    packet *before* the gap is muxed with a duration stretched to the next
    survivor. One lost packet doubles it; three quintuple it.

    This is why the duration check must NOT be moved into the normalization
    exemption. It is not reporting benign container metadata -- it is reporting
    that the clip window is missing video, which is precisely the loss ADR-0001
    exists to prevent.
    """
    packets = (_packet(1000, keyframe=True), _packet(2000), _packet(3000))
    muxed = tuple(_muxed_fact(packet, translation=-10) for packet in packets)

    # The packet at index 1 was dropped by the ring, so index 0's duration is
    # stretched across the hole. Its own declared duration never changed.
    stretched = (
        replace(muxed[0], duration=muxed[0].duration * 2),
        muxed[2],
    )
    surviving_sources = (packets[0], packets[2])

    with pytest.raises(ValueError, match="duration changed"):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            surviving_sources,
            stretched,
            packets[0].configuration,
            container_normalized_streams={_VIDEO_STREAM},
        )


def test_lease_backpressure_drops_are_counted_for_operators(caplog) -> None:
    """The condition that stalls clip finalization must be visible when it happens.

    A lease-pressure drop is the mechanism behind the live incident: select()
    pins the clip window, the ring cannot trim, and the arriving packet is
    discarded, leaving the still-recording window discontiguous. The counter for
    this already existed and was reported nowhere, so 1053 clips silently failed
    to finalize while the diagnostic sat unused.
    """
    import logging

    from worker.pipeline.output.evidence.packet_ring import PacketRingLimits, SourcePacketRing

    ring = SourcePacketRing(
        "camera-1",
        # Deliberately tiny: one packet's worth, so the second arrival is over
        # limit while the first is leased.
        PacketRingLimits(max_packets=2, max_bytes=64, max_duration_seconds=0.001),
    )
    assert ring.append(_packet(1000, keyframe=True))
    selection = ring.select(
        trigger_epoch=_packet(1000).epoch,
        trigger_pts=Fraction(1),
        pre_seconds=Fraction(10),
        post_seconds=Fraction(10),
    )
    assert selection.packets, "the window must be leased for backpressure to apply"

    with caplog.at_level(logging.WARNING):
        accepted = ring.append(_packet(2000))

    assert accepted is False, "an arriving packet must be dropped while the window is leased"
    assert ring.metrics.lease_backpressure_drops == 1
    assert "lease backpressure" in caplog.text
    assert "camera-1" in caplog.text


def test_the_lease_is_released_before_remux_so_arrivals_are_never_dropped() -> None:
    """The repair: hold packets, not the lease.

    ``SourcePacket`` is immutable and already materialized, so the ring lease
    buys nothing during remux -- but holding it pins the oldest entries, blocks
    eviction, and forces the ring to drop ARRIVING packets. Those drops hole the
    next clip window. Releasing first is independent of bitrate and of how long
    a remux takes, which is why no capacity tuning substitutes for it.
    """
    from worker.pipeline.output.evidence.packet_ring import PacketRingLimits, SourcePacketRing

    ring = SourcePacketRing(
        "camera-1",
        PacketRingLimits(max_packets=2, max_bytes=64, max_duration_seconds=0.001),
    )
    assert ring.append(_packet(1000, keyframe=True))

    selection = ring.select(
        trigger_epoch=_packet(1000).epoch,
        trigger_pts=Fraction(1),
        pre_seconds=Fraction(10),
        post_seconds=Fraction(10),
    )
    # What finalize() now does: detach the immutable packets, then release.
    detached = selection.packets
    selection.close()

    # A packet arriving during what would have been the remux window is accepted,
    # because nothing is leased and the ring is free to evict.
    assert ring.append(_packet(2000)) is True
    assert ring.metrics.lease_backpressure_drops == 0

    # And the detached tuple is still fully usable for the remux itself.
    assert len(detached) == 1
    assert detached[0].payload

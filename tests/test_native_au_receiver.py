from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import pytest

from worker.adapters.decode.native_au_receiver import NativeAuReceiver
from worker.types.source_packet import SourcePacket, StreamEpoch

_HEADER = struct.Struct("<4sIBBBBIQQqqqiiIIHHII")


# allow: MUTABLE_OK - synchronization fake records receiver effects.
@dataclass(slots=True)
class _Sink:
    packets: list[SourcePacket] = field(default_factory=list)
    epochs: list[StreamEpoch] = field(default_factory=list)
    accepted: threading.Event = field(default_factory=threading.Event)

    def register_camera(self, camera_id: str) -> None:
        assert camera_id == "camera-a"

    def roll_epoch(self, epoch: StreamEpoch) -> None:
        self.epochs.append(epoch)

    def append(self, packet: SourcePacket) -> bool:
        self.packets.append(packet)
        self.accepted.set()
        return True


def _frame(
    *,
    epoch: int,
    sequence: int,
    kind: int = 1,
    payload: bytes = b"\0\0\0\x02e\x80",
    generation: int = 3,
    width: int = 640,
    height: int = 360,
    time_base_denominator: int = 90_000,
    pts: int | None = None,
    framing: int = 2,
    codec_data: bytes = b"\x01d\0\x1f\xff",
    caps_suffix: bytes = b"",
) -> bytes:
    camera = b"camera-a"
    stream_format = b"avc" if framing == 2 else b"byte-stream"
    caps = b"video/x-h264,alignment=(string)au,stream-format=(string)" + stream_format + caps_suffix
    body = camera + caps + codec_data + payload
    return (
        _HEADER.pack(
            b"SAU1",
            len(body),
            kind,
            1,
            framing,
            1,
            generation,
            epoch,
            sequence,
            sequence * 3_000 if pts is None else pts,
            sequence * 3_000 if pts is None else pts,
            3_000,
            1,
            time_base_denominator,
            width,
            height,
            len(camera),
            len(caps),
            len(codec_data),
            len(payload),
        )
        + body
    )


def test_native_au_receiver_adapts_parser_facts_into_source_packet() -> None:
    # Given: a dedicated stream channel and the Python packet authority.
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    accepted: list[tuple[str, int, int, int]] = []
    receiver = NativeAuReceiver(
        parent,
        "boot-1",
        sink,
        lambda camera, reason: gaps.append((camera, reason)),
        accept_handler=lambda camera, pts, sequence, generation: accepted.append(
            (camera, pts, sequence, generation)
        ),
    )
    receiver.start()

    # When: one complete parser-aligned AU arrives.
    child.sendall(_frame(epoch=7, sequence=1))

    # Then: exact epoch/timeline/framing facts reach SourcePacket.
    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    packet = sink.packets[0]
    assert packet.epoch == StreamEpoch("boot-1", "camera-a", 7, 3)
    assert (packet.pts, packet.dts, packet.duration, packet.is_keyframe) == (
        3_000,
        3_000,
        3_000,
        True,
    )
    assert packet.configuration.streams[0].stream_format == "avc"
    assert packet.configuration.streams[0].nal_length_size == 4
    assert not gaps
    assert accepted == [("camera-a", 3_000, 1, 3)]


def test_a_sender_gap_marker_is_a_hole_not_a_rebuild() -> None:
    """The child's sender shedding units is the same event as this side shedding them.

    A marker used to request a source rebuild. On the live fleet the rebuild's
    opening burst congested the sender again, which emitted the next marker:
    five rebuilds a minute, each stalling the processing thread long enough
    to shed a second of every camera's units, and no clip with video. The
    marker admits nothing; the next unit's sequence jump marks the hole.
    """
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent, "boot-1", sink, lambda camera, reason: gaps.append((camera, reason))
    )
    receiver.start()
    try:
        child.sendall(_frame(epoch=7, sequence=1))
        # When: the bounded native sender declares it shed sequences 2-3.
        child.sendall(_frame(epoch=7, sequence=2, kind=2, payload=b""))
        child.sendall(_frame(epoch=7, sequence=4))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 2:
            time.sleep(0.01)
        time.sleep(0.2)
        # Then: the marker admitted nothing, nothing asked for a rebuild, and
        # the unit after the hole carries the mark.
        assert gaps == [], gaps
        assert [p.arrival_index for p in sink.packets] == [0, 3]
        assert sink.packets[-1].discontinuity == "sequence_gap:2->4"
    finally:
        receiver.close()
        child.close()
        parent.close()


def test_malformed_camera_au_does_not_kill_other_camera_drain() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gap_seen = threading.Event()
    gaps: list[tuple[str, str]] = []

    def gap(camera_id: str, category: str) -> None:
        gaps.append((camera_id, category))
        gap_seen.set()

    receiver = NativeAuReceiver(parent, "boot-1", sink, gap)
    receiver.start()
    child.sendall(_frame(epoch=7, sequence=1, width=0))
    assert gap_seen.wait(timeout=2.0)
    child.sendall(_frame(epoch=8, sequence=1))

    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    assert gaps == [("camera-a", "parser")]
    assert sink.packets[-1].epoch.stream_epoch == 8


def test_backward_epoch_is_dropped_without_recovery_but_higher_generation_readd_survives() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(parent, "boot-1", sink, lambda *gap: gaps.append(gap))
    receiver.start()
    child.sendall(_frame(epoch=2, sequence=1))
    assert sink.accepted.wait(timeout=2.0)
    sink.accepted.clear()
    child.sendall(_frame(epoch=1, sequence=1))
    child.sendall(_frame(epoch=1, sequence=1, generation=4))

    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    # CONTRACT CHANGE, deliberate and evidence-backed. This previously asserted
    # gaps == [("camera-a", "parser")]: a backward epoch asked for a source
    # rebuild. On the live 13-camera fleet that rule was self-sustaining --
    # access units already in flight when the epoch rolled arrive carrying the
    # superseded identity, each one requests a rebuild, each rebuild advances
    # the epoch and strands the next batch. Measured: 301 rebuilds in five
    # minutes with ZERO child-reported failures and no worker restart.
    #
    # A superseded epoch is now dropped the same way a retired generation
    # already was, a few lines above it in _accept. The unit is still rejected;
    # what changed is that rejection no longer triggers recovery.
    #
    # The other half of this test is unchanged and still load-bearing: a higher
    # generation re-add carrying a lower epoch must still be accepted.
    assert gaps == []
    assert sink.packets[-1].epoch == StreamEpoch("boot-1", "camera-a", 1, 4)


def test_timestamp_jump_requests_epoch_roll_without_appending_jump() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gap_seen = threading.Event()
    receiver = NativeAuReceiver(
        parent,
        "boot-1",
        sink,
        lambda _camera, _category: gap_seen.set(),
    )
    receiver.start()
    child.sendall(_frame(epoch=7, sequence=1))
    assert sink.accepted.wait(timeout=2.0)
    sink.accepted.clear()
    child.sendall(_frame(epoch=7, sequence=2, pts=900_000))

    assert gap_seen.wait(timeout=2.0)
    receiver.close()
    child.close()
    assert len(sink.packets) == 1


def test_caps_and_framing_changes_fail_closed_without_splicing() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gap_seen = threading.Event()
    gaps: list[tuple[str, str]] = []

    def gap(camera: str, category: str) -> None:
        gaps.append((camera, category))
        gap_seen.set()

    receiver = NativeAuReceiver(parent, "boot-1", sink, gap)
    receiver.start()
    child.sendall(_frame(epoch=7, sequence=1))
    assert sink.accepted.wait(timeout=2.0)
    sink.accepted.clear()
    child.sendall(_frame(epoch=7, sequence=2, caps_suffix=b",profile=(string)high"))
    assert gap_seen.wait(timeout=2.0)
    gap_seen.clear()
    child.sendall(
        _frame(
            epoch=8,
            sequence=1,
            framing=1,
            codec_data=b"",
            payload=b"\0\0\0\1\x65\x80",
        )
    )
    assert sink.accepted.wait(timeout=2.0)

    receiver.close()
    child.close()
    assert gaps == [("camera-a", "parser")]
    assert [packet.epoch.stream_epoch for packet in sink.packets] == [7, 8]


def test_short_avcc_codec_data_isolated_then_next_epoch_survives() -> None:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gap_seen = threading.Event()
    receiver = NativeAuReceiver(parent, "boot-1", sink, lambda _camera, _category: gap_seen.set())
    receiver.start()
    child.sendall(_frame(epoch=7, sequence=1, codec_data=b"\x01"))
    assert gap_seen.wait(timeout=2.0)
    child.sendall(_frame(epoch=8, sequence=1))

    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    assert sink.packets[-1].epoch.stream_epoch == 8


def test_in_flight_units_from_a_rolled_epoch_do_not_request_another_rebuild() -> None:
    """Regression for the self-sustaining rebuild storm (#424).

    Reporting a gap for every superseded-epoch unit made recovery feed itself:
    each rebuild advanced the epoch, stranding the units already in flight,
    which requested another rebuild. Measured on the live fleet at 301 rebuilds
    in five minutes with zero child-reported failures.
    """
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent,
        "boot-1",
        sink,
        lambda camera, category: gaps.append((camera, category)),
    )
    receiver.start()
    # Given: the receiver has adopted epoch 5
    child.sendall(_frame(epoch=5, sequence=1))
    assert sink.accepted.wait(timeout=2.0)
    sink.accepted.clear()
    accepted_before = len(sink.packets)
    # When: several units that were in flight during the roll arrive late
    for sequence in range(1, 4):
        child.sendall(_frame(epoch=4, sequence=sequence))
    child.sendall(_frame(epoch=5, sequence=2))
    assert sink.accepted.wait(timeout=2.0)
    receiver.close()
    child.close()
    # Then: none of them asked for recovery, and none of them entered the ring
    assert gaps == []
    assert len(sink.packets) == accepted_before + 1
    assert all(packet.epoch.stream_epoch == 5 for packet in sink.packets)


def test_a_rejected_ring_append_reports_a_named_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every gap path must pass a reason; a missed one becomes a TypeError storm.

    Adding the reason argument to _report_gap left two call sites unconverted.
    Both sat on paths the existing tests never exercised, so the suite stayed
    green while production raised TypeError inside the gap handler 432 times in
    four minutes -- each one caught by the same broad except that then reported
    another gap, feeding the exact loop this work is trying to remove.
    """
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    class _RejectingSink(_Sink):
        def append(self, packet: object) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
            super().append(packet)  # pyright: ignore[reportArgumentType]
            return False

    sink = _RejectingSink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent,
        "boot-1",
        sink,
        lambda camera, category: gaps.append((camera, category)),
    )
    receiver.start()
    # When: the ring refuses the packet
    child.sendall(_frame(epoch=1, sequence=1))
    deadline = time.monotonic() + 2.0
    while not gaps and time.monotonic() < deadline:
        time.sleep(0.02)
    receiver.close()
    child.close()
    # Then: a gap was requested, and it did not raise on the way there. The
    # reason lives only in the log, and a TypeError at the call site would be
    # swallowed by _process and reported as malformed_envelope instead.
    assert gaps, "ring rejection must request recovery"
    assert gaps[0][0] == "camera-a"
    messages = [record.getMessage() for record in caplog.records]
    assert any("reason=ring_append_rejected" in m for m in messages), messages
    assert not any("malformed_envelope" in m for m in messages), messages


def test_a_lost_opening_unit_no_longer_strands_the_receiver() -> None:
    """The receiver resynchronises on an epoch whose transition was lost.

    This test used to pin the opposite: the receiver saw epoch 2 begin at
    sequence 2, reported a discontinuity, refused to adopt it, and stayed
    stranded until a process restart. The remedy chosen then was sender-side
    only -- AuSender reserves the opening unit -- on the reasoning that the
    receiver's checks should not be relaxed (#424 storm).

    The live fleet showed that reservation is not enough: the loss also
    happens inside this process, and the earlier contract turned one lost unit
    into a permanent outage. The check is relaxed exactly for a strictly newer
    identity, which cannot be the #424 debris (superseded units are dropped
    before this point) and cannot splice epochs (the ring rolls first).
    """
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent, "boot-1", sink, lambda camera, reason: gaps.append((camera, reason))
    )
    receiver.start()
    try:
        child.sendall(_frame(epoch=1, sequence=1))
        child.sendall(_frame(epoch=1, sequence=2))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 2:
            time.sleep(0.01)
        assert len(sink.packets) == 2, f"epoch 1 baseline failed: {gaps}"

        # The opening unit of epoch 2 was lost; the first unit seen is sequence 2.
        child.sendall(_frame(epoch=2, sequence=2))
        child.sendall(_frame(epoch=2, sequence=3))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 4:
            time.sleep(0.01)

        assert [e.stream_epoch for e in sink.epochs] == [1, 2], gaps
        assert [p.epoch.stream_epoch for p in sink.packets] == [1, 1, 2, 2], gaps
        assert gaps == [], "adopting a newer identity must not request a rebuild"

        # A hole INSIDE the adopted epoch is marked on the next packet, not
        # turned into another rebuild.
        child.sendall(_frame(epoch=2, sequence=5))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 5:
            time.sleep(0.01)
        assert gaps == [], gaps
        assert sink.packets[-1].discontinuity == "sequence_gap:4->5"
    finally:
        receiver.close()
        child.close()
        parent.close()


def test_a_reserved_transition_is_adopted_with_no_receiver_change() -> None:
    """The remedy's whole point: adoption, not merely delivery.

    AuSender now reserves the unit that opens an epoch and drains it ahead of
    the gap as an ORDINARY access unit rather than a gap marker. That shape is
    what makes it effective: the receiver returns immediately on AuKind.GAP
    and never adopts the epoch a gap marker carries, so an earlier attempt
    that preserved the newest epoch in the gap slot delivered the transition
    to something that discards it. Delivery is not adoption, and this test
    asserts the difference.

    Because the receiver expects sequence 1 for a new (camera, generation,
    epoch) key, the reserved unit passes its existing checks unchanged. No
    receiver-side contract change is required, which is why this remedy does
    not reopen the regression caused by relaxing those checks.
    """
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent, "boot-1", sink, lambda camera, reason: gaps.append((camera, reason))
    )
    receiver.start()
    try:
        child.sendall(_frame(epoch=1, sequence=1))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 1:
            time.sleep(0.01)
        assert len(sink.packets) == 1, f"epoch 1 baseline failed: {gaps}"

        # The reserved transition arrives as an ordinary unit at sequence 1.
        child.sendall(_frame(epoch=2, sequence=1))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 2:
            time.sleep(0.01)

        adopted = [p.epoch.stream_epoch for p in sink.packets]
        assert 2 in adopted, (
            f"the receiver must ADOPT the epoch, not merely receive it; saw {adopted} gaps={gaps}"
        )
        assert not gaps, f"adoption must not require a rebuild first: {gaps}"

        # And the epoch continues normally from that baseline.
        child.sendall(_frame(epoch=2, sequence=2))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 3:
            time.sleep(0.01)
        assert [p.epoch.stream_epoch for p in sink.packets].count(2) == 2, (
            f"the adopted epoch did not continue: {gaps}"
        )
    finally:
        receiver.close()
        child.close()
        parent.close()


def test_a_newer_epoch_is_adopted_even_when_its_opening_unit_was_lost() -> None:
    """Regression for the residual #429 failure the reservation did not fix.

    Measured on the live fleet after the sender-side reservation shipped: a
    rebuild storm hit all 13 cameras within three seconds, the receiver's own
    drain queue overflowed (``transport_gap_marker``), and afterwards ten rings
    sat on ``active_epoch=1`` for half an hour while triggers carried epoch 2
    or 3. Every clip in that window published without video.

    The stuck state has two parts, and this test exercises both in the order
    the live log showed them:

    1. A gap was reported on epoch 1, so ``camera-a`` is already in the
       pending-gap set when the rebuild lands.
    2. The rebuilt epoch's sequence-1 unit never reaches ``_accept`` -- the
       reservation only protects the sender's queue, not this process's --
       so the first unit seen for epoch 2 is sequence 2.

    Before: the sequence check refuses it, the pending gap suppresses the
    report, and nothing ever happens again for this camera. After: a strictly
    newer identity is proof the rebuild happened, so the receiver adopts it
    from whatever unit arrives first instead of demanding a second rebuild
    that would trigger the same storm.
    """
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent, "boot-1", sink, lambda camera, reason: gaps.append((camera, reason))
    )
    receiver.start()
    try:
        child.sendall(_frame(epoch=1, sequence=1))
        child.sendall(_frame(epoch=1, sequence=2))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 2:
            time.sleep(0.01)
        assert len(sink.packets) == 2, f"epoch 1 baseline failed: {gaps}"

        # Step 1: a rebuild is outstanding for epoch 1 (a timestamp jump is
        # one of the conditions that still asks for one).
        child.sendall(_frame(epoch=1, sequence=3, pts=900_000))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not gaps:
            time.sleep(0.01)
        assert gaps == [("camera-a", "parser")], gaps

        # Step 2: the rebuilt epoch arrives without its opening unit.
        child.sendall(_frame(epoch=2, sequence=2))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 3:
            time.sleep(0.01)

        assert [e.stream_epoch for e in sink.epochs] == [1, 2], (
            "the ring must roll to the rebuilt epoch; it stayed on "
            f"{[e.stream_epoch for e in sink.epochs]} with gaps={gaps}"
        )
        assert sink.packets[-1].epoch.stream_epoch == 2
        assert len(gaps) == 1, f"adoption must not ask for yet another rebuild: {gaps}"

        # And the adopted epoch continues normally from that baseline.
        child.sendall(_frame(epoch=2, sequence=3))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 4:
            time.sleep(0.01)
        assert [p.epoch.stream_epoch for p in sink.packets] == [1, 1, 2, 2], gaps
        assert len(gaps) == 1, gaps
    finally:
        receiver.close()
        child.close()
        parent.close()


def test_units_lost_inside_an_epoch_mark_a_hole_instead_of_rebuilding() -> None:
    """Regression for the rebuild storm the first #429 fix set off on the live fleet.

    Every lost unit used to request a source rebuild, and the rebuild ran on
    the receiver's own processing thread. Each one stalled the drain queue,
    the stall shed more units, and every shed unit requested another rebuild:
    265 rebuilds a minute across 13 cameras, epochs in the sixties four
    minutes after boot, and not one clip with video. The stale per-camera
    suppression had been the only thing damping it.

    A hole inside an epoch costs the clip windows that cross it and nothing
    else: the next packet carries the mark, clip selection refuses to span
    it, and the stream keeps running. Duplicates and reordered units are
    dropped without comment.
    """
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent, "boot-1", sink, lambda camera, reason: gaps.append((camera, reason))
    )
    receiver.start()
    try:
        child.sendall(_frame(epoch=1, sequence=1))
        child.sendall(_frame(epoch=1, sequence=2))
        # sequences 3 and 4 are lost; 2 arrives again out of order
        child.sendall(_frame(epoch=1, sequence=5))
        child.sendall(_frame(epoch=1, sequence=2))
        child.sendall(_frame(epoch=1, sequence=6))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 4:
            time.sleep(0.01)
        time.sleep(0.2)
        assert gaps == [], f"a hole must not request a rebuild: {gaps}"
        assert [p.arrival_index for p in sink.packets] == [0, 1, 4, 5]
        assert [p.discontinuity for p in sink.packets] == [
            None,
            None,
            "sequence_gap:3->5",
            None,
        ]
        assert [e.stream_epoch for e in sink.epochs] == [1]
    finally:
        receiver.close()
        child.close()
        parent.close()


def test_a_hole_followed_by_a_dts_jump_is_still_a_hole_not_a_rebuild() -> None:
    """A stall that sheds seconds of units must not become thirteen rebuilds.

    On the live fleet every ~100 s stall shed >120 units per camera; the DTS
    jump across that hole tripped timestamp_discontinuity and rebuilt all 13
    sources at once. The jump on a contiguous sequence is a camera clock
    anomaly and still rebuilds; after a sequence gap it is the hole itself.
    """
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sink = _Sink()
    gaps: list[tuple[str, str]] = []
    receiver = NativeAuReceiver(
        parent, "boot-1", sink, lambda camera, reason: gaps.append((camera, reason))
    )
    receiver.start()
    try:
        child.sendall(_frame(epoch=1, sequence=1))
        child.sendall(_frame(epoch=1, sequence=2))
        # 300 units (20 s at 3000 ticks each) lost, then the stream resumes.
        child.sendall(_frame(epoch=1, sequence=303))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(sink.packets) < 3:
            time.sleep(0.01)
        time.sleep(0.2)
        assert gaps == [], gaps
        assert sink.packets[-1].discontinuity == "sequence_gap:3->303"
        # But a clock jump on a contiguous sequence still asks for a rebuild.
        child.sendall(_frame(epoch=1, sequence=304, pts=303 * 3_000 + 90_000 * 60))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not gaps:
            time.sleep(0.01)
        assert gaps == [("camera-a", "parser")], gaps
    finally:
        receiver.close()
        child.close()
        parent.close()

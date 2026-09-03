"""Pin the ``PyAvPreservingAdapter`` packet-tee seam that Wave 2 reuses.

These tests characterize the CURRENT behavior of the packet-preserving decode
session so the out-of-process decode refactor (plan todo 4) has a pinned
surface. Every assertion is on a real observable object -- real PyAV-demuxed
packet bytes, real ``SourcePacket`` values, real container-level mux templates
-- never on a mock call count.

Hermetic and CI-safe: the media is synthesized in-process by PyAV's bundled
libavcodec (no ``ffmpeg`` binary, no RTSP, no GPU), so this is deliberately
NOT marked ``real_stack``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import final

import av
import numpy as np
import pytest

from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.adapters.decode.pyav_preserving import (
    PyAvPreservingAdapter,
    PyAvPreservingSession,
)
from worker.interfaces.decode import DecodeAdapter, StreamIdentityDecodeSession
from worker.interfaces.source_packet import SourcePacketSink
from worker.pipeline.output.evidence.packet_ring import (
    PacketRingLimits,
    SourcePacketRing,
)
from worker.types.source_packet import SourcePacket

_GOP = 4
_KEYFRAME_INTERVAL_OPTIONS = {"g": str(_GOP)}


@final
class _RecordingSink:
    """Real sink capturing every ``SourcePacket`` the session tees out."""

    def __init__(self, *, accept: bool = True) -> None:
        self.packets: list[SourcePacket] = []
        self._accept = accept

    def append(self, packet: SourcePacket) -> bool:
        self.packets.append(packet)
        return self._accept


@dataclass(frozen=True, slots=True)
class _Media:
    path: Path
    payloads: tuple[bytes, ...]
    pts: tuple[int, ...]
    dts: tuple[int, ...]
    keyframes: tuple[bool, ...]
    extradata: bytes
    width: int
    height: int


def _encode(path: Path, *, width: int, height: int, frames: int) -> None:
    container = av.open(str(path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=30)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = _KEYFRAME_INTERVAL_OPTIONS
    for index in range(frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, index % 3] = 255
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _read_ground_truth(path: Path) -> _Media:
    container = av.open(str(path))
    try:
        video = container.streams.video[0]
        demuxed = [
            packet for packet in container.demux(video) if packet.dts is not None and bytes(packet)
        ]
        codec = video.codec_context
        return _Media(
            path=path,
            payloads=tuple(bytes(packet) for packet in demuxed),
            pts=tuple(packet.pts for packet in demuxed if packet.pts is not None),
            dts=tuple(packet.dts for packet in demuxed if packet.dts is not None),
            keyframes=tuple(packet.is_keyframe for packet in demuxed),
            extradata=bytes(codec.extradata or b""),
            width=codec.width,
            height=codec.height,
        )
    finally:
        container.close()


def _media(
    tmp_path: Path,
    name: str,
    *,
    width: int = 64,
    height: int = 48,
    frames: int = 12,
) -> _Media:
    path = tmp_path / name
    _encode(path, width=width, height=height, frames=frames)
    return _read_ground_truth(path)


def _open(media: _Media, sink: SourcePacketSink, *, epoch: int = 1) -> PyAvPreservingSession:
    session = PyAvPreservingAdapter(sink, decode_backend="cpu").open(
        CpuAvConfig(
            camera_id="camera-1",
            url=str(media.path),
            open_timeout_ms=2_000,
            read_timeout_ms=2_000,
        )
    )
    session.set_stream_identity("boot-1", epoch)
    assert session.wait_demux_complete(15)
    return session


def test_session_tees_byte_identical_source_packets_with_original_timestamps(
    tmp_path: Path,
) -> None:
    """SEAM A1: packet tee framing -- payload bytes, PTS/DTS, keyframe flags and
    arrival order reach the sink exactly as demuxed. Wave 2 must preserve this."""
    # Given
    media = _media(tmp_path, "tee.mp4")
    sink = _RecordingSink()

    # When
    session = _open(media, sink)
    session.close()

    # Then
    assert tuple(packet.payload for packet in sink.packets) == media.payloads
    assert tuple(packet.pts for packet in sink.packets) == media.pts
    assert tuple(packet.dts for packet in sink.packets) == media.dts
    assert tuple(packet.is_keyframe for packet in sink.packets) == media.keyframes
    assert tuple(packet.arrival_index for packet in sink.packets) == tuple(
        range(len(media.payloads))
    )
    assert {packet.discontinuity for packet in sink.packets} == {None}
    assert sink.packets[0].stream.extradata == media.extradata
    assert (sink.packets[0].stream.width, sink.packets[0].stream.height) == (
        media.width,
        media.height,
    )
    assert session.packet_drop_count == 0


def test_session_stamps_every_packet_with_the_assigned_stream_identity(
    tmp_path: Path,
) -> None:
    """SEAM A2: open/identity/close lifecycle -- identity is assigned exactly
    once, stamps every teed packet, and cannot be reassigned."""
    # Given
    media = _media(tmp_path, "identity.mp4")
    sink = _RecordingSink()

    # When
    session = _open(media, sink, epoch=7)

    # Then
    assert isinstance(session, StreamIdentityDecodeSession)
    identities = {
        (packet.epoch.worker_boot_id, packet.epoch.camera_id, packet.epoch.stream_epoch)
        for packet in sink.packets
    }
    assert identities == {("boot-1", "camera-1", 7)}
    with pytest.raises(RuntimeError, match="already assigned"):
        session.set_stream_identity("boot-2", 8)
    session.close()
    session.close()


def test_decoded_frames_carry_source_pts_and_the_sink_sees_the_same_stream(
    tmp_path: Path,
) -> None:
    """SEAM A3: the decoded-frame side of the session. Wave 2 moves decode out of
    process; the FramePacket fields asserted here are the contract it must keep
    emitting (camera_id, seq, dimensions, source PTS/DTS/time_base)."""
    # Given
    media = _media(tmp_path, "frames.mp4")
    sink = _RecordingSink()

    # When
    session = _open(media, sink)
    frame = session.read()
    session.close()

    # Then
    assert frame is not None
    assert frame.camera_id == "camera-1"
    assert (frame.width, frame.height) == (media.width, media.height)
    assert frame.seq == 0
    assert frame.frame.image.shape == (media.height, media.width, 3)
    assert frame.frame.image.dtype == np.uint8
    assert frame.worker_boot_id == "boot-1"
    assert frame.stream_epoch == 1
    assert frame.source_time_base is not None
    assert frame.source_pts is not None
    assert frame.pts == pytest.approx(float(frame.source_pts * frame.source_time_base))
    assert frame.source_time_base == sink.packets[0].stream.time_base
    frame.release()


def test_mux_template_is_a_real_replayable_container_header(tmp_path: Path) -> None:
    """SEAM A4: ``SourceStreamConfiguration.mux_template`` -- a real MP4 capsule
    built from the leading packets. Evidence remux depends on it; Wave 2 must
    keep producing it from the demux-only path."""
    # Given
    media = _media(tmp_path, "template.mp4")
    sink = _RecordingSink()

    # When
    session = _open(media, sink)
    session.close()

    # Then
    configuration = sink.packets[0].configuration
    assert configuration.mux_template
    assert all(
        packet.configuration.configuration_id == configuration.configuration_id
        for packet in sink.packets
    )
    capsule = av.open(io.BytesIO(configuration.mux_template))
    try:
        template_video = capsule.streams.video[0]
        assert template_video.codec_context.name == "h264"
        assert (template_video.codec_context.width, template_video.codec_context.height) == (
            media.width,
            media.height,
        )
        assert bytes(template_video.codec_context.extradata or b"") == media.extradata
    finally:
        capsule.close()


def test_sink_rejection_is_counted_and_never_stops_the_demux_tee(tmp_path: Path) -> None:
    """SEAM A5: backpressure contract -- a sink returning False increments
    ``packet_drop_count`` and the session keeps demuxing (no raise, no stall)."""
    # Given
    media = _media(tmp_path, "backpressure.mp4")
    sink = _RecordingSink(accept=False)

    # When
    session = _open(media, sink)
    session.close()

    # Then
    assert len(sink.packets) == len(media.payloads)
    assert session.packet_drop_count == len(media.payloads)


def test_multiple_video_streams_are_refused_at_open(tmp_path: Path) -> None:
    """SEAM A6: fail-closed on ambiguous sources. One video stream per session."""
    # Given
    path = tmp_path / "two-video.mp4"
    container = av.open(str(path), mode="w", format="mp4")
    streams = []
    for _ in range(2):
        stream = container.add_stream("libx264", rate=30)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        streams.append(stream)
    for index in range(4):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, index % 3] = 255
        for stream in streams:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
    for stream in streams:
        for packet in stream.encode():
            container.mux(packet)
    container.close()
    adapter = PyAvPreservingAdapter(_RecordingSink(), decode_backend="cpu")

    # When / Then
    assert isinstance(adapter, DecodeAdapter)
    with pytest.raises(RuntimeError, match="exactly one video stream"):
        _ = adapter.open(
            CpuAvConfig("camera-1", str(path), open_timeout_ms=2_000, read_timeout_ms=2_000)
        )


def test_unsupported_backend_and_unopenable_source_fail_closed(tmp_path: Path) -> None:
    """SEAM A7: adapter construction contract -- backend token validation and
    open failure are typed RuntimeErrors, never a silent fallback (ADR-0002)."""
    # Given
    sink = _RecordingSink()
    missing = tmp_path / "absent.mp4"

    # When / Then
    with pytest.raises(RuntimeError, match="unsupported packet-preserving decode backend"):
        _ = PyAvPreservingAdapter(sink, decode_backend="quicksync").open(
            CpuAvConfig("camera-1", str(missing), open_timeout_ms=200, read_timeout_ms=200)
        )
    with pytest.raises(RuntimeError, match="packet-preserving source open failed"):
        _ = PyAvPreservingAdapter(sink, decode_backend="cpu").open(
            CpuAvConfig("camera-1", str(missing), open_timeout_ms=200, read_timeout_ms=200)
        )
    assert sink.packets == []


def test_teed_packets_land_in_the_ring_byte_identically(tmp_path: Path) -> None:
    """SEAM A8 + B1: the composed tee -- session into a real ``SourcePacketRing``
    yields payloads byte-identical to the demuxed source, in arrival order."""
    # Given
    media = _media(tmp_path, "ring.mp4")
    ring = SourcePacketRing("camera-1", PacketRingLimits(1_000, 8 * 1024 * 1024, 60.0))

    # When
    session = _open(media, ring)
    session.close()

    # Then
    snapshot = ring.snapshot()
    assert tuple(packet.payload for packet in snapshot) == media.payloads
    assert ring.metrics.accepted_packets == len(media.payloads)
    assert ring.metrics.dropped_packets == 0
    assert ring.total_bytes == sum(len(payload) for payload in media.payloads)
    assert min(Fraction(packet.presentation_time) for packet in snapshot) == Fraction(0)

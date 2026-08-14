from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import pytest

from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.adapters.decode.pyav_preserving import PyAvPreservingAdapter
from worker.adapters.encode.packet_remuxer import PyAvPacketRemuxer
from worker.pipeline.output.evidence.evidence_media import inspect_finalized_media
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.packet_ring import (
    PacketRingLimits,
    SourcePacketRing,
)
from worker.types.source_packet import (
    PacketSelectionError,
    PacketTruncationReason,
    SourcePacket,
    StreamEpoch,
)

pytestmark = pytest.mark.real_stack


def _pts(packet: av.Packet | SourcePacket) -> int:
    assert packet.pts is not None
    return packet.pts


def _dts(packet: av.Packet | SourcePacket) -> int:
    assert packet.dts is not None
    return packet.dts


def _time_base(value: Fraction | None) -> Fraction:
    assert value is not None
    return value


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        pytest.skip("ffmpeg is required for source packet remux QA")
    return path


def _generate_source(path: Path) -> None:
    subprocess.run(
        (
            _require_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30000/1001:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=3",
            "-c:v",
            "libx264",
            "-bf",
            "2",
            "-g",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-y",
            str(path),
        ),
        check=True,
        stdin=subprocess.DEVNULL,
    )


def _generate_vfr_no_audio_source(path: Path) -> None:
    subprocess.run(
        (
            _require_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30:duration=6",
            "-vf",
            "select='not(eq(mod(n,7),0))',setpts=PTS-507/(15360*TB)",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-bf",
            "2",
            "-g",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-y",
            str(path),
        ),
        check=True,
        stdin=subprocess.DEVNULL,
    )


def test_real_vfr_b_frames_allow_only_measured_uniform_mp4_timestamp_translation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vfr-source.mp4"
    output = tmp_path / "vfr-clip.mp4"
    _generate_vfr_no_audio_source(source)
    repository = PacketRingRepository(
        ("camera-1",),
        per_camera_limits=PacketRingLimits(10_000, 16 * 1024 * 1024, 10),
        global_max_bytes=16 * 1024 * 1024,
    )
    session = PyAvPreservingAdapter(repository, decode_backend="cpu").open(
        CpuAvConfig("camera-1", str(source), open_timeout_ms=2_000, read_timeout_ms=2_000)
    )
    session.set_stream_identity("boot-1", 1)
    assert session.wait_demux_complete(10)
    session.close()
    packets = repository.ring("camera-1").snapshot()
    second_keyframe = tuple(packet for packet in packets if packet.is_keyframe)[1]
    selected = tuple(
        replace(packet, dts=None if packet.dts is None else packet.dts - 1024)
        for packet in packets
        if second_keyframe.arrival_index
        <= packet.arrival_index
        < second_keyframe.arrival_index + 25
    )
    assert selected[0].stream.time_base == Fraction(1, 15_360)
    assert (selected[0].pts, selected[0].dts) == (17_920, 15_360)

    artifact = PyAvPacketRemuxer().remux(selected, selected[0].configuration, output)

    clip = av.open(str(output))
    try:
        actual = tuple(
            packet
            for packet in clip.demux(clip.streams.video[0])
            if packet.dts is not None and bytes(packet)
        )
        assert (actual[0].pts, actual[0].dts) == (17_910, 15_350)
        assert tuple(bytes(packet) for packet in actual) == tuple(
            packet.payload for packet in selected
        )
        assert tuple(packet.duration for packet in actual) == tuple(
            packet.duration for packet in selected
        )
        assert tuple(packet.is_keyframe for packet in actual) == tuple(
            packet.is_keyframe for packet in selected
        )
        assert tuple(_pts(packet) - _pts(actual[0]) for packet in actual) == tuple(
            _pts(packet) - _pts(selected[0]) for packet in selected
        )
        assert tuple(_dts(packet) - _dts(actual[0]) for packet in actual) == tuple(
            _dts(packet) - _dts(selected[0]) for packet in selected
        )
        assert tuple(_pts(packet) - _dts(packet) for packet in actual) == tuple(
            _pts(packet) - _dts(packet) for packet in selected
        )
    finally:
        clip.close()
    assert artifact.streams[0].packet_count == len(selected)
    assert artifact.streams[0].timestamp_translation_ticks == -10
    assert artifact.media_origin_pts_sec == float(
        min(packet.presentation_time for packet in selected)
    )


def test_real_source_packets_stream_copy_with_exact_stream_and_timestamp_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    _generate_source(source)
    repository = PacketRingRepository(
        ("camera-1",),
        per_camera_limits=PacketRingLimits(10_000, 16 * 1024 * 1024, 10),
        global_max_bytes=16 * 1024 * 1024,
    )
    session = PyAvPreservingAdapter(repository, decode_backend="cpu").open(
        CpuAvConfig("camera-1", str(source), open_timeout_ms=2_000, read_timeout_ms=2_000)
    )
    session.set_stream_identity("boot-1", 1)
    assert session.wait_demux_complete(10)
    decoded = session.read()
    assert decoded is not None
    assert decoded.worker_boot_id == "boot-1"
    assert decoded.stream_epoch == 1
    decoded.release()
    session.close()
    ring = repository.ring("camera-1")
    packets = ring.snapshot()
    assert packets
    video_times = [
        packet.presentation_time for packet in packets if packet.stream.media_type == "video"
    ]
    trigger = max(video_times)

    with ring.select(
        trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
        trigger_pts=trigger,
        pre_seconds=Fraction(2),
        post_seconds=Fraction(0),
    ) as selection:
        artifact = PyAvPacketRemuxer().remux(selection.packets, selection.configuration, output)
        expected = tuple(
            (
                packet.stream_index,
                packet.pts,
                packet.dts,
                packet.duration,
                packet.is_keyframe,
            )
            for packet in selection.packets
        )

    clip = av.open(str(output))
    try:
        streams = tuple(stream for stream in clip.streams if stream.type in {"video", "audio"})
        actual = tuple(
            (
                selection.configuration.streams[packet.stream.index].index,
                packet.pts,
                packet.dts,
                packet.duration,
                packet.is_keyframe,
            )
            for packet in clip.demux(streams)
            if packet.dts is not None and bytes(packet)
        )
        assert tuple(stream.codec_context.name for stream in streams) == ("h264", "aac")
        assert tuple(_time_base(stream.time_base) for stream in streams) == tuple(
            stream.time_base for stream in selection.configuration.streams
        )
        assert actual == expected
    finally:
        clip.close()
    assert artifact.remux_method == "pyav-packet-stream-copy"
    assert artifact.packet_count == len(expected)
    assert artifact.configuration_id == selection.configuration.configuration_id
    assert artifact.path.stat().st_size > 0
    facts = inspect_finalized_media(output)
    assert facts.video_codec == "h264"
    assert facts.audio_codec == "aac"


def test_malformed_source_fails_closed_without_partial_clip(tmp_path: Path) -> None:
    source = tmp_path / "malformed.bin"
    source.write_bytes(b"not-an-av-container")
    repository = PacketRingRepository(
        ("camera-1",),
        per_camera_limits=PacketRingLimits(100, 1024 * 1024, 10),
        global_max_bytes=1024 * 1024,
    )

    with pytest.raises(RuntimeError):
        PyAvPreservingAdapter(repository, decode_backend="cpu").open(
            CpuAvConfig("camera-1", str(source), open_timeout_ms=2_000, read_timeout_ms=2_000)
        )

    assert not (tmp_path / "clip.mp4").exists()
    assert repository.total_bytes == 0


def test_real_source_timestamp_discontinuity_fails_before_remux(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _generate_source(source)
    repository = PacketRingRepository(
        ("camera-1",),
        per_camera_limits=PacketRingLimits(10_000, 16 * 1024 * 1024, 10),
        global_max_bytes=16 * 1024 * 1024,
    )
    session = PyAvPreservingAdapter(repository, decode_backend="cpu").open(
        CpuAvConfig("camera-1", str(source), open_timeout_ms=2_000, read_timeout_ms=2_000)
    )
    session.set_stream_identity("boot-1", 1)
    assert session.wait_demux_complete(10)
    session.close()
    packets = repository.ring("camera-1").snapshot()
    trigger = max(
        packet.presentation_time for packet in packets if packet.stream.media_type == "video"
    )
    discontinuous = SourcePacketRing(
        "camera-1",
        PacketRingLimits(10_000, 16 * 1024 * 1024, 10),
    )
    marked_index = len(packets) // 2
    for index, packet in enumerate(packets):
        assert discontinuous.append(
            replace(packet, discontinuity="real-source-dts-jump")
            if index == marked_index
            else packet
        )

    with pytest.raises(PacketSelectionError) as raised:
        discontinuous.select(
            trigger_epoch=StreamEpoch("boot-1", "camera-1", 1),
            trigger_pts=trigger,
            pre_seconds=Fraction(2),
            post_seconds=Fraction(0),
        )

    assert raised.value.reason is PacketTruncationReason.TIMESTAMP_DISCONTINUITY
    assert not (tmp_path / "clip.mp4").exists()

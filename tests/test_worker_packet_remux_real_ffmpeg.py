from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

import av
import numpy as np
import pytest
from fanout_benchmark_harness import build_recorded_clip, recorded_stream_fanout

from contracts.frame import Frame
from worker.adapters.decode.cpu_av.models import CpuAvConfig
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig
from worker.adapters.decode.pyav_nvdec import NvdecPacketTeeSession
from worker.adapters.decode.pyav_preserving import PyAvPreservingAdapter
from worker.adapters.encode import packet_remuxer
from worker.adapters.encode.packet_remuxer import REMUX_METHOD, PyAvPacketRemuxer
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderConfig
from worker.pipeline.output.evidence.clip_recorder_services import default_services
from worker.pipeline.output.evidence.evidence_media import inspect_finalized_media
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.packet_ring import (
    PacketRingLimits,
    SourcePacketRing,
)
from worker.types import BusinessEvent, FramePacket
from worker.types.source_packet import (
    PacketSelectionError,
    PacketTruncationReason,
    SourcePacket,
    StreamEpoch,
)

pytestmark = pytest.mark.real_stack

_EVIDENCE_CAMERA = "evidence-cam-1"
_EVIDENCE_BOOT = "real-stack-evidence-boot"
_EVIDENCE_EPOCH = 1
# A canonical UUIDv4: the published manifest rejects anything else.
_EVIDENCE_EVENT_ID = "4f0a1c26-6f3c-4a3f-9f1b-2c8b3d0e5a71"


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


def _as_muxed(payload: bytes) -> bytes:
    """The bytes an MP4 sample description must carry for this payload.

    An RTSP source arrives in Annex-B framing; MP4 requires length-prefixed
    NAL units, so the remuxer reframes it. The media is unchanged -- only the
    unit separators are -- and these tests still compare every byte.
    """
    reframed = packet_remuxer._annexb_to_length_prefixed(payload, 4)  # noqa: SLF001
    return payload if reframed is None else reframed


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
            _as_muxed(packet.payload) for packet in selected
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
        # Stream identity, timestamps and durations must match exactly.
        assert tuple(row[:4] for row in actual) == tuple(row[:4] for row in expected)
        # Keyframes may only be gained, never lost. Once the NAL units are
        # framed the way an MP4 sample description requires, the container
        # parses their types instead of trusting the demuxer's packet flag --
        # and it finds IDRs the flag missed (measured here: a packet flagged
        # False whose payload began 00 00 01 45, NAL type 5). A clip that is
        # more seekable than its source metadata claimed is the good direction;
        # a lost keyframe is the one that breaks seeking.
        assert all(muxed[4] for muxed, source in zip(actual, expected, strict=True) if source[4])
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


# ---------------------------------------------------------------------------
# Plan todo 11: ADR-0001 evidence correctness under the new decode boundary.
#
# Wave 2 moved NVDEC decode into a child ``ffmpeg`` process behind
# ``NvdecPacketTeeSession``; the parent only demuxes and tees compressed
# packets into ``SourcePacketRing``. These tests run that real boundary --
# real RTSP through mediamtx, a real CUVID decoder child on the GPU, the real
# ``ClipRecorder``/``PacketClipRecordingCoordinator``/``PyAvPacketRemuxer``
# chain -- and probe the produced file with the real ``ffprobe`` binary,
# never a stub. The hermetic counterpart is
# ``tests/test_evidence_decode_boundary.py``.
# ---------------------------------------------------------------------------


class _ProbeStream(NamedTuple):
    codec_name: str
    time_base: str
    width: int
    height: int
    avg_frame_rate: str
    r_frame_rate: str
    nb_packets: int
    pix_fmt: str


class _DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def _idle_disk(_path: Path) -> _DiskUsage:
    """Keep retention pressure out of a test about evidence preservation."""
    return _DiskUsage(total=100, used=10, free=90)


def _require_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        pytest.skip("ffprobe is required to verify produced evidence clips")
    return path


def _ffprobe_json(path: Path, *args: str) -> dict[str, Any]:
    completed = subprocess.run(
        (
            _require_ffprobe(),
            "-v",
            "error",
            "-of",
            "json",
            *args,
            str(path),
        ),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    document: dict[str, Any] = json.loads(completed.stdout)
    return document


def _probe_video_stream(path: Path) -> _ProbeStream:
    stream = _ffprobe_json(
        path,
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=codec_name,time_base,width,height,avg_frame_rate,"
        "r_frame_rate,nb_read_packets,pix_fmt",
    )["streams"][0]
    return _ProbeStream(
        codec_name=stream["codec_name"],
        time_base=stream["time_base"],
        width=int(stream["width"]),
        height=int(stream["height"]),
        avg_frame_rate=stream["avg_frame_rate"],
        r_frame_rate=stream["r_frame_rate"],
        nb_packets=int(stream["nb_read_packets"]),
        pix_fmt=stream["pix_fmt"],
    )


def _probe_packet_timestamps(path: Path) -> tuple[tuple[int, int], ...]:
    packets = _ffprobe_json(
        path,
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts,dts",
    )["packets"]
    return tuple((int(packet["pts"]), int(packet["dts"])) for packet in packets)


def _wait_for(predicate: Callable[[], bool], *, timeout: float, what: str) -> None:
    """Bounded wait on an observable state change (never a fixed sleep)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError(f"timed out after {timeout}s waiting for {what}")


def _evidence_event(time_sec: float) -> BusinessEvent:
    return BusinessEvent(
        domain="fall",
        event_type="fall.detected",
        identity=_EVIDENCE_EVENT_ID,
        camera_id=_EVIDENCE_CAMERA,
        facility_id="evidence-facility",
        time_sec=time_sec,
        probability=0.91,
    )


def _open_tee_session(
    url: str,
    repository: PacketRingRepository,
) -> NvdecPacketTeeSession:
    """Open the production nvdec adapter: demux here, decode in a child ffmpeg."""
    session = PyAvPreservingAdapter(repository, decode_backend="nvdec").open(
        NvdecCuvidConfig(
            camera_id=_EVIDENCE_CAMERA,
            url=url,
            open_timeout_ms=10_000,
            read_timeout_ms=5_000,
        )
    )
    assert isinstance(session, NvdecPacketTeeSession)
    session.set_stream_identity(_EVIDENCE_BOOT, _EVIDENCE_EPOCH)
    return session


def _video_packets(repository: PacketRingRepository) -> tuple[SourcePacket, ...]:
    return tuple(
        packet
        for packet in repository.ring(_EVIDENCE_CAMERA).snapshot()
        if packet.stream.media_type == "video"
    )


def _secure_store_dir(tmp_path: Path, name: str) -> Path:
    """A 0700 clip store.

    The worktree's own directory mode is looser than the clip store's security
    contract expects, and the store's path checks refuse a group/other-writable
    ancestor. Owning a private directory here keeps the production check
    untouched while making the test independent of the worktree's mode.
    """
    store_dir = tmp_path / name
    store_dir.mkdir(mode=0o700, parents=True)
    return store_dir


class _ProducedClip(NamedTuple):
    clip_path: Path
    clip_source: Path
    manifest: dict[str, Any]
    ring_packets: tuple[SourcePacket, ...]
    frame_geometry: tuple[int, int]


def _produce_real_event_clip(tmp_path: Path) -> _ProducedClip:
    """Drive one real event->clip through the whole production boundary.

    mediamtx serves the stream, ffmpeg/CUVID decodes it in a child process,
    the parent only demuxes and tees, and the clip is published by the
    production ``ClipRecorder`` -- no stubs anywhere on the path.
    """
    _require_ffmpeg()
    _require_ffprobe()
    clip_source = build_recorded_clip(tmp_path / "evidence-source.mp4")
    store_dir = _secure_store_dir(tmp_path, "clip-store")
    config = ClipRecorderConfig(
        store_dir=store_dir,
        pre_event_seconds=2.0,
        post_event_seconds=1.0,
        # The ring's duration budget is pre+post+grace. The recorded stream's
        # GOP is one second, so the budget must exceed the pre-window by more
        # than a GOP or the oldest surviving keyframe lands inside the window
        # and the clip is (correctly) reported truncated.
        finalize_grace_seconds=4.0,
    )
    repository = PacketRingRepository(
        (_EVIDENCE_CAMERA,),
        per_camera_limits=PacketRingLimits(
            config.packet_ring_max_packets,
            config.packet_ring_max_bytes_per_camera,
            config.pre_event_seconds + config.post_event_seconds + config.finalize_grace_seconds,
        ),
        global_max_bytes=config.packet_ring_global_max_bytes,
    )
    recorder = ClipRecorder(
        config,
        services=default_services(config, repository),
        disk_usage_provider=_idle_disk,
        is_clip_held=lambda _clip_id: False,
    )

    with recorded_stream_fanout(stream_count=1, clip=clip_source, tmp_path=tmp_path) as (
        _server,
        urls,
    ):
        session = _open_tee_session(urls[0], repository)
        recorder.start()
        try:
            # When: a real decoded frame from the decoder child -- one far
            # enough into the stream that the full pre-event window exists
            # behind it -- triggers the event, and the post window then fills.
            stream_start = None
            deadline = time.monotonic() + 60.0
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "no decoded frame arrived with a full pre-event window behind it"
                    )
                decoded = session.read()
                if decoded is None:
                    continue
                assert decoded.source_pts is not None
                assert decoded.source_time_base is not None
                source_pts = decoded.source_pts
                source_time_base = decoded.source_time_base
                trigger_pts = float(source_pts * source_time_base)
                frame_width, frame_height = decoded.width, decoded.height
                decoded.release()
                packets = _video_packets(repository)
                if not packets:
                    continue
                if stream_start is None:
                    stream_start = float(min(packet.presentation_time for packet in packets))
                keyframes_before_window = tuple(
                    packet
                    for packet in packets
                    if packet.is_keyframe
                    and float(packet.presentation_time) <= trigger_pts - config.pre_event_seconds
                )
                if (
                    trigger_pts - stream_start >= config.pre_event_seconds + 0.5
                    and keyframes_before_window
                ):
                    break
            _wait_for(
                lambda: (
                    bool(_video_packets(repository))
                    and float(
                        max(packet.presentation_time for packet in _video_packets(repository))
                    )
                    >= trigger_pts + config.post_event_seconds + 0.5
                ),
                timeout=60.0,
                what="the packet ring to cover the post-event window",
            )
            trigger = FramePacket(
                camera_id=_EVIDENCE_CAMERA,
                frame=Frame(
                    index=0,
                    time_sec=trigger_pts,
                    image=np.zeros((frame_height, frame_width, 3), dtype=np.uint8),
                ),
                pts=trigger_pts,
                seq=0,
                width=frame_width,
                height=frame_height,
                decode_time_ms=0.0,
                worker_boot_id=_EVIDENCE_BOOT,
                stream_epoch=_EVIDENCE_EPOCH,
                source_pts=source_pts,
                source_time_base=source_time_base,
            )
            # The ring is a bounded live window: it keeps appending while the
            # clip is cut and evicts its own head as it goes. Merge a snapshot
            # from each side of finalize, keyed by the tee's arrival index, so
            # the comparison set covers the whole selected interval regardless
            # of which packets the ring still held at either instant.
            observed: dict[int, SourcePacket] = {
                packet.arrival_index: packet for packet in _video_packets(repository)
            }
            clip_id = recorder.on_event(trigger, _evidence_event(trigger_pts))
            assert clip_id is not None
            assert recorder.flush(timeout=60.0)
            observed.update({packet.arrival_index: packet for packet in _video_packets(repository)})
            ring_packets = tuple(observed[index] for index in sorted(observed))
        finally:
            recorder.stop(timeout=60.0)
            session.close()
            repository.close()

    return _ProducedClip(
        clip_path=store_dir / "clips" / clip_id / "clip.mp4",
        clip_source=clip_source,
        manifest=json.loads(
            (store_dir / "clips" / clip_id / "manifest.json").read_text(encoding="utf-8")
        ),
        ring_packets=ring_packets,
        frame_geometry=(frame_width, frame_height),
    )


def test_real_nvdec_boundary_event_produces_a_stream_copied_primary_clip(
    tmp_path: Path,
) -> None:
    """Happy path: RTSP -> subprocess NVDEC tee -> event -> primary clip.

    ``ffprobe`` proves the published file is a stream copy of the source
    packets -- same codec, same time base, same PTS/DTS, same geometry and
    frame rate, byte-identical payloads.
    """
    # Given / When: one real event-driven clip off the production boundary.
    produced = _produce_real_event_clip(tmp_path)
    clip_path = produced.clip_path
    clip_source = produced.clip_source
    manifest = produced.manifest
    ring_packets = produced.ring_packets
    ring_time_base = ring_packets[0].stream.time_base
    ring_geometry = (ring_packets[0].stream.width, ring_packets[0].stream.height)
    ring_codec = ring_packets[0].stream.codec_name
    frame_width, frame_height = produced.frame_geometry

    # Then: the published clip is a byte-true stream copy of source packets.
    probe = _probe_video_stream(clip_path)
    source_probe = _probe_video_stream(clip_source)
    muxed_timestamps = _probe_packet_timestamps(clip_path)
    print(
        "EVIDENCE_CLIP_FFPROBE "
        f"clip={probe} source={source_probe} "
        f"first_packets={muxed_timestamps[:4]}",
        flush=True,
    )

    # No transcode: the muxed codec and pixel format are the source's, and the
    # codec matches what the live RTSP stream itself declared to the tee.
    assert probe.codec_name == source_probe.codec_name
    assert probe.codec_name == ring_codec
    assert probe.pix_fmt == source_probe.pix_fmt
    # No resize: geometry matches the source file, the live stream descriptor,
    # and the frame the decoder subprocess produced.
    assert (probe.width, probe.height) == (source_probe.width, source_probe.height)
    assert (probe.width, probe.height) == ring_geometry
    assert (probe.width, probe.height) == (frame_width, frame_height)
    # No fps normalization.
    assert probe.r_frame_rate == source_probe.r_frame_rate
    # Original time base: the live RTSP stream's own 90kHz RTP clock, carried
    # through unchanged. (The recorded file on disk is 1/15360; RTP re-times it
    # on the wire, and it is the WIRE stream this boundary must preserve.)
    assert probe.time_base == f"{ring_time_base.numerator}/{ring_time_base.denominator}"
    # Original timestamps and byte-identical payloads, for the contiguous run
    # the selection covered. MP4 muxing may shift the whole timeline by ONE
    # exact uniform tick offset (the source PTS does not start at zero); ADR-
    # 0001 permits only that, it must be identical for every packet, and the
    # manifest must declare it. Nothing else about the timestamps may move.
    translation_ticks = manifest["source_media"]["streams"][0]["timestamp_translation_ticks"]
    ring_timestamps = tuple(
        (packet.pts, packet.dts) for packet in ring_packets if packet.pts is not None
    )
    untranslated = tuple(
        (pts - translation_ticks, dts - translation_ticks) for pts, dts in muxed_timestamps
    )
    assert muxed_timestamps
    assert len(muxed_timestamps) == probe.nb_packets
    start = ring_timestamps.index(untranslated[0])
    assert untranslated == ring_timestamps[start : start + len(untranslated)]
    # Deltas -- the actual timeline -- survive with zero drift, which also
    # pins the translation as uniform rather than per-packet re-timing.
    assert tuple(pts - muxed_timestamps[0][0] for pts, _ in muxed_timestamps) == tuple(
        pts - ring_timestamps[start][0]
        for pts, _ in ring_timestamps[start : start + len(untranslated)]
    )
    muxed = av.open(str(clip_path))
    try:
        muxed_payloads = tuple(
            bytes(packet)
            for packet in muxed.demux(muxed.streams.video[0])
            if packet.dts is not None and bytes(packet)
        )
    finally:
        muxed.close()
    ring_payloads = tuple(_as_muxed(packet.payload) for packet in ring_packets)
    assert muxed_payloads == ring_payloads[start : start + len(muxed_payloads)]

    # And the manifest tells the same story.
    assert manifest["video_available"] is True
    assert manifest["encoder"] == "source-packet-remux"
    assert manifest["codec"] == source_probe.codec_name
    assert manifest["source_media"]["remux_method"] == REMUX_METHOD
    assert manifest["source_media"]["packet_count"] == len(muxed_payloads)
    assert manifest["source_media"]["streams"][0]["codec_name"] == source_probe.codec_name
    assert manifest["source_media"]["streams"][0]["time_base"] == probe.time_base
    # The declared shift is the one actually applied, and it is sub-millisecond
    # -- a mux-time origin offset, not a re-timing of the evidence.
    assert Fraction(manifest["source_media"]["timestamp_translation_seconds"]) == (
        translation_ticks * ring_time_base
    )
    assert abs(translation_ticks * ring_time_base) < Fraction(1, 1000)
    assert manifest["truncation_reasons"] == []
    assert manifest["event_ref"] == _EVIDENCE_EVENT_ID
    facts = inspect_finalized_media(clip_path)
    assert facts.video_codec == source_probe.codec_name
    assert facts.size_bytes == clip_path.stat().st_size


def test_real_nvdec_boundary_clip_decodes_every_packet_it_carries(
    tmp_path: Path,
) -> None:
    """The clip an operator downloads must decode to real frames.

    Stream-copy fidelity is necessary but not sufficient, so this asserts the
    playback property directly rather than inferring it from byte equality.
    """
    # Given / When: one real event-driven clip off the production boundary.
    produced = _produce_real_event_clip(tmp_path)

    # Then: a decoder gets as many frames out as there are packets in.
    decoded_frames = int(
        _ffprobe_json(
            produced.clip_path,
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
        )["streams"][0]["nb_read_frames"]
    )
    packet_count = _probe_video_stream(produced.clip_path).nb_packets
    print(
        f"EVIDENCE_CLIP_DECODABILITY packets={packet_count} decoded_frames={decoded_frames}",
        flush=True,
    )
    assert decoded_frames == packet_count


def test_real_nvdec_boundary_clip_preserves_keyframe_identity(tmp_path: Path) -> None:
    """Keyframes in the selected window must stay keyframes in the clip."""
    # Given / When: one real event-driven clip off the production boundary.
    produced = _produce_real_event_clip(tmp_path)

    # Then: the clip flags at least as many keyframes as it must to be seekable.
    flags = _ffprobe_json(
        produced.clip_path,
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=flags",
    )["packets"]
    muxed_keyframes = sum(1 for packet in flags if "K" in packet["flags"])
    print(
        f"EVIDENCE_CLIP_KEYFRAMES muxed_keyframes={muxed_keyframes} packets={len(flags)}",
        flush=True,
    )
    assert muxed_keyframes > 0


def test_real_nvdec_boundary_clip_owes_nothing_to_the_decoded_frame_lane(
    tmp_path: Path,
) -> None:
    """The tee feeds the ring at demux, upstream of any inference-side drop.

    One decoded frame is taken to obtain the trigger identity, then the
    decoded-frame lane is abandoned entirely -- the decoder child's stdout is
    never drained again, which is the worst case the inference path can
    impose. The packet ring must keep growing and the clip must still be a
    complete stream copy.
    """
    # Given: a live tee session and a baseline packet count.
    _require_ffmpeg()
    _require_ffprobe()
    clip_source = build_recorded_clip(tmp_path / "evidence-source.mp4")
    repository = PacketRingRepository(
        (_EVIDENCE_CAMERA,),
        per_camera_limits=PacketRingLimits(20_000, 64 * 1024 * 1024, 30.0),
        global_max_bytes=64 * 1024 * 1024,
    )

    with recorded_stream_fanout(stream_count=1, clip=clip_source, tmp_path=tmp_path) as (
        _server,
        urls,
    ):
        session = _open_tee_session(urls[0], repository)
        try:
            decoded = session.read()
            assert decoded is not None
            assert decoded.source_pts is not None
            assert decoded.source_time_base is not None
            decoded.release()
            # From here on the decoder child's stdout is never drained again.
            baseline_packets = _video_packets(repository)
            baseline = len(baseline_packets)
            baseline_latest = max(packet.presentation_time for packet in baseline_packets)

            # When: nothing ever reads a decoded frame again. The wait is
            # anchored on the ring's OWN head at this moment, so it can only
            # finish after 3 further source-seconds actually arrived.
            _wait_for(
                lambda: (
                    bool(_video_packets(repository))
                    and max(packet.presentation_time for packet in _video_packets(repository))
                    >= baseline_latest + 3
                ),
                timeout=60.0,
                what="the abandoned-frame-lane ring to advance 3 source seconds",
            )
            packets = _video_packets(repository)
            grown = len(packets)
            latest = max(packet.presentation_time for packet in packets)

            # Then: the ring advanced anyway, and a full clip still selects.
            with repository.ring(_EVIDENCE_CAMERA).select(
                trigger_epoch=StreamEpoch(_EVIDENCE_BOOT, _EVIDENCE_CAMERA, _EVIDENCE_EPOCH),
                trigger_pts=baseline_latest + 2,
                pre_seconds=Fraction(1),
                post_seconds=Fraction(1),
            ) as selection:
                artifact = PyAvPacketRemuxer().remux(
                    selection.packets,
                    selection.configuration,
                    tmp_path / "abandoned-lane-clip.mp4",
                )
                selected_payloads = tuple(_as_muxed(packet.payload) for packet in selection.packets)
                truncations = selection.truncations
        finally:
            session.close()
            repository.close()

    print(
        "ABANDONED_FRAME_LANE_RING_GROWTH "
        f"baseline_packets={baseline} grown_packets={grown} "
        f"latest_presentation_time={float(latest):.3f} "
        f"clip_packets={artifact.packet_count}",
        flush=True,
    )
    assert grown > baseline
    assert truncations == ()
    assert artifact.remux_method == REMUX_METHOD
    probe = _probe_video_stream(artifact.path)
    source_probe = _probe_video_stream(clip_source)
    assert probe.codec_name == source_probe.codec_name
    assert (probe.width, probe.height) == (source_probe.width, source_probe.height)
    assert probe.r_frame_rate == source_probe.r_frame_rate
    muxed = av.open(str(artifact.path))
    try:
        muxed_payloads = tuple(
            bytes(packet)
            for packet in muxed.demux(muxed.streams.video[0])
            if packet.dts is not None and bytes(packet)
        )
    finally:
        muxed.close()
    assert muxed_payloads == selected_payloads


def test_real_nvdec_boundary_undersized_ring_reports_its_truncation(
    tmp_path: Path,
) -> None:
    """Failure path: a ring too small for the pre-window says so out loud.

    Forcing eviction past the requested pre-window must surface
    ``HISTORY_UNAVAILABLE`` (and the ring's eviction counters), never a
    silently shortened clip.
    """
    # Given: a deliberately undersized ring on the real decode boundary.
    _require_ffmpeg()
    clip_source = build_recorded_clip(tmp_path / "evidence-source.mp4")
    repository = PacketRingRepository(
        (_EVIDENCE_CAMERA,),
        per_camera_limits=PacketRingLimits(40, 64 * 1024 * 1024, 30.0),
        global_max_bytes=64 * 1024 * 1024,
    )

    with recorded_stream_fanout(stream_count=1, clip=clip_source, tmp_path=tmp_path) as (
        _server,
        urls,
    ):
        session = _open_tee_session(urls[0], repository)
        ring = repository.ring(_EVIDENCE_CAMERA)
        try:
            # When: the ring has evicted well past a 10s pre-window.
            _wait_for(
                lambda: ring.metrics.evicted_packets > 60,
                timeout=60.0,
                what="the undersized ring to evict past its pre-window",
            )
            packets = _video_packets(repository)
            trigger_pts = max(packet.presentation_time for packet in packets)
            with ring.select(
                trigger_epoch=StreamEpoch(_EVIDENCE_BOOT, _EVIDENCE_CAMERA, _EVIDENCE_EPOCH),
                trigger_pts=trigger_pts,
                pre_seconds=Fraction(10),
                post_seconds=Fraction(0),
            ) as selection:
                truncations = selection.truncations
                selected_span = selection.selected_end - selection.selected_start
                requested_span = trigger_pts - selection.requested_start
            evicted_packets = ring.metrics.evicted_packets
            evicted_bytes = ring.metrics.evicted_bytes
            accepted_packets = ring.metrics.accepted_packets
        finally:
            session.close()
            repository.close()

    print(
        "UNDERSIZED_RING_TRUNCATION "
        f"accepted_packets={accepted_packets} evicted_packets={evicted_packets} "
        f"evicted_bytes={evicted_bytes} "
        f"requested_span_s={float(requested_span):.3f} "
        f"selected_span_s={float(selected_span):.3f} "
        f"truncations={[reason.value for reason in truncations]}",
        flush=True,
    )
    assert PacketTruncationReason.HISTORY_UNAVAILABLE in truncations
    assert selected_span < requested_span
    assert evicted_packets > 0
    assert evicted_bytes > 0

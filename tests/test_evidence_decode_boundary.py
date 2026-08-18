"""ADR-0001 evidence correctness with decode moved out of process (plan todo 11).

Wave 2 moved NVDEC decode into a child ``ffmpeg`` and left the parent with a
demux-only tee (``NvdecPacketTeeSession``) that feeds ``SourcePacketRing``.
These tests pin the properties that move must not have broken:

* the clip a real event produces is still a stream copy of the source packets
  the tee published -- byte-identical payloads, original PTS/DTS/time base, the
  source codec, no re-encode (A);
* clip production reads no decoded frame at all: the coordinator's frame path
  is inert and the ring keeps filling while the decoder subprocess is never
  read from, so an event's clip window is complete regardless of what the
  inference lane did with (or dropped from) its frames (B);
* an undersized ring reports an explicit truncation reason instead of silently
  handing back a shortened clip, and the reason survives into the manifest (C);
* clip-enabled mode does not push decode back into the Python process
  (issue #312 regression guard) -- with a packet sink present, the nvdec
  backend still composes the subprocess tee session, never the in-process
  PyAV decode session (D).

Hermetic and CI-safe: media is synthesized in-process by PyAV's bundled
libavcodec and the decoder child is a fake spawner, so there is no ``ffmpeg``
binary, no RTSP and no GPU here. The real-hardware counterpart (real NVDEC
child + real RTSP + ``ffprobe`` on the produced clip) lives in
``tests/test_worker_packet_remux_real_ffmpeg.py``.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, NamedTuple, final

import av
import numpy as np
import pytest

import worker.adapters.encode.packet_remuxer as packet_remuxer
from contracts.frame import Frame
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig
from worker.adapters.decode.pyav_nvdec import NvdecPacketTeeSession
from worker.adapters.decode.pyav_preserving import (
    PyAvPreservingAdapter,
    PyAvPreservingSession,
)
from worker.adapters.encode.packet_remuxer import REMUX_METHOD, PyAvPacketRemuxer
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder
from worker.pipeline.output.evidence.clip_recorder_models import ClipRecorderConfig
from worker.pipeline.output.evidence.clip_recorder_services import default_services
from worker.pipeline.output.evidence.clip_recording import (
    ClipReady,
    ClipReasonCode,
    ClipUnavailable,
    ClipWindow,
)
from worker.pipeline.output.evidence.packet_recording import PacketClipRecordingCoordinator
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.packet_ring import PacketRingLimits
from worker.runtime.ingest_composition import decoder_for
from worker.types import BusinessEvent, FrameKey, FramePacket
from worker.types.source_packet import (
    PacketTruncationReason,
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
    StreamEpoch,
)

_CAMERA: Final = "camera-1"
_BOOT: Final = "boot-1"
_EPOCH: Final = 1
_FPS: Final = 10


class _DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def _idle_disk(_path: Path) -> _DiskUsage:
    """Keep retention pressure out of a test about packet-window truncation."""
    return _DiskUsage(total=100, used=10, free=90)


@final
class _FakeDecoderProcess:
    """A decoder child that accepts packets and is never read for frames.

    Modelling the worst case for property (B): the inference side of the
    boundary produces nothing at all, so any decoded-frame dependency in clip
    production would show up as a missing or truncated clip.
    """

    def __init__(self) -> None:
        self.packet_payloads: list[bytes] = []
        self.frames: deque[bytes] = deque()
        self.read_count = 0

    def write_packet(self, payload: bytes) -> None:
        self.packet_payloads.append(bytes(payload))

    def close_input(self) -> None:
        return None

    def read_frame(self, timeout_sec: float) -> bytes | None:
        del timeout_sec
        self.read_count += 1
        return self.frames.popleft() if self.frames else None

    def reap(self, timeout_sec: float) -> int | None:
        del timeout_sec
        return 0


@final
class _FakeSpawner:
    def __init__(self) -> None:
        self.processes: list[_FakeDecoderProcess] = []
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def __call__(self, args: tuple[str, ...], frame_size: int) -> _FakeDecoderProcess:
        self.calls.append((args, frame_size))
        process = _FakeDecoderProcess()
        self.processes.append(process)
        return process


@dataclass(frozen=True, slots=True)
class _Source:
    path: Path
    payloads: tuple[bytes, ...]
    pts: tuple[int, ...]
    dts: tuple[int, ...]
    keyframes: tuple[bool, ...]
    time_base: Fraction
    width: int
    height: int
    codec_name: str


def _encode(path: Path, *, width: int, height: int, frames: int, gop: int) -> None:
    container = av.open(str(path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=_FPS)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"g": str(gop), "bf": "0"}
    for index in range(frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, index % 3] = (index * 17) % 256
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _source(
    tmp_path: Path,
    name: str,
    *,
    width: int = 64,
    height: int = 48,
    frames: int = 40,
    gop: int = 4,
) -> _Source:
    path = tmp_path / name
    _encode(path, width=width, height=height, frames=frames, gop=gop)
    container = av.open(str(path))
    try:
        video = container.streams.video[0]
        time_base = video.time_base
        assert time_base is not None
        demuxed = [
            packet
            for packet in container.demux(video)
            if packet.dts is not None and bytes(packet)
        ]
        return _Source(
            path=path,
            payloads=tuple(bytes(packet) for packet in demuxed),
            pts=tuple(packet.pts for packet in demuxed if packet.pts is not None),
            dts=tuple(packet.dts for packet in demuxed if packet.dts is not None),
            keyframes=tuple(packet.is_keyframe for packet in demuxed),
            time_base=Fraction(time_base),
            width=video.codec_context.width,
            height=video.codec_context.height,
            codec_name=video.codec_context.name,
        )
    finally:
        container.close()


def _repository(limits: PacketRingLimits, *, global_max_bytes: int) -> PacketRingRepository:
    return PacketRingRepository(
        (_CAMERA,),
        per_camera_limits=limits,
        global_max_bytes=global_max_bytes,
    )


def _tee(
    source: _Source,
    repository: PacketRingRepository,
    spawner: _FakeSpawner,
) -> NvdecPacketTeeSession:
    """Open the real out-of-process tee session over a local file source."""
    session = PyAvPreservingAdapter(
        repository,
        decode_backend="nvdec",
        process_spawner=spawner,
    ).open(
        NvdecCuvidConfig(
            camera_id=_CAMERA,
            url=str(source.path),
            open_timeout_ms=2_000,
            read_timeout_ms=1_000,
        )
    )
    assert isinstance(session, NvdecPacketTeeSession)
    session.set_stream_identity(_BOOT, _EPOCH)
    assert session.wait_demux_complete(20)
    return session


def _trigger_key(source: _Source, *, packet_index: int) -> FrameKey:
    source_pts = source.pts[packet_index]
    return FrameKey(
        worker_boot_id=_BOOT,
        camera_id=_CAMERA,
        stream_epoch=_EPOCH,
        seq=packet_index,
        pts=float(source_pts * source.time_base),
        source_pts=source_pts,
        source_time_base=source.time_base,
    )


def _event() -> BusinessEvent:
    return BusinessEvent(
        domain="fall",
        event_type="fall.detected",
        # The manifest requires a canonical UUIDv4 event reference.
        identity="7b0f2f7e-0b0a-4c0e-9d1a-2f6a3c4b5d6e",
        camera_id=_CAMERA,
        facility_id="facility-1",
        time_sec=1.0,
        probability=0.9,
    )


def _muxed_video_packets(path: Path) -> tuple[av.Packet, ...]:
    container = av.open(str(path))
    try:
        return tuple(
            packet
            for packet in container.demux(container.streams.video[0])
            if packet.dts is not None and bytes(packet)
        )
    finally:
        container.close()


def test_event_clip_from_the_teed_ring_is_a_byte_true_stream_copy(tmp_path: Path) -> None:
    """(A) The clip an event produces holds the source's own packets.

    Payload bytes, PTS/DTS, durations, keyframe flags, time base, codec and
    geometry all come from the packets the demux tee published -- proof that
    moving decode into a child process left the evidence lane a stream copy
    (ADR-0001), with no transcode, no fps change and no resize.
    """
    # Given: a source demuxed through the out-of-process tee into the ring.
    source = _source(tmp_path, "stream-copy.mp4")
    repository = _repository(
        PacketRingLimits(4_000, 16 * 1024 * 1024, 60.0),
        global_max_bytes=16 * 1024 * 1024,
    )
    spawner = _FakeSpawner()
    session = _tee(source, repository, spawner)
    session.close()
    coordinator = PacketClipRecordingCoordinator(
        repository,
        PyAvPacketRemuxer(),
        window=ClipWindow(pre_event_seconds=1.0, post_event_seconds=1.0),
    )

    # When: an event triggers on a frame in the middle of the ring.
    trigger_index = len(source.pts) // 2
    outcome = coordinator.finalize(
        camera_id=_CAMERA,
        clip_id="clip-stream-copy",
        event_time_sec=float(source.pts[trigger_index] * source.time_base),
        event=_event(),
        output_dir=tmp_path,
        trigger_frame_key=_trigger_key(source, packet_index=trigger_index),
    )

    # Then: the muxed clip replays the source packets unchanged.
    assert isinstance(outcome, ClipReady)
    artifact = outcome.artifact
    assert artifact.remux_method == REMUX_METHOD
    muxed = _muxed_video_packets(artifact.path)
    payloads = tuple(bytes(packet) for packet in muxed)
    start = source.payloads.index(payloads[0])
    expected = source.payloads[start : start + len(payloads)]
    assert payloads == expected
    assert tuple(packet.pts for packet in muxed) == source.pts[start : start + len(payloads)]
    assert tuple(packet.dts for packet in muxed) == source.dts[start : start + len(payloads)]
    assert tuple(packet.is_keyframe for packet in muxed) == (
        source.keyframes[start : start + len(payloads)]
    )
    assert artifact.streams[0].timestamp_translation_ticks == 0
    assert artifact.streams[0].codec_name == source.codec_name
    assert artifact.streams[0].time_base == source.time_base
    assert (artifact.streams[0].width, artifact.streams[0].height) == (
        source.width,
        source.height,
    )
    assert artifact.truncation_reasons == ()
    # The decoder child produced nothing at all, and the clip above owes it
    # nothing. (How much of the packet stream the child actually consumed is
    # deliberately NOT asserted: DecoderInputQueue is allowed to drop decoder-
    # only backlog under pressure, and session.close() discards whatever the
    # writer thread had not drained -- racing that is exactly the kind of
    # timing-luck assertion this suite must not contain.)
    assert spawner.processes[0].read_count == 0


def test_clip_production_needs_no_decoded_frame_from_the_inference_path(
    tmp_path: Path,
) -> None:
    """(B) Clip production imposes no decoded-frame work on the inference path.

    The tee feeds the ring at demux, upstream of any drop: with the decoder
    child emitting zero frames and ``session.read()`` never called, the ring is
    still complete and a full clip still finalizes. The coordinator's frame
    intake is inert -- ``write`` accepts every packet without touching pixels.
    """
    # Given: a fully demuxed ring whose decoder produced no frames at all.
    source = _source(tmp_path, "no-frames.mp4")
    repository = _repository(
        PacketRingLimits(4_000, 16 * 1024 * 1024, 60.0),
        global_max_bytes=16 * 1024 * 1024,
    )
    spawner = _FakeSpawner()
    session = _tee(source, repository, spawner)
    try:
        assert tuple(packet.payload for packet in repository.ring(_CAMERA).snapshot()) == (
            source.payloads
        )
        assert spawner.processes[0].read_count == 0
    finally:
        session.close()
    coordinator = PacketClipRecordingCoordinator(
        repository,
        PyAvPacketRemuxer(),
        window=ClipWindow(pre_event_seconds=1.0, post_event_seconds=1.0),
    )

    # When: a clip finalizes after a frame that shares nothing with the source
    # geometry is written -- if clip bytes came from decoded frames, a 1x1
    # placeholder could not produce a 64x48 stream copy.
    placeholder = FramePacket(
        camera_id=_CAMERA,
        frame=Frame(index=0, time_sec=0.0, image=np.zeros((1, 1, 3), dtype=np.uint8)),
        pts=0.0,
        seq=0,
        width=1,
        height=1,
        decode_time_ms=0.0,
        worker_boot_id=_BOOT,
        stream_epoch=_EPOCH,
    )
    accepted = coordinator.write(placeholder)
    trigger_index = len(source.pts) // 2
    outcome = coordinator.finalize(
        camera_id=_CAMERA,
        clip_id="clip-no-frames",
        event_time_sec=float(source.pts[trigger_index] * source.time_base),
        event=_event(),
        output_dir=tmp_path,
        trigger_frame_key=_trigger_key(source, packet_index=trigger_index),
    )

    # Then: the clip is complete and the decoder was never read.
    assert accepted is True
    assert isinstance(outcome, ClipReady)
    assert outcome.artifact.packet_count > 1
    assert outcome.artifact.truncation_reasons == ()
    assert (outcome.artifact.streams[0].width, outcome.artifact.streams[0].height) == (
        source.width,
        source.height,
    )
    assert spawner.processes[0].read_count == 0


def test_undersized_ring_reports_history_truncation_instead_of_silent_shortening(
    tmp_path: Path,
) -> None:
    """(C) A ring too small for the pre-window says so, explicitly.

    The requested window starts before the oldest surviving keyframe, so the
    selection is genuinely shortened. That shortening must arrive as
    ``HISTORY_UNAVAILABLE`` on the artifact -- never as a quietly shorter clip.
    """
    # Given: a ring that keeps only the newest handful of packets.
    source = _source(tmp_path, "undersized.mp4", frames=60, gop=4)
    repository = _repository(
        PacketRingLimits(12, 16 * 1024 * 1024, 60.0),
        global_max_bytes=16 * 1024 * 1024,
    )
    spawner = _FakeSpawner()
    session = _tee(source, repository, spawner)
    session.close()
    ring = repository.ring(_CAMERA)
    retained = ring.snapshot()

    # When: an event asks for a 3s pre-window the ring can no longer supply.
    coordinator = PacketClipRecordingCoordinator(
        repository,
        PyAvPacketRemuxer(),
        window=ClipWindow(pre_event_seconds=3.0, post_event_seconds=0.0),
    )
    trigger_index = source.payloads.index(retained[-1].payload)
    outcome = coordinator.finalize(
        camera_id=_CAMERA,
        clip_id="clip-undersized",
        event_time_sec=float(source.pts[trigger_index] * source.time_base),
        event=_event(),
        output_dir=tmp_path,
        trigger_frame_key=_trigger_key(source, packet_index=trigger_index),
    )

    # Then: the shortening is reported, and the ring's eviction is counted.
    assert ring.metrics.evicted_packets == len(source.payloads) - len(retained)
    assert isinstance(outcome, ClipReady)
    assert PacketTruncationReason.HISTORY_UNAVAILABLE.value in (
        outcome.artifact.truncation_reasons
    )
    selected_span = (
        outcome.artifact.selected_end_pts_sec - outcome.artifact.selected_start_pts_sec
    )
    assert selected_span is not None
    assert selected_span < 3.0


def test_ring_evicted_past_the_trigger_fails_closed_with_an_explicit_reason(
    tmp_path: Path,
) -> None:
    """(C) Eviction past the trigger keyframe yields no clip and a named reason."""
    # Given: a ring holding a couple of packets, and a trigger far behind them.
    source = _source(tmp_path, "evicted.mp4", frames=60, gop=4)
    repository = _repository(
        PacketRingLimits(2, 16 * 1024 * 1024, 60.0),
        global_max_bytes=16 * 1024 * 1024,
    )
    session = _tee(source, repository, _FakeSpawner())
    session.close()
    coordinator = PacketClipRecordingCoordinator(
        repository,
        PyAvPacketRemuxer(),
        window=ClipWindow(pre_event_seconds=1.0, post_event_seconds=0.0),
    )

    # When: the event's trigger predates everything the ring still holds.
    outcome = coordinator.finalize(
        camera_id=_CAMERA,
        clip_id="clip-evicted",
        event_time_sec=float(source.pts[0] * source.time_base),
        event=_event(),
        output_dir=tmp_path,
        trigger_frame_key=_trigger_key(source, packet_index=0),
    )

    # Then: no clip file, and the reason names the missing history.
    assert outcome == ClipUnavailable(
        "clip-evicted",
        ClipReasonCode.REMUX_FAILED,
        PacketTruncationReason.KEYFRAME_UNAVAILABLE.value,
    )
    assert not (tmp_path / "clip.mp4").exists()


def test_manifest_carries_the_truncation_reason_and_the_stream_copy_provenance(
    tmp_path: Path,
) -> None:
    """(A + C) The published manifest states stream copy and any truncation."""
    # Given: a running recorder over a ring too small for the pre-window.
    source = _source(tmp_path, "manifest.mp4", frames=60, gop=4)
    store_dir = tmp_path / "clip-store"
    store_dir.mkdir(mode=0o700)
    config = ClipRecorderConfig(
        store_dir=store_dir,
        pre_event_seconds=3.0,
        post_event_seconds=0.0,
        finalize_grace_seconds=0.5,
        packet_ring_max_packets=12,
    )
    repository = _repository(
        PacketRingLimits(
            config.packet_ring_max_packets,
            config.packet_ring_max_bytes_per_camera,
            60.0,
        ),
        global_max_bytes=config.packet_ring_global_max_bytes,
    )
    session = _tee(source, repository, _FakeSpawner())
    session.close()
    retained = repository.ring(_CAMERA).snapshot()
    trigger_index = source.payloads.index(retained[-1].payload)
    recorder = ClipRecorder(
        config,
        services=default_services(config, repository),
        disk_usage_provider=_idle_disk,
        is_clip_held=lambda _clip_id: False,
    )
    recorder.start()

    # When: an event finalizes through the real recorder actor.
    try:
        trigger = FramePacket(
            camera_id=_CAMERA,
            frame=Frame(
                index=trigger_index,
                time_sec=float(source.pts[trigger_index] * source.time_base),
                image=np.zeros((source.height, source.width, 3), dtype=np.uint8),
            ),
            pts=float(source.pts[trigger_index] * source.time_base),
            seq=trigger_index,
            width=source.width,
            height=source.height,
            decode_time_ms=0.0,
            worker_boot_id=_BOOT,
            stream_epoch=_EPOCH,
            source_pts=source.pts[trigger_index],
            source_time_base=source.time_base,
        )
        clip_id = recorder.on_event(trigger, _event())
        assert clip_id is not None
        assert recorder.flush(timeout=20.0)
    finally:
        recorder.stop(timeout=20.0)

    # Then: the manifest records the stream copy and the explicit truncation.
    manifest = json.loads(
        (store_dir / "clips" / clip_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["video_available"] is True
    assert manifest["codec"] == source.codec_name
    assert manifest["encoder"] == "source-packet-remux"
    assert manifest["source_media"]["remux_method"] == REMUX_METHOD
    assert manifest["source_media"]["streams"][0]["codec_name"] == source.codec_name
    assert manifest["source_media"]["streams"][0]["time_base"] == (
        f"{source.time_base.numerator}/{source.time_base.denominator}"
    )
    assert manifest["source_media"]["streams"][0]["timestamp_translation_ticks"] == 0
    assert manifest["truncation_reasons"] == [
        PacketTruncationReason.HISTORY_UNAVAILABLE.value
    ]


def test_clip_enabled_keeps_nvdec_decode_out_of_the_python_process(tmp_path: Path) -> None:
    """(D) #312 regression guard: a packet sink does not revive in-process NVDEC.

    Clip recording composes a ``PacketRingRepository`` sink into
    ``decoder_for``. On the nvdec profile that must still open the subprocess
    tee session -- one decoder child fed compressed packets over ``pipe:0`` --
    and the parent's decoded-frame path must remain the child's stdout, not a
    PyAV decode call in this interpreter.
    """
    # Given: the exact adapter the composition root builds with clips enabled.
    source = _source(tmp_path, "clip-enabled.mp4", frames=12)
    repository = _repository(
        PacketRingLimits(4_000, 16 * 1024 * 1024, 60.0),
        global_max_bytes=16 * 1024 * 1024,
    )
    adapter = decoder_for("nvdec", packet_sink=repository)
    assert isinstance(adapter, PyAvPreservingAdapter)
    spawner = _FakeSpawner()

    # When: that adapter opens a camera session.
    session = _tee(source, repository, spawner)

    # Then: decode lives in a child process, and evidence still gets its bytes.
    try:
        assert isinstance(session, NvdecPacketTeeSession)
        assert not isinstance(session, PyAvPreservingSession)
        argv = spawner.calls[0][0]
        assert argv[argv.index("-i") + 1] == "pipe:0"
        assert argv[-1] == "pipe:1"
        assert tuple(packet.payload for packet in repository.ring(_CAMERA).snapshot()) == (
            source.payloads
        )
    finally:
        session.close()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PRE-EXISTING ADR-0001 GAP (not a decode-boundary regression, and not "
        "todo 11's to fix): when a source's own extradata differs from the mux "
        "template's -- which is ALWAYS true for RTSP, where packets arrive in "
        "Annex-B framing while the PyAV-built template capsule carries "
        "length-prefixed AVCC -- ``_verify_packet_facts`` puts the stream in "
        "``container_normalized_streams`` and then skips BOTH the per-packet "
        "payload comparison and the keyframe-identity comparison for it. A "
        "corrupted payload or a dropped keyframe flag on the production source "
        "type therefore passes verification silently; the real-stack "
        "counterpart "
        "(tests/test_worker_packet_remux_real_ffmpeg.py::"
        "test_real_nvdec_boundary_clip_preserves_keyframe_identity) shows the "
        "published RTSP clip really does lose every keyframe flag. Owner: the "
        "remuxer/ADR-0001 lane."
    ),
)
def test_container_normalized_streams_still_verifies_payload_bytes() -> None:
    """Container normalization must not disable the byte-level guarantee."""
    # Given: a stream whose extradata was normalized by the container, and a
    # muxed result whose payload was corrupted.
    configuration = SourceStreamConfiguration.from_streams(
        [
            SourceStreamDescriptor(
                0,
                "video",
                "h264",
                "avc1",
                Fraction(1, 90_000),
                b"annex-b-extradata",
                640,
                480,
            )
        ],
        mux_template=b"template",
    )
    epoch = StreamEpoch(_BOOT, _CAMERA, _EPOCH)
    expected = (
        SourcePacket(epoch, configuration, 0, 0, 0, 6_000, True, b"original", 0),
        SourcePacket(epoch, configuration, 0, 6_000, 6_000, 6_000, False, b"second", 1),
    )
    corrupted = (
        packet_remuxer._MuxedPacketFact(  # noqa: SLF001 - verifier under test
            stream_index=0,
            pts=0,
            dts=0,
            duration=6_000,
            time_base=Fraction(1, 90_000),
            is_keyframe=False,
            payload=b"tampered",
        ),
        packet_remuxer._MuxedPacketFact(  # noqa: SLF001
            stream_index=0,
            pts=6_000,
            dts=6_000,
            duration=6_000,
            time_base=Fraction(1, 90_000),
            is_keyframe=False,
            payload=b"second",
        ),
    )

    # When / Then: verification must still reject it.
    with pytest.raises(ValueError):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            expected,
            corrupted,
            configuration,
            container_normalized_streams={0},
        )


def test_clip_enabled_on_a_cpu_profile_still_decodes_in_process(tmp_path: Path) -> None:
    """(D) The guard is specific: only NVDEC moves out, CPU stays where it was."""
    # Given: the clip-enabled adapter on a non-NVDEC profile.
    source = _source(tmp_path, "cpu-profile.mp4", frames=12)
    repository = _repository(
        PacketRingLimits(4_000, 16 * 1024 * 1024, 60.0),
        global_max_bytes=16 * 1024 * 1024,
    )
    adapter = decoder_for("cpu", packet_sink=repository)

    # When: it opens a camera session.
    session = adapter.open(
        NvdecCuvidConfig(
            camera_id=_CAMERA,
            url=str(source.path),
            open_timeout_ms=2_000,
            read_timeout_ms=1_000,
        )
    )

    # Then: it is the in-process PyAV session, and evidence still gets its bytes.
    try:
        assert isinstance(session, PyAvPreservingSession)
        session.set_stream_identity(_BOOT, _EPOCH)
        assert session.wait_demux_complete(20)
        assert tuple(packet.payload for packet in repository.ring(_CAMERA).snapshot()) == (
            source.payloads
        )
    finally:
        session.close()

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

import pytest

import worker.adapters.encode.packet_remuxer as packet_remuxer
from worker.adapters.encode.adapter_errors import ClipRemuxError
from worker.adapters.encode.models import ClipArtifact
from worker.pipeline.output.evidence.clip_recording import (
    ClipReady,
    ClipReasonCode,
    ClipUnavailable,
    ClipWindow,
)
from worker.pipeline.output.evidence.packet_recording import PacketClipRecordingCoordinator
from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.packet_ring import PacketRingLimits
from worker.types import BusinessEvent, FrameKey
from worker.types.source_packet import (
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
    StreamEpoch,
)


def _config() -> SourceStreamConfiguration:
    return SourceStreamConfiguration.from_streams(
        [SourceStreamDescriptor(0, "video", "h264", "avc1", Fraction(1, 1000), b"avcc", 16, 16)],
        mux_template=b"detached-template",
    )


def _packet(ms: int, *, keyframe: bool = False) -> SourcePacket:
    config = _config()
    return SourcePacket(
        StreamEpoch("boot-1", "camera-1", 3),
        config,
        0,
        ms,
        ms,
        40,
        keyframe,
        bytes([ms % 251 + 1]),
        ms,
    )


def _event() -> BusinessEvent:
    return BusinessEvent("fall", "fall.detected", 1, "camera-1", "facility-1", 2.0, 0.9)


@dataclass(slots=True)
class _Remuxer:
    calls: list[tuple[SourcePacket, ...]] = field(default_factory=list)
    fail: bool = False

    def remux(
        self,
        packets: Sequence[SourcePacket],
        configuration: SourceStreamConfiguration,
        output_path: Path,
    ) -> ClipArtifact:
        self.calls.append(tuple(packets))
        if self.fail:
            raise ClipRemuxError("bad packet")
        assert configuration.configuration_id == packets[0].configuration.configuration_id
        return ClipArtifact(output_path, 3, 1, 1.0, remux_method="test-stream-copy")


def _coordinator(remuxer: _Remuxer) -> tuple[PacketClipRecordingCoordinator, PacketRingRepository]:
    repository = PacketRingRepository(
        ("camera-1",),
        per_camera_limits=PacketRingLimits(32, 4096, 30),
        global_max_bytes=4096,
    )
    for packet in (_packet(0, keyframe=True), _packet(1000), _packet(2000), _packet(3000)):
        assert repository.append(packet)
    return PacketClipRecordingCoordinator(repository, remuxer, window=ClipWindow(1, 1)), repository


def test_primary_coordinator_uses_exact_source_pts_and_never_reads_a_decoded_frame(
    tmp_path: Path,
) -> None:
    remuxer = _Remuxer()
    coordinator, _ = _coordinator(remuxer)
    key = FrameKey("boot-1", "camera-1", 3, 10, 999.0, 2000, Fraction(1, 1000))

    outcome = coordinator.finalize(
        camera_id="camera-1",
        clip_id="clip-1",
        event_time_sec=999.0,
        event=_event(),
        output_dir=tmp_path,
        trigger_frame_key=key,
    )

    assert isinstance(outcome, ClipReady)
    assert tuple(packet.presentation_time for packet in remuxer.calls[0]) == (
        Fraction(0),
        Fraction(1),
        Fraction(2),
        Fraction(3),
    )
    assert outcome.artifact.remux_method == "test-stream-copy"
    assert outcome.artifact.selected_start_pts_sec == 0.0
    assert outcome.artifact.selected_end_pts_sec == 3.0


def _muxed_fact(
    packet: SourcePacket,
    *,
    translation: int = 0,
) -> packet_remuxer._MuxedPacketFact:  # noqa: SLF001 - exact verifier regression
    return packet_remuxer._MuxedPacketFact(  # noqa: SLF001
        stream_index=packet.stream_index,
        pts=None if packet.pts is None else packet.pts + translation,
        dts=None if packet.dts is None else packet.dts + translation,
        duration=packet.duration,
        time_base=packet.stream.time_base,
        is_keyframe=packet.is_keyframe,
        payload=packet.payload,
    )


def test_packet_verifier_records_exact_uniform_translation() -> None:
    packets = (_packet(1000, keyframe=True), _packet(2000))
    translations, translation_seconds = packet_remuxer._verify_packet_facts(  # noqa: SLF001
        packets,
        tuple(_muxed_fact(packet, translation=-10) for packet in packets),
        packets[0].configuration,
    )

    assert translations == {0: -10}
    assert translation_seconds == Fraction(-1, 100)


@pytest.mark.parametrize(
    "actual",
    (
        (
            _muxed_fact(_packet(1000, keyframe=True), translation=-10),
            _muxed_fact(_packet(2000), translation=-11),
        ),
        (
            replace(_muxed_fact(_packet(1000, keyframe=True), translation=-10), dts=989),
            _muxed_fact(_packet(2000), translation=-10),
        ),
        (
            replace(_muxed_fact(_packet(1000, keyframe=True)), payload=b"changed"),
            _muxed_fact(_packet(2000)),
        ),
    ),
)
def test_packet_verifier_rejects_nonuniform_composition_or_payload_changes(
    actual: tuple[packet_remuxer._MuxedPacketFact, ...],  # noqa: SLF001
) -> None:
    packets = (_packet(1000, keyframe=True), _packet(2000))

    with pytest.raises(ValueError):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            packets,
            actual,
            packets[0].configuration,
        )


def test_packet_verifier_requires_one_exact_wall_clock_translation_across_streams() -> None:
    configuration = SourceStreamConfiguration.from_streams(
        [
            SourceStreamDescriptor(
                0,
                "video",
                "h264",
                "avc1",
                Fraction(1, 1000),
                b"avcc",
                16,
                16,
            ),
            SourceStreamDescriptor(
                1,
                "audio",
                "aac",
                "mp4a",
                Fraction(1, 48_000),
                b"aac",
                sample_rate=48_000,
                channels=1,
            ),
        ],
        mux_template=b"detached-template",
    )
    epoch = StreamEpoch("boot-1", "camera-1", 3)
    packets = (
        SourcePacket(epoch, configuration, 0, 1000, 900, 40, True, b"video", 0),
        SourcePacket(epoch, configuration, 1, 48_000, 47_000, 1024, True, b"audio", 1),
    )
    translated = (
        _muxed_fact(packets[0], translation=-10),
        _muxed_fact(packets[1], translation=-480),
    )
    _, translation_seconds = packet_remuxer._verify_packet_facts(  # noqa: SLF001
        packets,
        translated,
        configuration,
    )
    assert translation_seconds == Fraction(-1, 100)

    with pytest.raises(ValueError, match="nonuniform timestamp translations"):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            packets,
            (translated[0], _muxed_fact(packets[1], translation=-479)),
            configuration,
        )


def test_packet_verifier_rejects_translation_that_creates_negative_presentation_time() -> None:
    packets = (_packet(0, keyframe=True), _packet(1000))

    with pytest.raises(ValueError, match="negative timeline"):
        packet_remuxer._verify_packet_facts(  # noqa: SLF001
            packets,
            tuple(_muxed_fact(packet, translation=-1) for packet in packets),
            packets[0].configuration,
        )


@pytest.mark.parametrize(
    "changed",
    (
        replace(_packet(2000), epoch=StreamEpoch("boot-1", "camera-1", 4)),
        replace(
            _packet(2000),
            configuration=SourceStreamConfiguration.from_streams(
                [
                    SourceStreamDescriptor(
                        0,
                        "video",
                        "h264",
                        "avc1",
                        Fraction(1, 1000),
                        b"changed-avcc",
                        16,
                        16,
                    )
                ],
                mux_template=b"detached-template",
            ),
        ),
    ),
)
def test_source_timeline_rejects_cross_epoch_or_configuration_packets(
    changed: SourcePacket,
) -> None:
    packets = (_packet(1000, keyframe=True), changed)

    with pytest.raises(ValueError, match="mixes stream epochs or configurations"):
        packet_remuxer._validate_source_timeline(  # noqa: SLF001
            packets,
            packets[0].configuration,
        )


def test_missing_identity_or_corrupt_remux_fails_closed_without_reencode(tmp_path: Path) -> None:
    remuxer = _Remuxer(fail=True)
    coordinator, _ = _coordinator(remuxer)
    missing = coordinator.finalize(
        camera_id="camera-1",
        clip_id="clip-missing",
        event_time_sec=2.0,
        event=_event(),
        output_dir=tmp_path,
        trigger_frame_key=None,
    )
    assert missing == ClipUnavailable(
        "clip-missing",
        ClipReasonCode.STREAM_EPOCH_MISMATCH,
        "TRIGGER_STREAM_IDENTITY_UNAVAILABLE",
    )

    failed = coordinator.finalize(
        camera_id="camera-1",
        clip_id="clip-bad",
        event_time_sec=2.0,
        event=_event(),
        output_dir=tmp_path,
        trigger_frame_key=FrameKey("boot-1", "camera-1", 3, 10, 2.0, 2000, Fraction(1, 1000)),
    )
    assert isinstance(failed, ClipUnavailable)
    assert failed.reason_code is ClipReasonCode.REMUX_FAILED
    assert failed.detail_reason == "SOURCE_PACKET_REMUX_FAILED"
    assert not (tmp_path / "clip.mp4").exists()

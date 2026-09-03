from __future__ import annotations

import inspect
import threading
from collections import deque
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any, final

import av
import numpy as np
import pytest

import worker.adapters.decode.pyav_preserving as preserving_module
from contracts.frame import Frame
from worker.adapters.decode.nvdec_cuvid.input_queue import DecoderInputQueue
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig
from worker.adapters.decode.pyav_demux import PyAvPacketDemuxer
from worker.adapters.decode.pyav_preserving import PyAvPreservingAdapter
from worker.pipeline.ingest.rtsp import RTSPSource
from worker.pipeline.output.evidence.packet_ring import (
    PacketRingLimits,
    SourcePacketRing,
)
from worker.types import FramePacket
from worker.types.source_packet import (
    PacketSelectionError,
    SourcePacket,
    SourceStreamConfiguration,
    SourceStreamDescriptor,
    StreamEpoch,
)


@final
class _FakeDecoderProcess:
    def __init__(self, frames: list[bytes] | None = None) -> None:
        self.frames = deque(frames or [])
        self.packet_payloads: list[bytes] = []
        self.read_timeouts: list[float] = []
        self.input_close_count = 0
        self.reap_count = 0
        self._written = threading.Condition()

    def write_packet(self, payload: bytes) -> None:
        with self._written:
            self.packet_payloads.append(bytes(payload))
            self._written.notify_all()

    def wait_for_writes(self, count: int, timeout: float) -> bool:
        """Block until the writer thread has handed ``count`` packets over."""
        with self._written:
            return self._written.wait_for(
                lambda: len(self.packet_payloads) >= count, timeout=timeout
            )

    def close_input(self) -> None:
        self.input_close_count += 1

    def read_frame(self, timeout_sec: float) -> bytes | None:
        self.read_timeouts.append(timeout_sec)
        return self.frames.popleft() if self.frames else None

    def reap(self, timeout_sec: float) -> int | None:
        del timeout_sec
        self.reap_count += 1
        return 0


@final
class _BlockingDecoderProcess(_FakeDecoderProcess):
    def __init__(self) -> None:
        super().__init__([])
        self.write_entered = threading.Event()
        self.input_closed = threading.Event()
        self._release_write = threading.Event()

    def write_packet(self, payload: bytes) -> None:
        self.write_entered.set()
        self._release_write.wait()
        super().write_packet(payload)

    def close_input(self) -> None:
        super().close_input()
        self.input_closed.set()

    def release_write(self) -> None:
        self._release_write.set()

    def reap(self, timeout_sec: float) -> int | None:
        self.release_write()
        return super().reap(timeout_sec)


@final
class _FakeSpawner:
    def __init__(self, *, emit_frame: bool = True) -> None:
        self.emit_frame = emit_frame
        self.calls: list[tuple[tuple[str, ...], int]] = []
        self.processes: list[_FakeDecoderProcess] = []

    def __call__(self, args: tuple[str, ...], frame_size: int) -> _FakeDecoderProcess:
        self.calls.append((args, frame_size))
        payload = bytes(index % 251 for index in range(frame_size))
        process = _FakeDecoderProcess([payload] if self.emit_frame else [])
        self.processes.append(process)
        return process


def _encode(path: Path, *, width: int = 8, height: int = 6, frames: int = 8) -> None:
    output = av.open(str(path), mode="w", format="mp4")
    stream = output.add_stream("libx264", rate=10)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"g": "2", "bf": "0"}
    for index in range(frames):
        image = np.full((height, width, 3), (index * 20) % 256, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()


def _payloads(path: Path) -> tuple[bytes, ...]:
    source = av.open(str(path))
    try:
        return tuple(
            bytes(packet)
            for packet in source.demux(source.streams.video[0])
            if packet.dts is not None and bytes(packet)
        )
    finally:
        source.close()


def _config(path: Path, *, camera_id: str = "camera-1") -> NvdecCuvidConfig:
    return NvdecCuvidConfig(
        camera_id=camera_id,
        url=str(path),
        open_timeout_ms=2_000,
        read_timeout_ms=125,
    )


def _ring() -> SourcePacketRing:
    return SourcePacketRing("camera-1", PacketRingLimits(1_000, 8 * 1024 * 1024, 60.0))


def test_nvdec_parent_tees_exact_packets_and_frames_come_from_subprocess_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    _encode(source)
    expected_packets = _payloads(source)
    original_open = av.open
    source_open_count = 0

    def count_source_open(file: Any, *args: Any, **kwargs: Any):
        nonlocal source_open_count
        mode = kwargs.get("mode", args[0] if args else "r")
        if str(file) == str(source) and mode == "r":
            source_open_count += 1
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(preserving_module.av, "open", count_source_open)
    ring = _ring()
    spawner = _FakeSpawner()
    session = PyAvPreservingAdapter(
        ring,
        decode_backend="nvdec",
        process_spawner=spawner,
    ).open(_config(source))

    session.set_stream_identity("boot-1", 1)
    assert session.wait_demux_complete(10)
    process = spawner.processes[0]
    # The demuxer only hands packets to the writer thread; a decoded frame can
    # only follow the packet that produced it, so wait for the real write.
    assert process.wait_for_writes(len(expected_packets), 10.0)
    frame = session.read()
    session.close()

    assert frame is not None
    assert source_open_count == 1
    assert tuple(packet.payload for packet in ring.snapshot()) == expected_packets
    assert tuple(process.packet_payloads) == expected_packets
    assert spawner.calls[0][0][spawner.calls[0][0].index("-i") + 1] == "pipe:0"
    expected_frame = np.frombuffer(
        bytes(index % 251 for index in range(frame.width * frame.height * 3)),
        dtype=np.uint8,
    ).reshape(frame.height, frame.width, 3)
    assert np.array_equal(frame.frame.image, expected_frame)
    assert frame.worker_boot_id == "boot-1"
    assert frame.stream_epoch == 1
    assert frame.source_pts is not None
    assert frame.source_time_base is not None
    assert frame.pts == pytest.approx(float(frame.source_pts * frame.source_time_base))
    assert process.input_close_count >= 1
    frame.release()


def test_stalled_decoder_input_never_blocks_evidence_tee(tmp_path: Path) -> None:
    source = tmp_path / "stalled-decoder.mp4"
    _encode(source, frames=160)
    expected_packets = _payloads(source)
    ring = _ring()
    process = _BlockingDecoderProcess()
    session = PyAvPreservingAdapter(
        ring,
        decode_backend="nvdec",
        process_spawner=lambda _args, _frame_size: process,
    ).open(_config(source))

    try:
        session.set_stream_identity("boot-1", 1)
        assert process.write_entered.wait(1.0)
        assert session.wait_demux_complete(2.0)
        snapshot = ring.snapshot()
        assert tuple(packet.payload for packet in snapshot) == expected_packets
        assert session.decoder_input_overflow_count > 0
        process.release_write()
        assert process.input_closed.wait(2.0)
        assert len(process.packet_payloads) >= 2
        keyframe_payloads = {packet.payload for packet in snapshot if packet.is_keyframe}
        assert process.packet_payloads[1] in keyframe_payloads
    finally:
        session.close()


def test_nvdec_session_parent_hot_path_contains_no_decode_or_to_ndarray(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    _encode(source)
    session = PyAvPreservingAdapter(
        _ring(), decode_backend="nvdec", process_spawner=_FakeSpawner()
    ).open(_config(source))
    try:
        implementation = inspect.getsource(type(session))
        assert ".decode(" not in implementation
        assert "to_ndarray" not in implementation
    finally:
        session.close()


def test_reconnect_respawns_decoder_and_rolls_away_stale_ring_history(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    _encode(first, width=8, height=6)
    _encode(second, width=10, height=8)
    ring = _ring()
    spawner = _FakeSpawner()
    adapter = PyAvPreservingAdapter(
        ring,
        decode_backend="nvdec",
        process_spawner=spawner,
    )

    first_session = adapter.open(_config(first))
    first_session.set_stream_identity("boot-1", 1)
    assert first_session.wait_demux_complete(10)
    old_epoch = StreamEpoch("boot-1", "camera-1", 1)
    old_trigger = max(packet.presentation_time for packet in ring.snapshot())
    first_session.close()

    second_session = adapter.open(_config(second))
    second_session.set_stream_identity("boot-1", 2)
    assert second_session.wait_demux_complete(10)
    second_session.close()

    new_epoch = StreamEpoch("boot-1", "camera-1", 2)
    assert len(spawner.processes) == 2
    assert ring.active_epoch == new_epoch
    assert {packet.epoch for packet in ring.snapshot()} == {new_epoch}
    with pytest.raises(PacketSelectionError):
        ring.select(
            trigger_epoch=old_epoch,
            trigger_pts=old_trigger,
            pre_seconds=old_trigger,
            post_seconds=old_trigger,
        )


@final
class _ExtradataChangingContainer:
    def __init__(
        self,
        inner: Any,
        *,
        change_after: int,
        written_before_change: Callable[[], bool] | None = None,
    ) -> None:
        self._inner = inner
        self.streams = inner.streams
        self._change_after = change_after
        self._written_before_change = written_before_change

    def demux(self, streams: Any):
        seen = 0
        video = self.streams.video[0]
        original = bytes(video.codec_context.extradata or b"")
        for packet in self._inner.demux(streams):
            if packet.dts is not None and bytes(packet):
                seen += 1
                if seen == self._change_after:
                    if self._written_before_change is not None:
                        # Respawn only once the first decoder really consumed a
                        # packet, so the assertion below is not a thread race.
                        assert self._written_before_change()
                    video.codec_context.extradata = original + b"\x00"
            yield packet

    def close(self) -> None:
        self._inner.close()


def test_extradata_change_respawns_decoder_and_rolls_same_session_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "changing.mp4"
    _encode(source, frames=12)
    original_open = av.open
    handed_out = False

    def open_with_change(file: Any, *args: Any, **kwargs: Any):
        nonlocal handed_out
        mode = kwargs.get("mode", args[0] if args else "r")
        if str(file) == str(source) and mode == "r" and not handed_out:
            handed_out = True
            return _ExtradataChangingContainer(
                original_open(file, *args, **kwargs),
                change_after=6,
                written_before_change=lambda: spawner.processes[0].wait_for_writes(1, 10.0),
            )
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(preserving_module.av, "open", open_with_change)
    ring = _ring()
    spawner = _FakeSpawner()
    session = PyAvPreservingAdapter(ring, decode_backend="nvdec", process_spawner=spawner).open(
        _config(source)
    )

    session.set_stream_identity("boot-1", 1)
    assert session.wait_demux_complete(10)
    assert len(spawner.processes) == 2
    # Writing happens on the input-queue thread; close() aborts its backlog, so
    # wait for the write itself instead of racing the shutdown.
    assert spawner.processes[1].wait_for_writes(1, 10.0)
    session.close()

    assert all(process.packet_payloads for process in spawner.processes)
    assert ring.active_epoch == StreamEpoch("boot-1", "camera-1", 1)
    assert len({packet.configuration.configuration_id for packet in ring.snapshot()}) == 1


def test_subprocess_spawn_failure_is_sanitized_and_fails_open_loudly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "spawn-failure.mp4"
    _encode(source)
    config = _config(source)
    secret = "credential=never-render-this"

    def fail_spawn(_args: tuple[str, ...], _frame_size: int) -> _FakeDecoderProcess:
        raise OSError(secret)

    with pytest.raises(RuntimeError, match="ffmpeg spawn failed") as raised:
        PyAvPreservingAdapter(_ring(), decode_backend="nvdec", process_spawner=fail_spawn).open(
            config
        )
    assert config.url not in str(raised.value)
    assert secret not in str(raised.value)


def test_decode_silence_drives_rtsp_reconnect_instead_of_escaping_ingest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reconnect-after-silence.mp4"
    _encode(source)
    silent_spawner = _FakeSpawner(emit_frame=False)
    silent = PyAvPreservingAdapter(
        _ring(), decode_backend="nvdec", process_spawner=silent_spawner
    ).open(_config(source))
    recovered_process = _FakeDecoderProcess([])
    recovered = FramePacket(
        camera_id="camera-1",
        frame=Frame(index=0, time_sec=0.0, image=np.zeros((1, 1, 3), dtype=np.uint8)),
        pts=0.0,
        seq=0,
        width=1,
        height=1,
        decode_time_ms=0.0,
    )

    class RecoveredSession:
        def read(self) -> FramePacket | None:
            return recovered

        def close(self) -> None:
            recovered_process.reap(1.0)

    class Adapter:
        def __init__(self) -> None:
            self.sessions = deque([silent, RecoveredSession()])
            self.open_count = 0

        def open(self, _config: NvdecCuvidConfig) -> Any:
            self.open_count += 1
            return self.sessions.popleft()

    adapter = Adapter()
    stream = iter(
        RTSPSource(
            _config(source),
            adapter,
            max_failures=1,
            reconnect_initial_backoff_sec=0.0,
            reconnect_max_backoff_sec=0.0,
            target_fps=1_000.0,
            worker_boot_id="boot-1",
        )
    )
    try:
        packet = next(stream)
        assert (packet.camera_id, packet.seq) == ("camera-1", 0)
        assert adapter.open_count == 2
        assert silent_spawner.processes[0].reap_count >= 1
    finally:
        stream.close()


def _source_packet(arrival_index: int) -> SourcePacket:
    configuration = SourceStreamConfiguration.from_streams(
        [
            SourceStreamDescriptor(
                index=0,
                media_type="video",
                codec_name="h264",
                codec_tag="avc1",
                time_base=Fraction(1, 1_000),
                extradata=b"\x00",
                width=8,
                height=6,
            )
        ]
    )
    return SourcePacket(
        epoch=StreamEpoch("boot-1", "camera-1", 1),
        configuration=configuration,
        stream_index=0,
        pts=arrival_index * 40,
        dts=arrival_index * 40,
        duration=40,
        is_keyframe=arrival_index == 0,
        payload=bytes((arrival_index + 1,)) * 4,
        arrival_index=arrival_index,
    )


@final
class _EmitDuringWriteProcess(_FakeDecoderProcess):
    """Decoder whose frame becomes readable *inside* ``write_packet``.

    This is the real race: ffmpeg can flush a decoded frame to stdout the
    instant the packet bytes land, so the reader thread may call
    ``pop_timing`` before the writer thread finishes the write call.
    """

    def __init__(self, queue_holder: list[DecoderInputQueue]) -> None:
        super().__init__([])
        self._queue_holder = queue_holder
        self.observed: list[SourcePacket | Exception] = []
        self.frame_read = threading.Event()

    def write_packet(self, payload: bytes) -> None:
        decoder_input = self._queue_holder[0]
        try:
            self.observed.append(decoder_input.pop_timing())
        except Exception as error:  # noqa: BLE001 - probe records the failure
            self.observed.append(error)
        self.frame_read.set()
        super().write_packet(payload)


@final
class _RejectingProcess(_FakeDecoderProcess):
    def __init__(self, failure: Exception) -> None:
        super().__init__([])
        self._failure = failure

    def write_packet(self, payload: bytes) -> None:
        del payload
        raise self._failure


def test_timing_is_readable_before_the_decoder_can_emit_its_frame() -> None:
    """A frame emitted synchronously with the write still finds its timing."""
    holder: list[DecoderInputQueue] = []
    process = _EmitDuringWriteProcess(holder)
    decoder_input = DecoderInputQueue(process)
    holder.append(decoder_input)
    packet = _source_packet(0)
    try:
        assert decoder_input.offer(packet)
        assert process.frame_read.wait(2.0)
    finally:
        decoder_input.abort()
        decoder_input.join(2.0)

    assert process.observed == [packet]
    assert decoder_input.error is None


def test_failed_packet_write_withdraws_its_timing_from_the_pairing() -> None:
    """A packet the decoder never accepted must not shift frame->timing pairs."""
    failure = OSError("broken pipe")
    process = _RejectingProcess(failure)
    decoder_input = DecoderInputQueue(process)
    try:
        assert decoder_input.offer(_source_packet(0))
        decoder_input.join(5.0)
        assert process.input_close_count == 1, "writer thread did not exit"
        assert decoder_input.error is failure
        with pytest.raises(RuntimeError, match="without source packet timing"):
            decoder_input.pop_timing()
    finally:
        decoder_input.abort()
        decoder_input.join(2.0)


def test_transient_decode_silence_returns_none_within_read_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "silent.mp4"
    _encode(source)
    spawner = _FakeSpawner(emit_frame=False)
    session = PyAvPreservingAdapter(_ring(), decode_backend="nvdec", process_spawner=spawner).open(
        _config(source)
    )
    session.set_stream_identity("boot-1", 1)
    assert session.wait_demux_complete(10)

    assert session.read() is None
    assert spawner.processes[0].read_timeouts == [pytest.approx(0.125)]
    session.close()


def test_current_wrappers_keep_generic_runtimeerror_and_cause_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline: both wrappers stay generic RuntimeError with raise-from."""
    from worker.adapters.decode.nvdec_cuvid.errors import NvdecUnavailableError

    # Given
    cause = NvdecUnavailableError("cuvid decode failed", returncode=69)
    process = _RejectingProcess(cause)
    decoder_input = DecoderInputQueue(process)

    # When / Then: input-queue wrapper
    try:
        assert decoder_input.offer(_source_packet(0))
        decoder_input.join(5.0)
        with pytest.raises(
            RuntimeError,
            match=r"packet-preserving NVDEC decode failed \(NvdecUnavailableError\)",
        ) as raised:
            decoder_input.raise_if_failed()
        assert type(raised.value) is RuntimeError
        assert raised.value.__cause__ is cause
        assert "cuvid decode failed" not in str(raised.value)
    finally:
        decoder_input.abort()
        decoder_input.join(2.0)

    # And: session demux wrapper keeps the same contract via the public adapter.
    source = tmp_path / "demux-wrapper.mp4"
    _encode(source)

    def failing_run(self: object, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise OSError("broken pipe")

    monkeypatch.setattr(PyAvPacketDemuxer, "run", failing_run)
    session = PyAvPreservingAdapter(
        _ring(),
        decode_backend="nvdec",
        process_spawner=lambda _args, _frame_size: _FakeDecoderProcess(),
    ).open(_config(source))
    session.set_stream_identity("boot-1", 1)
    assert session.wait_demux_complete(10)
    with pytest.raises(
        RuntimeError,
        match=r"packet-preserving NVDEC decode failed \(OSError\)",
    ) as demuxed:
        _ = session.read()
    assert type(demuxed.value) is RuntimeError
    assert isinstance(demuxed.value.__cause__, OSError)
    assert "broken pipe" not in str(demuxed.value)
    session.close()


def test_raise_if_failed_stores_safe_process_detail_on_nvdec_reason() -> None:
    """The wrapper stays generic; the chained NVDEC reason carries the safe line."""
    from worker.adapters.decode.nvdec_cuvid.errors import NvdecUnavailableError

    @final
    class _ExitedProcess:
        failure_returncode = 69
        failure_detail = "cuvid decode failed"

        def write_packet(self, payload: bytes) -> None:
            del payload

        def close_input(self) -> None:
            return

        def read_frame(self, timeout_sec: float) -> bytes | None:
            del timeout_sec
            return None

        def reap(self, timeout_sec: float) -> int | None:
            del timeout_sec
            return 69

    decoder_input = DecoderInputQueue(_ExitedProcess())
    try:
        with pytest.raises(NvdecUnavailableError) as raised:
            decoder_input.raise_if_failed()
        assert type(raised.value) is NvdecUnavailableError
        assert raised.value.returncode == 69
        assert "cuvid decode failed" in raised.value.reason
        assert raised.value.safe_log_detail == raised.value.reason
        wrapper = RuntimeError(
            f"packet-preserving NVDEC decode failed ({type(raised.value).__name__})"
        )
        wrapper.__cause__ = raised.value
        assert type(wrapper) is RuntimeError
        assert wrapper.__cause__ is raised.value
        assert "cuvid decode failed" not in str(wrapper)
    finally:
        decoder_input.abort()
        decoder_input.join(2.0)

"""Pin the ``NvdecCuvidAdapter`` subprocess seam that Wave 2 reuses.

Plan todo 4 reuses ``ffmpeg_decode_args``, ``cuvid_decoder_for`` and the
raw-frame framing from ``nvdec_cuvid`` rather than duplicating them. These
tests pin the exact ffmpeg argv (as a list, position by position), the codec
probe table, and the byte-level rawvideo framing contract, so the refactor
changes only what it must.

Every assertion is on a real observable object -- the full argv tuple, real
decoded pixel bytes, real exception instances -- never a mock call count.
Fakes stand in for the ffmpeg child process only (no GPU, no ffmpeg binary,
no RTSP), so this stays CI-safe and is deliberately not ``real_stack``.
"""

from __future__ import annotations

from typing import final

import pytest

from worker.adapters.decode.nvdec_cuvid.adapter import (
    NvdecCuvidAdapter,
    NvdecCuvidSession,
    ffmpeg_decode_args,
)
from worker.adapters.decode.nvdec_cuvid.errors import (
    NvdecReadError,
    NvdecUnavailableError,
    UnsupportedCodecError,
)
from worker.adapters.decode.nvdec_cuvid.models import NvdecCuvidConfig, StreamMetadata
from worker.adapters.decode.nvdec_cuvid.probe import (
    cuvid_decoder_for,
    ffprobe_args,
    ffprobe_binary,
    probe_stream_metadata,
)
from worker.adapters.decode.nvdec_cuvid.process import FFmpegDecodeProcess
from worker.interfaces.decode import DecodeAdapter, DecodeSession

_CONFIG = NvdecCuvidConfig(
    camera_id="camera-a",
    url="rtsp://camera.local/live",
    open_timeout_ms=4_000,
    read_timeout_ms=250,
)


@final
class _FakeDecoderProcess:
    def __init__(self, payloads: list[bytes | None]) -> None:
        self._payloads = payloads
        self.reap_calls = 0

    def read_frame(self, timeout_sec: float) -> bytes | None:
        del timeout_sec
        return self._payloads.pop(0) if self._payloads else None

    def reap(self, timeout_sec: float) -> int | None:
        del timeout_sec
        self.reap_calls += 1
        return 0


@final
class _ChunkStream:
    """Real pipe-shaped stdout that hands back arbitrary chunk boundaries."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        del size
        return self._chunks.pop(0) if self._chunks else b""

    def close(self) -> None:
        self.closed = True


@final
class _ChunkChild:
    def __init__(self, stdout: _ChunkStream) -> None:
        self.stdout = stdout
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


def _probe_runner(codec_name: str = "h264", *, width: int = 2, height: int = 1):
    def run(args: tuple[str, ...], timeout_sec: float) -> str:
        del args, timeout_sec
        return (
            f'{{"streams":[{{"width":{width},"height":{height},'
            f'"codec_name":"{codec_name}"}}]}}'
        )

    return run


def test_ffmpeg_decode_argv_is_pinned_position_by_position() -> None:
    """SEAM B1: the exact ffmpeg argv. Wave 2 reuses this builder for the
    per-camera decode subprocess; the hwaccel/decoder/rawvideo/rgb24/pipe:1
    shape below is the contract, and any change is a deliberate seam change."""
    # Given / When
    argv = ffmpeg_decode_args(_CONFIG, "h264_cuvid")

    # Then
    assert argv == (
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-hwaccel",
        "cuda",
        "-c:v",
        "h264_cuvid",
        "-i",
        "rtsp://camera.local/live",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    )
    # The pieces Wave 2 MUST keep, stated independently of argv order:
    assert argv[0] == _CONFIG.ffmpeg_bin
    assert argv[argv.index("-hwaccel") + 1] == "cuda"
    assert argv[argv.index("-c:v") + 1] == "h264_cuvid"
    assert argv[argv.index("-f") + 1] == "rawvideo"
    assert argv[argv.index("-pix_fmt") + 1] == "rgb24"
    assert argv[-1] == "pipe:1"
    # The input source is the RTSP URL TODAY. Wave 2 switches this to a stdin
    # pipe; this assertion is the tripwire that says the change was deliberate.
    assert argv[argv.index("-i") + 1] == _CONFIG.url
    assert "-nostdin" in argv


def test_ffmpeg_decode_argv_honours_a_custom_binary_and_decoder() -> None:
    """SEAM B2: ffmpeg binary and decoder token are parameters, not constants."""
    # Given
    config = NvdecCuvidConfig(
        camera_id="camera-a",
        url="rtsp://camera.local/second",
        ffmpeg_bin="/opt/ffmpeg/bin/ffmpeg",
    )

    # When
    argv = ffmpeg_decode_args(config, "hevc_cuvid")

    # Then
    assert argv[0] == "/opt/ffmpeg/bin/ffmpeg"
    assert argv[argv.index("-c:v") + 1] == "hevc_cuvid"
    assert argv[argv.index("-i") + 1] == "rtsp://camera.local/second"


@pytest.mark.parametrize(
    ("codec_name", "decoder"),
    [
        ("h264", "h264_cuvid"),
        ("avc", "h264_cuvid"),
        ("hevc", "hevc_cuvid"),
        ("h265", "hevc_cuvid"),
        ("av1", "av1_cuvid"),
        ("vp9", "vp9_cuvid"),
        ("vp8", "vp8_cuvid"),
        ("mjpeg", "mjpeg_cuvid"),
        ("  H264  ", "h264_cuvid"),
    ],
)
def test_cuvid_decoder_probe_table_is_pinned(codec_name: str, decoder: str) -> None:
    """SEAM B3: the codec->cuvid decoder table (probe.py:111). Wave 2 reuses it
    verbatim to choose the subprocess decoder."""
    assert cuvid_decoder_for(codec_name) == decoder


@pytest.mark.parametrize("codec_name", ["mpeg4", "vc1", "", "  "])
def test_unsupported_codec_fails_closed_with_a_typed_error(codec_name: str) -> None:
    """SEAM B4: unknown codecs raise ``UnsupportedCodecError`` -- never a guessed
    decoder, never a software fallback (ADR-0002)."""
    with pytest.raises(UnsupportedCodecError) as raised:
        _ = cuvid_decoder_for(codec_name)
    assert raised.value.codec_name == codec_name.strip().lower()
    assert isinstance(raised.value, NvdecUnavailableError)


def test_ffprobe_argv_and_binary_derivation_are_pinned() -> None:
    """SEAM B5: the codec probe's own subprocess contract. Wave 2 keeps probing
    the source for width/height/codec before spawning the decoder."""
    # Given / When
    argv = ffprobe_args(_CONFIG)

    # Then
    assert argv == (
        "ffprobe",
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name",
        "-of",
        "json",
        "rtsp://camera.local/live",
    )
    assert ffprobe_binary("/opt/ffmpeg/bin/ffmpeg") == "/opt/ffmpeg/bin/ffprobe"
    assert ffprobe_binary("ffmpeg") == "ffprobe"
    assert ffprobe_binary("/usr/bin/avconv") == "ffprobe"


def test_probe_returns_real_stream_metadata_and_uses_the_open_timeout() -> None:
    """SEAM B6: probe output is parsed into ``StreamMetadata`` and the ffprobe
    call is bounded by ``open_timeout_ms``."""
    # Given
    seen: list[tuple[tuple[str, ...], float]] = []

    def runner(args: tuple[str, ...], timeout_sec: float) -> str:
        seen.append((args, timeout_sec))
        return '{"streams":[{"width":1920,"height":1080,"codec_name":"hevc"}]}'

    # When
    metadata = probe_stream_metadata(_CONFIG, runner=runner)

    # Then
    assert metadata == StreamMetadata(1920, 1080, "hevc")
    assert seen == [(ffprobe_args(_CONFIG), 4.0)]


def test_unparsable_probe_output_is_sanitized_and_never_spawns() -> None:
    """SEAM B7: malformed probe output fails closed with a sanitized message
    (no URL credentials leak) and no decode subprocess is created."""
    # Given
    secret_url = "rtsp://operator:s3cr3t@camera.local/live?token=plain"
    config = NvdecCuvidConfig(camera_id="camera-a", url=secret_url)
    spawned: list[tuple[str, ...]] = []

    def spawn(args: tuple[str, ...], frame_size: int) -> _FakeDecoderProcess:
        del frame_size
        spawned.append(args)
        return _FakeDecoderProcess([])

    adapter = NvdecCuvidAdapter(
        probe_runner=lambda args, timeout: "not-json",
        process_spawner=spawn,
    )

    # When / Then
    with pytest.raises(NvdecUnavailableError) as raised:
        _ = adapter.open(config)
    assert "ffprobe metadata unusable" in str(raised.value)
    assert "s3cr3t" not in str(raised.value)
    assert secret_url not in str(raised.value)
    assert spawned == []


def test_adapter_spawns_with_probed_frame_size_and_frames_real_rgb_bytes() -> None:
    """SEAM B8: raw-frame framing -- ``width * height * 3`` bytes per frame,
    reshaped to (H, W, 3) RGB. Wave 2 keeps this framing when the subprocess
    reads from a demux pipe instead of RTSP."""
    # Given
    first_payload = bytes((255, 0, 0, 0, 255, 0))
    second_payload = bytes((0, 0, 255, 255, 255, 255))
    process = _FakeDecoderProcess([first_payload, second_payload])
    spawned: list[tuple[tuple[str, ...], int]] = []

    def spawn(args: tuple[str, ...], frame_size: int) -> _FakeDecoderProcess:
        spawned.append((args, frame_size))
        return process

    adapter = NvdecCuvidAdapter(probe_runner=_probe_runner(), process_spawner=spawn)

    # When
    session = adapter.open(_CONFIG)
    first = session.read()
    second = session.read()
    exhausted = session.read()
    session.close()

    # Then
    assert isinstance(adapter, DecodeAdapter)
    assert isinstance(session, NvdecCuvidSession)
    assert isinstance(session, DecodeSession)
    assert spawned == [(ffmpeg_decode_args(_CONFIG, "h264_cuvid"), 6)]
    assert first is not None and second is not None
    assert first.frame.image.tolist() == [[[255, 0, 0], [0, 255, 0]]]
    assert second.frame.image.tolist() == [[[0, 0, 255], [255, 255, 255]]]
    assert (first.camera_id, first.width, first.height) == ("camera-a", 2, 1)
    assert (first.seq, second.seq) == (0, 1)
    assert exhausted is None
    assert process.reap_calls == 1


def test_short_frame_fails_closed_and_closes_the_session() -> None:
    """SEAM B9: a truncated rawvideo frame raises ``NvdecReadError`` carrying the
    exact expected/actual sizes and closes the session -- never a partial frame."""
    # Given
    process = _FakeDecoderProcess([bytes((1, 2, 3, 4))])
    adapter = NvdecCuvidAdapter(
        probe_runner=_probe_runner(),
        process_spawner=lambda args, frame_size: process,
    )
    session = adapter.open(_CONFIG)

    # When / Then
    with pytest.raises(NvdecReadError) as raised:
        _ = session.read()
    assert (raised.value.expected_size, raised.value.actual_size) == (6, 4)
    assert session.read() is None
    assert process.reap_calls == 1


def test_stdout_reader_reassembles_frames_across_arbitrary_chunk_boundaries() -> None:
    """SEAM B10: ``FFmpegDecodeProcess`` framing over a real byte pipe -- chunk
    boundaries never align with frame boundaries on a real pipe, so frames are
    reassembled from the byte stream, in order, with no loss."""
    # Given
    frame_size = 6
    frames = [bytes([index] * frame_size) for index in range(4)]
    stream = b"".join(frames)
    chunks = [stream[0:4], stream[4:9], stream[9:20], stream[20:]]
    child = _ChunkChild(_ChunkStream(list(chunks)))
    process = FFmpegDecodeProcess(child, frame_size)

    # When
    read = [process.read_frame(2.0) for _ in range(4)]
    trailing = process.read_frame(0.2)
    returncode = process.reap(1.0)

    # Then
    assert read == frames
    assert trailing is None
    assert returncode == 0
    assert child.stdout.closed

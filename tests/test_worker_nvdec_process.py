from __future__ import annotations

import subprocess
import sys
import threading
import time
from typing import final

import pytest

from worker.adapters.decode.nvdec_cuvid.errors import NvdecUnavailableError
from worker.adapters.decode.nvdec_cuvid.input_queue import DecoderInputQueue
from worker.adapters.decode.nvdec_cuvid.process import (
    FFmpegDecodeProcess,
    spawn_decoder_process,
)


@final
class _ChunkStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        del size
        return self._chunks.pop(0) if self._chunks else b""

    def close(self) -> None:
        self.closed = True


@final
class _CaptureInput:
    def __init__(self) -> None:
        self.payload = bytearray()
        self.closed = False

    def write(self, payload: bytes | memoryview) -> int:
        self.payload.extend(payload)
        return len(payload)

    def close(self) -> None:
        self.closed = True


@final
class _BlockingStream:
    def __init__(self) -> None:
        self._closed = threading.Event()
        self.read_entered = threading.Event()
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        del size
        self.read_entered.set()
        _ = self._closed.wait()
        return b""

    def close(self) -> None:
        self.closed = True
        self._closed.set()


@final
class _EventGatedStream:
    """Emit a prefix, then wait on a test-owned event before the remainder."""

    def __init__(self, prefix: bytes, remainder: bytes, resume: threading.Event) -> None:
        self._prefix = prefix
        self._remainder = remainder
        self._resume = resume
        self.prefix_emitted = threading.Event()
        self.closed = False
        self._phase = 0

    def read(self, size: int = -1) -> bytes:
        del size
        if self._phase == 0:
            self._phase = 1
            self.prefix_emitted.set()
            return self._prefix
        if self._phase == 1:
            self._resume.wait()
            self._phase = 2
            return self._remainder
        return b""

    def close(self) -> None:
        self.closed = True
        self._resume.set()


@final
class _Child:
    def __init__(
        self,
        stdout: _ChunkStream | _BlockingStream | _EventGatedStream,
        wait_outcomes: list[int | subprocess.TimeoutExpired] | None = None,
        stderr: _ChunkStream | _BlockingStream | None = None,
    ) -> None:
        self.stdin = _CaptureInput()
        self.stdout = stdout
        self.stderr = stderr
        self.returncode: int | None = None
        self._wait_outcomes = wait_outcomes or [0]
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        outcome = self._wait_outcomes.pop(0)
        if isinstance(outcome, subprocess.TimeoutExpired):
            raise outcome
        self.returncode = outcome
        return outcome


def test_decode_process_assembles_partial_chunks_through_bounded_queue() -> None:
    # Given
    child = _Child(_ChunkStream([b"\x00\x01", b"\x02\x03\x04\x05", b""]))
    process = FFmpegDecodeProcess(child, frame_size=6)

    # When
    payload = process.read_frame(timeout_sec=0.1)
    returncode = process.reap(timeout_sec=0.1)

    # Then
    assert payload == b"\x00\x01\x02\x03\x04\x05"
    assert process.queue_capacity == 2
    assert returncode == 0
    assert child.terminate_calls == 1
    assert child.kill_calls == 0
    assert child.stdout.closed is True
    assert process.reader_alive is False
    assert child.stdin.closed is True


def test_decode_process_read_timeout_leaves_child_and_pending_intact() -> None:
    # Given: a silent child that never emits a frame
    child = _Child(_BlockingStream())
    process = FFmpegDecodeProcess(child, frame_size=3)

    # When: the read deadline expires with an empty stdout queue
    started_at = time.monotonic()
    payload = process.read_frame(timeout_sec=0.02)

    # Then: timeout is non-terminal — child, reader, input, and pending survive
    assert payload is None
    assert time.monotonic() - started_at < 0.25
    assert child.terminate_calls == 0
    assert child.stdout.closed is False
    assert process.reader_alive is True
    assert child.stdin.closed is False
    assert bytes(process._pending) == b""
    assert process.failure_returncode is None
    assert process._child is child
    returncode = process.reap(timeout_sec=0.1)
    assert returncode == 0
    assert child.terminate_calls == 1


def test_decode_process_read_timeout_preserves_partial_bytes_until_resume() -> None:
    # Given: stdout emits two bytes, then waits on a test-owned event
    resume = threading.Event()
    stdout = _EventGatedStream(b"\x00\x01", b"\x02\x03\x04\x05", resume)
    child = _Child(stdout)
    process = FFmpegDecodeProcess(child, frame_size=6)
    assert stdout.prefix_emitted.wait(1.0)

    # When: the first read expires before the remaining four bytes arrive
    first = process.read_frame(timeout_sec=0.05)

    # Then: timeout returns None without reaping or clearing the prefix
    assert first is None
    assert child.terminate_calls == 0
    assert process.reader_alive is True
    assert child.stdin.closed is False
    assert child.stdout.closed is False
    assert bytes(process._pending) == b"\x00\x01"
    assert process.failure_returncode is None
    assert process._child is child

    # When: the remainder is released
    resume.set()
    second = process.read_frame(timeout_sec=1.0)

    # Then: the next read returns the exact six-byte frame
    assert second == b"\x00\x01\x02\x03\x04\x05"
    assert bytes(process._pending) == b""
    assert child.terminate_calls == 0
    assert process.failure_returncode is None
    returncode = process.reap(timeout_sec=0.1)
    assert returncode == 0
    assert child.terminate_calls == 1


def test_decode_process_eof_with_nonzero_returncode_sets_failure_and_is_loud() -> None:
    # Given: a genuine EOF after a partial frame, with a nonzero child exit
    child = _Child(_ChunkStream([b"\x00\x01", b""]), wait_outcomes=[7])
    process = FFmpegDecodeProcess(child, frame_size=6)
    decoder_input = DecoderInputQueue(process)

    # When: the reader sentinel arrives before a complete frame
    payload = process.read_frame(timeout_sec=1.0)

    # Then: EOF still reaps, clears pending, and surfaces the failure loudly
    assert payload == b"\x00\x01"
    assert bytes(process._pending) == b""
    assert child.terminate_calls == 1
    assert process.reader_alive is False
    assert child.stdin.closed is True
    assert process.failure_returncode == 7
    with pytest.raises(NvdecUnavailableError) as raised:
        decoder_input.raise_if_failed()
    assert raised.value.returncode == 7
    assert process.reap(timeout_sec=0.1) == 7
    decoder_input.abort()
    decoder_input.join(1.0)


def test_decode_process_escalates_to_kill_and_reaps_only_once() -> None:
    # Given
    timeout = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=0.01)
    child = _Child(_BlockingStream(), [timeout, 137])
    process = FFmpegDecodeProcess(child, frame_size=3)

    # When
    first_returncode = process.reap(timeout_sec=0.01)
    second_returncode = process.reap(timeout_sec=0.01)

    # Then
    assert first_returncode == 137
    assert second_returncode == 137
    assert child.terminate_calls == 1
    assert child.kill_calls == 1
    assert child.wait_calls == [0.01, None]
    assert child.stdout.closed is True
    assert process.reader_alive is False
    assert child.stdin.closed is True


def test_nonzero_exit_preserves_returncode_and_has_no_failure_detail() -> None:
    """Baseline: a dead child already exposes its returncode, but no stderr detail."""
    # Given
    child = _Child(_ChunkStream([b""]), wait_outcomes=[69])
    process = FFmpegDecodeProcess(child, frame_size=1)

    # When
    payload = process.read_frame(timeout_sec=0.1)

    # Then
    assert payload is None
    assert process.failure_returncode == 69
    assert getattr(process, "failure_detail", None) is None


def test_clean_exit_exposes_no_failure_returncode_or_detail() -> None:
    """Baseline: a clean child exit is not a failure and carries no detail."""
    # Given
    child = _Child(
        _ChunkStream([b""]),
        wait_outcomes=[0],
        stderr=_ChunkStream([b"should not become failure detail\n"]),
    )
    process = FFmpegDecodeProcess(child, frame_size=1)

    # When
    payload = process.read_frame(timeout_sec=0.1)
    returncode = process.reap(timeout_sec=0.1)

    # Then
    assert payload is None
    assert returncode == 0
    assert process.failure_returncode is None
    assert process.failure_detail is None
    assert child.stderr is not None
    assert child.stderr.closed is True
    assert process.stderr_drain_started is True
    assert process.stderr_drain_alive is False


def test_nonzero_child_exposes_returncode_and_final_safe_stderr_line() -> None:
    """A dead decoder must surface its exit code and last useful stderr line."""
    # Given
    child = _Child(
        _ChunkStream([b""]),
        wait_outcomes=[69],
        stderr=_ChunkStream(
            [b"ignored earlier warning\n", b"cuvid decode failed: codec not supported\n"]
        ),
    )
    process = FFmpegDecodeProcess(child, frame_size=1)

    # When
    payload = process.read_frame(timeout_sec=0.1)

    # Then
    assert payload is None
    assert process.failure_returncode == 69
    assert process.failure_detail == "cuvid decode failed: codec not supported"
    assert child.stdout.closed is True
    assert child.stderr is not None
    assert child.stderr.closed is True
    assert process.reader_alive is False
    assert process.stderr_drain_started is True
    assert process.stderr_drain_alive is False


def test_stderr_tail_is_bounded_redacted_and_truncated() -> None:
    """Keep only an 8 KiB tail, one safe line, and at most 512 rendered chars."""
    # Given: more than 8 KiB of noise, then one long credential-bearing line.
    diagnostic = "cuvid-diagnostic"
    last_line = (("N" * 600) + f" {diagnostic} rtsp://admin:secret@camera/token=abc").encode()
    stderr = _ChunkStream([b"n" * 9000 + b"\n", last_line + b"\n"])
    child = _Child(_ChunkStream([b""]), wait_outcomes=[1], stderr=stderr)
    process = FFmpegDecodeProcess(child, frame_size=1)

    # When
    _ = process.read_frame(timeout_sec=0.1)
    detail = process.failure_detail

    # Then
    assert detail is not None
    assert detail.startswith("[truncated] ")
    assert len(detail) <= 512
    assert diagnostic in detail
    assert "secret" not in detail
    assert "abc" not in detail
    assert "admin" not in detail
    assert "\n" not in detail
    assert "\r" not in detail
    assert len(process.retained_stderr) <= 8192


def test_unicode_c1_control_is_removed_from_safe_stderr_line() -> None:
    """UTF-8 C1 controls such as U+009B must not survive sanitization."""
    # Given: CSI-style C1 (U+009B) encoded as UTF-8 C2 9B, plus a diagnostic token.
    child = _Child(
        _ChunkStream([b""]),
        wait_outcomes=[1],
        stderr=_ChunkStream([b"cuvid error\xc2\x9b31mINJECT\n"]),
    )
    process = FFmpegDecodeProcess(child, frame_size=1)

    # When
    _ = process.read_frame(timeout_sec=0.1)
    detail = process.failure_detail

    # Then
    assert detail is not None
    assert "cuvid error" in detail
    assert "INJECT" in detail
    assert "\x9b" not in detail
    assert "\u009b" not in detail


def test_malformed_stderr_controls_and_repeated_reap_stay_safe() -> None:
    """Control bytes become spaces; a second reap keeps the first safe detail."""
    # Given
    child = _Child(
        _ChunkStream([b""]),
        wait_outcomes=[2],
        stderr=_ChunkStream([b"cuvid\x00failed\x07now\xff\n"]),
    )
    process = FFmpegDecodeProcess(child, frame_size=1)

    # When
    _ = process.read_frame(timeout_sec=0.1)
    first = process.failure_detail
    second_returncode = process.reap(timeout_sec=0.1)

    # Then
    assert first is not None
    assert "cuvid" in first
    assert "failed" in first
    assert "now" in first
    assert "\x00" not in first
    assert "\x07" not in first
    assert second_returncode == 2
    assert process.failure_detail == first
    assert child.terminate_calls == 1


def test_spawned_nonzero_child_exposes_safe_stderr_and_closes_streams() -> None:
    """The real PIPE drain must keep a failed child's last line."""
    # Given
    script = (
        "import sys\n"
        "sys.stderr.buffer.write(b'noise\\ncuvid error: decoder not found\\n')\n"
        "sys.stderr.buffer.flush()\n"
        "sys.exit(69)\n"
    )
    process = spawn_decoder_process((sys.executable, "-c", script), frame_size=4)
    assert isinstance(process, FFmpegDecodeProcess)

    # When
    payload = process.read_frame(timeout_sec=2.0)
    _ = process.reap(timeout_sec=0.5)

    # Then
    assert payload is None
    assert process.failure_returncode == 69
    assert process.failure_detail is not None
    assert "cuvid error: decoder not found" in process.failure_detail
    assert process.reader_alive is False
    assert process.stderr_drain_started is True
    assert process.stderr_drain_alive is False


def test_stderr_retention_keeps_strict_8192_byte_tail_across_3000_byte_chunks() -> None:
    """Retention must equal the final 8192 bytes, not a whole-chunk drop."""
    # Given: legal 3000-byte chunks whose join is 9000 bytes.
    chunks = [b"a" * 3000, b"b" * 3000, b"c" * 3000]
    payload = b"".join(chunks)
    child = _Child(
        _ChunkStream([b""]),
        wait_outcomes=[1],
        stderr=_ChunkStream(list(chunks)),
    )
    process = FFmpegDecodeProcess(child, frame_size=1)
    assert process.join_stderr_drain(1.0) is True

    # When
    retained = process.retained_stderr

    # Then
    assert retained == payload[-8192:]
    assert len(retained) == 8192
    _ = process.reap(timeout_sec=0.1)


def test_reap_unblocks_blocked_stderr_drain_and_joins_before_return() -> None:
    """Close the blocked pipe first, then join; reap must not leave the drain up."""
    # Given
    stderr = _BlockingStream()
    child = _Child(_ChunkStream([b""]), wait_outcomes=[7], stderr=stderr)
    process = FFmpegDecodeProcess(child, frame_size=1)
    assert process.stderr_drain_started is True
    assert stderr.read_entered.wait(1.0)
    assert process.stderr_drain_alive is True

    # When
    returncode = process.reap(timeout_sec=0.1)

    # Then: stream close and drain death are the deterministic outcomes.
    assert returncode == 7
    assert stderr.closed is True
    assert process.stderr_drain_alive is False
    assert process.join_stderr_drain(2.0) is True

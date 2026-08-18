from __future__ import annotations

import subprocess
import threading
import time
from typing import final

import pytest

from worker.adapters.decode.nvdec_cuvid.errors import NvdecUnavailableError
from worker.adapters.decode.nvdec_cuvid.input_queue import DecoderInputQueue
from worker.adapters.decode.nvdec_cuvid.process import FFmpegDecodeProcess


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
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        del size
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
        stdout: _ChunkStream | _BlockingStream,
        wait_outcomes: list[int | subprocess.TimeoutExpired] | None = None,
    ) -> None:
        self.stdin = _CaptureInput()
        self.stdout = stdout
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

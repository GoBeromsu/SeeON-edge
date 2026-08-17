from __future__ import annotations

import subprocess
import threading
import time
from typing import final

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


def test_decode_process_read_timeout_reaps_hanging_child_and_reader() -> None:
    # Given
    child = _Child(_BlockingStream())
    process = FFmpegDecodeProcess(child, frame_size=3)

    # When
    started_at = time.monotonic()
    payload = process.read_frame(timeout_sec=0.02)

    # Then
    assert payload is None
    assert time.monotonic() - started_at < 0.25
    assert child.terminate_calls == 1
    assert child.stdout.closed is True
    assert process.reader_alive is False
    assert child.stdin.closed is True


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

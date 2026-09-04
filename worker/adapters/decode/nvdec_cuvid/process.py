from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
import unicodedata
from collections import deque
from contextlib import suppress
from typing import IO, Final, Protocol, final

from worker.adapters.decode.nvdec_cuvid.errors import (
    NvdecConfigError,
    NvdecUnavailableError,
    sanitized_nvdec_error,
)

_READ_QUEUE_CAPACITY: Final = 2
_QUEUE_PUT_TIMEOUT_SEC: Final = 0.05
_MIN_READER_JOIN_TIMEOUT_SEC: Final = 0.1
_STDERR_DRAIN_JOIN_TIMEOUT_SECONDS: Final = 2.0
_STDERR_DRAIN_CHUNK_BYTES: Final = 4096
_STDERR_TAIL_MAX_BYTES: Final = 8192
_STDERR_RENDER_MAX_CHARS: Final = 512
_STDERR_TRUNCATION_PREFIX: Final = "[truncated] "
_USERINFO_RE: Final = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)([^/@\s]+)@")
_SECRET_QUERY_KEYS: Final = "token|password|passwd|pwd|auth|key|secret|credential"
_SECRET_ASSIGNMENT_RE: Final = re.compile(rf"(?i)((?:^|[?&/;])(?:{_SECRET_QUERY_KEYS})=)([^&\s#]*)")


class ReadablePipe(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


class WritablePipe(Protocol):
    def write(self, payload: bytes | memoryview) -> int: ...

    def close(self) -> None: ...


class DecoderChild(Protocol):
    @property
    def stdin(self) -> WritablePipe | None: ...

    @property
    def stdout(self) -> ReadablePipe | None: ...

    @property
    def stderr(self) -> ReadablePipe | None: ...

    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class DecoderProcess(Protocol):
    def write_packet(self, payload: bytes) -> None: ...

    def close_input(self) -> None: ...

    def read_frame(self, timeout_sec: float) -> bytes | None: ...

    def reap(self, timeout_sec: float) -> int | None: ...


class ProcessSpawner(Protocol):
    def __call__(self, args: tuple[str, ...], frame_size: int, /) -> DecoderProcess: ...


@final
class _PopenReadablePipe:
    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        self._stream.close()


@final
class _PopenWritablePipe:
    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream

    def write(self, payload: bytes | memoryview) -> int:
        return self._stream.write(payload)

    def close(self) -> None:
        self._stream.close()


@final
class _PopenDecoderChild:
    def __init__(self, child: subprocess.Popen[bytes]) -> None:
        self._child = child
        self._stdin = None if child.stdin is None else _PopenWritablePipe(child.stdin)
        self._stdout = None if child.stdout is None else _PopenReadablePipe(child.stdout)
        self._stderr = None if child.stderr is None else _PopenReadablePipe(child.stderr)

    @property
    def stdin(self) -> WritablePipe | None:
        return self._stdin

    @property
    def stdout(self) -> ReadablePipe | None:
        return self._stdout

    @property
    def stderr(self) -> ReadablePipe | None:
        return self._stderr

    @property
    def returncode(self) -> int | None:
        return self._child.returncode

    def terminate(self) -> None:
        self._child.terminate()

    def kill(self) -> None:
        self._child.kill()

    def wait(self, timeout: float | None = None) -> int:
        return self._child.wait(timeout=timeout)


@final
class FFmpegDecodeProcess:
    """Own one decoder child, its packet input, and bounded stdout reader."""

    def __init__(self, child: DecoderChild, frame_size: int) -> None:
        if frame_size <= 0:
            raise NvdecConfigError("frame size must be positive")
        self._child: DecoderChild | None = child
        self._input = child.stdin
        self._returncode: int | None = None
        self._failure_returncode: int | None = None
        self._failure_detail: str | None = None
        self._frame_size = frame_size
        self._chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=_READ_QUEUE_CAPACITY)
        self._pending = bytearray()
        self._stop_reader = threading.Event()
        self._reap_lock = threading.Lock()
        self._stderr_chunks: deque[bytes] = deque()
        self._stderr_tail_len = 0
        self._stderr_lock = threading.Lock()
        self._reader_thread = threading.Thread(
            target=self._pump_stdout,
            args=(child.stdout,),
            name="nvdec-cuvid-stdout-reader",
            daemon=True,
        )
        self._reader_thread.start()
        stderr = getattr(child, "stderr", None)
        if stderr is None:
            self._stderr_thread = None
        else:
            self._stderr_thread = threading.Thread(
                target=self._pump_stderr,
                args=(stderr,),
                name="nvdec-cuvid-stderr-drain",
                daemon=True,
            )
            self._stderr_thread.start()

    @property
    def queue_capacity(self) -> int:
        return self._chunks.maxsize

    @property
    def reader_alive(self) -> bool:
        return self._reader_thread.is_alive()

    @property
    def stderr_drain_started(self) -> bool:
        return self._stderr_thread is not None

    @property
    def stderr_drain_alive(self) -> bool:
        thread = self._stderr_thread
        return thread is not None and thread.is_alive()

    @property
    def retained_stderr(self) -> bytes:
        return self._stderr_tail()

    def join_stderr_drain(self, timeout_sec: float) -> bool:
        thread = self._stderr_thread
        if thread is None:
            return True
        thread.join(timeout=timeout_sec)
        return not thread.is_alive()

    @property
    def failure_returncode(self) -> int | None:
        return self._failure_returncode

    @property
    def failure_detail(self) -> str | None:
        return self._failure_detail

    def write_packet(self, payload: bytes) -> None:
        if not payload:
            raise NvdecConfigError("compressed packet must not be empty")
        stream = self._input
        if stream is None:
            raise NvdecUnavailableError("ffmpeg decoder input is closed")
        remaining = memoryview(payload)
        try:
            while remaining:
                written = stream.write(remaining)
                if written <= 0:
                    _raise_zero_byte_write()
                remaining = remaining[written:]
        except (BrokenPipeError, OSError, ValueError) as error:
            raise sanitized_nvdec_error("ffmpeg packet write failed", error) from None

    def close_input(self) -> None:
        stream = self._input
        self._input = None
        if stream is not None:
            with suppress(OSError, ValueError):
                stream.close()

    def read_frame(self, timeout_sec: float) -> bytes | None:
        if timeout_sec <= 0:
            raise NvdecConfigError("read timeout must be positive")
        if self._child is None:
            return None
        deadline = time.monotonic() + timeout_sec
        while len(self._pending) < self._frame_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._finish_read(timeout_sec, timed_out=True)
            try:
                chunk = self._chunks.get(timeout=remaining)
            except queue.Empty:
                return self._finish_read(timeout_sec, timed_out=True)
            if chunk is None:
                return self._finish_read(timeout_sec, timed_out=False)
            self._pending.extend(chunk)
        payload = bytes(self._pending[: self._frame_size])
        del self._pending[: self._frame_size]
        return payload

    def _finish_read(self, timeout_sec: float, *, timed_out: bool) -> bytes | None:
        if timed_out:
            return None
        partial = bytes(self._pending)
        self._pending.clear()
        returncode = self.reap(timeout_sec=min(timeout_sec, _QUEUE_PUT_TIMEOUT_SEC))
        if returncode not in (None, 0):
            self._failure_returncode = returncode
        return partial or None

    def reap(self, timeout_sec: float) -> int | None:
        if timeout_sec <= 0:
            raise NvdecConfigError("reap timeout must be positive")
        with self._reap_lock:
            child = self._child
            if child is None:
                return self._returncode
            self._child = None
            self._stop_reader.set()
            with suppress(OSError):
                child.terminate()
            try:
                returncode = child.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                with suppress(OSError):
                    child.kill()
                try:
                    returncode = child.wait()
                except OSError:
                    returncode = child.returncode
            except OSError:
                returncode = child.returncode
            self.close_input()
            stream = child.stdout
            if stream is not None:
                with suppress(OSError, ValueError):
                    stream.close()
            self._reader_thread.join(timeout=max(timeout_sec, _MIN_READER_JOIN_TIMEOUT_SEC))
            stderr = getattr(child, "stderr", None)
            if stderr is not None:
                with suppress(OSError, ValueError):
                    stderr.close()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=_STDERR_DRAIN_JOIN_TIMEOUT_SECONDS)
            if returncode not in (None, 0) and self._failure_detail is None:
                self._failure_detail = _render_safe_stderr_line(self._stderr_tail())
            self._returncode = returncode
            return returncode

    def _pump_stdout(self, stream: ReadablePipe | None) -> None:
        if stream is None:
            self._put_chunk(None)
            return
        try:
            while not self._stop_reader.is_set():
                chunk = stream.read(self._frame_size)
                if not chunk:
                    self._put_chunk(None)
                    return
                self._put_chunk(chunk)
        except (OSError, ValueError):
            self._put_chunk(None)

    def _put_chunk(self, chunk: bytes | None) -> None:
        while not self._stop_reader.is_set():
            try:
                self._chunks.put(chunk, timeout=_QUEUE_PUT_TIMEOUT_SEC)
            except queue.Full:
                continue
            return

    def _pump_stderr(self, stream: ReadablePipe) -> None:
        try:
            while True:
                chunk = stream.read(_STDERR_DRAIN_CHUNK_BYTES)
                if not chunk:
                    return
                self._retain_stderr_tail(chunk)
        except (OSError, ValueError):
            return

    def _retain_stderr_tail(self, chunk: bytes) -> None:
        with self._stderr_lock:
            self._stderr_chunks.append(chunk)
            self._stderr_tail_len += len(chunk)
            while self._stderr_tail_len > _STDERR_TAIL_MAX_BYTES and self._stderr_chunks:
                overflow = self._stderr_tail_len - _STDERR_TAIL_MAX_BYTES
                oldest = self._stderr_chunks[0]
                if overflow >= len(oldest):
                    dropped = self._stderr_chunks.popleft()
                    self._stderr_tail_len -= len(dropped)
                    continue
                self._stderr_chunks[0] = oldest[overflow:]
                self._stderr_tail_len = _STDERR_TAIL_MAX_BYTES

    def _stderr_tail(self) -> bytes:
        with self._stderr_lock:
            return b"".join(self._stderr_chunks)


def _raise_zero_byte_write() -> None:
    raise BrokenPipeError("ffmpeg decoder accepted zero packet bytes")


def _render_safe_stderr_line(payload: bytes) -> str | None:
    if not payload:
        return None
    text = payload.decode("utf-8", errors="replace")
    last_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            last_line = stripped
    if not last_line:
        return None
    sanitized = "".join(
        " " if unicodedata.category(character) == "Cc" else character for character in last_line
    )
    redacted = _USERINFO_RE.sub(r"\1***:***@", sanitized)
    redacted = _SECRET_ASSIGNMENT_RE.sub(r"\1***", redacted)
    if len(redacted) <= _STDERR_RENDER_MAX_CHARS:
        return redacted
    kept = _STDERR_RENDER_MAX_CHARS - len(_STDERR_TRUNCATION_PREFIX)
    return f"{_STDERR_TRUNCATION_PREFIX}{redacted[-kept:]}"


def spawn_decoder_process(args: tuple[str, ...], frame_size: int) -> DecoderProcess:
    try:
        child = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as error:
        raise sanitized_nvdec_error("ffmpeg spawn failed", error) from None
    return FFmpegDecodeProcess(_PopenDecoderChild(child), frame_size)


__all__ = [
    "DecoderChild",
    "DecoderProcess",
    "FFmpegDecodeProcess",
    "ProcessSpawner",
    "ReadablePipe",
    "WritablePipe",
    "spawn_decoder_process",
]

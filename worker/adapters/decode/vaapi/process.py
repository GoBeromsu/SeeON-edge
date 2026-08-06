from __future__ import annotations

import queue
import subprocess
import threading
import time
from contextlib import suppress
from typing import IO, Final, Protocol, final

from worker.adapters.decode.vaapi.errors import (
    VaapiConfigError,
    sanitized_vaapi_error,
)

_READ_QUEUE_CAPACITY: Final = 2
_QUEUE_PUT_TIMEOUT_SEC: Final = 0.05
_MIN_READER_JOIN_TIMEOUT_SEC: Final = 0.1


class ReadablePipe(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


class DecoderChild(Protocol):
    @property
    def stdout(self) -> ReadablePipe | None: ...

    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class DecoderProcess(Protocol):
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
class _PopenDecoderChild:
    def __init__(self, child: subprocess.Popen[bytes]) -> None:
        self._child = child
        stream = child.stdout
        self._stdout = None if stream is None else _PopenReadablePipe(stream)

    @property
    def stdout(self) -> ReadablePipe | None:
        return self._stdout

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
    """Own one mutable decoder child and its bounded stdout reader."""

    def __init__(self, child: DecoderChild, frame_size: int) -> None:
        if frame_size <= 0:
            raise VaapiConfigError("frame size must be positive")
        self._child: DecoderChild | None = child
        self._returncode: int | None = None
        self._frame_size = frame_size
        self._chunks: queue.Queue[bytes | None] = queue.Queue(
            maxsize=_READ_QUEUE_CAPACITY
        )
        self._pending = bytearray()
        self._stop_reader = threading.Event()
        self._reap_lock = threading.Lock()
        self._reader_thread = threading.Thread(
            target=self._pump_stdout,
            args=(child.stdout,),
            name="vaapi-stdout-reader",
            daemon=True,
        )
        self._reader_thread.start()

    @property
    def queue_capacity(self) -> int:
        return self._chunks.maxsize

    @property
    def reader_alive(self) -> bool:
        return self._reader_thread.is_alive()

    def read_frame(self, timeout_sec: float) -> bytes | None:
        if timeout_sec <= 0:
            raise VaapiConfigError("read timeout must be positive")
        if self._child is None:
            return None
        deadline = time.monotonic() + timeout_sec
        while len(self._pending) < self._frame_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _ = self.reap(timeout_sec=min(timeout_sec, _QUEUE_PUT_TIMEOUT_SEC))
                return None
            try:
                chunk = self._chunks.get(timeout=remaining)
            except queue.Empty:
                _ = self.reap(timeout_sec=min(timeout_sec, _QUEUE_PUT_TIMEOUT_SEC))
                return None
            if chunk is None:
                _ = self.reap(timeout_sec=min(timeout_sec, _QUEUE_PUT_TIMEOUT_SEC))
                return None
            self._pending.extend(chunk)
        payload = bytes(self._pending[: self._frame_size])
        del self._pending[: self._frame_size]
        return payload

    def reap(self, timeout_sec: float) -> int | None:
        if timeout_sec <= 0:
            raise VaapiConfigError("reap timeout must be positive")
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
            stream = child.stdout
            if stream is not None:
                with suppress(OSError, ValueError):
                    stream.close()
            self._reader_thread.join(
                timeout=max(timeout_sec, _MIN_READER_JOIN_TIMEOUT_SEC)
            )
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


def spawn_decoder_process(args: tuple[str, ...], frame_size: int) -> DecoderProcess:
    try:
        child = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
    except OSError as error:
        raise sanitized_vaapi_error("ffmpeg spawn failed", error) from None
    return FFmpegDecodeProcess(_PopenDecoderChild(child), frame_size)


__all__ = [
    "DecoderChild",
    "DecoderProcess",
    "FFmpegDecodeProcess",
    "ProcessSpawner",
    "ReadablePipe",
    "spawn_decoder_process",
]

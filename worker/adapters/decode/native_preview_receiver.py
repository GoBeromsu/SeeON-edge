"""Bounded native NVJPEG frames into the existing latest-frame surface."""

from __future__ import annotations

import socket
import struct
import threading
from contextlib import suppress
from typing import Final, Protocol, final

_HEADER: Final = struct.Struct("<4sIQQHI")
_MAX_PREVIEW_BYTES: Final = 2 * 1024 * 1024


class NativePreviewSink(Protocol):
    def register_camera(self, camera_id: str) -> None: ...
    def publish_jpeg(
        self, camera_id: str, jpeg: bytes, *, frame_index: int, seq: int | None = None
    ) -> None: ...


@final
class NativePreviewReceiver:
    def __init__(self, endpoint: socket.socket, store: NativePreviewSink) -> None:
        self._endpoint = endpoint
        self._store = store
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="deepstream-preview-receiver", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with suppress(OSError):
            self._endpoint.shutdown(socket.SHUT_RDWR)
        self._endpoint.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                header = _recv_exact(self._endpoint, _HEADER.size)
                magic, body_size, sequence, _pts, camera_size, jpeg_size = _HEADER.unpack(header)
                if (
                    magic != b"SJP1"
                    or body_size > _MAX_PREVIEW_BYTES
                    or camera_size + jpeg_size != body_size
                ):
                    return
                body = _recv_exact(self._endpoint, body_size)
                camera = body[:camera_size].decode()
                jpeg = body[camera_size:]
                if not camera or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                    continue
            except (ConnectionError, OSError, UnicodeDecodeError):
                return
            self._store.register_camera(camera)
            self._store.publish_jpeg(camera, jpeg, frame_index=sequence, seq=sequence)


def _recv_exact(endpoint: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = endpoint.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("native preview stream closed")
        chunks.extend(chunk)
    return bytes(chunks)


__all__ = ["NativePreviewReceiver", "NativePreviewSink"]

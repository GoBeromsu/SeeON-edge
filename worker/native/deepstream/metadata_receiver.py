"""Resilient metadata wake consumer over inherited or owner-only AF_UNIX datagrams."""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Protocol, final, override

from worker.native.deepstream.ipc import MetadataFrame
from worker.native.deepstream.metadata_slot import LatestMetadataSlot

_MAX_DATAGRAM: Final = 512


def _require_socket(endpoint: object) -> socket.socket:
    if not isinstance(endpoint, socket.socket):
        raise TypeError("inherited datagram wake socket required")
    return endpoint


@dataclass(frozen=True, slots=True)
class MetadataPullFailure(Exception):
    code: str

    @override
    def __str__(self) -> str:
        return self.code


class MetadataPullStopped(Exception):
    """Confirmed child/control EOF; the dark runner must fail fast."""


class MetadataPuller(Protocol):
    def pull_latest(self, camera_id: str) -> MetadataFrame | None: ...


@final
class MetadataReceiver:
    """Pull latest metadata after wakeups; recover isolated malformed/pull failures."""

    def __init__(
        self,
        endpoint: socket.socket,
        slot: LatestMetadataSlot,
        puller: MetadataPuller,
    ) -> None:
        self._endpoint = _require_socket(endpoint)
        self._slot = slot
        self._puller = puller
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._fatal = threading.Event()

    @property
    def fatal_event(self) -> threading.Event:
        return self._fatal

    def __enter__(self) -> MetadataReceiver:
        self._socket = self._endpoint
        self._thread = threading.Thread(target=self._run, name="deepstream-metadata", daemon=True)
        self._thread.start()
        return self

    def pull_now(self, camera_id: str) -> MetadataFrame | None:
        if camera_id == "":
            self._slot.mark_malformed()
            return None
        try:
            metadata = self._puller.pull_latest(camera_id)
        except MetadataPullFailure:
            self._slot.mark_pull_failure()
            return None
        except MetadataPullStopped:
            self._fatal.set()
            raise
        if metadata is None:
            return None
        _ = self._slot.publish(metadata)
        return metadata

    def _run(self) -> None:
        receiver = self._socket
        if receiver is None:
            return
        while not self._stopping.is_set():
            try:
                data = receiver.recv(_MAX_DATAGRAM)
            except OSError:
                if not self._stopping.is_set():
                    self._fatal.set()
                return
            if self._stopping.is_set():
                return
            if data == b"":
                self._slot.mark_malformed()
                continue
            try:
                camera_id = data.decode()
            except UnicodeDecodeError:
                self._slot.mark_malformed()
            else:
                try:
                    _ = self.pull_now(camera_id)
                except MetadataPullStopped:
                    return

    def close(self) -> None:
        self._stopping.set()
        receiver, self._socket = self._socket, None
        if receiver is not None:
            try:
                receiver.shutdown(socket.SHUT_RD)
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if receiver is not None:
            try:
                receiver.close()
            except OSError:
                pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "MetadataPullFailure",
    "MetadataPuller",
    "MetadataPullStopped",
    "MetadataReceiver",
]

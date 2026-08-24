"""Reliable inherited native-failure receiver owned by the Python supervisor."""

from __future__ import annotations

import socket
import threading
import uuid
from collections.abc import Callable
from typing import Final, Literal, assert_never, final

from worker.native.deepstream.ipc import IpcProtocolError, MessageKind, decode_control_message

_MAX_FAILURE: Final = 65_535
FailureKind = Literal[MessageKind.SOURCE_FAILURE, MessageKind.FATAL]


def _failure_kind(kind: MessageKind) -> FailureKind | None:
    match kind:
        case MessageKind.SOURCE_FAILURE | MessageKind.FATAL:
            return kind
        case _:
            return None


@final
class NativeFailureReceiver:
    def __init__(
        self,
        receiver: socket.socket,
        worker_boot_id: uuid.UUID,
        child_instance_id: uuid.UUID,
        on_source: Callable[[str, str], None],
        on_fatal: Callable[[str], None],
    ) -> None:
        self._receiver = receiver
        self._worker_boot_id = worker_boot_id
        self._child_instance_id = child_instance_id
        self._on_source = on_source
        self._on_fatal = on_fatal
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="deepstream-native-failures",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stopping.set()
        try:
            self._receiver.shutdown(socket.SHUT_RD)
        except OSError:
            pass
        self._receiver.close()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                raw = self._receiver.recv(_MAX_FAILURE)
            except InterruptedError:
                continue
            except OSError:
                if not self._stopping.is_set():
                    self._on_fatal("failure_ipc")
                return
            if raw == b"":
                if not self._stopping.is_set():
                    self._on_fatal("child_exit")
                return
            try:
                message = decode_control_message(raw)
                category = message.payload.decode()
            except (IpcProtocolError, UnicodeDecodeError):
                self._on_fatal("failure_ipc")
                return
            if (
                message.worker_boot_id != self._worker_boot_id
                or message.child_instance_id != self._child_instance_id
            ):
                self._on_fatal("failure_identity")
                return
            kind = _failure_kind(message.kind)
            if kind is None:
                self._on_fatal("failure_kind")
                return
            match kind:
                case MessageKind.SOURCE_FAILURE:
                    self._on_source(message.camera_id, category)
                    continue
                case MessageKind.FATAL:
                    self._on_fatal(category)
                    return
            assert_never(kind)


__all__ = ["NativeFailureReceiver"]

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from worker.types import FramePacket

_DecodeConfigT = TypeVar("_DecodeConfigT", contravariant=True)


@runtime_checkable
class DecodeSession(Protocol):
    def read(self) -> FramePacket | None: ...

    def close(self) -> None: ...


@runtime_checkable
class StreamIdentityDecodeSession(DecodeSession, Protocol):
    def set_stream_identity(self, worker_boot_id: str, stream_epoch: int) -> None: ...


@runtime_checkable
class DecodeAdapter(Protocol[_DecodeConfigT]):
    def open(self, config: _DecodeConfigT) -> DecodeSession: ...


__all__ = ["DecodeAdapter", "DecodeSession", "StreamIdentityDecodeSession"]

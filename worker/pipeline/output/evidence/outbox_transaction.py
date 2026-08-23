"""The filesystem delivery queue publishes entries atomically."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Self


class ImmediateTransaction(AbstractContextManager["ImmediateTransaction"]):
    """Compatibility context for callers that no longer own durable state."""

    def __init__(self, _connection: object) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args


def write_transaction(connection: object) -> ImmediateTransaction:
    return ImmediateTransaction(connection)


__all__ = ["ImmediateTransaction", "write_transaction"]

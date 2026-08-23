"""Removed SQLite outbox database boundary."""

from __future__ import annotations


def open_connection(_path: object) -> object:
    raise RuntimeError("the evidence outbox uses the delivery queue, not a database")

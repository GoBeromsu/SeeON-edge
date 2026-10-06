"""Refusals raised by the SQLite-to-PostgreSQL cutover tooling."""

from __future__ import annotations


class MigrationError(RuntimeError):
    """The cutover step refused; nothing past the last durable step changed."""


__all__ = ["MigrationError"]

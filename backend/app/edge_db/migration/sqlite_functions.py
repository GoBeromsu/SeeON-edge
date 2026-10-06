"""Deterministic SQLite functions the migration registers on source connections."""

from __future__ import annotations

import sqlite3

from backend.app.edge_db.functions import audit_record_hash


def register_edge_db_functions(connection: sqlite3.Connection) -> None:
    """Install deterministic helpers. This is not DDL."""
    connection.create_function("seeon_audit_record_hash", 2, audit_record_hash, deterministic=True)


__all__ = ["register_edge_db_functions"]

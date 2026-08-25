"""Deterministic SQLite functions registered on migrator and runtime connections."""

from __future__ import annotations

import hashlib
import json
import sqlite3


def audit_record_hash(previous_hash: str, payload_json: str) -> str:
    """SHA-256(previous-hash bytes || canonical sorted-key compact UTF-8 JSON)."""
    payload = json.loads(payload_json)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(bytes.fromhex(previous_hash) + canonical).hexdigest()


def register_edge_db_functions(connection: sqlite3.Connection) -> None:
    """Install deterministic helpers. This is not DDL."""
    connection.create_function("seeon_audit_record_hash", 2, audit_record_hash, deterministic=True)


__all__ = ["audit_record_hash", "register_edge_db_functions"]

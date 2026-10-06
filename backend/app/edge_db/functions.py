"""Audit-chain record hash shared by the PostgreSQL audit store and the migration."""

from __future__ import annotations

import hashlib
import json


def audit_record_hash(previous_hash: str, payload_json: str) -> str:
    """SHA-256(previous-hash bytes || canonical sorted-key compact UTF-8 JSON)."""
    payload = json.loads(payload_json)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(bytes.fromhex(previous_hash) + canonical).hexdigest()


__all__ = ["audit_record_hash"]

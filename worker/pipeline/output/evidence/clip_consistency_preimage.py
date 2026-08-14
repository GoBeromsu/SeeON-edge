"""Deterministic logical database preimages for repair recovery boundaries."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from typing import TypeAlias, cast

from worker.pipeline.output.evidence.clip_consistency_schema import schema_fingerprint

_JsonValue: TypeAlias = None | bool | int | float | str | list["_JsonValue"]


def non_relation_preimage_sha256(connection: sqlite3.Connection) -> str:
    """Hash schema plus every logical row outside the repair-owned relation table."""
    tables = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name != 'clip_events' ORDER BY name"
        )
    )
    payload: list[_JsonValue] = [_json_value(schema_fingerprint(connection))]
    for table in tables:
        identifier = '"' + table.replace('"', '""') + '"'
        encoded_rows = [
            [_encode_sqlite_value(value) for value in row]
            for row in connection.execute(f"SELECT * FROM {identifier}")
        ]
        encoded_rows.sort(key=_canonical_json)
        table_payload: list[_JsonValue] = [table, _json_value(encoded_rows)]
        payload.append(table_payload)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _encode_sqlite_value(value: object) -> _JsonValue:
    if value is None:
        return ["null"]
    if isinstance(value, int):
        return ["integer", str(value)]
    if isinstance(value, float):
        return ["real", value.hex()]
    if isinstance(value, str):
        return ["text", value]
    if isinstance(value, bytes):
        return ["blob", base64.b64encode(value).decode("ascii")]
    raise TypeError(f"unsupported SQLite value: {type(value).__name__}")


def _json_value(value: object) -> _JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        items = cast(tuple[object, ...] | list[object], value)
        return [_json_value(item) for item in items]
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def _canonical_json(value: _JsonValue) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


__all__ = ["non_relation_preimage_sha256"]

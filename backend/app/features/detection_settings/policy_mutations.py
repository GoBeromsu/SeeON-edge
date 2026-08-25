"""Current-plus-previous policy mutation SQL."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from backend.app.edge_db.configuration import utc_now
from backend.app.features.detection_settings.policy_rows import (
    PolicyRecord,
    decode_policy_values,
)
from shared.detection_policies import NumericPolicy, PolicyDocumentError, policy_values_dict


@dataclass(frozen=True, slots=True)
class PolicyWrite:
    facility_id: str
    camera_id: str | None
    module_id: str
    module_version: int
    schema_id: str
    schema_version: int
    active_values: NumericPolicy | None
    previous_present: bool
    previous_values: NumericPolicy | None
    generation: int


def save_policy(
    connection: sqlite3.Connection,
    raw: tuple[object, ...] | None,
    write: PolicyWrite,
) -> int:
    active_json, active_hash = encode_policy(write.active_values)
    previous_json, previous_hash = encode_policy(write.previous_values)
    now = utc_now()
    params = (
        write.facility_id,
        write.camera_id,
        write.module_id,
        write.module_version,
        write.schema_id,
        write.schema_version,
        active_json,
        active_hash,
        int(write.previous_present),
        previous_json,
        previous_hash,
        write.generation,
        now,
        now,
    )
    if raw is None:
        cursor = connection.execute(
            "INSERT INTO policies(facility_id,camera_id,module_id,module_version,schema_id,"
            "schema_version,active_values_json,active_content_sha256,previous_present,"
            "previous_values_json,previous_content_sha256,activation_generation,status,"
            "activated_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'pending',?,?)",
            params,
        )
        if cursor.lastrowid is None:
            raise PolicyDocumentError("policy insert did not return a row id")
        return cursor.lastrowid
    policy_id = int(raw[0])
    connection.execute(
        "UPDATE policies SET schema_id=?,schema_version=?,active_values_json=?,"
        "active_content_sha256=?,previous_present=?,previous_values_json=?,"
        "previous_content_sha256=?,activation_generation=?,status='pending',"
        "refusal_reason=NULL,activated_at=?,applied_at=NULL,updated_at=? WHERE policy_id=?",
        (params[4], params[5], *params[6:12], now, now, policy_id),
    )
    return policy_id


def encode_policy(values: NumericPolicy | None) -> tuple[str | None, str | None]:
    if values is None:
        return None, None
    encoded = json.dumps(policy_values_dict(values), sort_keys=True, separators=(",", ":"))
    return encoded, hashlib.sha256(encoded.encode()).hexdigest()


def previous_state(
    raw: tuple[object, ...] | None,
    record: PolicyRecord | None,
    camera_id: str | None,
) -> tuple[bool, NumericPolicy | None]:
    if record is not None and record.status != "failed":
        return True, record.active_values
    if raw is not None:
        previous = None if raw[10] is None else decode_policy_values(raw[10], raw[11], raw)
        return bool(raw[9]), previous
    return camera_id is not None, None


def next_generation(connection: sqlite3.Connection, facility_id: str) -> int:
    row = connection.execute(
        "SELECT max(activation_generation) FROM policies WHERE facility_id=?", (facility_id,)
    ).fetchone()
    return 1 if row is None or row[0] is None else int(row[0]) + 1

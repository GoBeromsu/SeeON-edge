"""Current-plus-previous policy mutation SQL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

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
    connection: psycopg.Connection,
    raw: Mapping[str, Any] | None,
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
        with connection.cursor(row_factory=dict_row) as cursor:
            row = cursor.execute(
                "INSERT INTO policies(facility_id,camera_id,module_id,module_version,schema_id,"
                "schema_version,active_values_json,active_content_sha256,previous_present,"
                "previous_values_json,previous_content_sha256,activation_generation,status,"
                "activated_at,updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'pending',%s,%s) RETURNING policy_id",
                params,
            ).fetchone()
        if row is None:
            raise PolicyDocumentError("policy insert did not return a row id")
        return int(row["policy_id"])
    with connection.cursor(row_factory=dict_row) as cursor:
        row = cursor.execute(
            "UPDATE policies SET schema_id=%s,schema_version=%s,active_values_json=%s,"
            "active_content_sha256=%s,previous_present=%s,previous_values_json=%s,"
            "previous_content_sha256=%s,activation_generation=%s,status='pending',"
            "refusal_reason=NULL,activated_at=%s,applied_at=NULL,updated_at=%s "
            "WHERE policy_id=%s RETURNING policy_id",
            (params[4], params[5], *params[6:12], now, now, int(raw["policy_id"])),
        ).fetchone()
    if row is None:
        raise PolicyDocumentError("policy row is missing")
    return int(row["policy_id"])


def encode_policy(values: NumericPolicy | None) -> tuple[str | None, str | None]:
    if values is None:
        return None, None
    encoded = json.dumps(policy_values_dict(values), sort_keys=True, separators=(",", ":"))
    return encoded, hashlib.sha256(encoded.encode()).hexdigest()


def previous_state(
    raw: Mapping[str, Any] | None,
    record: PolicyRecord | None,
    camera_id: str | None,
) -> tuple[bool, NumericPolicy | None]:
    if record is not None and record.status != "failed":
        return True, record.active_values
    if raw is not None:
        previous = decode_policy_values(
            raw["previous_values_json"], raw["previous_content_sha256"], raw
        )
        return bool(raw["previous_present"]), previous
    return camera_id is not None, None


def current_generation(connection: psycopg.Connection, facility_id: str) -> int:
    with connection.cursor(row_factory=dict_row) as cursor:
        row = cursor.execute(
            "SELECT max(activation_generation) AS generation FROM policies WHERE facility_id=%s",
            (facility_id,),
        ).fetchone()
    return 0 if row is None or row["generation"] is None else int(row["generation"])


def next_generation(connection: psycopg.Connection, facility_id: str) -> int:
    return current_generation(connection, facility_id) + 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

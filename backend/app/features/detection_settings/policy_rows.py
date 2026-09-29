"""Policy row decoding, scope lookup, and public activation projection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import psycopg
from psycopg.rows import dict_row

from backend.app.edge_db.postgres import PostgresError
from backend.app.features.detection_settings.policy_models import (
    ActivationStatus,
    PolicyActivation,
    PolicyActivationRefused,
)
from shared.detection_policies import (
    EffectivePolicy,
    NumericPolicy,
    PolicyDocumentError,
    make_effective_policy,
    parse_policy_values,
    policy_definition,
)


@dataclass(frozen=True, slots=True)
class PolicyRecord:
    policy_id: int
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
    status: ActivationStatus
    refusal_reason: str | None


class DetectionPolicyNotInitialized(PostgresError):
    def __init__(self) -> None:
        super().__init__("detection policy bootstrap row is missing")


class InvalidPolicyRecord(PolicyActivationRefused):
    """A malformed snapshot row, retained only for a fenced compare-and-mark."""

    def __init__(self, row: Mapping[str, Any], reason: str) -> None:
        self.row = row
        super().__init__(int(row["policy_id"]), reason)


_RAW_SELECT = (
    "SELECT p.policy_id,p.facility_id,p.camera_id,p.module_id,p.module_version,p.schema_id,"
    "p.schema_version,p.active_values_json,p.active_content_sha256,p.previous_present,"
    "p.previous_values_json,p.previous_content_sha256,p.activation_generation,p.status,"
    "p.refusal_reason,c.backend_camera_id FROM policies p "
    "LEFT JOIN cameras c ON c.camera_id=p.camera_id"
)


def require_policy_site(connection: psycopg.Connection, *, lock: bool = False) -> None:
    with connection.cursor(row_factory=dict_row) as cursor:
        row = cursor.execute(
            "SELECT id FROM edge_site WHERE id=1" + (" FOR UPDATE" if lock else "")
        ).fetchone()
    if row is None:
        raise DetectionPolicyNotInitialized()


def effective_policy(
    connection: psycopg.Connection,
    facility_id: str,
    camera_id: str | None,
    module_id: str,
    module_version: int,
) -> EffectivePolicy:
    facility = policy_record(connection, facility_id, None, module_id, module_version)
    camera = (
        None
        if camera_id is None
        else policy_record(connection, facility_id, camera_id, module_id, module_version)
    )
    selected = camera if camera is not None and camera.active_values is not None else facility
    if selected is None or selected.active_values is None:
        values = policy_definition(module_id, module_version).image_default
        source = "image-default"
    else:
        values = selected.active_values
        source = "camera-override" if camera is selected else "facility-default"
    return make_effective_policy(
        module_id=module_id,
        module_version=module_version,
        values=values,
        source=source,
        facility_revision_id=None if facility is None else facility.generation,
        camera_revision_id=(
            camera.generation if camera is not None and camera.active_values is not None else None
        ),
    )


def policy_record(
    connection: psycopg.Connection,
    facility_id: str,
    camera_id: str | None,
    module_id: str,
    module_version: int,
) -> PolicyRecord | None:
    raw = raw_policy_record(connection, facility_id, camera_id, module_id, module_version)
    if raw is None:
        return None
    try:
        record = decode_policy_record(raw)
    except (PolicyDocumentError, TypeError, ValueError) as error:
        # Snapshot projections are READ ONLY. The owner marks this exact row
        # failed in a separate fenced transaction, never from the read cursor.
        raise InvalidPolicyRecord(raw, str(error)) from error
    if record.status == "failed":
        raise PolicyActivationRefused(
            record.policy_id, record.refusal_reason or "activation is marked failed"
        )
    return record


def raw_policy_record(
    connection: psycopg.Connection,
    facility_id: str,
    camera_id: str | None,
    module_id: str,
    module_version: int,
) -> dict[str, Any] | None:
    camera_clause = "p.camera_id IS NULL" if camera_id is None else "p.camera_id=%s"
    params = (
        (facility_id, module_id, module_version)
        if camera_id is None
        else (facility_id, camera_id, module_id, module_version)
    )
    with connection.cursor(row_factory=dict_row) as cursor:
        return cursor.execute(
            _RAW_SELECT + f" WHERE p.facility_id=%s AND {camera_clause}"
            " AND p.module_id=%s AND p.module_version=%s",
            params,
        ).fetchone()


def decode_policy_record(row: Mapping[str, Any]) -> PolicyRecord:
    status = _status(row["status"])
    active = (
        None
        if status == "failed"
        else decode_policy_values(row["active_values_json"], row["active_content_sha256"], row)
    )
    previous = (
        None
        if status == "failed"
        else decode_policy_values(row["previous_values_json"], row["previous_content_sha256"], row)
    )
    return PolicyRecord(
        int(row["policy_id"]),
        str(row["facility_id"]),
        None if row["camera_id"] is None else str(row["camera_id"]),
        str(row["module_id"]),
        int(row["module_version"]),
        str(row["schema_id"]),
        int(row["schema_version"]),
        active,
        bool(row["previous_present"]),
        previous,
        int(row["activation_generation"]),
        status,
        None if row["refusal_reason"] is None else str(row["refusal_reason"]),
    )


def decode_policy_values(
    value: object, digest: object, row: Mapping[str, Any]
) -> NumericPolicy | None:
    if value is None:
        return None
    encoded = str(value)
    if hashlib.sha256(encoded.encode()).hexdigest() != str(digest):
        raise PolicyDocumentError("policy content hash mismatch")
    return parse_policy_values(
        module_id=str(row["module_id"]),
        module_version=int(row["module_version"]),
        schema_id=str(row["schema_id"]),
        schema_version=int(row["schema_version"]),
        values=json.loads(encoded),
    )


def record_by_id(connection: psycopg.Connection, policy_id: int) -> PolicyRecord:
    with connection.cursor(row_factory=dict_row) as cursor:
        row = cursor.execute(_RAW_SELECT + " WHERE p.policy_id=%s", (policy_id,)).fetchone()
    if row is None:
        raise PolicyDocumentError("policy row is missing")
    return decode_policy_record(row)


def activation(record: PolicyRecord, external_camera_id: str | None) -> PolicyActivation:
    active_revision = None if record.active_values is None else record.generation
    previous_revision = (
        None
        if not record.previous_present
        else (0 if record.previous_values is None else max(1, record.generation - 1))
    )
    return PolicyActivation(
        record.policy_id,
        record.facility_id,
        external_camera_id,
        record.module_id,
        record.module_version,
        active_revision,
        previous_revision,
        record.generation,
        record.status,
        record.refusal_reason,
    )


def database_camera_id(connection: psycopg.Connection, camera_id: str | None) -> str | None:
    if camera_id is None:
        return None
    with connection.cursor(row_factory=dict_row) as cursor:
        # Worker/Hub IDs win over local IDs when the namespaces overlap.
        row = cursor.execute(
            "SELECT camera_id FROM cameras WHERE camera_id=%s OR backend_camera_id=%s "
            "ORDER BY CASE WHEN backend_camera_id=%s THEN 0 ELSE 1 END, "
            'camera_id COLLATE "C" LIMIT 1',
            (camera_id, camera_id, camera_id),
        ).fetchone()
    return camera_id if row is None else str(row["camera_id"])


def external_camera_id(row: Mapping[str, Any]) -> str | None:
    return (
        None
        if row["backend_camera_id"] is None and row["camera_id"] is None
        else str(row["backend_camera_id"] or row["camera_id"])
    )


def try_policy_record(raw: Mapping[str, Any] | None) -> PolicyRecord | None:
    if raw is None:
        return None
    try:
        return decode_policy_record(raw)
    except (PolicyDocumentError, TypeError, ValueError):
        return None


def record_token(record: PolicyRecord | None) -> int:
    return 0 if record is None else record.generation


def raw_token(raw: Mapping[str, Any] | None) -> int:
    return 0 if raw is None else int(raw["activation_generation"])


def raw_select() -> str:
    return _RAW_SELECT


def _status(value: object) -> ActivationStatus:
    if value not in {"pending", "applied", "failed"}:
        raise PolicyDocumentError("stored policy activation status is unknown")
    return cast(ActivationStatus, value)

"""Policy row decoding, scope lookup, and public activation projection."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import cast

from backend.app.edge_db.configuration import utc_now
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


_RAW_SELECT = (
    "SELECT p.policy_id,p.facility_id,p.camera_id,p.module_id,p.module_version,p.schema_id,"
    "p.schema_version,p.active_values_json,p.active_content_sha256,p.previous_present,"
    "p.previous_values_json,p.previous_content_sha256,p.activation_generation,p.status,"
    "p.refusal_reason,c.backend_camera_id FROM policies p "
    "LEFT JOIN cameras c ON c.camera_id=p.camera_id"
)


def effective_policy(
    connection: sqlite3.Connection,
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
    connection: sqlite3.Connection,
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
        reason = str(error)
        connection.execute(
            "UPDATE policies SET status='failed',refusal_reason=?,applied_at=NULL,updated_at=? "
            "WHERE policy_id=?",
            (reason, utc_now(), int(raw[0])),
        )
        raise PolicyActivationRefused(int(raw[0]), reason) from error
    if record.status == "failed":
        raise PolicyActivationRefused(
            record.policy_id, record.refusal_reason or "activation is marked failed"
        )
    return record


def raw_policy_record(
    connection: sqlite3.Connection,
    facility_id: str,
    camera_id: str | None,
    module_id: str,
    module_version: int,
) -> tuple[object, ...] | None:
    camera_clause = "p.camera_id IS NULL" if camera_id is None else "p.camera_id=?"
    params = (
        (facility_id, module_id, module_version)
        if camera_id is None
        else (facility_id, camera_id, module_id, module_version)
    )
    return connection.execute(
        _RAW_SELECT
        + f" WHERE p.facility_id=? AND {camera_clause} AND p.module_id=? AND p.module_version=?",
        params,
    ).fetchone()


def decode_policy_record(row: tuple[object, ...]) -> PolicyRecord:
    status = _status(row[13])
    active = None if status == "failed" else decode_policy_values(row[7], row[8], row)
    previous = None if status == "failed" else decode_policy_values(row[10], row[11], row)
    return PolicyRecord(
        int(row[0]),
        str(row[1]),
        None if row[2] is None else str(row[2]),
        str(row[3]),
        int(row[4]),
        str(row[5]),
        int(row[6]),
        active,
        bool(row[9]),
        previous,
        int(row[12]),
        status,
        None if row[14] is None else str(row[14]),
    )


def decode_policy_values(
    value: object, digest: object, row: tuple[object, ...]
) -> NumericPolicy | None:
    if value is None:
        return None
    encoded = str(value)
    if hashlib.sha256(encoded.encode()).hexdigest() != str(digest):
        raise PolicyDocumentError("policy content hash mismatch")
    return parse_policy_values(
        module_id=str(row[3]),
        module_version=int(row[4]),
        schema_id=str(row[5]),
        schema_version=int(row[6]),
        values=json.loads(encoded),
    )


def record_by_id(connection: sqlite3.Connection, policy_id: int) -> PolicyRecord:
    row = connection.execute(_RAW_SELECT + " WHERE p.policy_id=?", (policy_id,)).fetchone()
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


def database_camera_id(connection: sqlite3.Connection, camera_id: str | None) -> str | None:
    if camera_id is None:
        return None
    row = connection.execute(
        "SELECT camera_id FROM cameras WHERE camera_id=? OR backend_camera_id=?",
        (camera_id, camera_id),
    ).fetchone()
    return camera_id if row is None else str(row[0])


def external_camera_id(row: tuple[object, ...]) -> str | None:
    return None if row[15] is None and row[2] is None else str(row[15] or row[2])


def try_policy_record(raw: tuple[object, ...] | None) -> PolicyRecord | None:
    if raw is None:
        return None
    try:
        return decode_policy_record(raw)
    except (PolicyDocumentError, TypeError, ValueError):
        return None


def record_token(record: PolicyRecord | None) -> int:
    return 0 if record is None else record.generation


def raw_token(raw: tuple[object, ...] | None) -> int:
    return 0 if raw is None else int(raw[12])


def raw_select() -> str:
    return _RAW_SELECT


def _status(value: object) -> ActivationStatus:
    if value not in {"pending", "applied", "failed"}:
        raise PolicyDocumentError("stored policy activation status is unknown")
    return cast(ActivationStatus, value)

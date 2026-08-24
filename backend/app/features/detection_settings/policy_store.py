"""Schema-18 current-plus-previous detection policy authority."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.connection import write_transaction
from backend.app.edge_db.configuration import open_configuration_database, utc_now
from backend.app.features.detection_settings.policy_models import (
    ActivationStatus,
    PolicyActivation,
    PolicyActivationRefused,
    PolicyCameraIdentity,
    PolicyDiff,
    PolicyRevisionConflict,
    PolicyRollbackUnavailable,
)
from shared.detection_policies import (
    LATEST_POLICY_VERSIONS,
    EffectivePolicy,
    NumericPolicy,
    PolicyBundle,
    PolicyDocumentError,
    default_policy_bundle,
    make_effective_policy,
    parse_policy_values,
    policy_definition,
    policy_values_dict,
)


@dataclass(frozen=True, slots=True)
class _PolicyRecord:
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


class DetectionPolicyStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with closing(open_configuration_database(self.path)):
            pass

    @classmethod
    def from_env(cls) -> DetectionPolicyStore:
        return cls(EDGE_DATABASE_PATH)

    def generation(self, facility_id: str | None) -> int:
        if facility_id is None:
            return 0
        with closing(open_configuration_database(self.path)) as connection:
            row = connection.execute(
                "SELECT max(activation_generation) FROM policies WHERE facility_id=?",
                (facility_id,),
            ).fetchone()
        return 0 if row is None or row[0] is None else int(row[0])

    def diff(
        self,
        *,
        facility_id: str,
        module_id: str,
        module_version: int,
        schema_id: str,
        schema_version: int,
        camera_id: str | None,
        values: object,
    ) -> PolicyDiff:
        parsed = self._parse_input(
            module_id, module_version, schema_id, schema_version, camera_id, values
        )
        with closing(open_configuration_database(self.path)) as connection:
            database_camera_id = _database_camera_id(connection, camera_id)
            record = _record(connection, facility_id, database_camera_id, module_id, module_version)
            current = _effective(
                connection, facility_id, database_camera_id, module_id, module_version
            )
            facility = _effective(connection, facility_id, None, module_id, module_version)
        proposed = (
            facility
            if parsed is None
            else make_effective_policy(
                module_id=module_id,
                module_version=module_version,
                values=parsed,
                source="facility-default" if camera_id is None else "camera-override",
                facility_revision_id=0 if camera_id is None else facility.facility_revision_id,
                camera_revision_id=None if camera_id is None else 0,
            )
        )
        changed = (
            current.source == "camera-override"
            if parsed is None
            else (current.source, current.values) != (proposed.source, proposed.values)
        )
        compared_payload: dict[str, object] = {
            "module_id": module_id,
            "module_version": module_version,
            "schema_id": schema_id,
            "schema_version": schema_version,
            "camera_id": camera_id,
            "values": None if parsed is None else policy_values_dict(parsed),
        }
        return PolicyDiff(changed, current, proposed, compared_payload, _token(record))

    def apply(
        self,
        *,
        facility_id: str,
        module_id: str,
        module_version: int,
        schema_id: str,
        schema_version: int,
        camera_id: str | None,
        values: object | None,
        expected_revision_id: int,
    ) -> PolicyActivation:
        if expected_revision_id < 0:
            raise PolicyDocumentError("expected_revision_id must be >= 0")
        parsed = self._parse_input(
            module_id, module_version, schema_id, schema_version, camera_id, values
        )
        with (
            closing(open_configuration_database(self.path)) as connection,
            write_transaction(connection),
        ):
            database_camera_id = _database_camera_id(connection, camera_id)
            raw = _raw_record(
                connection, facility_id, database_camera_id, module_id, module_version
            )
            record = _try_record(raw)
            if _raw_token(raw) != expected_revision_id:
                raise PolicyRevisionConflict(
                    "detection policy activation changed since the submitted diff"
                )
            if record is not None and record.status != "failed" and record.active_values == parsed:
                return _activation(record, camera_id)
            if raw is None and parsed is None:
                raise PolicyRevisionConflict("camera policy already inherits its default")
            generation = _next_generation(connection, facility_id)
            previous_present, previous_values = _previous_state(raw, record, camera_id)
            policy_id = _save_policy(
                connection,
                raw,
                facility_id=facility_id,
                camera_id=database_camera_id,
                module_id=module_id,
                module_version=module_version,
                schema_id=schema_id,
                schema_version=schema_version,
                active_values=parsed,
                previous_present=previous_present,
                previous_values=previous_values,
                generation=generation,
            )
            saved = _record_by_id(connection, policy_id)
            return _activation(saved, camera_id)

    def rollback(
        self,
        *,
        facility_id: str,
        module_id: str,
        module_version: int,
        camera_id: str | None,
        expected_revision_id: int,
    ) -> PolicyActivation:
        if expected_revision_id < 0:
            raise PolicyDocumentError("expected_revision_id must be >= 0")
        with (
            closing(open_configuration_database(self.path)) as connection,
            write_transaction(connection),
        ):
            database_camera_id = _database_camera_id(connection, camera_id)
            raw = _raw_record(
                connection, facility_id, database_camera_id, module_id, module_version
            )
            record = None if raw is None else _decode_record(raw)
            if _token(record) != expected_revision_id:
                raise PolicyRevisionConflict(
                    "detection policy activation changed since the submitted rollback"
                )
            if record is None or not record.previous_present:
                raise PolicyRollbackUnavailable("no prior policy state is available for rollback")
            generation = _next_generation(connection, facility_id)
            values_json, digest = _encoded(record.previous_values)
            connection.execute(
                "UPDATE policies SET active_values_json=?,active_content_sha256=?,"
                "previous_present=0,previous_values_json=NULL,previous_content_sha256=NULL,"
                "activation_generation=?,status='pending',refusal_reason=NULL,activated_at=?,"
                "applied_at=NULL,updated_at=? WHERE policy_id=?",
                (values_json, digest, generation, utc_now(), utc_now(), record.policy_id),
            )
            return _activation(_record_by_id(connection, record.policy_id), camera_id)

    def resolve_bundle(
        self, facility_id: str | None, cameras: tuple[PolicyCameraIdentity, ...]
    ) -> PolicyBundle:
        base = default_policy_bundle(tuple(camera.camera_id for camera in cameras))
        if facility_id is None:
            return base
        with closing(open_configuration_database(self.path)) as connection:
            try:
                defaults = {
                    module_id: _effective(connection, facility_id, None, module_id, version)
                    for module_id, version in LATEST_POLICY_VERSIONS.items()
                }
                bundle = PolicyBundle(base.schema_version, defaults, {})
                for camera in cameras:
                    database_camera_id = _database_camera_id(connection, camera.camera_id)
                    policies = {
                        module_id: _effective(
                            connection, facility_id, database_camera_id, module_id, version
                        )
                        for module_id, version in LATEST_POLICY_VERSIONS.items()
                    }
                    bundle = bundle.with_camera(camera.camera_id, policies)
            except (PolicyDocumentError, sqlite3.Error, TypeError, ValueError) as error:
                raise PolicyActivationRefused(0, str(error)) from error
        return bundle

    def mark_applied(self, facility_id: str, activation_generation: int) -> None:
        with (
            closing(open_configuration_database(self.path)) as connection,
            write_transaction(connection),
        ):
            now = utc_now()
            connection.execute(
                "UPDATE policies SET status='applied',refusal_reason=NULL,"
                "applied_at=?,updated_at=? "
                "WHERE facility_id=? AND status='pending' AND activation_generation<=?",
                (now, now, facility_id, activation_generation),
            )

    def activations(self, facility_id: str) -> tuple[PolicyActivation, ...]:
        with closing(open_configuration_database(self.path)) as connection:
            rows = connection.execute(
                _RAW_SELECT + " WHERE p.facility_id=? ORDER BY p.policy_id", (facility_id,)
            ).fetchall()
            return tuple(_activation(_decode_record(row), _external_camera_id(row)) for row in rows)

    @staticmethod
    def _parse_input(
        module_id: str,
        module_version: int,
        schema_id: str,
        schema_version: int,
        camera_id: str | None,
        values: object | None,
    ) -> NumericPolicy | None:
        if values is None:
            if camera_id is None:
                raise PolicyDocumentError("facility default policy values cannot be null")
            return None
        return parse_policy_values(
            module_id=module_id,
            module_version=module_version,
            schema_id=schema_id,
            schema_version=schema_version,
            values=values,
        )


def _effective(
    connection: sqlite3.Connection,
    facility_id: str,
    camera_id: str | None,
    module_id: str,
    module_version: int,
) -> EffectivePolicy:
    facility = _record(connection, facility_id, None, module_id, module_version)
    camera = (
        None
        if camera_id is None
        else _record(connection, facility_id, camera_id, module_id, module_version)
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


def _record(
    connection: sqlite3.Connection,
    facility_id: str,
    camera_id: str | None,
    module_id: str,
    module_version: int,
) -> _PolicyRecord | None:
    raw = _raw_record(connection, facility_id, camera_id, module_id, module_version)
    if raw is None:
        return None
    try:
        record = _decode_record(raw)
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


def _raw_record(
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


def _decode_record(row: tuple[object, ...]) -> _PolicyRecord:
    status = _status(row[13])
    active = None if status == "failed" else _decode_values(row[7], row[8], row)
    previous = None if status == "failed" else _decode_values(row[10], row[11], row)
    return _PolicyRecord(
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


def _decode_values(value: object, digest: object, row: tuple[object, ...]) -> NumericPolicy | None:
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


def _save_policy(
    connection: sqlite3.Connection,
    raw: tuple[object, ...] | None,
    **values: object,
) -> int:
    active_json, active_hash = _encoded(cast(NumericPolicy | None, values["active_values"]))
    previous_json, previous_hash = _encoded(cast(NumericPolicy | None, values["previous_values"]))
    now = utc_now()
    params = (
        values["facility_id"],
        values["camera_id"],
        values["module_id"],
        values["module_version"],
        values["schema_id"],
        values["schema_version"],
        active_json,
        active_hash,
        int(bool(values["previous_present"])),
        previous_json,
        previous_hash,
        values["generation"],
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


def _record_by_id(connection: sqlite3.Connection, policy_id: int) -> _PolicyRecord:
    row = connection.execute(_RAW_SELECT + " WHERE p.policy_id=?", (policy_id,)).fetchone()
    if row is None:
        raise PolicyDocumentError("policy row is missing")
    return _decode_record(row)


def _encoded(values: NumericPolicy | None) -> tuple[str | None, str | None]:
    if values is None:
        return None, None
    encoded = json.dumps(policy_values_dict(values), sort_keys=True, separators=(",", ":"))
    return encoded, hashlib.sha256(encoded.encode()).hexdigest()


def _previous_state(
    raw: tuple[object, ...] | None,
    record: _PolicyRecord | None,
    camera_id: str | None,
) -> tuple[bool, NumericPolicy | None]:
    if record is not None and record.status != "failed":
        return True, record.active_values
    if raw is not None:
        return bool(raw[9]), None if raw[10] is None else _decode_values(raw[10], raw[11], raw)
    return camera_id is not None, None


def _activation(record: _PolicyRecord, external_camera_id: str | None) -> PolicyActivation:
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


def _next_generation(connection: sqlite3.Connection, facility_id: str) -> int:
    row = connection.execute(
        "SELECT max(activation_generation) FROM policies WHERE facility_id=?", (facility_id,)
    ).fetchone()
    return 1 if row is None or row[0] is None else int(row[0]) + 1


def _database_camera_id(connection: sqlite3.Connection, camera_id: str | None) -> str | None:
    if camera_id is None:
        return None
    row = connection.execute(
        "SELECT camera_id FROM cameras WHERE camera_id=? OR backend_camera_id=?",
        (camera_id, camera_id),
    ).fetchone()
    return camera_id if row is None else str(row[0])


def _external_camera_id(row: tuple[object, ...]) -> str | None:
    return None if row[15] is None and row[2] is None else str(row[15] or row[2])


def _try_record(raw: tuple[object, ...] | None) -> _PolicyRecord | None:
    if raw is None:
        return None
    try:
        return _decode_record(raw)
    except (PolicyDocumentError, TypeError, ValueError):
        return None


def _token(record: _PolicyRecord | None) -> int:
    return 0 if record is None else record.generation


def _raw_token(raw: tuple[object, ...] | None) -> int:
    return 0 if raw is None else int(raw[12])


def _status(value: object) -> ActivationStatus:
    if value not in {"pending", "applied", "failed"}:
        raise PolicyDocumentError("stored policy activation status is unknown")
    return cast(ActivationStatus, value)


_RAW_SELECT = (
    "SELECT p.policy_id,p.facility_id,p.camera_id,p.module_id,p.module_version,p.schema_id,"
    "p.schema_version,p.active_values_json,p.active_content_sha256,p.previous_present,"
    "p.previous_values_json,p.previous_content_sha256,p.activation_generation,p.status,"
    "p.refusal_reason,c.backend_camera_id FROM policies p "
    "LEFT JOIN cameras c ON c.camera_id=p.camera_id"
)

__all__ = [
    "DetectionPolicyStore",
    "PolicyActivation",
    "PolicyActivationRefused",
    "PolicyCameraIdentity",
    "PolicyDiff",
    "PolicyRevisionConflict",
    "PolicyRollbackUnavailable",
]

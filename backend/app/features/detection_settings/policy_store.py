"""API-owned immutable detection-policy revisions and activation pointers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

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
from shared.edge_db import EDGE_DATABASE_PATH
from shared.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction

ActivationStatus = Literal["pending", "applied", "failed"]


class PolicyActivationRefused(RuntimeError):
    def __init__(self, activation_id: int, reason: str) -> None:
        self.activation_id = activation_id
        self.reason = reason
        super().__init__(f"detection policy activation {activation_id} refused: {reason}")


class PolicyRevisionConflict(RuntimeError):
    pass


class PolicyRollbackUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PolicyCameraIdentity:
    camera_id: str


@dataclass(frozen=True, slots=True)
class PolicyActivation:
    activation_id: int
    facility_id: str
    camera_id: str | None
    module_id: str
    module_version: int
    active_revision_id: int | None
    previous_revision_id: int | None
    activation_generation: int
    status: ActivationStatus
    refusal_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "activation_id": self.activation_id,
            "facility_id": self.facility_id,
            "camera_id": self.camera_id,
            "module_id": self.module_id,
            "module_version": self.module_version,
            "active_revision_id": self.active_revision_id,
            "previous_revision_id": self.previous_revision_id,
            "activation_generation": self.activation_generation,
            "status": self.status,
            "refusal_reason": self.refusal_reason,
        }


@dataclass(frozen=True, slots=True)
class PolicyDiff:
    changed: bool
    current: EffectivePolicy
    proposed: EffectivePolicy
    compared_payload: dict[str, object]
    concurrency_token: int

    def as_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "current": self.current.as_dict(),
            "proposed": self.proposed.as_dict(),
            "compared_payload": dict(self.compared_payload),
            "concurrency_token": self.concurrency_token,
        }


@dataclass(frozen=True, slots=True)
class _Revision:
    revision_id: int
    facility_id: str
    camera_id: str | None
    module_id: str
    module_version: int
    values: NumericPolicy


class DetectionPolicyStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def from_env(cls) -> DetectionPolicyStore:
        return cls(EDGE_DATABASE_PATH)

    def generation(self, facility_id: str | None) -> int:
        if facility_id is None:
            return 0
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT activation_generation FROM control_detection_policy_state "
                "WHERE facility_id = ?",
                (facility_id,),
            ).fetchone()
            return 0 if row is None else int(row[0])
        finally:
            connection.close()

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
        if values is None and camera_id is None:
            raise PolicyDocumentError("facility default policy values cannot be null")
        proposed_values = (
            None
            if values is None
            else parse_policy_values(
                module_id=module_id,
                module_version=module_version,
                schema_id=schema_id,
                schema_version=schema_version,
                values=values,
            )
        )
        connection = self._connect()
        try:
            activation = self._activation(
                connection, facility_id, camera_id, module_id, module_version
            )
            concurrency_token = _concurrency_token(activation)
            facility = self._resolve_scope(connection, facility_id, None, module_id, module_version)
            facility_effective = self._effective_from_scope(
                module_id,
                module_version,
                facility,
                None,
            )
            if camera_id is None:
                assert proposed_values is not None
                current = facility_effective
                proposed = make_effective_policy(
                    module_id=module_id,
                    module_version=module_version,
                    values=proposed_values,
                    source="facility-default",
                    facility_revision_id=-1,
                    camera_revision_id=None,
                )
            else:
                camera = self._resolve_scope(
                    connection, facility_id, camera_id, module_id, module_version
                )
                current = self._effective_from_scope(module_id, module_version, facility, camera)
                proposed = (
                    facility_effective
                    if proposed_values is None
                    else make_effective_policy(
                        module_id=module_id,
                        module_version=module_version,
                        values=proposed_values,
                        source="camera-override",
                        facility_revision_id=(None if facility is None else facility.revision_id),
                        camera_revision_id=-1,
                    )
                )
        finally:
            connection.close()
        changed = (
            current.source == "camera-override"
            if proposed_values is None
            else (current.source, current.values) != (proposed.source, proposed.values)
        )
        # Proposal sentinel revisions are for a non-persisting diff only. Build
        # their stable preview identity from zero instead of accepting negative
        # persisted revision identities at the shared boundary.
        if proposed_values is not None:
            proposed = _preview_policy(proposed, camera_id=camera_id)
        compared_payload: dict[str, object] = {
            "module_id": module_id,
            "module_version": module_version,
            "schema_id": schema_id,
            "schema_version": schema_version,
            "camera_id": camera_id,
            "values": (None if proposed_values is None else policy_values_dict(proposed_values)),
        }
        return PolicyDiff(changed, current, proposed, compared_payload, concurrency_token)

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
        if values is None and camera_id is None:
            raise PolicyDocumentError("facility default policy values cannot be null")
        if expected_revision_id < 0:
            raise PolicyDocumentError("expected_revision_id must be >= 0")
        parsed = (
            None
            if values is None
            else parse_policy_values(
                module_id=module_id,
                module_version=module_version,
                schema_id=schema_id,
                schema_version=schema_version,
                values=values,
            )
        )
        connection = self._connect()
        try:
            with write_transaction(connection):
                activation = self._activation(
                    connection, facility_id, camera_id, module_id, module_version
                )
                current_token = _concurrency_token(activation)
                if current_token != expected_revision_id:
                    raise PolicyRevisionConflict(
                        "detection policy activation changed since the submitted diff"
                    )
                current_revision = None if activation is None else activation.active_revision_id
                active_revision_is_valid = activation is None or activation.status != "failed"
                if active_revision_is_valid:
                    try:
                        same_active_values = self._same_active_values(
                            connection,
                            current_revision,
                            parsed,
                            active_is_inherit=values is None,
                        )
                    except (PolicyDocumentError, TypeError, ValueError):
                        active_revision_is_valid = False
                        same_active_values = False
                else:
                    same_active_values = False
                if same_active_values:
                    if activation is None:
                        raise PolicyRevisionConflict("camera policy already inherits its default")
                    return activation
                revision_id = (
                    None
                    if parsed is None
                    else self._insert_revision(
                        connection,
                        facility_id=facility_id,
                        camera_id=camera_id,
                        module_id=module_id,
                        module_version=module_version,
                        schema_id=schema_id,
                        schema_version=schema_version,
                        values=parsed,
                    )
                )
                generation = self._next_generation(connection, facility_id)
                now = _utc_now()
                if activation is None:
                    cursor = connection.execute(
                        "INSERT INTO control_detection_policy_activations "
                        "(facility_id,camera_id,module_id,module_version,active_revision_id,"
                        "previous_revision_id,activation_generation,status,refusal_reason,"
                        "activated_at,applied_at) VALUES (?,?,?,?,?,?,?,'pending',NULL,?,NULL)",
                        (
                            facility_id,
                            camera_id,
                            module_id,
                            module_version,
                            revision_id,
                            None,
                            generation,
                            now,
                        ),
                    )
                    activation_id = _last_insert_id(cursor)
                else:
                    activation_id = activation.activation_id
                    previous_revision_id = (
                        activation.active_revision_id
                        if active_revision_is_valid
                        else activation.previous_revision_id
                    )
                    connection.execute(
                        "UPDATE control_detection_policy_activations SET "
                        "previous_revision_id=?, active_revision_id=?, "
                        "activation_generation=?, status='pending', refusal_reason=NULL, "
                        "activated_at=?, applied_at=NULL WHERE activation_id=?",
                        (
                            previous_revision_id,
                            revision_id,
                            generation,
                            now,
                            activation_id,
                        ),
                    )
                resolved = self._activation_by_id(connection, activation_id)
                assert resolved is not None
                return resolved
        finally:
            connection.close()

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
        connection = self._connect()
        try:
            with write_transaction(connection):
                activation = self._activation(
                    connection, facility_id, camera_id, module_id, module_version
                )
                current_token = _concurrency_token(activation)
                if current_token != expected_revision_id:
                    raise PolicyRevisionConflict(
                        "detection policy activation changed since the submitted rollback"
                    )
                if activation is None or activation.previous_revision_id is None:
                    raise PolicyRollbackUnavailable(
                        "no prior immutable policy revision is available for rollback"
                    )
                rollback_revision_id = activation.previous_revision_id
                predecessor_revision_id = self._prior_revision_id(
                    connection,
                    facility_id=facility_id,
                    camera_id=camera_id,
                    module_id=module_id,
                    module_version=module_version,
                    before_revision_id=rollback_revision_id,
                )
                generation = self._next_generation(connection, facility_id)
                connection.execute(
                    "UPDATE control_detection_policy_activations SET "
                    "active_revision_id=?, previous_revision_id=?, activation_generation=?, "
                    "status='pending', refusal_reason=NULL, "
                    "activated_at=?, applied_at=NULL WHERE activation_id=?",
                    (
                        rollback_revision_id,
                        predecessor_revision_id,
                        generation,
                        _utc_now(),
                        activation.activation_id,
                    ),
                )
                resolved = self._activation_by_id(connection, activation.activation_id)
                assert resolved is not None
                return resolved
        finally:
            connection.close()

    def resolve_bundle(
        self,
        facility_id: str | None,
        cameras: tuple[PolicyCameraIdentity, ...],
    ) -> PolicyBundle:
        base = default_policy_bundle(tuple(camera.camera_id for camera in cameras))
        if facility_id is None:
            return base
        connection = self._connect()
        try:
            defaults: dict[str, EffectivePolicy] = {}
            facility_revisions: dict[str, _Revision | None] = {}
            for module_id, module_version in LATEST_POLICY_VERSIONS.items():
                facility_revision = self._resolve_scope(
                    connection, facility_id, None, module_id, module_version
                )
                facility_revisions[module_id] = facility_revision
                defaults[module_id] = self._effective_from_scope(
                    module_id, module_version, facility_revision, None
                )
            bundle = PolicyBundle(base.schema_version, defaults, {})
            for camera in cameras:
                policies: dict[str, EffectivePolicy] = {}
                for module_id, module_version in LATEST_POLICY_VERSIONS.items():
                    camera_revision = self._resolve_scope(
                        connection,
                        facility_id,
                        camera.camera_id,
                        module_id,
                        module_version,
                    )
                    policies[module_id] = self._effective_from_scope(
                        module_id,
                        module_version,
                        facility_revisions[module_id],
                        camera_revision,
                    )
                bundle = bundle.with_camera(camera.camera_id, policies)
        except PolicyActivationRefused:
            raise
        except (PolicyDocumentError, sqlite3.Error, TypeError, ValueError) as exc:
            raise PolicyActivationRefused(0, str(exc)) from exc
        finally:
            connection.close()
        return bundle

    def mark_applied(self, facility_id: str, activation_generation: int) -> None:
        connection = self._connect()
        try:
            with write_transaction(connection):
                connection.execute(
                    "UPDATE control_detection_policy_activations SET status='applied', "
                    "refusal_reason=NULL, applied_at=? WHERE facility_id=? "
                    "AND status='pending' AND activation_generation <= ?",
                    (_utc_now(), facility_id, activation_generation),
                )
        finally:
            connection.close()

    def activations(self, facility_id: str) -> tuple[PolicyActivation, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                _ACTIVATION_SELECT + " WHERE facility_id=? ORDER BY activation_id",
                (facility_id,),
            ).fetchall()
            return tuple(_activation_from_row(row) for row in rows)
        finally:
            connection.close()

    def _resolve_scope(
        self,
        connection: sqlite3.Connection,
        facility_id: str,
        camera_id: str | None,
        module_id: str,
        module_version: int,
    ) -> _Revision | None:
        activation = self._activation(connection, facility_id, camera_id, module_id, module_version)
        if activation is None:
            return None
        return self._revision_for_activation(connection, activation)

    def _revision_for_activation(
        self, connection: sqlite3.Connection, activation: PolicyActivation
    ) -> _Revision | None:
        if activation.status == "failed":
            raise PolicyActivationRefused(
                activation.activation_id,
                activation.refusal_reason or "activation is marked failed",
            )
        revision_id = activation.active_revision_id
        if revision_id is None:
            return None
        try:
            revision = self._validated_active_revision(connection, revision_id, activation)
        except (PolicyDocumentError, TypeError, ValueError) as exc:
            reason = str(exc)
            connection.execute(
                "UPDATE control_detection_policy_activations SET status='failed', "
                "refusal_reason=?, applied_at=NULL WHERE activation_id=?",
                (reason, activation.activation_id),
            )
            connection.commit()
            raise PolicyActivationRefused(activation.activation_id, reason) from exc
        return revision

    def _validated_active_revision(
        self,
        connection: sqlite3.Connection,
        revision_id: int,
        activation: PolicyActivation,
    ) -> _Revision:
        revision = self._revision(connection, revision_id)
        if revision is None:
            raise PolicyDocumentError("active revision row is missing")
        if (
            revision.facility_id,
            revision.camera_id,
            revision.module_id,
            revision.module_version,
        ) != (
            activation.facility_id,
            activation.camera_id,
            activation.module_id,
            activation.module_version,
        ):
            raise PolicyDocumentError("active revision scope does not match activation")
        return revision

    def _revision(self, connection: sqlite3.Connection, revision_id: int) -> _Revision | None:
        row = connection.execute(
            "SELECT revision_id,facility_id,camera_id,module_id,module_version,schema_id,"
            "schema_version,values_json,content_sha256 "
            "FROM control_detection_policy_revisions WHERE revision_id=?",
            (revision_id,),
        ).fetchone()
        if row is None:
            return None
        values_json = str(row[7])
        if hashlib.sha256(values_json.encode()).hexdigest() != str(row[8]):
            raise PolicyDocumentError("policy revision content hash mismatch")
        try:
            raw_values = json.loads(values_json)
        except json.JSONDecodeError as exc:
            raise PolicyDocumentError("policy revision JSON is malformed") from exc
        values = parse_policy_values(
            module_id=str(row[3]),
            module_version=int(row[4]),
            schema_id=str(row[5]),
            schema_version=int(row[6]),
            values=raw_values,
        )
        return _Revision(
            int(row[0]),
            str(row[1]),
            None if row[2] is None else str(row[2]),
            str(row[3]),
            int(row[4]),
            values,
        )

    def _effective_from_scope(
        self,
        module_id: str,
        module_version: int,
        facility: _Revision | None,
        camera: _Revision | None,
    ) -> EffectivePolicy:
        definition = policy_definition(module_id, module_version)
        if camera is not None:
            return make_effective_policy(
                module_id=module_id,
                module_version=module_version,
                values=camera.values,
                source="camera-override",
                facility_revision_id=None if facility is None else facility.revision_id,
                camera_revision_id=camera.revision_id,
            )
        if facility is not None:
            return make_effective_policy(
                module_id=module_id,
                module_version=module_version,
                values=facility.values,
                source="facility-default",
                facility_revision_id=facility.revision_id,
                camera_revision_id=None,
            )
        return make_effective_policy(
            module_id=module_id,
            module_version=module_version,
            values=definition.image_default,
            source="image-default",
            facility_revision_id=None,
            camera_revision_id=None,
        )

    def _insert_revision(
        self,
        connection: sqlite3.Connection,
        *,
        facility_id: str,
        camera_id: str | None,
        module_id: str,
        module_version: int,
        schema_id: str,
        schema_version: int,
        values: NumericPolicy,
    ) -> int:
        values_json = json.dumps(
            policy_values_dict(values),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        cursor = connection.execute(
            "INSERT INTO control_detection_policy_revisions "
            "(facility_id,camera_id,module_id,module_version,schema_id,schema_version,"
            "values_json,content_sha256,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                facility_id,
                camera_id,
                module_id,
                module_version,
                schema_id,
                schema_version,
                values_json,
                hashlib.sha256(values_json.encode()).hexdigest(),
                _utc_now(),
            ),
        )
        return _last_insert_id(cursor)

    def _same_active_values(
        self,
        connection: sqlite3.Connection,
        revision_id: int | None,
        values: NumericPolicy | None,
        *,
        active_is_inherit: bool,
    ) -> bool:
        if active_is_inherit:
            return revision_id is None
        if revision_id is None or values is None:
            return False
        revision = self._revision(connection, revision_id)
        return revision is not None and revision.values == values

    def _prior_revision_id(
        self,
        connection: sqlite3.Connection,
        *,
        facility_id: str,
        camera_id: str | None,
        module_id: str,
        module_version: int,
        before_revision_id: int,
    ) -> int | None:
        camera_clause = "camera_id IS NULL" if camera_id is None else "camera_id=?"
        parameters: tuple[object, ...] = (
            (facility_id, module_id, module_version, before_revision_id)
            if camera_id is None
            else (
                facility_id,
                camera_id,
                module_id,
                module_version,
                before_revision_id,
            )
        )
        row = connection.execute(
            "SELECT max(revision_id) FROM control_detection_policy_revisions "
            f"WHERE facility_id=? AND {camera_clause} AND module_id=? "
            "AND module_version=? AND revision_id < ?",
            parameters,
        ).fetchone()
        return None if row is None or row[0] is None else _db_int(row[0], "revision_id")

    def _next_generation(self, connection: sqlite3.Connection, facility_id: str) -> int:
        row = connection.execute(
            "SELECT activation_generation FROM control_detection_policy_state WHERE facility_id=?",
            (facility_id,),
        ).fetchone()
        generation = 1 if row is None else int(row[0]) + 1
        connection.execute(
            "INSERT INTO control_detection_policy_state(facility_id,activation_generation) "
            "VALUES (?,?) ON CONFLICT(facility_id) DO UPDATE SET "
            "activation_generation=excluded.activation_generation",
            (facility_id, generation),
        )
        return generation

    def _activation(
        self,
        connection: sqlite3.Connection,
        facility_id: str,
        camera_id: str | None,
        module_id: str,
        module_version: int,
    ) -> PolicyActivation | None:
        camera_clause = "camera_id IS NULL" if camera_id is None else "camera_id=?"
        parameters: tuple[object, ...] = (
            (facility_id, module_id, module_version)
            if camera_id is None
            else (facility_id, camera_id, module_id, module_version)
        )
        row = connection.execute(
            _ACTIVATION_SELECT
            + f" WHERE facility_id=? AND {camera_clause} AND module_id=? AND module_version=?",
            parameters,
        ).fetchone()
        return None if row is None else _activation_from_row(row)

    def _activation_by_id(
        self, connection: sqlite3.Connection, activation_id: int
    ) -> PolicyActivation | None:
        row = connection.execute(
            _ACTIVATION_SELECT + " WHERE activation_id=?", (activation_id,)
        ).fetchone()
        return None if row is None else _activation_from_row(row)

    def _connect(self) -> sqlite3.Connection:
        return open_runtime_database(self.path, actor=RuntimeActor.API)


_ACTIVATION_SELECT = (
    "SELECT activation_id,facility_id,camera_id,module_id,module_version,"
    "active_revision_id,previous_revision_id,activation_generation,status,refusal_reason "
    "FROM control_detection_policy_activations"
)


def _activation_from_row(row: tuple[object, ...]) -> PolicyActivation:
    return PolicyActivation(
        activation_id=_db_int(row[0], "activation_id"),
        facility_id=str(row[1]),
        camera_id=None if row[2] is None else str(row[2]),
        module_id=str(row[3]),
        module_version=_db_int(row[4], "module_version"),
        active_revision_id=(None if row[5] is None else _db_int(row[5], "active_revision_id")),
        previous_revision_id=(None if row[6] is None else _db_int(row[6], "previous_revision_id")),
        activation_generation=_db_int(row[7], "activation_generation"),
        status=_activation_status(row[8]),
        refusal_reason=None if row[9] is None else str(row[9]),
    )


def _last_insert_id(cursor: sqlite3.Cursor) -> int:
    value = cursor.lastrowid
    if value is None:
        raise PolicyDocumentError("policy insert did not return a row id")
    return value


def _db_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyDocumentError(f"stored policy {field_name} is not an integer")
    return value


def _activation_status(value: object) -> ActivationStatus:
    if value not in {"pending", "applied", "failed"}:
        raise PolicyDocumentError("stored policy activation status is unknown")
    return cast(ActivationStatus, value)


def _preview_policy(policy: EffectivePolicy, *, camera_id: str | None) -> EffectivePolicy:
    return make_effective_policy(
        module_id=policy.module_id,
        module_version=policy.module_version,
        values=policy.values,
        source="facility-default" if camera_id is None else "camera-override",
        facility_revision_id=0 if camera_id is None else policy.facility_revision_id,
        camera_revision_id=None if camera_id is None else 0,
    )


def _concurrency_token(activation: PolicyActivation | None) -> int:
    """Stable CAS token for a scope, including generation-zero / inherited state.

    Image-default facility rows and camera scopes that currently inherit have no
    active revision row identity. Those states still participate in compare-and-swap
    with token ``0`` so a missing expected token can never mean "unchecked write".
    """
    if activation is None or activation.active_revision_id is None:
        return 0
    return activation.active_revision_id


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DetectionPolicyStore",
    "PolicyActivation",
    "PolicyActivationRefused",
    "PolicyCameraIdentity",
    "PolicyDiff",
    "PolicyRevisionConflict",
    "PolicyRollbackUnavailable",
]

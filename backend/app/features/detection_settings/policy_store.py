"""Schema-18 current-plus-previous detection policy authority."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.configuration import open_configuration_database, utc_now
from backend.app.edge_db.connection import write_transaction
from backend.app.features.detection_settings.policy_diff import PolicyProposal, build_policy_diff
from backend.app.features.detection_settings.policy_models import (
    PolicyActivation,
    PolicyActivationRefused,
    PolicyCameraIdentity,
    PolicyDiff,
    PolicyRevisionConflict,
    PolicyRollbackUnavailable,
)
from backend.app.features.detection_settings.policy_mutations import (
    PolicyWrite,
    encode_policy,
    next_generation,
    previous_state,
    save_policy,
)
from backend.app.features.detection_settings.policy_rows import (
    activation,
    database_camera_id,
    decode_policy_record,
    effective_policy,
    external_camera_id,
    raw_policy_record,
    raw_select,
    raw_token,
    record_by_id,
    record_token,
    try_policy_record,
)
from shared.detection_policies import (
    LATEST_POLICY_VERSIONS,
    NumericPolicy,
    PolicyBundle,
    PolicyDocumentError,
    default_policy_bundle,
    parse_policy_values,
)


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
        return build_policy_diff(
            self.path,
            PolicyProposal(
                facility_id,
                module_id,
                module_version,
                schema_id,
                schema_version,
                camera_id,
                values,
            ),
        )

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
            camera_key = database_camera_id(connection, camera_id)
            raw = raw_policy_record(connection, facility_id, camera_key, module_id, module_version)
            record = try_policy_record(raw)
            if raw_token(raw) != expected_revision_id:
                raise PolicyRevisionConflict(
                    "detection policy activation changed since the submitted diff"
                )
            if record is not None and record.status != "failed" and record.active_values == parsed:
                return activation(record, camera_id)
            if raw is None and parsed is None:
                raise PolicyRevisionConflict("camera policy already inherits its default")
            generation = next_generation(connection, facility_id)
            previous_present, previous_values = previous_state(raw, record, camera_id)
            policy_id = save_policy(
                connection,
                raw,
                PolicyWrite(
                    facility_id=facility_id,
                    camera_id=camera_key,
                    module_id=module_id,
                    module_version=module_version,
                    schema_id=schema_id,
                    schema_version=schema_version,
                    active_values=parsed,
                    previous_present=previous_present,
                    previous_values=previous_values,
                    generation=generation,
                ),
            )
            saved = record_by_id(connection, policy_id)
            return activation(saved, camera_id)

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
            camera_key = database_camera_id(connection, camera_id)
            raw = raw_policy_record(connection, facility_id, camera_key, module_id, module_version)
            record = None if raw is None else decode_policy_record(raw)
            if record_token(record) != expected_revision_id:
                raise PolicyRevisionConflict(
                    "detection policy activation changed since the submitted rollback"
                )
            if record is None or not record.previous_present:
                raise PolicyRollbackUnavailable("no prior policy state is available for rollback")
            generation = next_generation(connection, facility_id)
            values_json, digest = encode_policy(record.previous_values)
            connection.execute(
                "UPDATE policies SET active_values_json=?,active_content_sha256=?,"
                "previous_present=0,previous_values_json=NULL,previous_content_sha256=NULL,"
                "activation_generation=?,status='pending',refusal_reason=NULL,activated_at=?,"
                "applied_at=NULL,updated_at=? WHERE policy_id=?",
                (values_json, digest, generation, utc_now(), utc_now(), record.policy_id),
            )
            return activation(record_by_id(connection, record.policy_id), camera_id)

    def resolve_bundle(
        self, facility_id: str | None, cameras: tuple[PolicyCameraIdentity, ...]
    ) -> PolicyBundle:
        base = default_policy_bundle(tuple(camera.camera_id for camera in cameras))
        if facility_id is None:
            return base
        with closing(open_configuration_database(self.path)) as connection:
            try:
                defaults = {
                    module_id: effective_policy(connection, facility_id, None, module_id, version)
                    for module_id, version in LATEST_POLICY_VERSIONS.items()
                }
                bundle = PolicyBundle(base.schema_version, defaults, {})
                for camera in cameras:
                    camera_key = database_camera_id(connection, camera.camera_id)
                    policies = {
                        module_id: effective_policy(
                            connection, facility_id, camera_key, module_id, version
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
                raw_select() + " WHERE p.facility_id=? ORDER BY p.policy_id", (facility_id,)
            ).fetchall()
            return tuple(
                activation(decode_policy_record(row), external_camera_id(row)) for row in rows
            )

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


__all__ = [
    "DetectionPolicyStore",
    "PolicyActivation",
    "PolicyActivationRefused",
    "PolicyCameraIdentity",
    "PolicyDiff",
    "PolicyRevisionConflict",
    "PolicyRollbackUnavailable",
]

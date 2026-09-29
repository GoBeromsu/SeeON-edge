"""Current-plus-previous detection policies on the API-owned PostgreSQL pool."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import psycopg
from psycopg.rows import dict_row

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase
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
    current_generation,
    encode_policy,
    next_generation,
    previous_state,
    save_policy,
    utc_now,
)
from backend.app.features.detection_settings.policy_rows import (
    DetectionPolicyNotInitialized,
    InvalidPolicyRecord,
    activation,
    database_camera_id,
    decode_policy_record,
    decode_policy_values,
    effective_policy,
    external_camera_id,
    raw_policy_record,
    raw_select,
    raw_token,
    record_by_id,
    record_token,
    require_policy_site,
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

_Result = TypeVar("_Result")


class DetectionPolicyStore:
    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        self.database = database
        self.authority = authority

    def generation(self, facility_id: str | None) -> int:
        if facility_id is None:
            return 0
        return self._read_snapshot(lambda connection: current_generation(connection, facility_id))

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
        proposal = PolicyProposal(
            facility_id,
            module_id,
            module_version,
            schema_id,
            schema_version,
            camera_id,
            values,
        )
        return self._read_snapshot(lambda connection: build_policy_diff(connection, proposal))

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
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> PolicyActivation:
        def persist(connection: psycopg.Connection) -> PolicyActivation:
            if expected_revision_id < 0:
                raise PolicyDocumentError("expected_revision_id must be >= 0")
            parsed = self._parse_input(
                module_id, module_version, schema_id, schema_version, camera_id, values
            )
            camera_key = database_camera_id(connection, camera_id)
            raw = raw_policy_record(connection, facility_id, camera_key, module_id, module_version)
            record = try_policy_record(raw)
            if raw_token(raw) != expected_revision_id:
                raise PolicyRevisionConflict(
                    "detection policy activation changed since the submitted diff"
                )
            if record is not None and record.status != "failed" and record.active_values == parsed:
                if after_write is not None:
                    after_write(connection)
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
            if after_write is not None:
                after_write(connection)
            return activation(saved, camera_id)

        return self._mutate(persist)

    def rollback(
        self,
        *,
        facility_id: str,
        module_id: str,
        module_version: int,
        camera_id: str | None,
        expected_revision_id: int,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> PolicyActivation:
        def persist(connection: psycopg.Connection) -> PolicyActivation:
            if expected_revision_id < 0:
                raise PolicyDocumentError("expected_revision_id must be >= 0")
            camera_key = database_camera_id(connection, camera_id)
            raw = raw_policy_record(connection, facility_id, camera_key, module_id, module_version)
            record = None if raw is None else decode_policy_record(raw)
            if record_token(record) != expected_revision_id:
                raise PolicyRevisionConflict(
                    "detection policy activation changed since the submitted rollback"
                )
            if raw is None or record is None or not record.previous_present:
                raise PolicyRollbackUnavailable("no prior policy state is available for rollback")
            # Failed records hide values for listing; rollback must validate raw history.
            previous = decode_policy_values(
                raw["previous_values_json"], raw["previous_content_sha256"], raw
            )
            values_json, digest = encode_policy(previous)
            generation = next_generation(connection, facility_id)
            connection.execute(
                "UPDATE policies SET active_values_json=%s,active_content_sha256=%s,"
                "previous_present=0,previous_values_json=NULL,previous_content_sha256=NULL,"
                "activation_generation=%s,status='pending',refusal_reason=NULL,activated_at=%s,"
                "applied_at=NULL,updated_at=%s WHERE policy_id=%s",
                (values_json, digest, generation, utc_now(), utc_now(), record.policy_id),
            )
            saved = activation(record_by_id(connection, record.policy_id), camera_id)
            if after_write is not None:
                after_write(connection)
            return saved

        return self._mutate(persist)

    def resolve_bundle(
        self, facility_id: str | None, cameras: tuple[PolicyCameraIdentity, ...]
    ) -> PolicyBundle:
        base = default_policy_bundle(tuple(camera.camera_id for camera in cameras))
        if facility_id is None:
            return base

        def read(connection: psycopg.Connection) -> PolicyBundle:
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
            except (PolicyDocumentError, TypeError, ValueError) as error:
                raise PolicyActivationRefused(0, str(error)) from error
            return bundle

        return self._read_snapshot(read)

    def acknowledge_applied(self, facility_id: str) -> None:
        """Mark every pending activation at or below the latest generation as applied.

        The caller must establish that the heartbeat acknowledges the current
        desired configuration. This source API has no acknowledged-generation
        argument: its bound is the latest generation inside this transaction.
        """

        def persist(connection: psycopg.Connection) -> None:
            activation_generation = current_generation(connection, facility_id)
            with connection.cursor(row_factory=dict_row) as cursor:
                pending = cursor.execute(
                    "SELECT 1 FROM policies WHERE facility_id=%s AND status='pending'"
                    " AND activation_generation<=%s LIMIT 1",
                    (facility_id, activation_generation),
                ).fetchone()
            if pending is None:
                return
            now = utc_now()
            connection.execute(
                "UPDATE policies SET status='applied',refusal_reason=NULL,"
                "applied_at=%s,updated_at=%s "
                "WHERE facility_id=%s AND status='pending' AND activation_generation<=%s",
                (now, now, facility_id, activation_generation),
            )

        self._mutate(persist)

    def activations(self, facility_id: str) -> tuple[PolicyActivation, ...]:
        def read(connection: psycopg.Connection) -> tuple[PolicyActivation, ...]:
            with connection.cursor(row_factory=dict_row) as cursor:
                rows = cursor.execute(
                    raw_select() + " WHERE p.facility_id=%s ORDER BY p.policy_id", (facility_id,)
                ).fetchall()
            return tuple(
                activation(decode_policy_record(row), external_camera_id(row)) for row in rows
            )

        return self._read_snapshot(read)

    def _mutate(self, callback: Callable[[psycopg.Connection], _Result]) -> _Result:
        def persist(connection: psycopg.Connection) -> _Result:
            require_authority(connection, self.authority)
            require_policy_site(connection, lock=True)
            return callback(connection)

        try:
            return self.database.transact(persist)
        except psycopg.Error:
            # Native diagnostics can include the SQL and complete row values.
            # Authority, canonical validation and unknown-COMMIT errors are
            # deliberately not caught here.
            raise PolicyActivationRefused(0, "policy database operation failed") from None

    def _read_snapshot(self, callback: Callable[[psycopg.Connection], _Result]) -> _Result:
        def read(connection: psycopg.Connection) -> _Result:
            require_policy_site(connection)
            return callback(connection)

        try:
            return self.database.read_snapshot(read)
        except InvalidPolicyRecord as error:
            self._mark_failed(error)
            raise
        except psycopg.Error:
            raise PolicyActivationRefused(0, "policy database operation failed") from None

    def _mark_failed(self, error: InvalidPolicyRecord) -> None:
        def persist(connection: psycopg.Connection) -> None:
            observed = error.row
            current = raw_policy_record(
                connection,
                observed["facility_id"],
                observed["camera_id"],
                observed["module_id"],
                observed["module_version"],
            )
            # A concurrent repair must never be overwritten by an old read.
            if current != observed:
                return
            # The outward canonical validation error stays intact; persisted
            # diagnostics must fit PostgreSQL text and the schema's 256-char cap.
            reason = error.reason.replace("\x00", r"\u0000")[:256]
            connection.execute(
                "UPDATE policies SET status='failed',refusal_reason=%s,applied_at=NULL,"
                "updated_at=%s WHERE policy_id=%s",
                (reason, utc_now(), error.activation_id),
            )

        self._mutate(persist)

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
    "DetectionPolicyNotInitialized",
    "DetectionPolicyStore",
    "PolicyActivation",
    "PolicyActivationRefused",
    "PolicyCameraIdentity",
    "PolicyDiff",
    "PolicyRevisionConflict",
    "PolicyRollbackUnavailable",
]

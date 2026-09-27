"""Exercise native constraints in test-owned PostgreSQL namespaces, not SQLite.

SEEON_TEST_POSTGRES_DSN must target an isolated service. A missing DSN skips the
ordinary suite and does not establish field acceptance.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from backend.app.edge_db.compact_schema import COMPACT_APPLICATION_TABLES
from backend.app.edge_db.execution_records_ddl import EXECUTION_RECORD_TABLES
from backend.app.edge_db.functions import audit_record_hash

_DDL = Path(__file__).resolve().parents[1] / "backend" / "app" / "edge_db"
_NOW = "2026-09-27T03:00:00.123456Z"
_HASH = "a" * 64
_ZERO = "0" * 64
_AUDIT_SHADOW_PAYLOAD = '{"shadow":"namespace"}'


@dataclass
class _Schemas:
    connection: psycopg.Connection = field(repr=False)
    product: str
    diagnostics: str

    def use(self, name: str) -> psycopg.Connection:
        self.connection.execute(
            sql.SQL("SET search_path TO {}, pg_catalog, pg_temp").format(sql.Identifier(name))
        )
        return self.connection


@pytest.fixture
def postgres_schemas():
    dsn = os.environ.get("SEEON_TEST_POSTGRES_DSN")
    if dsn is None:
        pytest.skip("requires an isolated SEEON_TEST_POSTGRES_DSN; not field acceptance")
    if not dsn.strip() or "\x00" in dsn:
        pytest.fail("SEEON_TEST_POSTGRES_DSN must be nonblank without NUL bytes", pytrace=False)
    try:
        connection = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
    except (psycopg.Error, OSError, ValueError, TypeError):
        connection = None
    if connection is None:
        pytest.fail("the configured PostgreSQL test service is unavailable", pytrace=False)
    product = "seeon_product_test_" + uuid4().hex
    diagnostics = "seeon_diagnostics_test_" + uuid4().hex
    schemas = _Schemas(connection, product, diagnostics)
    try:
        connection.execute("SET statement_timeout = '10s'")
        connection.execute("SET lock_timeout = '3s'")
        with connection.transaction():
            for name in (product, diagnostics):
                connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
                schemas.use(name)
                if name == product:
                    connection.execute((_DDL / "postgres_product.sql").read_text(), prepare=False)
                connection.execute((_DDL / "postgres_diagnostics.sql").read_text(), prepare=False)
        schemas.use(product)
        yield schemas
    finally:
        for name in (diagnostics, product):
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name))
            )
        connection.close()


@pytest.fixture
def postgres_delivery_schema(postgres_schemas):
    connection = postgres_schemas.connection
    with connection.transaction():
        connection.execute((_DDL / "postgres_delivery.sql").read_text(), prepare=False)
    return connection


@pytest.mark.parametrize("dsn", [None, "", " \t\n", "synthetic\x00dsn"])
def test_schema_fixture_rejects_invalid_opt_in_before_connect(monkeypatch, dsn):
    monkeypatch.setattr(os, "environ", {} if dsn is None else {"SEEON_TEST_POSTGRES_DSN": dsn})

    def forbidden_connect(*args, **kwargs):
        pytest.fail("invalid opt-in must not open an ambient connection")

    monkeypatch.setattr(psycopg, "connect", forbidden_connect)
    expected = pytest.skip.Exception if dsn is None else pytest.fail.Exception
    message = "requires an isolated" if dsn is None else "nonblank without NUL"
    with pytest.raises(expected, match=message):
        next(postgres_schemas.__wrapped__())


@pytest.mark.parametrize("failure", [psycopg.OperationalError, OSError, ValueError, TypeError])
def test_schema_fixture_connection_failure_has_no_sensitive_exception_context(monkeypatch, failure):
    sentinel = "synthetic-schema-connection-secret"
    monkeypatch.setenv("SEEON_TEST_POSTGRES_DSN", "host=synthetic-isolated-test")

    def failed_connect(*args, **kwargs):
        raise failure(sentinel)

    monkeypatch.setattr(psycopg, "connect", failed_connect)
    with pytest.raises(pytest.fail.Exception, match="test service is unavailable") as caught:
        next(postgres_schemas.__wrapped__())
    assert sentinel not in str(caught.value)
    assert caught.value.__context__ is None


def _insert(connection, table, values):
    return connection.execute(
        sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, values)),
            sql.SQL(", ").join(sql.Placeholder() for _ in values),
        ),
        tuple(values.values()),
    )


def _reject(connection, table, values, error=psycopg.errors.CheckViolation):
    with pytest.raises(error), connection.transaction():
        _insert(connection, table, values)


def _camera(**changes):
    return {
        "camera_id": "camera-1",
        "label": "camera",
        "rtsp_url": "rtsp://test.invalid/stream",
        "normalized_stream_identity": "stream-1",
        "mapping_state": "UNMAPPED",
        "never_connected": 1,
        "revision": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    } | changes


def _policy(**changes):
    return {
        "facility_id": "facility-1",
        "module_id": "fall",
        "module_version": 1,
        "schema_id": "fall-policy",
        "schema_version": 1,
        "active_values_json": "{}",
        "active_content_sha256": _HASH,
        "previous_present": 0,
        "activation_generation": 1,
        "status": "pending",
        "activated_at": _NOW,
        "updated_at": _NOW,
    } | changes


def _incident(**changes):
    return {
        "incident_id": "incident-1",
        "edge_event_id": "event-1",
        "facility_id": "facility-1",
        "camera_id": "camera-1",
        "event_type": "fall",
        "detected_at": _NOW,
        "lifecycle_state": "OPEN",
        "provenance_state": "MISSING",
        "provenance_missing_reason": "not-recorded",
        "review_version": 0,
        "revision": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    } | changes


def _clip(**changes):
    return {
        "clip_id": "clip-1",
        "camera_id": "camera-1",
        "event_facet": "fall",
        "started_at": _NOW,
        "local_state": "UNAVAILABLE",
        "local_reason": "not-recorded",
        "publish_state": "WAITING",
        "retention_state": "RETAINED",
        "revision": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    } | changes


def _artifact(**changes):
    return {
        "incident_id": "incident-1",
        "kind": "PRIMARY_CLIP",
        "state": "PENDING",
        "revision": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    } | changes


def _audit(**changes):
    values = {
        "occurred_at": _NOW,
        "recorded_at": _NOW,
        "clock_quality": "trusted",
        "actor_type": "user",
        "actor_id": "operator-1",
        "auth_mechanism": "session",
        "action": "review",
        "target_type": "incident",
        "target_id": "incident-1",
        "outcome": "success",
        "reason": None,
        "request_id": None,
        "interaction_id": None,
        "detail_json": None,
        "previous_hash": _ZERO,
        "retention_class": "standard",
        "hold_reference": None,
    } | changes
    return values | {
        "record_hash": audit_record_hash(
            values["previous_hash"], json.dumps(values, ensure_ascii=False)
        )
    }


def _delivery_parents(connection, edge_event_id="event-1"):
    _insert(
        connection,
        "incidents",
        _incident(incident_id=f"incident:{edge_event_id}", edge_event_id=edge_event_id),
    )
    _insert(
        connection,
        "event_outbox",
        {
            "edge_event_id": edge_event_id,
            "envelope": "{}",
            "envelope_sha256": hashlib.sha256(b"{}").hexdigest(),
            "envelope_bytes": 2,
            "backend_camera_id": "hub-camera",
            "state": "PENDING",
            "accepted_generation": 1,
            "accepted_at": _NOW,
            "retry_at": _NOW,
        },
    )
    attempt = {
        "attempt_id": uuid4(),
        "edge_event_id": edge_event_id,
        "ordinal": 1,
        "writer_generation": 1,
        "started_at": _NOW,
    }
    _insert(connection, "event_delivery_attempts", attempt)
    return attempt


def _observation(attempt, **changes):
    return {
        "attempt_id": attempt["attempt_id"],
        "edge_event_id": attempt["edge_event_id"],
        "ordinal": attempt["ordinal"],
        "observed_at": _NOW,
        "outcome": "SENT",
        "reason": "ACCEPTED",
        "http_status": 202,
        "backend_event_id": "central-1",
    } | changes


def _diagnostic_parents(connection):
    _insert(
        connection,
        "execution_provenance",
        {
            "provenance_id": _HASH,
            "worker_build_revision": "build",
            "worker_image_digest": "image",
            "model_digest": "model",
            "calibration_digest": "calibration",
            "preprocessing_identity": "preprocessing",
            "config_digest": "config",
            "policy_identity": "policy",
            "backend_build_revision": "backend",
            "first_seen_ns": 2**40,
        },
    )
    _insert(
        connection,
        "execution_segments",
        {
            "segment_id": 2**40,
            "camera_id": "camera-1",
            "worker_boot_id": "boot-1",
            "source_generation": 0,
            "stream_epoch": 0,
            "segment_ordinal": 0,
            "storage_state": "OPEN",
            "opened_at_ns": 2**40,
        },
    )
    _insert(
        connection,
        "execution_units",
        {
            "causal_unit_id": "unit-1",
            "camera_id": "camera-1",
            "worker_boot_id": "boot-1",
            "source_generation": 0,
            "stream_epoch": 0,
            "causal_state": "COMPLETE",
            "first_observed_ns": 2**40,
            "last_observed_ns": 2**40,
        },
    )


def _record(**changes):
    payload = ' { "한글": [1, {"x": "é"}], "n": null } '
    return {
        "record_id": "b" * 64,
        "record_kind": "policy.decision",
        "camera_id": "camera-1",
        "worker_boot_id": "boot-1",
        "source_generation": 0,
        "stream_epoch": 0,
        "producer": "policy",
        "producer_sequence": 2**40,
        "source_pts_ns": -1,
        "observed_at_ns": 2**40,
        "time_quality": "trusted",
        "causal_unit_id": "unit-1",
        "parent_record_id": "c" * 64,
        "segment_id": 2**40,
        "provenance_id": _HASH,
        "outcome": "accepted",
        "payload": payload,
        "payload_bytes": len(payload.encode()),
        "committed_at_ns": 2**40,
    } | changes


def test_namespace_tables_integer_widths_and_unstamped_ledger(postgres_schemas):
    connection = postgres_schemas.connection
    for schema, expected in (
        (postgres_schemas.product, COMPACT_APPLICATION_TABLES | EXECUTION_RECORD_TABLES),
        (postgres_schemas.diagnostics, EXECUTION_RECORD_TABLES),
    ):
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_type = 'BASE TABLE'",
            (schema,),
        ).fetchall()
        assert {row[0] for row in tables} == expected
        wrong_width = connection.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND data_type IN ('integer', 'smallint')",
            (schema,),
        ).fetchall()
        assert wrong_width == []
    assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone() == (0,)
    types = connection.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = 'credentials' "
        "AND column_name IN ('salt', 'password_hash')",
        (postgres_schemas.product,),
    ).fetchall()
    assert dict(types) == {"salt": "bytea", "password_hash": "bytea"}


def test_delivery_schema_explicit_product_table_inventory(postgres_delivery_schema):
    connection = postgres_delivery_schema
    tables = connection.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema=current_schema() AND table_type='BASE TABLE'"
    ).fetchall()
    assert {row[0] for row in tables} == COMPACT_APPLICATION_TABLES | EXECUTION_RECORD_TABLES | {
        "deployment_authority",
        "event_outbox",
        "event_delivery_attempts",
        "event_delivery_results",
        "event_delivery_observations",
    }


@pytest.mark.parametrize("invalid", ["missing_attempt", "cross_event", "ordinal"])
def test_delivery_observation_fk_binds_attempt_event_and_ordinal(postgres_delivery_schema, invalid):
    connection = postgres_delivery_schema
    attempt = _delivery_parents(connection)
    other = _delivery_parents(connection, edge_event_id="event-2")
    if invalid == "missing_attempt":
        changes = {"attempt_id": uuid4()}
    elif invalid == "cross_event":
        changes = {"edge_event_id": other["edge_event_id"]}
    else:
        changes = {"ordinal": attempt["ordinal"] + 1}
    _reject(
        connection,
        "event_delivery_observations",
        _observation(attempt, **changes),
        psycopg.errors.ForeignKeyViolation,
    )
    assert connection.execute("SELECT count(*) FROM event_delivery_observations").fetchone() == (0,)
    assert connection.execute("SELECT count(*) FROM event_delivery_attempts").fetchone() == (2,)
    _insert(connection, "event_delivery_observations", _observation(attempt))
    assert connection.execute(
        "SELECT attempt_id,edge_event_id,ordinal FROM event_delivery_observations"
    ).fetchone() == (attempt["attempt_id"], attempt["edge_event_id"], attempt["ordinal"])


def test_delivery_attempt_fk_still_requires_accepted_outbox(postgres_delivery_schema):
    connection = postgres_delivery_schema
    attempt = _delivery_parents(connection)
    _reject(
        connection,
        "event_delivery_attempts",
        attempt | {"attempt_id": uuid4(), "edge_event_id": "missing-event"},
        psycopg.errors.ForeignKeyViolation,
    )
    assert connection.execute("SELECT count(*) FROM event_delivery_attempts").fetchone() == (1,)


def test_delivery_observation_allows_only_one_response_per_attempt(postgres_delivery_schema):
    connection = postgres_delivery_schema
    attempt = _delivery_parents(connection)
    _insert(connection, "event_delivery_observations", _observation(attempt))
    before = connection.execute("SELECT * FROM event_delivery_observations").fetchall()
    for changes in ({}, {"backend_event_id": "central-other"}):
        _reject(
            connection,
            "event_delivery_observations",
            _observation(attempt, **changes),
            psycopg.errors.UniqueViolation,
        )
        assert connection.execute("SELECT * FROM event_delivery_observations").fetchall() == before


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE event_delivery_observations SET reason='DIFFERENT'",
        "UPDATE event_delivery_observations SET outcome=outcome",
        "DELETE FROM event_delivery_observations",
        "TRUNCATE event_delivery_observations",
    ],
)
def test_delivery_observation_history_is_immutable(postgres_delivery_schema, statement):
    connection = postgres_delivery_schema
    attempt = _delivery_parents(connection)
    _insert(connection, "event_delivery_observations", _observation(attempt))
    before = connection.execute("SELECT * FROM event_delivery_observations").fetchall()
    with pytest.raises(psycopg.errors.CheckViolation), connection.transaction():
        connection.execute(statement)
    assert connection.execute("SELECT * FROM event_delivery_observations").fetchall() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"ordinal": 0},
        {"outcome": "UNCLASSIFIED"},
        {"reason": ""},
        {"reason": "raw response text"},
        {"reason": "A" * 65},
        {"http_status": 99},
        {"http_status": 600},
        {"backend_event_id": None},
        {"backend_event_id": ""},
        {"backend_event_id": "x" * 129},
        {"outcome": "UNKNOWN"},
    ],
)
def test_delivery_observation_rejects_invalid_response_metadata(postgres_delivery_schema, changes):
    connection = postgres_delivery_schema
    attempt = _delivery_parents(connection)
    _reject(connection, "event_delivery_observations", _observation(attempt, **changes))
    assert connection.execute("SELECT count(*) FROM event_delivery_observations").fetchone() == (0,)


@pytest.mark.parametrize(
    "outcome,http_status",
    [("SENT", 100), ("SENT", 599), ("RETRY", 503), ("REJECTED", 422), ("UNKNOWN", None)],
)
def test_delivery_observation_accepts_classified_boundary_metadata(
    postgres_delivery_schema, outcome, http_status
):
    connection = postgres_delivery_schema
    attempt = _delivery_parents(connection)
    backend_event_id = "가" * 128 if outcome == "SENT" else None
    _insert(
        connection,
        "event_delivery_observations",
        _observation(
            attempt,
            outcome=outcome,
            reason="A" * 64,
            http_status=http_status,
            backend_event_id=backend_event_id,
        ),
    )
    assert connection.execute(
        "SELECT outcome,reason,http_status,backend_event_id FROM event_delivery_observations"
    ).fetchone() == (outcome, "A" * 64, http_status, backend_event_id)


@pytest.mark.parametrize(
    "value",
    [
        "0000-02-29T00:00:00Z",
        "2000-02-29T23:59:59Z",
        "2026-09-27T00:00:00.1Z",
        "2026-09-27T00:00:00.123456Z",
        "9999-12-31T23:59:59Z",
    ],
)
def test_strict_utc_valid_boundaries_preserve_text(postgres_schemas, value):
    connection = postgres_schemas.connection
    _insert(
        connection,
        "credentials",
        {
            "id": 1,
            "username": "가" * 128,
            "algorithm": "scrypt",
            "salt": bytes(range(16)),
            "password_hash": bytes(range(64)),
            "updated_at": value,
        },
    )
    assert connection.execute(
        "SELECT salt, password_hash, updated_at FROM credentials"
    ).fetchone() == (
        bytes(range(16)),
        bytes(range(64)),
        value,
    )


@pytest.mark.parametrize(
    "value",
    [
        "2026-02-29T00:00:00Z",
        "1900-02-29T00:00:00Z",
        "2026-04-31T00:00:00Z",
        "2026-13-01T00:00:00Z",
        "2026-00-01T00:00:00Z",
        "2026-01-00T00:00:00Z",
        "2026-01-01T24:00:00Z",
        "2026-01-01T00:60:00Z",
        "2026-01-01T00:00:60Z",
        "2026-01-01 00:00:00Z",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00z",
        "2026-01-01T00:00:00.Z",
        "2026-01-01T00:00:00.1234567Z",
        "infinity",
        "",
    ],
)
def test_strict_utc_rejects_invalid_calendar_and_noncanonical_forms(postgres_schemas, value):
    _reject(
        postgres_schemas.connection,
        "credentials",
        {
            "id": 1,
            "username": "operator",
            "algorithm": "scrypt",
            "salt": bytes(16),
            "password_hash": bytes(64),
            "updated_at": value,
        },
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"id": 2},
        {"username": ""},
        {"username": "a" * 129},
        {"algorithm": "sha256"},
        {"salt": bytes(15)},
        {"salt": bytes(17)},
        {"password_hash": bytes(63)},
    ],
)
def test_credentials_binary_lengths_singleton_and_algorithm(postgres_schemas, changes):
    _reject(
        postgres_schemas.connection,
        "credentials",
        {
            "id": 1,
            "username": "operator",
            "algorithm": "scrypt",
            "salt": bytes(16),
            "password_hash": bytes(64),
            "updated_at": _NOW,
        }
        | changes,
    )


def test_edge_site_group_presence_and_schedule_bounds(postgres_schemas):
    connection = postgres_schemas.connection
    for changes in (
        {"facility_token": "partial-enrollment"},
        {"fall_on": 1},
        {"fall_on": 1, "fall_mode": "window", "fall_start_time": "23:00"},
        {"fall_on": 1, "fall_mode": "window", "fall_start_time": "24:00", "fall_end_time": "00:00"},
        {"bed_exit_on": 1, "bed_exit_mode": "always", "bed_exit_start_time": "00:00"},
        {"topology_pending_body": b"{}"},
        {"topology_dirty_registry_version": 1},
        {"topology_confirmation_id": "partial"},
        {"storage_state": "degraded"},
        {"registry_version": -1},
        {"clip_export_enabled": 2},
    ):
        _reject(connection, "edge_site", {"id": 1, "updated_at": _NOW} | changes)
    _insert(
        connection,
        "edge_site",
        {
            "id": 1,
            "updated_at": _NOW,
            "registry_version": 2**40,
            "fall_on": 1,
            "fall_mode": "window",
            "fall_start_time": "23:59",
            "fall_end_time": "00:00",
            "bed_exit_on": 0,
            "bed_exit_mode": "always",
            "topology_pending_snapshot_id": "snapshot",
            "topology_pending_body": b"{\n}",
            "topology_pending_registry_version": 2**40,
            "topology_pending_client_revision": 1,
            "topology_pending_expected_server_revision": 0,
            "topology_dirty_registry_version": 1,
            "topology_dirty_created_at": _NOW,
        },
    )
    assert connection.execute(
        "SELECT registry_version, topology_pending_body FROM edge_site"
    ).fetchone() == (2**40, b"{\n}")


@pytest.mark.parametrize(
    "path", ["", "/absolute", "../escape", "a/../b", "a/..", "a\\b", "x" * 513]
)
def test_contained_media_paths_reject_escape_and_length(postgres_schemas, path):
    _reject(
        postgres_schemas.connection,
        "edge_site",
        {"id": 1, "updated_at": _NOW, "clip_store_subdir": path},
    )


def test_locations_camera_room_uniqueness_and_restricted_deletes(postgres_schemas):
    connection = postgres_schemas.connection
    base = {"name": "location", "order_index": 0, "created_at": _NOW, "updated_at": _NOW}
    _insert(connection, "locations", base | {"location_id": "floor", "kind": "FLOOR"})
    room = base | {
        "location_id": "room",
        "kind": "ROOM",
        "parent_location_id": "floor",
        "parent_kind": "FLOOR",
    }
    _reject(
        connection,
        "locations",
        room | {"parent_location_id": "absent"},
        psycopg.errors.ForeignKeyViolation,
    )
    _reject(connection, "locations", room | {"parent_kind": None})
    _insert(connection, "locations", room)
    _insert(
        connection,
        "cameras",
        _camera(room_location_id="room", room_location_kind="ROOM", edge_ref="edge"),
    )
    _reject(
        connection,
        "cameras",
        _camera(
            camera_id="camera-2",
            room_location_id="room",
            room_location_kind="ROOM",
            edge_ref="edge",
        ),
        psycopg.errors.UniqueViolation,
    )
    for statement in (
        "DELETE FROM locations WHERE kind = 'FLOOR'",
        "DELETE FROM locations WHERE kind = 'ROOM'",
    ):
        with pytest.raises(psycopg.errors.RestrictViolation), connection.transaction():
            connection.execute(statement)
    _insert(connection, "policies", _policy(camera_id="camera-1"))
    with pytest.raises(psycopg.errors.RestrictViolation), connection.transaction():
        connection.execute("DELETE FROM cameras")


@pytest.mark.parametrize(
    "changes",
    [
        {"camera_id": ""},
        {"camera_id": "x" * 129},
        {"mapping_state": "MAPPED"},
        {"backend_camera_id": "backend"},
        {"never_connected": 2},
        {"revision": 0},
        {"bed_polygon_json": "[]"},
        {"room_location_id": "room", "edge_ref": "edge"},
    ],
)
def test_camera_identity_mapping_and_group_constraints(postgres_schemas, changes):
    _reject(postgres_schemas.connection, "cameras", _camera(**changes))


def test_postgres_rejects_nul_text(postgres_schemas):
    _reject(postgres_schemas.connection, "cameras", _camera(label="nul\x00label"), psycopg.Error)


def test_camera_json_is_validated_without_normalizing_bytes(postgres_schemas):
    connection = postgres_schemas.connection
    polygon = "[ [0, 1],\n [2,3] ]"
    bed = {
        "bed_polygon_json": polygon,
        "bed_image_width": 1920,
        "bed_image_height": 1080,
        "bed_recognized_at": _NOW,
    }
    _reject(connection, "cameras", _camera(**(bed | {"bed_polygon_json": "{}"})))
    _reject(connection, "cameras", _camera(**(bed | {"bed_polygon_json": "[not json]"})))
    _insert(connection, "cameras", _camera(**bed))
    assert connection.execute("SELECT bed_polygon_json FROM cameras").fetchone() == (polygon,)


def test_policy_json_byte_limits_pairing_status_and_scope_uniqueness(postgres_schemas):
    connection = postgres_schemas.connection
    for changes in (
        {"active_values_json": "[]"},
        {"active_values_json": "{bad}"},
        {"active_values_json": json.dumps({"s": "가" * 6000}, ensure_ascii=False)},
        {"active_values_json": None},
        {"active_content_sha256": "A" * 64},
        {"previous_values_json": "{}", "previous_content_sha256": _HASH},
        {"status": "applied"},
        {"status": "failed"},
    ):
        _reject(connection, "policies", _policy(**changes))
    encoded = ' { "z": [1, 2], "한글": {"a": true} } '
    _insert(connection, "policies", _policy(active_values_json=encoded))
    _reject(connection, "policies", _policy(), psycopg.errors.UniqueViolation)
    assert connection.execute("SELECT active_values_json FROM policies").fetchone() == (encoded,)


@pytest.mark.parametrize(
    "table,values", [("cameras", _camera(revision=2**40)), ("clips", _clip(revision=2**40))]
)
def test_revision_cas_rejects_stale_writers_with_64_bit_revisions(postgres_schemas, table, values):
    connection = postgres_schemas.connection
    _insert(connection, table, values)
    update = sql.SQL("UPDATE {} SET revision = revision + 1 WHERE revision = %s").format(
        sql.Identifier(table)
    )
    assert connection.execute(update, (2**40,)).rowcount == 1
    assert connection.execute(update, (2**40,)).rowcount == 0
    assert connection.execute(
        sql.SQL("SELECT revision FROM {}").format(sql.Identifier(table))
    ).fetchone() == (2**40 + 1,)


@pytest.mark.parametrize(
    "changes",
    [
        {"duration_ms": 0},
        {"duration_ms": 120001},
        {"local_state": "AVAILABLE", "local_reason": None},
        {"manifest_relpath": "manifest.json"},
        {"publish_state": "PUBLISHED"},
        {"retention_state": "PURGED"},
        {"revision": 0},
    ],
)
def test_clip_state_metadata_pairing_and_time_bounds(postgres_schemas, changes):
    _reject(postgres_schemas.connection, "clips", _clip(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"probability": -0.01},
        {"probability": 1.01},
        {"probability": float("nan")},
        {"lifecycle_state": "FAILED"},
        {"failure_reason": "unexpected"},
        {"provenance_state": "QUALIFIED", "provenance_missing_reason": None},
        {"review_version": 1},
        {"review_actor": "unexpected"},
    ],
)
def test_incident_probability_provenance_and_review_constraints(postgres_schemas, changes):
    _reject(postgres_schemas.connection, "incidents", _incident(**changes))


def test_incident_cas_immutable_identity_and_legal_lifecycle(postgres_schemas):
    connection = postgres_schemas.connection
    _insert(connection, "incidents", _incident())
    for statement in (
        "UPDATE incidents SET probability = 0.5",
        "UPDATE incidents SET revision = revision + 2",
        "UPDATE incidents SET camera_id = 'different', revision = revision + 1",
    ):
        with pytest.raises(psycopg.errors.CheckViolation), connection.transaction():
            connection.execute(statement)
    change = (
        "UPDATE incidents SET lifecycle_state = 'COMPLETE', revision = revision + 1 "
        "WHERE incident_id = 'incident-1' AND revision = 1"
    )
    assert connection.execute(change).rowcount == 1
    assert connection.execute(change).rowcount == 0
    with pytest.raises(psycopg.errors.CheckViolation, match="lifecycle"), connection.transaction():
        connection.execute("UPDATE incidents SET lifecycle_state = 'OPEN', revision = revision + 1")
    connection.execute(
        "UPDATE incidents SET lifecycle_state = 'FAILED', failure_reason = 'missing', "
        "revision = revision + 1"
    )
    assert connection.execute("SELECT lifecycle_state, revision FROM incidents").fetchone() == (
        "FAILED",
        3,
    )


def test_artifact_transition_identity_revision_and_fk_guards(postgres_schemas):
    connection = postgres_schemas.connection
    _insert(connection, "incidents", _incident())
    _insert(connection, "clips", _clip())
    _reject(
        connection, "artifacts", _artifact(incident_id="absent"), psycopg.errors.ForeignKeyViolation
    )
    _insert(connection, "artifacts", _artifact())
    with pytest.raises(psycopg.errors.CheckViolation, match="revision"), connection.transaction():
        connection.execute("UPDATE artifacts SET updated_at = updated_at")
    connection.execute(
        "UPDATE artifacts SET state = 'AVAILABLE', artifact_id = 'artifact-1', clip_id = 'clip-1', "
        "contained_relpath = 'clip.mp4', content_sha256 = %s, size_bytes = %s, "
        "mime_type = 'video/mp4', revision = 2",
        (_HASH, 2**40),
    )
    for statement in (
        (
            "UPDATE artifacts SET state = 'CORRUPT', reason = 'hash', "
            "contained_relpath = 'different.mp4', revision = 3"
        ),
        "UPDATE artifacts SET artifact_id = 'different', revision = 3",
        "UPDATE artifacts SET state = 'PENDING', revision = 3",
    ):
        with pytest.raises(psycopg.errors.CheckViolation), connection.transaction():
            connection.execute(statement)
    for table in ("incidents", "clips"):
        with pytest.raises(psycopg.errors.RestrictViolation), connection.transaction():
            connection.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table)))
    connection.execute("UPDATE artifacts SET state = 'CORRUPT', reason = 'hash', revision = 3")
    connection.execute(
        "UPDATE artifacts SET state = 'PURGED', reason = 'retention', "
        "contained_relpath = NULL, revision = 4"
    )
    assert connection.execute(
        "SELECT state, content_sha256, size_bytes, revision FROM artifacts"
    ).fetchone() == ("PURGED", _HASH, 2**40, 4)


def test_audit_hash_matches_python_unicode_controls_and_nested_json_string(postgres_schemas):
    connection = postgres_schemas.connection
    detail = ' { "z": [1, "한글", {"literal": "\\u00e9"}], "a": "𝄞" } '
    first = _audit(
        actor_id="é/한글/𝄞",
        target_id='quoted"\\/' + "".join(chr(i) for i in range(1, 32)),
        detail_json=detail,
    )
    payload = {key: value for key, value in first.items() if key != "record_hash"}
    assert connection.execute(
        "SELECT seeon_audit_record_hash(%s, %s)", (_ZERO, json.dumps(payload, ensure_ascii=False))
    ).fetchone() == (first["record_hash"],)
    _insert(connection, "audit_events", first)
    second = _audit(
        previous_hash=first["record_hash"],
        actor_id="é",
        retention_class="legal_hold",
        hold_reference="hold",
    )
    _insert(connection, "audit_events", second)
    assert connection.execute(
        "SELECT detail_json, record_hash FROM audit_events ORDER BY audit_id"
    ).fetchall() == [
        (detail, first["record_hash"]),
        (None, second["record_hash"]),
    ]


def test_audit_rejects_changed_payload_wrong_chain_and_invalid_detail(postgres_schemas):
    connection = postgres_schemas.connection
    first = _audit()
    _reject(connection, "audit_events", first | {"actor_id": "tampered"})
    _reject(connection, "audit_events", first | {"record_hash": "f" * 64})
    _reject(connection, "audit_events", _audit(detail_json="[]"))
    _reject(connection, "audit_events", _audit(retention_class="legal_hold"))
    _insert(connection, "audit_events", first)
    _reject(connection, "audit_events", _audit(actor_id="fork"))
    successor = _audit(previous_hash=first["record_hash"], request_id="next")
    _insert(connection, "audit_events", successor)
    assert connection.execute("SELECT count(*) FROM audit_events").fetchone() == (2,)


@contextmanager
def _fresh_audit_connection(schemas):
    # The schema fixture has validated the explicit opt-in. Open a new backend
    # only after product shadows commit, before any PL/pgSQL audit bindings exist.
    try:
        connection = psycopg.connect(
            os.environ["SEEON_TEST_POSTGRES_DSN"], autocommit=True, connect_timeout=5
        )
    except (psycopg.Error, OSError, ValueError, TypeError):
        connection = None
    if connection is None:
        pytest.fail("the configured PostgreSQL test service is unavailable", pytrace=False)
    with connection:
        connection.execute("SET statement_timeout = '10s'")
        connection.execute("SET lock_timeout = '3s'")
        connection.execute(
            sql.SQL("SET search_path TO {}, pg_catalog, pg_temp").format(
                sql.Identifier(schemas.product)
            )
        )
        with connection.transaction(force_rollback=True):
            yield connection


def _assert_audit_namespace_guards(connection):
    first_id = 2**40
    first = _audit(request_id="namespace-first")
    second = _audit(previous_hash=first["record_hash"], request_id="namespace-second")
    _insert(connection, "audit_events", first | {"audit_id": first_id})
    _insert(connection, "audit_events", second | {"audit_id": first_id + 1})
    select_chain = "SELECT audit_id, previous_hash, record_hash FROM audit_events ORDER BY audit_id"
    healthy_chain = [
        (first_id, first["previous_hash"], first["record_hash"]),
        (first_id + 1, second["previous_hash"], second["record_hash"]),
    ]
    assert connection.execute(select_chain).fetchall() == healthy_chain

    successor = _audit(previous_hash=second["record_hash"], request_id="namespace-successor")
    # This is the exact hash a substituted JSON payload would produce, not a
    # malformed hash or an existing unique key that could mask a missing guard.
    forged_hash = audit_record_hash(successor["previous_hash"], _AUDIT_SHADOW_PAYLOAD)
    wrong_chain = _audit(previous_hash=_HASH, request_id="namespace-wrong-chain")
    assert (
        len(
            {
                first["record_hash"],
                second["record_hash"],
                successor["record_hash"],
                forged_hash,
                wrong_chain["record_hash"],
            }
        )
        == 5
    )
    assert wrong_chain["previous_hash"] not in {
        first["previous_hash"],
        second["previous_hash"],
        successor["previous_hash"],
    }
    for values, message in (
        (
            successor | {"audit_id": first_id + 2, "record_hash": forged_hash},
            "audit record hash is invalid",
        ),
        (
            wrong_chain | {"audit_id": first_id + 2},
            "audit hash chain is invalid",
        ),
        (
            successor | {"audit_id": first_id - 1},
            "audit identity must advance",
        ),
    ):
        with pytest.raises(psycopg.errors.CheckViolation) as caught, connection.transaction():
            _insert(connection, "audit_events", values)
        assert caught.value.diag.message_primary == message
        assert connection.execute(select_chain).fetchall() == healthy_chain

    _insert(connection, "audit_events", successor | {"audit_id": first_id + 2})
    assert connection.execute(select_chain).fetchall() == [
        *healthy_chain,
        (first_id + 2, successor["previous_hash"], successor["record_hash"]),
    ]


@pytest.mark.parametrize(
    "ddl,probe,expected",
    [
        pytest.param(
            """
            CREATE FUNCTION audit_shadow_equal(pg_catalog.text, pg_catalog.text)
            RETURNS pg_catalog.bool LANGUAGE sql IMMUTABLE
            AS $$ SELECT true $$;
            CREATE OPERATOR = (
                LEFTARG = pg_catalog.text, RIGHTARG = pg_catalog.text,
                FUNCTION = audit_shadow_equal
            );
            """,
            "SELECT 'left'::pg_catalog.text = 'right'::pg_catalog.text, "
            "'left'::pg_catalog.text OPERATOR(pg_catalog.=) 'right'::pg_catalog.text",
            (True, False),
            id="text-equality-true",
        ),
        pytest.param(
            """
            CREATE FUNCTION audit_shadow_equal(pg_catalog.text, pg_catalog.text)
            RETURNS pg_catalog.bool LANGUAGE sql IMMUTABLE
            AS $$ SELECT NULL::pg_catalog.bool $$;
            CREATE OPERATOR = (
                LEFTARG = pg_catalog.text, RIGHTARG = pg_catalog.text,
                FUNCTION = audit_shadow_equal
            );
            """,
            "SELECT 'left'::pg_catalog.text = 'right'::pg_catalog.text, "
            "'left'::pg_catalog.text OPERATOR(pg_catalog.=) 'right'::pg_catalog.text",
            (None, False),
            id="text-equality-null",
        ),
        pytest.param(
            """
            CREATE FUNCTION audit_shadow_le(pg_catalog.int8, pg_catalog.int8)
            RETURNS pg_catalog.bool LANGUAGE sql IMMUTABLE
            AS $$ SELECT false $$;
            CREATE OPERATOR <= (
                LEFTARG = pg_catalog.int8, RIGHTARG = pg_catalog.int8,
                FUNCTION = audit_shadow_le
            );
            """,
            "SELECT 1::pg_catalog.int8 <= 2::pg_catalog.int8, "
            "1::pg_catalog.int8 OPERATOR(pg_catalog.<=) 2::pg_catalog.int8",
            (False, True),
            id="int8-int8-less-equal-false",
        ),
        pytest.param(
            """
            CREATE FUNCTION audit_shadow_ge(pg_catalog.int8, pg_catalog.int4)
            RETURNS pg_catalog.bool LANGUAGE sql IMMUTABLE
            AS $$ SELECT true $$;
            CREATE OPERATOR >= (
                LEFTARG = pg_catalog.int8, RIGHTARG = pg_catalog.int4,
                FUNCTION = audit_shadow_ge
            );
            """,
            "SELECT 0::pg_catalog.int8 >= 1000000::pg_catalog.int4, "
            "0::pg_catalog.int8 OPERATOR(pg_catalog.>=) 1000000::pg_catalog.int4",
            (True, False),
            id="int8-int4-greater-equal-true",
        ),
        pytest.param(
            """
            CREATE FUNCTION repeat(pg_catalog.text, pg_catalog.int4)
            RETURNS pg_catalog.text LANGUAGE sql IMMUTABLE
            AS $$ SELECT pg_catalog.repeat('a'::pg_catalog.text, 64::pg_catalog.int4) $$;
            """,
            "SELECT repeat('0'::pg_catalog.text, 64::pg_catalog.int4), "
            "pg_catalog.repeat('0'::pg_catalog.text, 64::pg_catalog.int4)",
            (_HASH, _ZERO),
            id="repeat-forged-genesis",
        ),
        pytest.param(
            """
            CREATE FUNCTION audit_shadow_count_step(pg_catalog.int8)
            RETURNS pg_catalog.int8 LANGUAGE sql IMMUTABLE
            AS $$ SELECT 1000000::pg_catalog.int8 $$;
            CREATE AGGREGATE count(*) (
                SFUNC = audit_shadow_count_step,
                STYPE = pg_catalog.int8,
                INITCOND = '1000000'
            );
            """,
            "SELECT count(*), pg_catalog.count(*) FROM (VALUES (1), (2)) AS bounded(value)",
            (1000000, 2),
            id="count-forged-capacity",
        ),
        pytest.param(
            """
            CREATE FUNCTION json_build_object(VARIADIC pg_catalog.text[])
            RETURNS pg_catalog.json LANGUAGE sql IMMUTABLE
            AS $$ SELECT {payload}::pg_catalog.json $$;
            """,
            "SELECT pg_catalog.json_extract_path_text("
            "json_build_object('action'::pg_catalog.text, 'review'::pg_catalog.text), "
            "'shadow'::pg_catalog.text)",
            ("namespace",),
            id="json-build-object-substituted-payload",
        ),
        pytest.param(
            """
            CREATE DOMAIN text AS pg_catalog.text CHECK (VALUE IS NULL);
            """,
            "SELECT pg_catalog.pg_typeof(NULL::text) OPERATOR(pg_catalog.=) "
            "'pg_catalog.text'::pg_catalog.regtype",
            (False,),
            id="text-domain-rejecting-nonnull",
        ),
    ],
)
def test_audit_guards_ignore_product_namespace_shadows(postgres_schemas, ddl, probe, expected):
    # Only the fixture-owned schema retains these objects; its teardown drops
    # them. A separate backend cannot see uncommitted product shadows.
    with postgres_schemas.connection.transaction():
        postgres_schemas.connection.execute(
            sql.SQL(ddl).format(payload=sql.Literal(_AUDIT_SHADOW_PAYLOAD)), prepare=False
        )
    with _fresh_audit_connection(postgres_schemas) as connection:
        assert connection.execute(probe).fetchone() == expected
        _assert_audit_namespace_guards(connection)


def test_audit_hash_survives_temporary_json_casts(postgres_schemas):
    with _fresh_audit_connection(postgres_schemas) as connection:
        # Types, functions and casts all roll back with this fresh session's
        # transaction. Neither audit function has been called in this backend.
        connection.execute(
            sql.SQL(
                """
                CREATE TYPE pg_temp.json AS (payload pg_catalog.text);
                CREATE FUNCTION pg_temp.audit_shadow_text_to_json(pg_catalog.text)
                RETURNS pg_temp.json LANGUAGE sql IMMUTABLE STRICT
                AS $$ SELECT ROW($1)::pg_temp.json $$;
                CREATE FUNCTION pg_temp.audit_shadow_json_to_json(pg_temp.json)
                RETURNS pg_catalog.json LANGUAGE sql IMMUTABLE STRICT
                AS $$ SELECT {payload}::pg_catalog.json $$;
                CREATE CAST (pg_catalog.text AS pg_temp.json)
                    WITH FUNCTION pg_temp.audit_shadow_text_to_json(pg_catalog.text);
                CREATE CAST (pg_temp.json AS pg_catalog.json)
                    WITH FUNCTION pg_temp.audit_shadow_json_to_json(pg_temp.json) AS IMPLICIT;
                """
            ).format(payload=sql.Literal(_AUDIT_SHADOW_PAYLOAD)),
            prepare=False,
        )
        payload = json.dumps(
            {key: value for key, value in _audit().items() if key != "record_hash"},
            ensure_ascii=False,
        )
        # PostgreSQL's unquoted JSON type syntax still denotes pg_catalog.json.
        # Prove the adversarial casts work when explicitly selected, without
        # claiming that the original unqualified ::json was shadowable.
        with connection.transaction(force_rollback=True):
            connection.execute("SET LOCAL search_path TO pg_catalog")
            assert connection.execute(
                "SELECT 'json'::pg_catalog.regtype OPERATOR(pg_catalog.=) "
                "'pg_temp.json'::pg_catalog.regtype"
            ).fetchone() == (False,)
            assert connection.execute(
                "SELECT key, value FROM pg_catalog.json_each_text("
                "%s::pg_catalog.text::pg_temp.json)",
                (payload,),
            ).fetchall() == [("shadow", "namespace")]
            assert connection.execute(
                "SELECT pg_catalog.json_extract_path_text("
                "%s::pg_catalog.text::json, 'actor_id'::pg_catalog.text)",
                (payload,),
            ).fetchone() == ("operator-1",)
            assert connection.execute(
                "SELECT pg_catalog.json_extract_path_text("
                "%s::pg_catalog.text::pg_catalog.json, 'actor_id'::pg_catalog.text)",
                (payload,),
            ).fetchone() == ("operator-1",)
        _assert_audit_namespace_guards(connection)


def test_audit_hash_ignores_implicit_temporary_text_domain(postgres_schemas):
    with _fresh_audit_connection(postgres_schemas) as connection:
        connection.execute("CREATE DOMAIN pg_temp.text AS pg_catalog.text CHECK (VALUE IS NULL)")
        with connection.transaction(force_rollback=True):
            connection.execute("SET LOCAL search_path TO pg_catalog")
            assert connection.execute(
                "SELECT 'text'::pg_catalog.regtype OPERATOR(pg_catalog.=) "
                "'pg_temp.text'::pg_catalog.regtype"
            ).fetchone() == (True,)
            with pytest.raises(psycopg.errors.CheckViolation), connection.transaction():
                connection.execute("SELECT 'nonnull'::text")
            assert connection.execute("SELECT 'nonnull'::pg_catalog.text").fetchone() == (
                "nonnull",
            )
        _assert_audit_namespace_guards(connection)


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE audit_events SET actor_id = 'tampered'",
        "DELETE FROM audit_events",
        "TRUNCATE audit_events",
    ],
)
def test_audit_history_is_immutable_including_truncate(postgres_schemas, statement):
    connection = postgres_schemas.connection
    first = _audit()
    _insert(connection, "audit_events", first)
    with pytest.raises(psycopg.errors.CheckViolation, match="immutable"), connection.transaction():
        connection.execute(statement)
    assert connection.execute("SELECT record_hash FROM audit_events").fetchone() == (
        first["record_hash"],
    )


def test_diagnostic_namespaces_preserve_bytes_and_have_no_cross_namespace_fk(postgres_schemas):
    for name in (postgres_schemas.product, postgres_schemas.diagnostics):
        connection = postgres_schemas.use(name)
        _diagnostic_parents(connection)
        record = _record()
        _insert(connection, "execution_records", record)
        assert connection.execute(
            "SELECT payload, payload_bytes, source_pts_ns FROM execution_records"
        ).fetchone() == (
            record["payload"],
            len(record["payload"].encode()),
            -1,
        )
    connection.execute("DELETE FROM execution_units")
    assert connection.execute("SELECT count(*) FROM execution_records").fetchone() == (0,)
    postgres_schemas.use(postgres_schemas.product)
    assert connection.execute("SELECT count(*) FROM execution_records").fetchone() == (1,)


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"payload_bytes": 1}, psycopg.errors.CheckViolation),
        ({"payload": "not-json", "payload_bytes": 8}, psycopg.errors.CheckViolation),
        ({"record_kind": "unknown"}, psycopg.errors.CheckViolation),
        ({"record_id": "B" * 64}, psycopg.errors.CheckViolation),
        ({"observed_at_ns": -1}, psycopg.errors.CheckViolation),
        ({"producer_sequence": -1}, psycopg.errors.CheckViolation),
        ({"frame_seq": -1}, psycopg.errors.CheckViolation),
        ({"causal_unit_id": "absent"}, psycopg.errors.ForeignKeyViolation),
        ({"segment_id": 999}, psycopg.errors.ForeignKeyViolation),
        ({"provenance_id": "d" * 64}, psycopg.errors.ForeignKeyViolation),
    ],
)
def test_diagnostic_record_constraints(postgres_schemas, changes, error):
    connection = postgres_schemas.use(postgres_schemas.diagnostics)
    _diagnostic_parents(connection)
    _reject(connection, "execution_records", _record(**changes), error)


def test_diagnostic_cascade_keeps_receipts_and_parent_deletes_restrict(postgres_schemas):
    connection = postgres_schemas.use(postgres_schemas.diagnostics)
    _diagnostic_parents(connection)
    _insert(connection, "execution_records", _record())
    batch = {
        "batch_id": "e" * 64,
        "camera_id": "camera-1",
        "worker_boot_id": "boot-1",
        "received_at_ns": 2**40,
        "accepted_records": 1,
        "duplicate_records": 0,
        "rejected_records": 0,
        "receipt": '{ "accepted": 1 }',
    }
    _insert(connection, "execution_batches", batch)
    _reject(connection, "execution_records", _record(), psycopg.errors.UniqueViolation)
    _reject(connection, "execution_batches", batch, psycopg.errors.UniqueViolation)
    assert (
        connection.execute(
            "INSERT INTO execution_batches SELECT * FROM execution_batches "
            "ON CONFLICT (batch_id) DO NOTHING"
        ).rowcount
        == 0
    )
    for table in ("execution_segments", "execution_provenance"):
        with pytest.raises(psycopg.errors.ForeignKeyViolation), connection.transaction():
            connection.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table)))
    connection.execute("DELETE FROM execution_units")
    assert connection.execute("SELECT count(*) FROM execution_records").fetchone() == (0,)
    assert connection.execute("SELECT receipt FROM execution_batches").fetchone() == (
        batch["receipt"],
    )
    assert connection.execute("SELECT count(*) FROM execution_segments").fetchone() == (1,)
    assert connection.execute("SELECT count(*) FROM execution_provenance").fetchone() == (1,)


def test_diagnostic_segment_unit_and_coverage_time_bounds(postgres_schemas):
    connection = postgres_schemas.use(postgres_schemas.diagnostics)
    _diagnostic_parents(connection)
    for statement in (
        "UPDATE execution_segments SET sealed_at_ns = opened_at_ns - 1",
        "UPDATE execution_segments SET storage_state = 'UNKNOWN'",
        "UPDATE execution_units SET last_observed_ns = first_observed_ns - 1",
        "UPDATE execution_units SET terminal = 2",
        "UPDATE execution_units SET causal_state = 'UNKNOWN'",
    ):
        with pytest.raises(psycopg.errors.CheckViolation), connection.transaction():
            connection.execute(statement)
    coverage = {
        "camera_id": "camera-1",
        "worker_boot_id": "boot-1",
        "source_generation": 0,
        "stream_epoch": 0,
        "coverage_kind": "MISSING_NOT_RECORDED",
        "from_sequence": 0,
        "to_sequence": 2**40,
        "from_ns": 2**40,
        "to_ns": 2**40,
        "record_count": 1,
        "exact": 1,
        "cause": "capacity",
        "recorded_at_ns": 2**40,
    }
    for changes in (
        {"to_ns": 0},
        {"from_sequence": None},
        {"from_sequence": 10, "to_sequence": 9},
        {"exact": 2},
        {"coverage_kind": "invented"},
        {"record_count": -1},
    ):
        _reject(connection, "execution_coverage", coverage | changes)
    _insert(connection, "execution_coverage", coverage)
    assert connection.execute(
        "SELECT from_sequence, to_sequence FROM execution_coverage"
    ).fetchone() == (0, 2**40)

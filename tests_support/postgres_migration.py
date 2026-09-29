"""Real PostgreSQL targets and schema-19 SQLite sources for the migration tool tests."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from backend.app.edge_db.functions import audit_record_hash
from backend.app.edge_db.migration.mapping import diagnostics_schema_name
from backend.app.edge_db.migration.provision import (
    ProvisionResult,
    provision,
    set_runtime_password,
)
from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase
from backend.app.features.diagnostics.records import (
    BatchReceipt,
    ExecutionRecordInput,
    IngestBatch,
    Provenance,
    RecordKind,
)
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore
from tests_support.sqlite_source import create_schema19_source, open_source_writer

NOW = "2026-09-27T03:00:00.123456Z"
IN_FLIGHT_ATTEMPT = UUID("00000000-0000-4000-8000-000000000001")
_HASH = "a" * 64
_LIVE_PROVENANCE = Provenance(
    worker_build_revision="worker-rev",
    worker_image_digest="sha256:worker",
    model_digest="sha256:model",
    calibration_digest="sha256:cal",
    preprocessing_identity="pre-v1",
    config_digest="sha256:cfg",
    policy_identity="policy-v1",
    backend_build_revision="backend-rev",
)
_GENESIS = "0" * 64
_BUDGET = PoolBudget(
    max_connections=4,
    max_waiting=8,
    acquire_timeout_sec=5.0,
    statement_timeout_ms=5000,
    lock_timeout_ms=3000,
    startup_timeout_sec=5.0,
)


@dataclass(frozen=True, slots=True)
class MigrationNames:
    admin: psycopg.Connection = field(repr=False)
    dsn: str = field(repr=False)
    schema: str
    runtime_role: str


@pytest.fixture
def migration_names() -> Iterator[MigrationNames]:
    """Per-test schema, diagnostics schema and runtime role on the isolated PG18, dropped after."""
    dsn = os.environ.get("SEEON_TEST_POSTGRES_DSN")
    if dsn is None:
        pytest.fail(
            "SEEON_TEST_POSTGRES_DSN is required; point it at an isolated test database",
            pytrace=False,
        )
    if not dsn.strip() or "\x00" in dsn:
        pytest.fail("SEEON_TEST_POSTGRES_DSN must be nonblank without NUL bytes", pytrace=False)
    try:
        admin = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
    except (psycopg.Error, OSError, ValueError, TypeError):
        admin = None
    if admin is None:
        # Outside the except block: do not chain a libpq error containing the DSN.
        pytest.fail("isolated PostgreSQL test database is unreachable", pytrace=False)
    names = MigrationNames(
        admin=admin,
        dsn=dsn,
        schema=f"seeon_mig_test_{uuid4().hex}",
        runtime_role=f"seeon_mig_rt_{uuid4().hex}",
    )
    try:
        admin.execute("SET statement_timeout TO 5000")
        admin.execute("SET lock_timeout TO 3000")
        yield names
    finally:
        try:
            admin.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {}, {} CASCADE").format(
                    sql.Identifier(names.schema),
                    sql.Identifier(diagnostics_schema_name(names.schema)),
                )
            )
            role_exists = admin.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %s)",
                (names.runtime_role,),
            ).fetchone()[0]
            if role_exists:
                role = sql.Identifier(names.runtime_role)
                admin.execute(sql.SQL("DROP OWNED BY {}").format(role))
                admin.execute(sql.SQL("DROP ROLE {}").format(role))
        finally:
            admin.close()


def provision_target(names: MigrationNames, authority_path: Path) -> ProvisionResult:
    return provision(
        names.dsn,
        schema=names.schema,
        runtime_role=names.runtime_role,
        authority_path=authority_path,
        statement_timeout_ms=5000,
        lock_timeout_ms=2000,
    )


def set_target_runtime_password(names: MigrationNames, password: str) -> bool:
    return set_runtime_password(
        names.dsn,
        schema=names.schema,
        runtime_role=names.runtime_role,
        password=password,
        statement_timeout_ms=5000,
        lock_timeout_ms=2000,
    )


def runtime_verifier(admin: psycopg.Connection, role: str) -> str | None:
    row = admin.execute(
        "SELECT rolpassword FROM pg_catalog.pg_authid WHERE rolname = %s", (role,)
    ).fetchone()
    assert row is not None
    return row[0]


def scram_verifier_accepts(verifier: str, password: str) -> bool:
    """Recompute RFC 5802 StoredKey and ServerKey from the stored salt and iterations."""
    prefix = "SCRAM-SHA-256$"
    assert verifier.startswith(prefix)
    parameters, keys = verifier.removeprefix(prefix).split("$")
    iterations, salt = parameters.split(":")
    stored_key, server_key = keys.split(":")
    salted = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), base64.b64decode(salt), int(iterations)
    )
    client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    expected_stored = base64.b64encode(hashlib.sha256(client_key).digest()).decode()
    expected_server = base64.b64encode(
        hmac.new(salted, b"Server Key", hashlib.sha256).digest()
    ).decode()
    return (expected_stored, expected_server) == (stored_key, server_key)


@contextmanager
def runtime_database(dsn: str, schema: str) -> Iterator[PostgresDatabase]:
    """The application transaction path the tool runs on; no outer transaction."""
    database = PostgresDatabase(dsn, schema, _BUDGET)
    database.start()
    try:
        yield database
    finally:
        database.close(timeout_sec=3.0)


@contextmanager
def runtime_role_database(
    dsn: str, schema: str, runtime_role: str, budget: PoolBudget = _BUDGET
) -> Iterator[PostgresDatabase]:
    """A pool whose every session runs as the runtime role, with only its grants."""
    conninfo = make_conninfo(dsn, options=f"-c role={runtime_role}")
    database = PostgresDatabase(conninfo, schema, budget)
    database.start()
    try:
        yield database
    finally:
        database.close(timeout_sec=3.0)


def ingest_live_record(database: PostgresDatabase, label: str) -> BatchReceipt:
    """One synthetic live execution record through lane E's real ingest path."""

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    record = ExecutionRecordInput(
        record_id=digest(f"record-{label}"),
        record_kind=RecordKind.SDK_FRAME,
        camera_id="cam-live",
        worker_boot_id="boot-live",
        source_generation=0,
        stream_epoch=0,
        producer="sdk",
        producer_sequence=0,
        observed_at_ns=100,
        time_quality="trusted",
        causal_unit_id=f"unit-{label}",
        outcome="ok",
        payload={},
    )
    batch = IngestBatch(
        batch_id=digest(f"batch-{label}"),
        camera_id="cam-live",
        worker_boot_id="boot-live",
        provenance=_LIVE_PROVENANCE,
        records=(record,),
        gaps=(),
    )
    store = ExecutionRecordStore(
        database, RetentionBudget(total_bytes=2**20), clock=lambda: 1_000_000
    )
    return store.ingest_batch(batch)


@dataclass(frozen=True, slots=True)
class MigrationTarget:
    admin: psycopg.Connection = field(repr=False)
    dsn: str = field(repr=False)
    schema: str
    runtime_role: str
    authority_path: Path
    database: PostgresDatabase = field(repr=False)


@pytest.fixture
def migration_target(migration_names: MigrationNames, tmp_path: Path) -> Iterator[MigrationTarget]:
    """A provisioned, fenced generation-1 schema with its owner-only authority file."""
    authority_directory = tmp_path / "authority"
    authority_directory.mkdir(mode=0o700)
    authority_path = authority_directory / "authority.json"
    provision_target(migration_names, authority_path)
    with runtime_database(migration_names.dsn, migration_names.schema) as database:
        yield MigrationTarget(
            admin=migration_names.admin,
            dsn=migration_names.dsn,
            schema=migration_names.schema,
            runtime_role=migration_names.runtime_role,
            authority_path=authority_path,
            database=database,
        )


def authority_row(admin: psycopg.Connection, schema: str) -> tuple[int, UUID, bool, bool]:
    (row,) = admin.execute(
        sql.SQL("SELECT generation, writer_token, accepting, egress_enabled FROM {}").format(
            sql.Identifier(schema, "deployment_authority")
        )
    ).fetchall()
    return tuple(row)


def authority_file_token(path: Path) -> tuple[int, UUID]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["generation"], UUID(payload["writer_token"])


def table_counts(admin: psycopg.Connection, schema: str) -> dict[str, int]:
    tables = admin.execute(
        "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = %s ORDER BY tablename",
        (schema,),
    ).fetchall()
    return {
        name: admin.execute(
            sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(schema, name))
        ).fetchone()[0]
        for (name,) in tables
    }


def ledger(admin: psycopg.Connection, schema: str) -> list[tuple[object, ...]]:
    rows = admin.execute(
        sql.SQL(
            "SELECT version, name, checksum, source_schema_version, source_db_sha256, "
            "reconciliation_sha256 FROM {} ORDER BY version"
        ).format(sql.Identifier(schema, "schema_migrations"))
    ).fetchall()
    return [tuple(row) for row in rows]


def insert_row(connection: sqlite3.Connection, table: str, values: Mapping[str, object]) -> None:
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values()))


def seed_source(state_dir: Path) -> Path:
    """Create the OLD schema-19 database and fill it with synthetic rows only."""
    source = create_schema19_source(state_dir / "edge.sqlite3")
    with closing(open_source_writer(source)) as writer:
        insert_row(
            writer,
            "credentials",
            {
                "id": 1,
                "username": "operator",
                "algorithm": "scrypt",
                "salt": bytes(16),
                "password_hash": bytes(range(64)),
                "updated_at": NOW,
            },
        )
        insert_row(writer, "edge_site", {"id": 1, "updated_at": NOW})
        insert_row(
            writer,
            "locations",
            {
                "location_id": "floor-1",
                "kind": "FLOOR",
                "parent_location_id": None,
                "parent_kind": None,
                "name": "1층",
                "order_index": 0,
                "capacity": None,
                "created_at": NOW,
                "updated_at": NOW,
            },
        )
        insert_row(
            writer,
            "locations",
            {
                "location_id": "room-1",
                "kind": "ROOM",
                "parent_location_id": "floor-1",
                "parent_kind": "FLOOR",
                "name": "101호",
                "order_index": 0,
                "capacity": 4,
                "created_at": NOW,
                "updated_at": NOW,
            },
        )
        insert_row(
            writer,
            "cameras",
            {
                "camera_id": "camera-1",
                "label": "camera",
                "rtsp_url": "rtsp://camera.invalid/stream",
                "normalized_stream_identity": "stream-1",
                "mapping_state": "UNMAPPED",
                "never_connected": 1,
                "revision": 1,
                "created_at": NOW,
                "updated_at": NOW,
            },
        )
        insert_row(
            writer,
            "policies",
            {
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
                "activated_at": NOW,
                "updated_at": NOW,
            },
        )
        insert_row(
            writer,
            "clips",
            {
                "clip_id": "clip-1",
                "camera_id": "camera-1",
                "event_facet": "fall",
                "started_at": NOW,
                "local_state": "UNAVAILABLE",
                "local_reason": "not-recorded",
                "publish_state": "WAITING",
                "retention_state": "RETAINED",
                "revision": 1,
                "created_at": NOW,
                "updated_at": NOW,
            },
        )
        add_incident(writer, 1)
        insert_row(
            writer,
            "artifacts",
            {
                "incident_id": "incident-1",
                "kind": "PRIMARY_CLIP",
                "state": "PENDING",
                "revision": 1,
                "created_at": NOW,
                "updated_at": NOW,
            },
        )
        append_audit(writer)
        append_audit(writer)
    return source


def source_and_destination(root: Path) -> tuple[Path, Path]:
    """The seeded old database under root/state and an unused snapshot path beside it."""
    state = root / "state"
    state.mkdir()
    snapshots = root / "snapshots"
    snapshots.mkdir()
    return seed_source(state), snapshots / "edge.snapshot.sqlite3"


def add_incident(connection: sqlite3.Connection, number: int) -> None:
    insert_row(
        connection,
        "incidents",
        {
            "incident_id": f"incident-{number}",
            "edge_event_id": f"event-{number}",
            "facility_id": "facility-1",
            "camera_id": "camera-1",
            "event_type": "fall",
            "probability": 0.1,
            "detected_at": NOW,
            "lifecycle_state": "OPEN",
            "provenance_state": "MISSING",
            "provenance_missing_reason": "not-recorded",
            "review_version": 0,
            "revision": 1,
            "created_at": NOW,
            "updated_at": NOW,
        },
    )


def append_audit(connection: sqlite3.Connection) -> None:
    """Append one record chained to the tail, as the old runtime's audit writer does."""
    tail = connection.execute(
        "SELECT record_hash FROM audit_events ORDER BY audit_id DESC LIMIT 1"
    ).fetchone()
    previous = _GENESIS if tail is None else tail[0]
    values = {
        "occurred_at": NOW,
        "recorded_at": NOW,
        "clock_quality": "trusted",
        "actor_type": "user",
        "actor_id": "operator-1",
        "auth_mechanism": "session",
        "action": "review",
        "target_type": "incident",
        "target_id": "incident-1",
        "outcome": "success",
        "previous_hash": previous,
        "retention_class": "standard",
        "reason": None,
        "request_id": None,
        "interaction_id": None,
        "detail_json": None,
        "hold_reference": None,
    }
    record_hash = audit_record_hash(previous, json.dumps(values))
    insert_row(connection, "audit_events", {**values, "record_hash": record_hash})


def make_worker_state(root: Path) -> Path:
    """A stopped old worker's state: lease and queue lock files, queued and dead-letter alerts."""
    state = root / "worker-state"
    queue = state / "delivery-queue"
    dead_letter = state / "delivery-queue-dead-letter"
    queue.mkdir(parents=True)
    dead_letter.mkdir()
    (state / ".gpu.lease").write_bytes(b"")
    (queue / ".delivery-queue.lock").write_bytes(b"")
    (queue / "0001-event-1.json").write_bytes(b'{"synthetic":1}\n')
    (queue / "0002-event-2.json").write_bytes(b'{"synthetic":2}\n')
    (queue / ".0003-event-3.json.tmp").write_bytes(b'{"synthetic":3')
    (dead_letter / "0000-event-0.json").write_bytes(b'{"synthetic":0}\n')
    return state


def insert_in_flight_outbox(admin: psycopg.Connection, schema: str) -> None:
    """A live sender's claim of event-1: PENDING insert, attempt insert, then IN_FLIGHT."""
    outbox = sql.Identifier(schema, "event_outbox")
    envelope = '{"edge_event_id":"event-1"}'
    with admin.transaction():
        admin.execute(
            sql.SQL(
                "INSERT INTO {} (edge_event_id, envelope, envelope_sha256, envelope_bytes, "
                "backend_camera_id, state, accepted_generation, accepted_at, retry_at) "
                "VALUES ('event-1', %s, encode(sha256(convert_to(%s, 'UTF8')), 'hex'), "
                "octet_length(%s), 'camera-1', 'PENDING', 1, now(), now())"
            ).format(outbox),
            (envelope, envelope, envelope),
        )
        admin.execute(
            sql.SQL(
                "INSERT INTO {} (attempt_id, edge_event_id, ordinal, writer_generation, "
                "started_at) VALUES (%s, 'event-1', 1, 1, now())"
            ).format(sql.Identifier(schema, "event_delivery_attempts")),
            (IN_FLIGHT_ATTEMPT,),
        )
        admin.execute(
            sql.SQL(
                "UPDATE {} SET state = 'IN_FLIGHT', attempt_count = 1, active_attempt = %s, "
                "lease_until = clock_timestamp() + interval '1 hour' "
                "WHERE edge_event_id = 'event-1'"
            ).format(outbox),
            (IN_FLIGHT_ATTEMPT,),
        )


__all__ = [
    "IN_FLIGHT_ATTEMPT",
    "NOW",
    "MigrationNames",
    "MigrationTarget",
    "add_incident",
    "append_audit",
    "authority_file_token",
    "authority_row",
    "insert_in_flight_outbox",
    "insert_row",
    "ledger",
    "make_worker_state",
    "migration_names",
    "migration_target",
    "open_source_writer",
    "provision_target",
    "runtime_database",
    "runtime_verifier",
    "scram_verifier_accepts",
    "seed_source",
    "set_target_runtime_password",
    "source_and_destination",
    "table_counts",
]

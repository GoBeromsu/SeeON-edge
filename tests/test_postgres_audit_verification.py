"""Opt-in real PostgreSQL component evidence, not readiness/deployment qualification.

The existing sandbox owns the namespace/pool and requires an explicit test DSN.
Lowered-limit tests exercise refusal branches, not million-row cost/capacity.
Fault hooks wrap real owner transactions; none substitutes a fake database.
"""

from __future__ import annotations

import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from importlib.resources import files
from queue import Queue
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING

import psycopg
import pytest
from psycopg import sql
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row

from backend.app.edge_db.functions import audit_record_hash
from backend.app.edge_db.postgres import (
    CommitOutcomeUnknown,
    PoolBudget,
    PostgresDatabase,
    PostgresUnavailable,
)
from backend.app.features.audit import postgres_store
from backend.app.features.audit import postgres_verification as native
from backend.app.features.audit.catalog import AuditAction, camera_probe_detail, empty_detail
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.audit.postgres_verification import PostgresAuditCheckpoint
from backend.app.features.audit.store import AuditEvent, _payload
from backend.app.features.audit.verification import (
    AUDIT_ROW_COLUMNS,
    GENESIS_HASH,
    AuditVerificationError,
    verify_row,
)

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_TIME = "2026-09-27T04:00:00.123Z"
_SENTINEL = "sensitive-audit-verification-sentinel"
_FAILURE = "audit verification failed"


@pytest.fixture
def audit_store(postgres_product_sandbox: ProductSandbox) -> PostgresAuditStore:
    return PostgresAuditStore(postgres_product_sandbox.database, postgres_product_sandbox.authority)


def _event(target: str = "protected") -> AuditEvent:
    return AuditEvent(
        occurred_at=_TIME,
        actor_id="test-operator",
        action=AuditAction.AUDIT_LIST,
        target_id=target,
        detail=empty_detail(AuditAction.AUDIT_LIST),
    )


def _product_sql() -> str:
    return files("backend.app.edge_db").joinpath("postgres_product.sql").read_text(encoding="utf-8")


def _function_sql(name: str) -> str:
    match = re.search(
        rf"^CREATE FUNCTION {re.escape(name)}\(.*?\$\$;", _product_sql(), re.MULTILINE | re.DOTALL
    )
    assert match is not None
    return match.group(0)


def _trigger_sql(name: str) -> str:
    match = re.search(
        rf"^CREATE TRIGGER {re.escape(name)} .*?;", _product_sql(), re.MULTILINE | re.DOTALL
    )
    assert match is not None
    return match.group(0)


def _replace_table(connection: psycopg.Connection) -> None:
    # Test-only replacement from the trusted product DDL, including real guards.
    source = _product_sql()
    table = re.search(r"^CREATE TABLE audit_events \(.*?\n\);", source, re.MULTILINE | re.DOTALL)
    assert table is not None
    connection.execute("DROP TABLE audit_events")
    connection.execute(table.group(0))
    for statement in re.findall(
        r"^CREATE (?:TRIGGER audit_events_|(?:UNIQUE )?INDEX audit_events_).*?;",
        source,
        re.MULTILINE | re.DOTALL,
    ):
        connection.execute(statement)


def _insert_ids(connection: psycopg.Connection, ids: tuple[int, ...]) -> None:
    previous_row = connection.execute(
        "SELECT record_hash FROM audit_events ORDER BY audit_id DESC LIMIT 1"
    ).fetchone()
    previous = GENESIS_HASH if previous_row is None else previous_row[0]
    for audit_id in ids:
        payload = _payload(_event(str(audit_id)), _TIME, previous)
        record_hash = audit_record_hash(previous, json.dumps(payload))
        values = {"audit_id": audit_id, **payload, "record_hash": record_hash}
        connection.execute(
            sql.SQL("INSERT INTO audit_events ({}) VALUES ({})").format(
                sql.SQL(",").join(map(sql.Identifier, values)),
                sql.SQL(",").join(sql.Placeholder() for _ in values),
            ),
            tuple(values.values()),
        )
        previous = record_hash


def _assert_private(failure, caplog) -> None:
    assert type(failure.value) is AuditVerificationError
    assert str(failure.value) == failure.value.reason == _FAILURE
    assert failure.value.__cause__ is None
    assert failure.value.__suppress_context__
    assert _SENTINEL not in "".join(traceback.format_exception(failure.value))
    assert _SENTINEL not in caplog.text


def _refuses(store, caplog, checkpoint=None) -> None:
    with pytest.raises(AuditVerificationError) as failure:
        store.verify(checkpoint)
    _assert_private(failure, caplog)


def _wait_for_lock(sandbox: ProductSandbox, pid: int) -> None:
    deadline = monotonic() + 1.5
    pacing = Event()
    while True:
        waiting = sandbox.database.read(
            lambda connection: connection.execute(
                "SELECT 1 FROM pg_catalog.pg_locks WHERE pid=%s AND NOT granted LIMIT 1", (pid,)
            ).fetchone()
        )
        if waiting == (1,):
            return
        assert monotonic() < deadline, "transaction never reached its real lock wait"
        pacing.wait(0.005)


def test_empty_checkpoint_is_immutable_and_extends_without_zero_sentinel(audit_store) -> None:
    empty = audit_store.verify()
    assert isinstance(empty, PostgresAuditCheckpoint)
    assert empty.row_count == 0 and empty.audit_id is None and empty.record_hash == GENESIS_HASH
    assert len(empty.identity) == 4 and all(value > 0 for value in empty.identity)
    assert not hasattr(empty, "__dict__")
    with pytest.raises(FrozenInstanceError):
        empty.row_count = 1
    row = audit_store.append(_event())
    checkpoint = audit_store.verify(empty)
    assert (checkpoint.row_count, checkpoint.audit_id, checkpoint.record_hash) == (
        1,
        row.audit_id,
        row.record_hash,
    )
    assert checkpoint.identity == empty.identity
    assert checkpoint.guard_fingerprint == empty.guard_fingerprint
    assert audit_store.verify(checkpoint) == audit_store.verify() == checkpoint


def test_quoted_mixed_case_namespace_uses_exact_canonical_captured_path(
    postgres_product_sandbox,
    caplog,
) -> None:
    sandbox = postgres_product_sandbox
    schema = sandbox.schema + 'Mixed"Case'
    database = PostgresDatabase(
        sandbox.dsn,
        schema,
        PoolBudget(
            max_connections=1,
            max_waiting=1,
            acquire_timeout_sec=1.0,
            statement_timeout_ms=5000,
            lock_timeout_ms=3000,
            startup_timeout_sec=5.0,
        ),
    )
    sandbox.admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        with sandbox.admin.transaction():
            sandbox.admin.execute(
                sql.SQL("SET LOCAL search_path TO {}, pg_catalog, pg_temp").format(
                    sql.Identifier(schema)
                )
            )
            sandbox.admin.execute(_product_sql(), prepare=False)
        database.start()
        store = PostgresAuditStore(database, sandbox.authority)
        empty = store.verify()
        assert (empty.row_count, empty.audit_id, empty.record_hash) == (0, None, GENESIS_HASH)
        database.transact(lambda connection: _insert_ids(connection, (0, 7)))
        checkpoint = store.verify(empty)
        assert (checkpoint.row_count, checkpoint.audit_id) == (2, 7)
        assert checkpoint.guard_fingerprint == empty.guard_fingerprint
        assert store.verify() == store.verify(checkpoint) == checkpoint
        # The other healthy product namespace is not an acceptable substitute.
        sandbox.admin.execute(
            sql.SQL("ALTER FUNCTION {}() SET search_path TO {}, pg_catalog, pg_temp").format(
                sql.Identifier(schema, "seeon_audit_insert"), sql.Identifier(sandbox.schema)
            )
        )
        _refuses(store, caplog)
        _refuses(store, caplog, checkpoint)
    finally:
        database.close()
        sandbox.admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_unicode_and_canonical_detail_use_the_shared_row_owner(audit_store, monkeypatch) -> None:
    rows = audit_store.append_batch(
        tuple(
            replace(
                _event('대상 "침대"\\' + suffix),
                actor_id="운영자 " + suffix,
                action=AuditAction.CAMERA_PROBE,
                detail=camera_probe_detail(True, None),
            )
            for suffix in ("é", "é")
        )
    )
    checked = []

    def observed(row, previous):
        checked.append(row)
        return verify_row(row, previous)

    monkeypatch.setattr(native, "verify_row", observed)
    checkpoint = audit_store.verify()
    assert [row[0] for row in checked] == [row.audit_id for row in rows]
    assert all(len(row) == len(AUDIT_ROW_COLUMNS) == 19 for row in checked)
    assert checked[0][14] == rows[0].detail.json
    assert (checkpoint.row_count, checkpoint.record_hash) == (2, rows[-1].record_hash)


@pytest.mark.parametrize("ids", [(-9, 0, 7, 43), (0,), (-(2**63), -11, 2**63 - 1)])
def test_explicit_nonpositive_and_gapped_ids_are_not_skipped(
    postgres_product_sandbox,
    audit_store,
    ids,
) -> None:
    sandbox = postgres_product_sandbox
    sandbox.database.transact(lambda connection: _insert_ids(connection, ids))
    checkpoint = audit_store.verify()
    assert (checkpoint.row_count, checkpoint.audit_id) == (len(ids), ids[-1])
    assert audit_store.verify(checkpoint) == checkpoint


def test_complete_multi_page_scan_rechecks_every_row_and_closes_inside_callback(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
) -> None:
    records = audit_store.append_batch(tuple(_event(str(i)) for i in range(1003)))
    fetchmany = psycopg.ServerCursor.fetchmany
    callback = postgres_store._verify_snapshot
    sizes, checked, closes = [], [], []

    def bounded_fetch(cursor, size=0):
        assert cursor.name == "seeon_audit_verification"
        assert 0 < size <= 1000
        assert cursor.connection.execute(
            "SELECT is_holdable,is_scrollable FROM pg_catalog.pg_cursors WHERE name=%s",
            (cursor.name,),
        ).fetchone() == (False, False)
        rows = fetchmany(cursor, size)
        sizes.append(len(rows))
        return rows

    def observed_row(row, previous):
        checked.append(row[0])
        return verify_row(row, previous)

    def observed_callback(connection, schema, checkpoint):
        result = callback(connection, schema, checkpoint)
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        assert (
            connection.execute(
                "SELECT name FROM pg_catalog.pg_cursors WHERE name='seeon_audit_verification'"
            ).fetchone()
            is None
        )
        closes.append(1)
        return result

    monkeypatch.setattr(psycopg.ServerCursor, "fetchmany", bounded_fetch)
    monkeypatch.setattr(native, "verify_row", observed_row)
    monkeypatch.setattr(postgres_store, "_verify_snapshot", observed_callback)
    first = audit_store.verify()
    assert sizes == [1000, 3, 0]
    assert checked == [record.audit_id for record in records]
    checked.clear()
    sizes.clear()
    assert audit_store.verify(first) == first
    assert checked == [record.audit_id for record in records]
    assert sizes == [1000, 3, 0] and closes == [1, 1]


@pytest.mark.parametrize("pause", ["count", "page"])
def test_concurrent_append_commits_before_reader_finishes_but_not_in_its_snapshot(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    pause,
) -> None:
    sandbox = postgres_product_sandbox
    records = audit_store.append_batch(tuple(_event(str(i)) for i in range(1001)))
    entered, release = Event(), Event()
    scan, fetchmany = native._scan, psycopg.ServerCursor.fetchmany

    def hold():
        if not entered.is_set():
            entered.set()
            assert release.wait(4), "test did not release its reader"

    def paused_scan(connection, table, count, checkpoint):
        assert (
            connection.execute(
                "SELECT 1 FROM pg_catalog.pg_locks WHERE pid=pg_catalog.pg_backend_pid() "
                "AND locktype='advisory' LIMIT 1"
            ).fetchone()
            is None
        )
        if pause == "count":
            hold()
        return scan(connection, table, count, checkpoint)

    def paused_fetch(cursor, size=0):
        result = fetchmany(cursor, size)
        if pause == "page":
            hold()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(native, "_scan", paused_scan)
        patch.setattr(psycopg.ServerCursor, "fetchmany", paused_fetch)
        with ThreadPoolExecutor(max_workers=2) as executor:
            reader = executor.submit(audit_store.verify)
            try:
                assert entered.wait(2)
                writer = executor.submit(audit_store.append, _event("concurrent"))
                appended = writer.result(timeout=2)
                assert not reader.done()
                assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (
                    1002,
                )
            finally:
                release.set()
            prefix = reader.result(timeout=2)
    assert (prefix.row_count, prefix.record_hash) == (1001, records[-1].record_hash)
    extended = audit_store.verify(prefix)
    assert (extended.row_count, extended.audit_id, extended.record_hash) == (
        1002,
        appended.audit_id,
        appended.record_hash,
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("action", _SENTINEL),
        ("auth_mechanism", _SENTINEL),
        ("detail_json", '{"version":2}'),
        ("detail_json", '{"version":1,"unexpected":"' + _SENTINEL + '"}'),
        ("detail_json", None),
        ("previous_hash", "f" * 64),
        ("record_hash", "e" * 64),
        ("actor_id", _SENTINEL),
    ],
)
def test_restored_guards_do_not_hide_old_semantic_or_hash_corruption_without_append(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    caplog,
    column,
    value,
) -> None:
    sandbox = postgres_product_sandbox
    first, _ = audit_store.append_batch((_event("first"), _event("last")))
    checkpoint = audit_store.verify()
    with sandbox.admin.transaction():
        sandbox.admin.execute("ALTER TABLE audit_events DISABLE TRIGGER USER")
        sandbox.admin.execute(
            sql.SQL("UPDATE audit_events SET {}=%s WHERE audit_id=%s").format(
                sql.Identifier(column)
            ),
            (value, first.audit_id),
        )
        sandbox.admin.execute("ALTER TABLE audit_events ENABLE TRIGGER USER")
    checked = []

    def observed(row, previous):
        checked.append(row[0])
        return verify_row(row, previous)

    monkeypatch.setattr(native, "verify_row", observed)
    _refuses(audit_store, caplog, checkpoint)
    _refuses(audit_store, caplog)
    assert checked == [first.audit_id, first.audit_id]


@pytest.mark.parametrize(
    ("column", "byte_limit"), [("actor_id", 512), ("reason", 1024), ("detail_json", 16_384)]
)
def test_oversized_row_fields_are_rejected_before_transfer_to_shared_validator(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    caplog,
    column,
    byte_limit,
) -> None:
    sandbox = postgres_product_sandbox
    record = audit_store.append(_event())
    checkpoint = audit_store.verify()
    oversized = _SENTINEL + "x" * (byte_limit + 1 - len(_SENTINEL.encode("utf-8")))
    with sandbox.admin.transaction():
        checks = sandbox.admin.execute(
            "SELECT conname FROM pg_constraint "
            "WHERE conrelid='audit_events'::regclass AND contype='c'"
        ).fetchall()
        for (name,) in checks:
            sandbox.admin.execute(
                sql.SQL("ALTER TABLE audit_events DROP CONSTRAINT {}").format(sql.Identifier(name))
            )
        sandbox.admin.execute("ALTER TABLE audit_events DISABLE TRIGGER USER")
        # One byte over the transfer ceiling still fits the actor secondary
        # index; all original indexes, including unique protections, stay put.
        length = sandbox.admin.execute(
            sql.SQL("UPDATE audit_events SET {}=%s RETURNING pg_catalog.octet_length({})").format(
                sql.Identifier(column), sql.Identifier(column)
            ),
            (oversized,),
        ).fetchone()
        assert length == (byte_limit + 1,)
        sandbox.admin.execute("ALTER TABLE audit_events ENABLE TRIGGER USER")
    checked, transferred = [], []
    fetchmany = psycopg.ServerCursor.fetchmany

    def observed_fetch(cursor, size=0):
        rows = fetchmany(cursor, size)
        transferred.extend(rows)
        return rows

    def observed(row, previous):
        checked.append(row)
        return verify_row(row, previous)

    monkeypatch.setattr(psycopg.ServerCursor, "fetchmany", observed_fetch)
    monkeypatch.setattr(native, "verify_row", observed)
    _refuses(audit_store, caplog)
    _refuses(audit_store, caplog, checkpoint)
    assert len(transferred) == 2
    column_index = AUDIT_ROW_COLUMNS.index(column)
    assert all(
        row[0] == record.audit_id and row[column_index] is None and row[-1] is False
        for row in transferred
    )
    assert not checked


@pytest.mark.parametrize(
    "tamper",
    [
        "missing",
        "disabled",
        "replica",
        "always",
        "additional",
        "wrong_binding",
        "wrong_event",
        "after",
        "column_filter",
        "when",
        "arguments",
        "wrong_namespace",
        "wrong_relation",
    ],
)
def test_exact_trigger_contract_refuses_impersonation_and_extra_behavior(
    postgres_product_sandbox,
    audit_store,
    caplog,
    tamper,
) -> None:
    admin = postgres_product_sandbox.admin
    name = "audit_events_immutable_update"
    if tamper == "disabled":
        admin.execute(f"ALTER TABLE audit_events DISABLE TRIGGER {name}")
    elif tamper in {"replica", "always"}:
        admin.execute(f"ALTER TABLE audit_events ENABLE {tamper.upper()} TRIGGER {name}")
    elif tamper == "additional":
        admin.execute(_trigger_sql(name).replace(name, "audit_events_extra"))
    else:
        admin.execute(f"DROP TRIGGER {name} ON audit_events")
        statement = _trigger_sql(name)
        if tamper == "missing":
            _refuses(audit_store, caplog)
            return
        if tamper == "wrong_binding":
            statement = statement.replace("seeon_audit_immutable()", "seeon_audit_serialize()")
        elif tamper == "wrong_event":
            statement = statement.replace("BEFORE UPDATE", "BEFORE DELETE")
        elif tamper == "after":
            statement = statement.replace("BEFORE UPDATE", "AFTER UPDATE")
        elif tamper == "column_filter":
            statement = statement.replace("UPDATE ON", "UPDATE OF actor_id ON")
        elif tamper == "when":
            statement = statement.replace(
                "EXECUTE FUNCTION", "WHEN (OLD.audit_id > 0) EXECUTE FUNCTION"
            )
        elif tamper == "arguments":
            statement = statement.replace(
                "seeon_audit_immutable()", "seeon_audit_immutable('" + _SENTINEL * 8000 + "')"
            )
        elif tamper == "wrong_namespace":
            admin.execute("CREATE TEMP TABLE namespace_anchor (id bigint)")
            admin.execute(
                _function_sql("seeon_audit_immutable").replace(
                    "CREATE FUNCTION seeon_audit_immutable",
                    "CREATE FUNCTION pg_temp.seeon_audit_immutable",
                )
            )
            statement = statement.replace(
                "FUNCTION seeon_audit_immutable", "FUNCTION pg_temp.seeon_audit_immutable"
            )
        elif tamper == "wrong_relation":
            admin.execute("CREATE TABLE trigger_decoy (LIKE audit_events)")
            statement = statement.replace("ON audit_events", "ON trigger_decoy")
        admin.execute(statement)
    _refuses(audit_store, caplog)


@pytest.mark.parametrize(
    ("name", "before", "after"),
    [
        ("seeon_audit_immutable", "audit events are immutable", "audit  events are immutable"),
        ("seeon_audit_serialize", "pg_advisory_xact_lock(TG_RELID::bigint)", "1"),
        (
            "seeon_audit_insert",
            "(NEW.record_hash OPERATOR(pg_catalog.=) expected_hash) IS NOT TRUE",
            "false",
        ),
        ("seeon_audit_record_hash", "'hex'", "'hex '"),
    ],
)
def test_each_function_body_is_exact_including_whitespace_inside_literals(
    postgres_product_sandbox,
    audit_store,
    caplog,
    name,
    before,
    after,
) -> None:
    statement = _function_sql(name).replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1)
    assert before in statement
    postgres_product_sandbox.admin.execute(statement.replace(before, after))
    _refuses(audit_store, caplog)


def test_product_text_equality_operator_cannot_bless_a_tampered_guard(
    postgres_product_sandbox,
    audit_store,
    caplog,
) -> None:
    sandbox = postgres_product_sandbox
    admin = sandbox.admin
    audit_store.append_batch((_event("first"), _event("last")))
    checkpoint = audit_store.verify()
    trusted = _function_sql("seeon_audit_immutable")
    function_oid = admin.execute(
        "SELECT 'seeon_audit_immutable()'::pg_catalog.regprocedure::pg_catalog.oid"
    ).fetchone()[0]
    equality = sql.Identifier(sandbox.schema, "test_text_equal")
    admin.execute(
        sql.SQL(
            "CREATE FUNCTION {}(pg_catalog.text, pg_catalog.text) "
            "RETURNS pg_catalog.bool LANGUAGE sql IMMUTABLE STRICT "
            "SET search_path TO pg_catalog AS $$ SELECT true $$"
        ).format(equality)
    )
    admin.execute(
        sql.SQL(
            "CREATE OPERATOR {}.= (FUNCTION = {}, "
            "LEFTARG = pg_catalog.text, RIGHTARG = pg_catalog.text)"
        ).format(sql.Identifier(sandbox.schema), equality)
    )
    assert audit_store.verify(checkpoint) == checkpoint

    replacement = trusted.replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1)
    guard = "RAISE EXCEPTION 'audit events are immutable' USING ERRCODE = '23514';"
    assert guard in replacement
    admin.execute(replacement.replace(guard, "RETURN OLD;"))
    body = trusted.split("$$", 2)[1]
    # Prove actual source equality is hijacked, while history and every other
    # function property stay unchanged. Only the verifier's path is isolated.
    assert admin.execute(
        "SELECT p.prosrc = %s, p.prosrc OPERATOR(pg_catalog.=) %s "
        "FROM pg_catalog.pg_proc p WHERE p.oid = %s",
        (body, body, function_oid),
    ).fetchone() == (True, False)
    _refuses(audit_store, caplog)
    _refuses(audit_store, caplog, checkpoint)
    admin.execute(replacement)
    assert audit_store.verify() == audit_store.verify(checkpoint) == checkpoint


@pytest.mark.parametrize(
    "alteration",
    [
        "seeon_audit_immutable() SECURITY DEFINER",
        "seeon_audit_serialize() STABLE",
        "seeon_audit_insert() STRICT",
        "seeon_audit_record_hash(text,text) CALLED ON NULL INPUT",
        "seeon_audit_record_hash(text,text) COST 1",
        "seeon_audit_record_hash(text,text) PARALLEL SAFE",
        "seeon_audit_immutable() SET statement_timeout TO 0",
        "seeon_audit_serialize() SET search_path TO public, pg_catalog",
        "seeon_audit_record_hash(text,text) SET search_path TO pg_catalog, pg_temp",
        "seeon_audit_insert() RESET search_path",
    ],
)
def test_function_execution_properties_and_complete_settings_are_required(
    postgres_product_sandbox,
    audit_store,
    caplog,
    alteration,
) -> None:
    postgres_product_sandbox.admin.execute("ALTER FUNCTION " + alteration)
    _refuses(audit_store, caplog)


@pytest.mark.parametrize(
    "suffix", [", pg_catalog", ", pg_temp, pg_catalog", ", pg_catalog, pg_temp, public"]
)
def test_insert_function_requires_exact_captured_path(
    postgres_product_sandbox,
    audit_store,
    caplog,
    suffix,
) -> None:
    sandbox = postgres_product_sandbox
    sandbox.admin.execute(
        sql.SQL("ALTER FUNCTION seeon_audit_insert() SET search_path TO {}" + suffix).format(
            sql.Identifier(sandbox.schema)
        )
    )
    _refuses(audit_store, caplog)


@pytest.mark.parametrize("tamper", ["signature", "overload", "huge_body", "huge_settings"])
def test_function_metadata_is_checked_without_unbounded_client_materialization(
    postgres_product_sandbox,
    audit_store,
    caplog,
    tamper,
) -> None:
    admin = postgres_product_sandbox.admin
    if tamper in {"signature", "overload"}:
        if tamper == "signature":
            admin.execute("DROP FUNCTION seeon_audit_record_hash(text,text)")
        admin.execute(
            "CREATE FUNCTION seeon_audit_record_hash(value text) RETURNS text "
            "LANGUAGE sql AS $$ SELECT value $$"
        )
    elif tamper == "huge_body":
        statement = (
            _function_sql("seeon_audit_immutable")
            .replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1)
            .replace("BEGIN", "BEGIN\n-- " + _SENTINEL * 8000 + "\n", 1)
        )
        admin.execute(statement)
    else:
        admin.execute(
            sql.SQL("ALTER FUNCTION seeon_audit_insert() SET search_path TO {}").format(
                sql.SQL(", ").join(sql.Identifier(_SENTINEL + str(index)) for index in range(8000))
            )
        )
    _refuses(audit_store, caplog)


@pytest.mark.parametrize(
    "definition",
    [
        "CREATE INDEX audit_events_one_successor_idx ON audit_events(previous_hash)",
        (
            "CREATE UNIQUE INDEX audit_events_one_successor_idx "
            "ON audit_events(previous_hash) WHERE audit_id>0"
        ),
        "CREATE UNIQUE INDEX audit_events_one_successor_idx ON audit_events((previous_hash || ''))",
        (
            "CREATE UNIQUE INDEX audit_events_one_successor_idx "
            "ON audit_events(previous_hash,audit_id)"
        ),
    ],
)
def test_same_index_name_is_not_a_unique_unconditional_exact_key_protection(
    postgres_product_sandbox,
    audit_store,
    caplog,
    definition,
) -> None:
    admin = postgres_product_sandbox.admin
    admin.execute("DROP INDEX audit_events_one_successor_idx")
    admin.execute(definition)
    _refuses(audit_store, caplog)


@pytest.mark.parametrize("key", ["pkey", "record_hash_key", "deferred_hash"])
def test_primary_and_record_hash_protections_are_required(
    postgres_product_sandbox, audit_store, caplog, key
) -> None:
    admin = postgres_product_sandbox.admin
    constraint = "audit_events_pkey" if key == "pkey" else "audit_events_record_hash_key"
    admin.execute(
        sql.SQL("ALTER TABLE audit_events DROP CONSTRAINT {}").format(sql.Identifier(constraint))
    )
    if key == "deferred_hash":
        admin.execute(
            "ALTER TABLE audit_events ADD CONSTRAINT audit_events_record_hash_key "
            "UNIQUE(record_hash) DEFERRABLE INITIALLY IMMEDIATE"
        )
    _refuses(audit_store, caplog)


@pytest.mark.parametrize(
    "alteration",
    [
        "ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE audit_events FORCE ROW LEVEL SECURITY",
        "CREATE POLICY hidden_history ON audit_events USING (false)",
        "CREATE RULE hidden_history AS ON INSERT TO audit_events DO INSTEAD NOTHING",
        "CREATE TABLE hidden_history () INHERITS (audit_events)",
        "ALTER TABLE audit_events SET UNLOGGED",
        "ALTER TABLE audit_events ADD COLUMN hidden text",
        "ALTER TABLE audit_events ALTER COLUMN reason TYPE varchar(256)",
        'ALTER TABLE audit_events ALTER COLUMN actor_id TYPE text COLLATE "C"',
        "ALTER TABLE audit_events ALTER COLUMN audit_id SET GENERATED ALWAYS",
    ],
)
def test_relation_shape_cannot_hide_or_reinterpret_history(
    postgres_product_sandbox,
    audit_store,
    caplog,
    alteration,
) -> None:
    postgres_product_sandbox.admin.execute(alteration)
    _refuses(audit_store, caplog)


@pytest.mark.parametrize("kind", ["view", "partitioned"])
def test_alternate_relation_kinds_are_refused(
    postgres_product_sandbox, audit_store, caplog, kind
) -> None:
    admin = postgres_product_sandbox.admin
    admin.execute("ALTER TABLE audit_events RENAME TO old_audit_events")
    if kind == "view":
        admin.execute("CREATE VIEW audit_events AS SELECT * FROM old_audit_events WHERE false")
    else:
        admin.execute(
            "CREATE TABLE audit_events (LIKE old_audit_events INCLUDING DEFAULTS "
            "INCLUDING CONSTRAINTS INCLUDING IDENTITY) PARTITION BY RANGE (audit_id)"
        )
    _refuses(audit_store, caplog)


def test_qualified_table_is_not_shadowed_by_temp_history(
    postgres_product_sandbox, audit_store
) -> None:
    sandbox = postgres_product_sandbox
    record = audit_store.append(_event())
    sandbox.admin.execute("CREATE TEMP TABLE audit_events (audit_id bigint)")
    # Deliberately hostile session lookup; the private native reads must still
    # qualify the explicit schema. Product verify() never accepts this connection.
    with sandbox.admin.transaction():
        sandbox.admin.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        sandbox.admin.execute("SET LOCAL search_path TO pg_temp, pg_catalog")
        checkpoint = native._verify_snapshot(sandbox.admin, sandbox.schema, None)
        assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
    assert checkpoint.row_count == 1 and checkpoint.audit_id == record.audit_id


def test_prior_anchor_cannot_bless_a_valid_rehash_of_the_same_length(
    postgres_product_sandbox,
    audit_store,
    caplog,
) -> None:
    sandbox = postgres_product_sandbox
    audit_store.append_batch((_event("first"), _event("second")))
    checkpoint = audit_store.verify()
    with sandbox.admin.transaction(), sandbox.admin.cursor(row_factory=dict_row) as cursor:
        history = cursor.execute("SELECT * FROM audit_events ORDER BY audit_id").fetchall()
        sandbox.admin.execute("ALTER TABLE audit_events DISABLE TRIGGER USER")
        previous = GENESIS_HASH
        for index, row in enumerate(history):
            if index == 0:
                row["actor_id"] = _SENTINEL
            row["previous_hash"] = previous
            payload = {
                name: value
                for name, value in row.items()
                if name not in {"audit_id", "record_hash"}
            }
            previous = audit_record_hash(row["previous_hash"], json.dumps(payload))
            sandbox.admin.execute(
                "UPDATE audit_events SET actor_id=%s,previous_hash=%s,record_hash=%s "
                "WHERE audit_id=%s",
                (row["actor_id"], row["previous_hash"], previous, row["audit_id"]),
            )
        sandbox.admin.execute("ALTER TABLE audit_events ENABLE TRIGGER USER")
    cold = audit_store.verify()  # Explicit cold-start limitation, not continuity.
    assert cold.row_count == checkpoint.row_count and cold.record_hash != checkpoint.record_hash
    assert cold.identity == checkpoint.identity
    _refuses(audit_store, caplog, checkpoint)


@pytest.mark.parametrize("tamper", ["delete_anchor", "rename_anchor", "shorten_to_empty"])
def test_missing_changed_or_shortened_anchor_is_not_discarded(
    postgres_product_sandbox,
    audit_store,
    caplog,
    tamper,
) -> None:
    sandbox = postgres_product_sandbox
    audit_store.append_batch((_event("first"), _event("second")))
    checkpoint = audit_store.verify()
    with sandbox.admin.transaction():
        sandbox.admin.execute("ALTER TABLE audit_events DISABLE TRIGGER USER")
        if tamper == "delete_anchor":
            sandbox.admin.execute(
                "DELETE FROM audit_events WHERE audit_id=%s", (checkpoint.audit_id,)
            )
        elif tamper == "rename_anchor":
            sandbox.admin.execute("UPDATE audit_events SET audit_id=audit_id+10")
        else:
            sandbox.admin.execute("DELETE FROM audit_events")
        sandbox.admin.execute("ALTER TABLE audit_events ENABLE TRIGGER USER")
    # All are valid-looking chains from genesis; the prior external seam matters.
    assert audit_store.verify().identity == checkpoint.identity
    _refuses(audit_store, caplog, checkpoint)


@pytest.mark.parametrize("change", ["replacement", "rewrite", "truncate"])
def test_relation_identity_changes_refuse_even_when_the_new_chain_is_valid(
    postgres_product_sandbox,
    audit_store,
    caplog,
    change,
) -> None:
    sandbox = postgres_product_sandbox
    audit_store.append(_event())
    checkpoint = audit_store.verify()
    if change == "replacement":
        with sandbox.admin.transaction():
            _replace_table(sandbox.admin)
    elif change == "rewrite":
        sandbox.admin.execute("CLUSTER audit_events USING audit_events_pkey")
    else:
        with sandbox.admin.transaction():
            sandbox.admin.execute("ALTER TABLE audit_events DISABLE TRIGGER USER")
            sandbox.admin.execute("TRUNCATE audit_events")
            sandbox.admin.execute("ALTER TABLE audit_events ENABLE TRIGGER USER")
    cold = audit_store.verify()
    assert cold.identity != checkpoint.identity
    _refuses(audit_store, caplog, checkpoint)


@pytest.mark.parametrize(
    "changes",
    [
        {"row_count": 1},
        {"row_count": 3},
        {"audit_id": None},
        {"record_hash": "f" * 64},
        {"guard_fingerprint": "f" * 64},
        {"row_count": 0, "audit_id": 0, "record_hash": GENESIS_HASH},
    ],
)
def test_anchor_requires_the_exact_former_prefix_position(audit_store, caplog, changes) -> None:
    audit_store.append_batch((_event("first"), _event("second")))
    checkpoint = audit_store.verify()
    _refuses(audit_store, caplog, replace(checkpoint, **changes))


@pytest.mark.parametrize("change", ["truncate", "replacement"])
def test_lock_precedes_snapshot_when_concurrent_ddl_already_owns_relation(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    change,
) -> None:
    sandbox = postgres_product_sandbox
    audit_store.append(_event("old"))
    callback, pids = postgres_store._verify_snapshot, Queue()

    def observed(connection, schema, checkpoint):
        pids.put(connection.info.backend_pid)
        return callback(connection, schema, checkpoint)

    monkeypatch.setattr(postgres_store, "_verify_snapshot", observed)
    sandbox.admin.execute("BEGIN")
    try:
        sandbox.admin.execute("LOCK TABLE audit_events IN ACCESS EXCLUSIVE MODE")
        if change == "replacement":
            _replace_table(sandbox.admin)
        else:
            sandbox.admin.execute("ALTER TABLE audit_events DISABLE TRIGGER USER")
            sandbox.admin.execute("TRUNCATE audit_events")
            sandbox.admin.execute("ALTER TABLE audit_events ENABLE TRIGGER USER")
        _insert_ids(sandbox.admin, (-3, 0))
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(audit_store.verify)
            try:
                pid = pids.get(timeout=2)
                _wait_for_lock(sandbox, pid)
                assert not pending.done()
            finally:
                sandbox.admin.commit()
            checkpoint = pending.result(timeout=2)
        assert (checkpoint.row_count, checkpoint.audit_id) == (2, 0)
        assert checkpoint == audit_store.verify()
    finally:
        sandbox.admin.rollback()


@pytest.mark.parametrize("change", ["truncate", "replacement"])
def test_access_share_pins_relation_until_snapshot_reader_exits(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    caplog,
    change,
) -> None:
    sandbox = postgres_product_sandbox
    audit_store.append(_event())
    scan, entered, release, pids = native._scan, Event(), Event(), Queue()

    def paused(connection, table, count, checkpoint):
        entered.set()
        assert release.wait(4)
        return scan(connection, table, count, checkpoint)

    def rewrite(connection):
        pids.put(connection.info.backend_pid)
        connection.execute("LOCK TABLE audit_events IN ACCESS EXCLUSIVE MODE")
        if change == "replacement":
            _replace_table(connection)
        else:
            connection.execute("ALTER TABLE audit_events DISABLE TRIGGER USER")
            connection.execute("TRUNCATE audit_events")
            connection.execute("ALTER TABLE audit_events ENABLE TRIGGER USER")

    with monkeypatch.context() as patch:
        patch.setattr(native, "_scan", paused)
        with ThreadPoolExecutor(max_workers=2) as executor:
            reader = executor.submit(audit_store.verify)
            try:
                assert entered.wait(2)
                writer = executor.submit(sandbox.database.transact, rewrite)
                _wait_for_lock(sandbox, pids.get(timeout=2))
                assert not writer.done()
            finally:
                release.set()
            checkpoint = reader.result(timeout=2)
            writer.result(timeout=2)
    assert checkpoint.row_count == 1
    _refuses(audit_store, caplog, checkpoint)


def test_lock_wait_is_bounded_and_failed_callback_publishes_nothing(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    caplog,
) -> None:
    sandbox = postgres_product_sandbox
    read_snapshot, calls, published = sandbox.database.read_snapshot, [], []

    def bounded(callback):
        def run(connection):
            connection.execute("SET LOCAL lock_timeout TO 75")
            calls.append(1)
            return callback(connection)

        return read_snapshot(run)

    monkeypatch.setattr(sandbox.database, "read_snapshot", bounded)
    with sandbox.admin.transaction():
        sandbox.admin.execute("LOCK TABLE audit_events IN ACCESS EXCLUSIVE MODE")
        started = monotonic()
        with pytest.raises(AuditVerificationError) as failure:
            published.append(audit_store.verify())
        _assert_private(failure, caplog)
        assert monotonic() - started < 2
    assert calls == [1] and not published
    assert audit_store.verify().row_count == 0


def test_function_change_after_snapshot_is_next_observation_not_a_relation_lock_guarantee(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    caplog,
) -> None:
    sandbox = postgres_product_sandbox
    audit_store.append(_event())
    checkpoint = audit_store.verify()
    verify_functions, entered, release = native._verify_functions, Event(), Event()

    def paused(connection, namespace, functions):
        entered.set()
        assert release.wait(4)
        return verify_functions(connection, namespace, functions)

    with monkeypatch.context() as patch:
        patch.setattr(native, "_verify_functions", paused)
        with ThreadPoolExecutor(max_workers=1) as executor:
            reader = executor.submit(audit_store.verify, checkpoint)
            try:
                assert entered.wait(2)
                sandbox.admin.execute(
                    _function_sql("seeon_audit_serialize")
                    .replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1)
                    .replace("pg_advisory_xact_lock(TG_RELID::bigint)", "1")
                )
                assert not reader.done()  # Function DDL committed despite ACCESS SHARE.
            finally:
                release.set()
            assert reader.result(timeout=2) == checkpoint
    _refuses(audit_store, caplog, checkpoint)


@pytest.mark.parametrize("count", [3, 4, 5])
def test_lowered_capacity_boundary_is_branch_evidence_only(
    audit_store, monkeypatch, caplog, count
) -> None:
    audit_store.append_batch(tuple(_event(str(i)) for i in range(count)))
    monkeypatch.setattr(native, "MAX_AUDIT_ROWS", 4)
    if count < 4:
        assert audit_store.verify().row_count == 3
    else:
        _refuses(audit_store, caplog)


@pytest.mark.parametrize("operation", ["read", "transact"])
def test_private_entry_refuses_wrong_transaction_mode(
    postgres_product_sandbox, caplog, operation
) -> None:
    sandbox = postgres_product_sandbox
    with pytest.raises(AuditVerificationError) as failure:
        getattr(sandbox.database, operation)(
            lambda connection: native._verify_snapshot(connection, sandbox.schema, None)
        )
    _assert_private(failure, caplog)


@pytest.mark.parametrize("role", ["replica", "local"])
def test_private_entry_refuses_non_origin_mode(postgres_product_sandbox, caplog, role) -> None:
    sandbox = postgres_product_sandbox
    with sandbox.admin.transaction():
        sandbox.admin.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        sandbox.admin.execute(
            sql.SQL("SET LOCAL session_replication_role TO {}").format(sql.Literal(role))
        )
        with pytest.raises(AuditVerificationError) as failure:
            native._verify_snapshot(sandbox.admin, sandbox.schema, None)
        _assert_private(failure, caplog)


@pytest.mark.parametrize("stage", ["callback_sql", "commit_receipt", "pool_exit"])
def test_owner_failure_never_publishes_or_replays_checkpoint(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    caplog,
    stage,
) -> None:
    sandbox = postgres_product_sandbox
    audit_store.append(_event())
    callback, commit, acquire = (
        postgres_store._verify_snapshot,
        psycopg.Connection.commit,
        sandbox.database._pool.connection,
    )
    calls, candidates, published, exits = [], [], [], []

    def observed(connection, schema, checkpoint):
        calls.append(connection.info.backend_pid)
        candidate = callback(connection, schema, checkpoint)
        candidates.append(candidate)
        if stage == "callback_sql":
            connection.execute("SELECT %s::bigint", (_SENTINEL,))
        return candidate

    def lose_receipt(connection):
        if connection.info.backend_pid in calls:
            commit(connection)  # Real COMMIT, then inject only loss of its receipt.
            raise psycopg.OperationalError(_SENTINEL)
        commit(connection)

    @contextmanager
    def fail_release(*, timeout):
        with acquire(timeout=timeout) as connection:
            yield connection
            assert connection.info.transaction_status is TransactionStatus.IDLE
        exits.append(1)
        raise psycopg.OperationalError(_SENTINEL)

    expected = {
        "callback_sql": AuditVerificationError,
        "commit_receipt": CommitOutcomeUnknown,
        "pool_exit": PostgresUnavailable,
    }[stage]
    with monkeypatch.context() as patch:
        patch.setattr(postgres_store, "_verify_snapshot", observed)
        if stage == "commit_receipt":
            patch.setattr(psycopg.Connection, "commit", lose_receipt)
        elif stage == "pool_exit":
            patch.setattr(sandbox.database._pool, "connection", fail_release)
        with pytest.raises(expected) as failure:
            published.append(audit_store.verify())
        if stage == "callback_sql":
            _assert_private(failure, caplog)
        else:
            assert _SENTINEL not in str(failure.value)
            assert _SENTINEL not in caplog.text
    assert len(calls) == len(candidates) == 1 and not published
    assert exits == ([1] if stage == "pool_exit" else [])
    assert set(vars(audit_store)) == {"database", "authority"}
    assert audit_store.verify() == candidates[0]


@pytest.mark.parametrize("committed", [False, True])
def test_unknown_commit_cannot_be_reinterpreted_by_a_reload_or_retry(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    committed,
) -> None:
    sandbox = postgres_product_sandbox
    callback, commit = postgres_store._verify_snapshot, psycopg.Connection.commit
    calls, commits, published = [], [], []

    def observed(connection, schema, checkpoint):
        calls.append(connection.info.backend_pid)
        return callback(connection, schema, checkpoint)

    def lose_receipt(connection):
        if connection.info.backend_pid in calls:
            commits.append(connection.info.backend_pid)
            if committed:
                commit(connection)
            else:
                connection.rollback()
            raise psycopg.InterfaceError(_SENTINEL)
        commit(connection)

    def no_other_owner_path(*args, **kwargs):
        pytest.fail("verification must borrow read_snapshot only, without reload/replay")

    with monkeypatch.context() as patch:
        patch.setattr(postgres_store, "_verify_snapshot", observed)
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        patch.setattr(sandbox.database, "read", no_other_owner_path)
        patch.setattr(sandbox.database, "transact", no_other_owner_path)
        with pytest.raises(CommitOutcomeUnknown):
            published.append(audit_store.verify())
    assert len(calls) == 1 and commits == calls and not published


@pytest.mark.parametrize("error", [KeyboardInterrupt, AssertionError, TypeError])
def test_interruptions_and_programming_defects_remain_distinct(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    error,
) -> None:
    scan, calls, published = native._scan, [], []

    def fail(connection, table, count, checkpoint):
        scan(connection, table, count, checkpoint)
        calls.append(1)
        raise error("test-only defect")

    with monkeypatch.context() as patch:
        patch.setattr(native, "_scan", fail)
        with pytest.raises(error):
            published.append(audit_store.verify())
    assert calls == [1] and not published
    assert audit_store.verify().row_count == 0


def test_missing_or_unparseable_trusted_bundle_refuses_without_live_learning(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    for content in (None, b"unparseable", b"x" * (native._MAX_BUNDLE_BYTES + 1)):
        resource = tmp_path / "postgres_product.sql"
        if content is not None:
            resource.write_bytes(content)
        with monkeypatch.context() as patch:
            patch.setattr(native, "files", lambda package: tmp_path)
            _refuses(audit_store, caplog)


def test_driver_failure_closes_the_server_cursor_before_owner_rollback(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
    caplog,
) -> None:
    audit_store.append(_event())
    fetchmany, close = psycopg.ServerCursor.fetchmany, psycopg.ServerCursor.close
    closes = []

    def fail_after_real_fetch(cursor, size=0):
        fetchmany(cursor, size)
        cursor.connection.execute("SELECT %s::bigint", (_SENTINEL,))

    def observed_close(cursor):
        closes.append(cursor.connection.info.transaction_status)
        return close(cursor)

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.ServerCursor, "fetchmany", fail_after_real_fetch)
        patch.setattr(psycopg.ServerCursor, "close", observed_close)
        _refuses(audit_store, caplog)
    assert closes == [TransactionStatus.INERROR]
    assert audit_store.verify().row_count == 1


def test_checkpoint_release_follows_actual_commit_and_pool_exit(
    postgres_product_sandbox,
    audit_store,
    monkeypatch,
) -> None:
    sandbox = postgres_product_sandbox
    callback = postgres_store._verify_snapshot
    commit = psycopg.Connection.commit
    acquire = sandbox.database._pool.connection
    events, pids, published, paths = [], [], [], []

    def observed(connection, schema, checkpoint):
        pids.append(connection.info.backend_pid)
        paths.append(connection.execute("SHOW search_path").fetchone())
        candidate = callback(connection, schema, checkpoint)
        assert connection.execute("SHOW search_path").fetchone() == ("pg_catalog, pg_temp",)
        events.append("callback")
        assert not published
        return candidate

    def observed_commit(connection):
        active = (
            connection.info.backend_pid in pids
            and connection.info.transaction_status is TransactionStatus.INTRANS
        )
        commit(connection)
        # The pool context also calls commit() on the already-idle connection;
        # only the owner's actual transaction COMMIT is evidence here.
        if active:
            assert connection.info.transaction_status is TransactionStatus.IDLE
            # The transaction-local verifier path expires at COMMIT, before
            # the pool reset could conceal an accidental session-level change.
            assert connection.execute("SHOW search_path").fetchone() == paths[0]
            events.append("commit")
            assert not published

    @contextmanager
    def observed_release(*, timeout):
        with acquire(timeout=timeout) as connection:
            yield connection
            assert events == ["callback", "commit"] and not published
        events.append("release")
        assert not published

    with monkeypatch.context() as patch:
        patch.setattr(postgres_store, "_verify_snapshot", observed)
        patch.setattr(psycopg.Connection, "commit", observed_commit)
        patch.setattr(sandbox.database._pool, "connection", observed_release)
        published.append(audit_store.verify())
        events.append("returned")
    assert events == ["callback", "commit", "release", "returned"]
    assert len(pids) == len(published) == 1


@pytest.mark.parametrize("count", [1, 3])
def test_scan_refuses_exact_count_disagreement_with_real_cursor(
    postgres_product_sandbox,
    audit_store,
    caplog,
    count,
) -> None:
    sandbox = postgres_product_sandbox
    audit_store.append_batch((_event("first"), _event("second")))
    with pytest.raises(AuditVerificationError) as failure:
        sandbox.database.read_snapshot(
            lambda connection: native._scan(
                connection,
                sql.Identifier(sandbox.schema, "audit_events"),
                count,
                None,
            )
        )
    assert str(failure.value) == _FAILURE
    assert _SENTINEL not in caplog.text


def test_catalog_result_budget_refuses_excess_indexes(
    postgres_product_sandbox,
    audit_store,
    caplog,
) -> None:
    admin = postgres_product_sandbox.admin
    # Six canonical indexes plus twenty-seven extras cross the finite metadata
    # budget without removing any of the actual required unique protections.
    for number in range(27):
        admin.execute(
            sql.SQL("CREATE INDEX {} ON audit_events(recorded_at)").format(
                sql.Identifier("test_extra_audit_index_" + str(number))
            )
        )
    _refuses(audit_store, caplog)


def test_public_verification_has_no_caller_connection_path(
    postgres_product_sandbox,
    audit_store,
) -> None:
    with pytest.raises(TypeError):
        audit_store.verify(connection=postgres_product_sandbox.admin)
    assert audit_store.verify().row_count == 0


def test_already_safe_owner_unavailability_remains_typed(
    postgres_product_sandbox,
    audit_store,
) -> None:
    postgres_product_sandbox.database.close()
    with pytest.raises(PostgresUnavailable, match="owner is not running"):
        audit_store.verify()

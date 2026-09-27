"""Full, bounded PostgreSQL audit verification as of one repeatable-read snapshot.

An OID/relfilenode anchor is continuity within a database, not authenticated cluster
identity. Function DDL after the snapshot is detected on the next observation;
ACCESS SHARE pins the relation, not functions. Cold starts cannot detect a valid
rehash of history. audit_id is ordered/anchored but is not in the canonical hash.
No checkpoint here is a freshness claim, persistent state, or incremental cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Final

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus
from psycopg.rows import tuple_row

from backend.app.features.audit.verification import (
    AUDIT_ROW_COLUMNS,
    GENESIS_HASH,
    MAX_AUDIT_ROWS,
    AuditVerificationError,
    verify_row,
)

_FAILURE: Final = "audit verification failed"
_PAGE_SIZE: Final = 1_000
_MAX_BUNDLE_BYTES: Final = 131_072
_MAX_INDEXES: Final = 32
# UTF-8 byte ceilings cover the product's character limits, without trusting its
# CHECK constraints to protect client memory. A separate validity bit prevents a
# truncated/oversized nullable field from being mistaken for a legitimate NULL.
_TEXT_BYTES: Final = (
    0,
    128,
    128,
    64,
    64,
    512,
    256,
    256,
    256,
    512,
    64,
    1024,
    512,
    512,
    16_384,
    64,
    64,
    64,
    512,
)
_NULLABLE: Final = frozenset({12, 13, 14, 15, 19})  # one-based attribute numbers
_TRIGGER_CONTRACT: Final = {
    "audit_events_immutable_update": ("seeon_audit_immutable", 19),
    "audit_events_immutable_delete": ("seeon_audit_immutable", 11),
    "audit_events_immutable_truncate": ("seeon_audit_immutable", 34),
    "audit_events_serialize": ("seeon_audit_serialize", 6),
    "audit_events_insert_guard": ("seeon_audit_insert", 7),
}
_FUNCTION_NAMES: Final = frozenset(name for name, _ in _TRIGGER_CONTRACT.values()) | {
    "seeon_audit_record_hash"
}


@dataclass(frozen=True, slots=True)
class PostgresAuditCheckpoint:
    identity: tuple[int, int, int, int]
    guard_fingerprint: str
    row_count: int
    audit_id: int | None
    record_hash: str


@dataclass(frozen=True, slots=True)
class _FunctionContract:
    definition: str
    body: str
    argument_names: tuple[str, ...]
    return_oid: int
    volatility: str
    strict: bool
    search_path: str


def _require(condition: bool) -> None:
    if not condition:
        raise AuditVerificationError(_FAILURE)


def _trusted_guards(schema_identifier: str) -> tuple[dict[str, _FunctionContract], str]:
    """Recognize only the bundled DDL's narrow grammar, never parse live SQL.

    Header defaults below are PostgreSQL CREATE FUNCTION defaults. Any new DDL
    syntax needs an explicit contract update; bodies are compared byte-for-byte,
    including whitespace inside literals. No deparser/syscache is an authority.
    schema_identifier is the canonical rendering of the configured namespace,
    not a value recovered from a live function's settings.
    """
    with files("backend.app.edge_db").joinpath("postgres_product.sql").open("rb") as resource:
        encoded = resource.read(_MAX_BUNDLE_BYTES + 1)
    _require(len(encoded) <= _MAX_BUNDLE_BYTES)
    source = encoded.decode("utf-8")
    pattern = re.compile(
        r"^CREATE FUNCTION (seeon_audit_\w+)\(([^()]*)\) RETURNS (text|trigger)\n"
        r"LANGUAGE plpgsql (IMMUTABLE STRICT )?SET search_path (= pg_catalog|FROM CURRENT) AS \$\$"
        r"(.*?)\$\$;",
        re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(source))
    _require(len(matches) == len(_FUNCTION_NAMES))
    _require(len(re.findall(r"^CREATE FUNCTION seeon_audit_", source, re.MULTILINE)) == 4)
    functions = {}
    for match in matches:
        name, arguments, returns, immutable, path, body = match.groups()
        _require(name in _FUNCTION_NAMES and name not in functions)
        is_hash = name == "seeon_audit_record_hash"
        _require(arguments == ("previous_hash text, payload_json text" if is_hash else ""))
        _require(returns == ("text" if is_hash else "trigger"))
        _require(bool(immutable) == is_hash)
        _require(path == ("FROM CURRENT" if name == "seeon_audit_insert" else "= pg_catalog"))
        functions[name] = _FunctionContract(
            definition=match.group(0),
            body=body,
            argument_names=("previous_hash", "payload_json") if is_hash else (),
            return_oid=25 if is_hash else 2279,  # pg_catalog.text / trigger
            volatility="i" if is_hash else "v",
            strict=is_hash,
            search_path=(
                schema_identifier + ", pg_catalog, pg_temp"
                if path == "FROM CURRENT"
                else "pg_catalog"
            ),
        )
    trigger_pattern = re.compile(
        r"^CREATE TRIGGER (audit_events_\w+) BEFORE "
        r"(UPDATE|DELETE|TRUNCATE|INSERT) ON audit_events\n"
        r"    FOR EACH (ROW|STATEMENT) EXECUTE FUNCTION (seeon_audit_\w+)\(\);",
        re.MULTILINE,
    )
    triggers = list(trigger_pattern.finditer(source))
    _require(len(triggers) == 5)
    _require(len(re.findall(r"^CREATE TRIGGER audit_events_", source, re.MULTILINE)) == 5)
    event_bits = {"INSERT": 4, "DELETE": 8, "UPDATE": 16, "TRUNCATE": 32}
    actual = {}
    for match in triggers:
        name, event, level, function = match.groups()
        _require(name not in actual)
        actual[name] = (function, 2 | event_bits[event] | int(level == "ROW"))
    _require(actual == _TRIGGER_CONTRACT)
    # This digest describes only the validated contract, not a data-version or
    # change counter. Function OIDs are separately checked at trigger bindings.
    contract = {
        "functions": {
            name: (spec.definition, spec.search_path) for name, spec in functions.items()
        },
        "triggers": sorted(match.group(0) for match in triggers),
        "columns": AUDIT_ROW_COLUMNS,
        "nullable": sorted(_NULLABLE),
        "unique_keys": ("audit_id:primary", "previous_hash", "record_hash"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return functions, fingerprint


def _relation_identity(connection: psycopg.Connection, schema: str) -> tuple[int, int, int, int]:
    rows = connection.execute(
        "SELECT d.oid, n.oid, c.oid, c.relfilenode, "
        "c.relkind = 'r' AND c.relpersistence = 'p' AND c.relnatts = 19 "
        "AND NOT c.relrowsecurity AND NOT c.relforcerowsecurity "
        "AND NOT c.relispartition AND NOT c.relhassubclass AND NOT c.relhasrules "
        "AND c.relhastriggers AND c.relhasindex "
        "AND c.relrewrite = 0 AND am.amname = 'heap' AND am.amtype = 't' "
        "AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_inherits i "
        "WHERE i.inhrelid = c.oid OR i.inhparent = c.oid) "
        "AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_rewrite r WHERE r.ev_class = c.oid) "
        "AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_policy p WHERE p.polrelid = c.oid) "
        "FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_catalog.pg_am am ON am.oid = c.relam "
        "CROSS JOIN pg_catalog.pg_database d "
        "WHERE n.nspname = %s AND c.relname = 'audit_events' "
        "AND d.datname = pg_catalog.current_database() LIMIT 2",
        (schema,),
    ).fetchall()
    _require(len(rows) == 1 and rows[0][4] is True)
    identity = tuple(rows[0][:4])
    _require(all(type(value) is int and value > 0 for value in identity))
    return identity


def _verify_columns(connection: psycopg.Connection, relation: int) -> None:
    # Names have PostgreSQL's fixed name width; all remaining results are scalar.
    rows = connection.execute(
        "SELECT attnum, attname, atttypid, attnotnull, attidentity, "
        "NOT attisdropped AND attgenerated = '' AND atttypmod = -1 "
        "AND NOT atthasmissing AND attinhcount = 0 AND attislocal "
        "AND attcollation = CASE WHEN attnum = 1 THEN 0 ELSE 100 END "
        "FROM pg_catalog.pg_attribute WHERE attrelid = %s AND attnum > 0 "
        "ORDER BY attnum LIMIT 20",
        (relation,),
    ).fetchall()
    expected = [
        (
            number,
            name,
            20 if number == 1 else 25,
            number not in _NULLABLE,
            "d" if number == 1 else "",
            True,
        )
        for number, name in enumerate(AUDIT_ROW_COLUMNS, 1)
    ]
    _require(rows == expected)


def _verify_functions(
    connection: psycopg.Connection,
    namespace: int,
    functions: dict[str, _FunctionContract],
) -> dict[str, int]:
    identities = {}
    for name, spec in functions.items():
        # Compare raw MVCC fields on the server: hostile prosrc/proconfig/arrays
        # never cross the wire, even when arbitrarily larger than the bundle.
        rows = connection.execute(
            "SELECT p.oid, p.prosrc = %s AND p.probin IS NULL AND p.prosqlbody IS NULL "
            "AND p.prokind = 'f' AND NOT p.proretset AND p.prorettype = %s "
            "AND l.lanname = 'plpgsql' AND l.lanispl AND l.lanpltrusted "
            "AND p.provolatile = %s AND p.proisstrict = %s "
            "AND NOT p.prosecdef AND NOT p.proleakproof AND p.proparallel = 'u' "
            "AND p.procost = 100 AND p.prorows = 0 AND p.prosupport = 0 "
            "AND p.provariadic = 0 AND p.pronargdefaults = 0 AND p.proargdefaults IS NULL "
            "AND p.pronargs = %s AND p.proargtypes = %s::pg_catalog.oidvector "
            "AND p.proallargtypes IS NULL AND p.proargmodes IS NULL AND p.protrftypes IS NULL "
            "AND p.proargnames IS NOT DISTINCT FROM %s::pg_catalog.text[] "
            "AND p.proconfig = %s::pg_catalog.text[] "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid = p.prolang "
            "WHERE p.pronamespace = %s AND p.proname = %s LIMIT 2",
            (
                spec.body,
                spec.return_oid,
                spec.volatility,
                spec.strict,
                len(spec.argument_names),
                "25 25" if spec.argument_names else "",
                list(spec.argument_names) if spec.argument_names else None,
                ["search_path=" + spec.search_path],
                namespace,
                name,
            ),
        ).fetchall()
        _require(len(rows) == 1 and rows[0][1] is True)
        identities[name] = rows[0][0]
    return identities


def _verify_triggers(
    connection: psycopg.Connection, relation: int, functions: dict[str, int]
) -> None:
    rows = connection.execute(
        "SELECT tgname, tgfoid, tgtype, tgenabled = 'O' AND NOT tgisinternal "
        "AND tgparentid = 0 AND tgconstrrelid = 0 AND tgconstrindid = 0 AND tgconstraint = 0 "
        "AND NOT tgdeferrable AND NOT tginitdeferred AND tgnargs = 0 "
        "AND tgargs = ''::pg_catalog.bytea AND tgattr = ''::pg_catalog.int2vector "
        "AND tgqual IS NULL AND tgoldtable IS NULL AND tgnewtable IS NULL "
        "FROM pg_catalog.pg_trigger WHERE tgrelid = %s ORDER BY tgname LIMIT 6",
        (relation,),
    ).fetchall()
    expected = sorted(
        (name, functions[function], bits, True)
        for name, (function, bits) in _TRIGGER_CONTRACT.items()
    )
    _require(rows == expected)


def _verify_indexes(connection: psycopg.Connection, relation: int, namespace: int) -> None:
    rows = connection.execute(
        "SELECT i.indisprimary, "
        "CASE WHEN i.indkey = '1'::pg_catalog.int2vector THEN 1 "
        "WHEN i.indkey = '16'::pg_catalog.int2vector THEN 16 "
        "WHEN i.indkey = '17'::pg_catalog.int2vector THEN 17 ELSE 0 END, "
        "i.indisunique AND i.indisvalid AND i.indisready AND i.indislive "
        "AND i.indimmediate AND NOT i.indisexclusion AND NOT i.indcheckxmin "
        "AND i.indnatts = 1 AND i.indnkeyatts = 1 AND i.indpred IS NULL AND i.indexprs IS NULL "
        "AND c.relkind = 'i' AND c.relpersistence = 'p' AND c.relnamespace = %s "
        "AND am.amname = 'btree' AND am.amtype = 'i' "
        "AND n.nspname = 'pg_catalog' AND op.opcdefault "
        "AND op.opcmethod = c.relam "
        "AND op.opcname = CASE WHEN i.indisprimary THEN 'int8_ops' ELSE 'text_ops' END "
        "AND i.indcollation = CASE WHEN i.indisprimary THEN '0'::pg_catalog.oidvector "
        "ELSE '100'::pg_catalog.oidvector END "
        "AND i.indoption = '0'::pg_catalog.int2vector "
        "AND (NOT i.indisprimary OR EXISTS (SELECT 1 FROM pg_catalog.pg_constraint co "
        "WHERE co.conrelid = i.indrelid AND co.conindid = i.indexrelid AND co.contype = 'p' "
        "AND co.convalidated AND NOT co.condeferrable AND NOT co.condeferred "
        "AND co.conkey = ARRAY[1]::pg_catalog.int2[])) "
        "FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid "
        "JOIN pg_catalog.pg_am am ON am.oid = c.relam "
        "LEFT JOIN pg_catalog.pg_opclass op ON op.oid = i.indclass[0] "
        "LEFT JOIN pg_catalog.pg_namespace n ON n.oid = op.opcnamespace "
        "WHERE i.indrelid = %s LIMIT %s",
        (namespace, relation, _MAX_INDEXES + 1),
    ).fetchall()
    _require(len(rows) <= _MAX_INDEXES)
    protections = {(primary, key) for primary, key, valid in rows if valid is True}
    _require({(True, 1), (False, 16), (False, 17)} <= protections)


def _bounded_projection() -> sql.Composed:
    columns = [sql.Identifier("a", AUDIT_ROW_COLUMNS[0])]
    bounded = []
    for name, limit in zip(AUDIT_ROW_COLUMNS[1:], _TEXT_BYTES[1:], strict=True):
        column = sql.Identifier("a", name)
        fits = sql.SQL("pg_catalog.octet_length({}) <= {}").format(column, sql.Literal(limit))
        columns.append(
            sql.SQL("CASE WHEN {} THEN {} END AS {}").format(fits, column, sql.Identifier(name))
        )
        bounded.append(sql.SQL("({} IS NULL OR {})").format(column, fits))
    columns.append(sql.SQL(" AND ").join(bounded))
    return sql.SQL(", ").join(columns)


def _check_checkpoint(checkpoint: PostgresAuditCheckpoint) -> None:
    _require(isinstance(checkpoint, PostgresAuditCheckpoint))
    _require(type(checkpoint.identity) is tuple and len(checkpoint.identity) == 4)
    _require(all(type(value) is int and value > 0 for value in checkpoint.identity))
    _require(type(checkpoint.row_count) is int and 0 <= checkpoint.row_count < MAX_AUDIT_ROWS)
    for value in (checkpoint.guard_fingerprint, checkpoint.record_hash):
        _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None)
    if checkpoint.row_count == 0:
        _require(checkpoint.audit_id is None and checkpoint.record_hash == GENESIS_HASH)
    else:
        _require(type(checkpoint.audit_id) is int and -(2**63) <= checkpoint.audit_id < 2**63)


def _scan(
    connection: psycopg.Connection,
    table: sql.Identifier,
    count: int,
    checkpoint: PostgresAuditCheckpoint | None,
) -> tuple[int | None, str]:
    scanned = 0
    last_id = None
    previous = GENESIS_HASH
    # First page has no lower-ID predicate: BY DEFAULT identity admits <= 0.
    with connection.cursor(
        name="seeon_audit_verification", withhold=False, scrollable=False, row_factory=tuple_row
    ) as cursor:
        cursor.execute(
            sql.SQL("SELECT {} FROM {} AS a ORDER BY a.audit_id LIMIT %s").format(
                _bounded_projection(), table
            ),
            (MAX_AUDIT_ROWS,),
        )
        while rows := cursor.fetchmany(_PAGE_SIZE):
            for row in rows:
                _require(len(row) == len(AUDIT_ROW_COLUMNS) + 1 and row[-1] is True)
                audit_id, record_hash = verify_row(row[:-1], previous)
                _require(last_id is None or audit_id > last_id)
                scanned += 1
                _require(scanned <= count)
                if checkpoint is not None and scanned == checkpoint.row_count:
                    _require(
                        (audit_id, record_hash) == (checkpoint.audit_id, checkpoint.record_hash)
                    )
                last_id, previous = audit_id, record_hash
    _require(scanned == count)
    return last_id, previous


def _verify_snapshot(
    connection: psycopg.Connection, schema: str, checkpoint: PostgresAuditCheckpoint | None
) -> PostgresAuditCheckpoint:
    """Private callback; only the owner may publish its result after pool exit.

    SET LOCAL and SHOW do not acquire a data snapshot. LOCK must precede *every*
    SELECT so a concurrent rewrite/TRUNCATE cannot leave a pre-lock snapshot of
    a new heap. ONLY avoids recursively locking hostile inheritance before
    shape validation.
    """
    try:
        _require(connection.info.transaction_status is TransactionStatus.INTRANS)
        # Product/user operators must not decide whether catalog fields match.
        # This expires at the owner's transaction boundary, not at pool reset.
        connection.execute("SET LOCAL search_path TO pg_catalog, pg_temp")
        _require(
            connection.execute("SHOW transaction_isolation").fetchone() == ("repeatable read",)
        )
        _require(connection.execute("SHOW transaction_read_only").fetchone() == ("on",))
        _require(connection.execute("SHOW session_replication_role").fetchone() == ("origin",))
        table = sql.Identifier(schema, "audit_events")
        connection.execute(sql.SQL("LOCK TABLE ONLY {} IN ACCESS SHARE MODE").format(table))
        # GUC capture uses PostgreSQL's canonical quoting, not Identifier's
        # always-quoted SQL spelling. Render only the trusted owner namespace.
        row = connection.execute("SELECT pg_catalog.quote_ident(%s)", (schema,)).fetchone()
        _require(row is not None and isinstance(row[0], str))
        functions, fingerprint = _trusted_guards(row[0])
        identity = _relation_identity(connection, schema)
        if checkpoint is not None:
            _check_checkpoint(checkpoint)
            _require(
                checkpoint.identity == identity and checkpoint.guard_fingerprint == fingerprint
            )
        _verify_columns(connection, identity[2])
        function_ids = _verify_functions(connection, identity[1], functions)
        _verify_triggers(connection, identity[2], function_ids)
        _verify_indexes(connection, identity[2], identity[1])
        row = connection.execute(
            sql.SQL(
                "SELECT pg_catalog.count(*) FROM (SELECT 1 FROM {} LIMIT %s) AS bounded_audit"
            ).format(table),
            (MAX_AUDIT_ROWS,),
        ).fetchone()
        _require(row is not None and type(row[0]) is int and 0 <= row[0] < MAX_AUDIT_ROWS)
        count = row[0]
        _require(checkpoint is None or count >= checkpoint.row_count)
        last_id, record_hash = _scan(connection, table, count, checkpoint)
        return PostgresAuditCheckpoint(identity, fingerprint, count, last_id, record_hash)
    except (AuditVerificationError, psycopg.Error, OSError, ValueError, RecursionError):
        # SQL diagnostics, enum values and JSON errors may contain protected data.
        # Interruptions and programming defects (AssertionError/TypeError/etc.)
        # remain distinct; safe typed owner errors are not caught here.
        raise AuditVerificationError(_FAILURE) from None


__all__ = ["PostgresAuditCheckpoint"]

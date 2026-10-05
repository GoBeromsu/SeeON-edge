# EDGE DB KNOWLEDGE BASE

Backend-owned SQLite foundation: schema contract, DDL-free runtime connections,
one-shot bootstrap. Earned its file as a distinct domain (score 12: 15 modules,
~3.7k LOC, 94 importing files).

## Where to look

| Task | File | Notes |
| --- | --- | --- |
| Open the product DB | `connection.py` | `open_runtime_database(path, *, actor, compatibility, busy_policy, check_same_thread)`, `write_transaction`, `best_effort_zero_wait_write` |
| Long-lived store connection | `configuration.py` | `open_configuration_database` = `RuntimeActor.API` + `check_same_thread=False`; also `ensure_edge_site` |
| Open diagnostics DB | `diagnostics_connection.py` | `open_diagnostics_database`; opens and verifies only |
| Create or extend schema | `bootstrap.py`, `__main__.py` | `python -m backend.app.edge_db [--database ...]`; prints `EDGE_DB_BOOTSTRAP_OK` / `EDGE_DIAGNOSTICS_DB_BOOTSTRAP_OK` |
| Table DDL | `compact_schema_ddl.py`, `execution_records_ddl.py` | Raw CREATE strings only (2006 + 205 lines) |
| Table sets | `compact_schema.py`, `ownership.py` | `APPLICATION_API_TABLES`, `Writer`, `writer_for_table`, `SCHEMA_LEDGER_TABLE` |
| Schema verification | `compatibility.py`, `schema18_manifest.py` | `classify_schema`, `verify_runtime_schema`, `verify_schema18_contract`, error types |
| Paths and file modes | `paths.py` | `EDGE_STATE_DIRECTORY`, `EDGE_DATABASE_PATH`, `DIAGNOSTICS_DATABASE_FILENAME`, `schema18_backup_path` |
| SQLite UDF | `functions.py` | `seeon_audit_record_hash` (deterministic) |
| Review value types | `reviews.py` | `EvidenceReview`, `ReviewDisposition` |

`__init__.py` is the DDL-free facade. Import from it unless you need a submodule symbol.

## Conventions

- Two files, one bootstrap run, one exclusive `deployment.lock`: product
  `edge.sqlite3` (ledger table `schema_migrations`) and `edge-diagnostics.sqlite3`
  (flat `PRAGMA user_version`, no ledger, one table family, no write authorizer).
- Runtime connections install an authorizer: every DDL action is denied, and
  INSERT/UPDATE/DELETE is denied on any table the `RuntimeActor` does not own.
  `RuntimeActor` has one member, `API`. WAL is required.
- Bootstrap creates schema 19 or extends an exact schema 18. Extension first
  writes `<name>.schema18-backup.sqlite3`. No other version is migrated and
  there is no downgrade path.
- The schema number lives in `shared/release_identity.py`
  (`EDGE_DATABASE_SCHEMA_VERSION`). Bump it together with the DDL; the worker
  release-pair check reads the same constant.
- The ledger CREATE text is byte-for-byte what deployed schema-18 databases
  carry, so fresh and deployed files compile to one structural manifest.
- `open_runtime_database` defaults `check_same_thread=True`; only
  `open_configuration_database` relaxes it.
- Ruff: `SLF001` is ignored for this package (store/cutover handshake).
- Tests get a per-test bootstrapped tmp `edge.sqlite3`; `tests/conftest.py`
  monkeypatches `EDGE_DATABASE_PATH` in each store module.

## Anti-patterns

- Importing `backend.app.edge_db.bootstrap` from any runtime module, or from
  `compatibility` / `schema18_manifest`. Import-linter forbids both.
- Lazy bootstrap at runtime. A missing or older file is an error, not a create.
- ALTER or DROP on schema-18 tables or rows.
- Nested `write_transaction` (raises `NestedTransactionError`).
- Putting execution-record traffic back on the product file: the diagnostics
  writer lock and WAL must not be shared with incidents, alerts, or policy
  writes. The product file's `execution_*` tables stay untouched (#583).
- A new table without a declared writer in `ownership.py`.
- Opening this database from `worker`, `shared`, or `contracts`.

## Focused tests

```bash
uv run pytest -q tests/test_edge_db_bootstrap.py tests/test_edge_db_concurrency.py \
  tests/test_edge_db_ddl_boundary.py tests/test_edge_db_schema18_constraints.py \
  tests/test_edge_db_schema18_manifest.py tests/test_sqlite_ownership_boundary.py \
  tests/test_redteam_observability_schema19.py
```

`tests/sqlite_ownership_baseline.txt` is the ownership baseline that
`test_sqlite_ownership_boundary.py` compares against.

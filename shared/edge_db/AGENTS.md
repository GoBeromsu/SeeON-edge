# EDGE DATABASE KNOWLEDGE BASE

Own the one local SQLite file shared by co-located `ml-api` and `ml-worker`.
Persistence only. Command and event traffic stay on HTTP.

## Single database

- Path: `/var/lib/seeon-state/edge.sqlite3`. WAL and SHM stay beside it.
- One file, one Linux unit, one API process, one worker process.
- `paths.py` creates `0700` / `0600` inodes. Refuse NFS and NAS.
- Public surface is DDL-free: paths, `open_runtime_database`, `write_transaction`, `best_effort_zero_wait_write`, `CURRENT_SCHEMA_RANGE`.

## Writer families

`ownership.py` is the writer contract. Prefix wins after the released-name allowlists.

- API writes `control_*` and `qa_*`, plus `API_LEGACY_TABLES`.
- Worker writes `runtime_*`, `evidence_*`, and `derivative_*`, plus `WORKER_LEGACY_TABLES`.
- Migrator alone writes `schema_*`.
- `writer_for_table()` is the only lookup. Unknown names have no writer.
- Cross-family writes are a design error. Runtime SQL authorizer denies them.

## Migrator-only DDL

- `schema.py` is the forward-only ledger. Sibling statement modules feed it. Runtimes never execute those strings.
- `compatibility.py` holds checksum identities. Current range is 16..16. Ledger drift fails verification.
- `migrator.py` is the sole DDL owner. CLI: `python -m shared.edge_db`.
- Runtime code imports `connection` and `compatibility`, never `schema` or `migrator`.
- `importer.py` is stopped-runtime cutover of the three released DBs. It takes the exclusive lock, then migrates.

## Transactions and concurrency

- Migrator takes exclusive `deployment.lock`. Runtimes take a non-blocking shared lock. A live runtime blocks migrate.
- `open_runtime_database` refuses a missing file or a non-WAL journal. It never creates schema.
- Writes use `write_transaction`: one `BEGIN IMMEDIATE`, no nesting. Bounded busy wait is 5s.
- Fatal paths use `best_effort_zero_wait_write`. Contention returns `False`.
- Authorizer denies DDL and non-read pragmas for `RuntimeActor.API` and `WORKER`.
- Two writers serialize on SQLite. Keep transactions short. Do not retry as IPC.

## SQLite is not IPC

- HTTP is the command and event boundary. Do not poll this file for the other process.
- Do not share one DB across workers. Do not store media BLOBs in rows.
- Evidence bytes live on disk. Rows hold paths, hashes, and state.

## Focused tests

```bash
uv run pytest -q tests/test_edge_db_migrations.py tests/test_edge_db_concurrency.py tests/test_edge_db_import.py -vv
```

Ledger, authorizer, lock, WAL, and two-process write serialization live in those files.

## Anti-patterns

- Importing `schema.py` from API or worker runtime.
- Opening raw `sqlite3.connect` on `edge.sqlite3` and skipping `open_runtime_database`.
- Nested transactions or a long-held `BEGIN IMMEDIATE`.
- Using the file as a mailbox, readiness signal, or camera-admission channel.

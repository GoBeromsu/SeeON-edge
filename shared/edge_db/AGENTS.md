# EDGE DATABASE KNOWLEDGE BASE

This dependency-light package owns the co-located SQLite foundation shared by the
single `ml-api` and single `ml-worker` processes on one Linux edge release unit.
It is persistence only: HTTP remains the command/event notification boundary.

## Ownership

- `schema.py` and `migrator.py`: forward-only DDL ledger. Only the one-shot
  migrator may import or execute these modules.
- `connection.py`: DDL-free runtime connections, pragmas, short transaction
  helpers, and SQL-authorizer enforcement.
- `compatibility.py`: supported schema ranges and exact runtime verification.
- `ownership.py`: `control_*`/`qa_*` are API-written;
  `runtime_*`/`evidence_*`/`derivative_*` are worker-written; `schema_*` is
  migrator-written.

Never use this database on NFS/NAS, with multiple workers, or as polling IPC.
Never put media BLOBs in it. The legacy worker outbox remains schema 8 and is not
moved by this foundation increment; legacy import/cutover belongs to Todo 6.

## Focused tests

```bash
uv run pytest -q tests/test_edge_db_migrations.py tests/test_edge_db_concurrency.py -vv
```

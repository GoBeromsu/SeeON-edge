# Edge database schema 19

Schema 19 is schema 18 plus six STRICT execution-record tables. Backend alone
owns SQLite. The worker never opens it. Bootstrap is the sole DDL owner.

## What bootstrap does

| On-disk `user_version` | Action |
| --- | --- |
| 0, no tables | Create schema 19 in one `BEGIN IMMEDIATE`. Ledger contains only row 19. |
| 18, exact schema-18 contract | Extend. See below. |
| 19 | Verify. Mutate nothing. |
| greater than 19 | Refuse (`NewerSchemaError`). |
| any other version, or tables with no version | Refuse. |

Create-only relative to schema 18: never `ALTER` / `DROP` / rewrite the ten
compact tables or their rows. Extension adds the six execution tables, their
indexes, and ledger row 19.

## Extension from schema 18

1. Verify the live file is exact schema 18: ledger ends at the frozen
   schema-18 identity, ten STRICT tables, schema-18 structural manifest.
2. If `<database name>.schema18-backup.sqlite3` already exists in the same
   state directory, refuse (`SchemaLedgerError`). Move that file aside before
   retrying.
3. Take a consistent backup with `sqlite3.Connection.backup()` to that path,
   mode `0600`.
4. In one `BEGIN IMMEDIATE`, create the six execution tables and indexes,
   insert ledger row 19 (`source_schema_version=18`,
   `source_db_sha256` = SHA-256 of the backup file bytes), set
   `PRAGMA user_version=19`.
5. Verify the runtime schema-19 contract.

The ledger after a successful extension is either `[18, 19]` (fresh schema-18
file) or `[1..17, 18, 19]` (deployed file that still carries the retired
v1–v18 rows). A fresh empty volume records only `[19]`.

## Rollback to a schema-18 image

There is no in-process downgrade.

A schema-18 image refuses a schema-19 database by design. Rolling back to a
schema-18 image after extension requires:

1. Stop the stack (`ml-worker`, `ml-api`, `edge-db-migrator`).
2. Restore `<database name>.schema18-backup.sqlite3` over `edge.sqlite3`.
3. Remove `edge.sqlite3-wal` and `edge.sqlite3-shm`.
4. Start the schema-18 images.

That restore discards every application write made after the extension,
including compact-table writes and every execution-record row.

## Query truth

Absence of an execution record is never rendered as "no person / no fall".
The vocabulary is `AVAILABLE`, `MISSING_NOT_RECORDED`, `DELETED_BY_CAPACITY`,
`UNKNOWN_COARSENED`, `UNKNOWN`.

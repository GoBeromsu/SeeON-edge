# PostgreSQL cutover

The edge moves its durable state from SQLite schema 19 (`edge.sqlite3`) to
PostgreSQL 18, schema `seeon_edge`. The move is one fenced copy, one
reconciliation and one authority transfer. There is no CDC, no dual-write and
no startup DDL: the runtime never creates or migrates the schema. Only the
migration tool below does. Live execution diagnostics go to a second schema,
`seeon_edge_diagnostics`, which the tool provisions empty and never reconciles
or copies back (see [Live diagnostics schema](#live-diagnostics-schema)).

The tool ships in the candidate `ml-api` image:

```sh
python -m backend.app.edge_db.migration <command> ...
```

Every command prints one `EDGE_PG_MIGRATION_<COMMAND>_OK ...` line on stdout
and exits 0, or prints one `EDGE_PG_MIGRATION_<COMMAND>_FAILED: <reason>` line
on stderr and exits 1. No command prints a DSN, a password or a row value.
Database errors print as error class and SQLSTATE only.

## Deployment inputs

`compose.edge.yaml` carries these. The env file names the three host secret
files; none has a default, so a render without them fails.

| Input | Contract |
| --- | --- |
| `postgres` service | PostgreSQL 18, pinned to the digest CI tests against. Data on the named volume `edge-pgdata`, a `pg_isready` healthcheck, no published port: only services on the project network reach it. |
| `PG_SUPERUSER_PASSWORD_HOST_FILE` | The `postgres` superuser password, read once by the image when it initialises an empty `edge-pgdata`. Mounted only into `postgres`. |
| `PG_OWNER_DSN_HOST_FILE` | libpq DSN of that superuser (`host=postgres user=postgres dbname=postgres password=...`). The owner creates the schema and the runtime role and sets the runtime password, which needs superuser. Mounted only into `edge-db-migrator` and `edge-db-cutover`. |
| `PG_RUNTIME_DSN_HOST_FILE` | libpq DSN of `seeon_edge_runtime` (`host=postgres user=seeon_edge_runtime dbname=postgres password=...`). `ml-api` reads it as `API_POSTGRES_DSN_FILE`; `edge-db-migrator` sets the role's password from it. |
| `API_POSTGRES_AUTHORITY_FILE` | `/run/seeon-authority/authority.json` on the named volume `edge-pg-authority`. JSON `{"generation": N, "writer_token": "<uuid>"}` written by the tool, mode `0600`. |
| `API_POSTGRES_SCHEMA` | `seeon_edge`, the same name `edge-db-migrator` passes as `--schema`. |

Make each host file a regular file, mode `0600`. Compose bind-mounts it with
that mode, and the tool refuses a DSN file with any group or other bit. The
containers read them as root.

`ml-api` mounts the authority **volume** read-only at `/run/seeon-authority`.
`transfer` and `freeze` replace the file with `os.replace`, which gives it a
new inode. A single-file bind mount keeps the old inode, so a container
created before the transfer would read the old, fenced generation and refuse
to start.

`ml-api` reads the authority file once at startup and refuses to start unless
it matches the live `deployment_authority` row and that row accepts writes.

`edge-db-migrator` runs on every `docker compose up`. Once `postgres` is
healthy it runs `provision` (below), and `ml-api` starts only after it exits 0.
`docker compose down` keeps every volume. Never run `down -v` on a deployed
edge: it deletes the database, the authority file, the legacy SQLite state and
the cutover snapshot.

`down -v` removes only the volumes that the services of the active profiles
mount. `edge-migration` is mounted only by `edge-db-cutover`, which is in the
`ops` profile, so a plain `docker compose down -v` leaves
`<project>_edge-migration` behind. To remove a rehearsal project completely,
add the profile:

```sh
docker compose -p <project> --profile ops down -v
```

## Cutover container

Run each migration step as a one-off `edge-db-cutover` container. It runs the
tool from the same `ML_API_IMAGE` as `ml-api`, waits for a healthy `postgres`,
and is in the `ops` profile, so it never starts with the stack:

```sh
docker compose run --rm edge-db-cutover <command> ...
```

| Mount | Mode | Why |
| --- | --- | --- |
| `edge-state` at `/var/lib/seeon-state` | read-write | `export` takes `deployment.lock` exclusively. It opens `edge.sqlite3` read-only. |
| `worker-local-state` at `/var/lib/seeon-worker-state` | read-only | Queue digest and liveness probe. Uses `flock` only. |
| `edge-migration` at `/var/lib/seeon-migration` | read-write | Snapshot, reports, digests. A volume of its own, outside both state volumes. |
| `pg_owner_dsn` secret at `/run/secrets/pg_owner_dsn` | read-only | The owner DSN. |
| `edge-pg-authority` at `/run/seeon-authority` | read-write | `authority.json` and its `.authority.json.pending` file |

In the commands below, `$OWNER` stands for:

```sh
--owner-dsn-file /run/secrets/pg_owner_dsn --schema seeon_edge
```

`--statement-timeout-ms` (default 600000) and `--lock-timeout-ms` (default
10000) bound every PostgreSQL command.

## Before the window

The old stack keeps running.

1. Write the three host secret files and set their paths and the candidate
   `ML_API_IMAGE` in the env file.
2. Start `postgres` and the provision job alone:

   ```sh
   docker compose up -d postgres edge-db-migrator
   docker compose logs edge-db-migrator
   ```

   Naming the two services leaves the running `ml-api` and `ml-worker` alone.
   Do not run a bare `docker compose up` before the window: it recreates
   `ml-api` on the candidate image, which refuses the fenced authority.

   The job runs:

   ```sh
   provision --owner-dsn-file /run/secrets/pg_owner_dsn --schema seeon_edge \
     --runtime-dsn-file /run/secrets/pg_runtime_dsn \
     --authority-file /run/seeon-authority/authority.json
   ```

   It provisions the schema, the live diagnostics schema, the runtime role
   and the fenced authority in one transaction. After that commits, a second
   transaction sets the runtime password. Expect
   `EDGE_PG_MIGRATION_PROVISION_OK schema=seeon_edge schema_created=true
   diagnostics_schema=seeon_edge_diagnostics diagnostics_schema_created=true
   role_created=true authority_created=true generation=1
   runtime_password_set=true`. A rerun changes nothing and prints `false` for
   all five. The server must be PostgreSQL 18 or later.
3. The role is created `LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE
   NOREPLICATION NOBYPASSRLS NOINHERIT`. Do not set its password by hand
   with `\password` or `ALTER ROLE`: the job sets it from the runtime DSN
   file. That file must be readable only by its owner, name
   `seeon_edge_runtime` as its user and carry a non-empty printable ASCII
   password; otherwise the job fails before it touches the server and names
   only the file path. The tool hashes the password as SCRAM-SHA-256 on the
   client and sends only the verifier, so the plaintext never reaches the
   server log. To rotate it, change the file and rerun the job; it prints
   `runtime_password_set=true` once. If the password step fails after
   provisioning committed, fix the file and rerun: provisioning is a no-op
   the second time.
4. The authority is provisioned fenced (`generation=1`, not accepting, egress
   off). Nothing can write to PostgreSQL yet, and the candidate `ml-api`
   refuses to start until `transfer`.

## Cutover

Do not run `docker compose up` or `edge-refused-evidence` during the window.
`up` restarts the writers, and `edge-refused-evidence` writes the worker state
the tool is reading.

1. **Fence.** Stop the writers, worker first:

   ```sh
   docker compose stop ml-worker
   docker compose stop ml-api
   ```

   `restart: unless-stopped` keeps an explicitly stopped container stopped,
   also across a Docker or host restart. The tools below refuse while any old
   process still holds its lock, so a missed stop fails closed.
2. **Account for in-flight work.** The old API delivered to the Hub
   synchronously, so the only pending deliveries are the worker's file queue
   and dead letters:

   ```sh
   queue-digest --worker-state-dir /var/lib/seeon-worker-state
   ```

   Expect `EDGE_PG_MIGRATION_QUEUE_DIGEST_OK queued=N temporary=N
   dead_lettered=N sha256=<queue sha>`. It refuses while `.gpu.lease` or
   `delivery-queue/.delivery-queue.lock` is held. Record the line.
3. **Snapshot and fence the source.**

   ```sh
   export --source /var/lib/seeon-state/edge.sqlite3 \
     --snapshot /var/lib/seeon-migration/edge-schema19.sqlite3
   ```

   Expect `EDGE_PG_MIGRATION_EXPORT_OK snapshot=... sha256=<snapshot sha>
   source_schema=19`. It refuses with `source database is in use by a running
   runtime` while any runtime holds `deployment.lock`. It also refuses a
   source that is not exact schema 19 or has a table without a mapping
   decision, a snapshot that fails the integrity or foreign-key check, a
   missing snapshot directory and an existing snapshot path. The snapshot is
   an online backup, so commits still in the source WAL are included. It holds
   resident data, so it is written mode `0600`, like the reports.

   Then fence the source, so the old stack refuses it:

   ```sh
   fence-sqlite --source /var/lib/seeon-state/edge.sqlite3 \
     --snapshot /var/lib/seeon-migration/edge-schema19.sqlite3 \
     --authority-file /run/seeon-authority/authority.json \
     --fence-receipt /var/lib/seeon-migration/fence-receipt.json
   ```

   Expect `EDGE_PG_MIGRATION_FENCE_SQLITE_OK generation=1 user_version=1000001
   source_present=true sha256=<fenced sha>`. The generation comes from the
   authority file. The source must still export to the snapshot's bytes, so
   fence right after `export`. The fence keeps the pre-fence bytes beside the
   receipt as `fence-receipt.pre-fence.sqlite3`, writes the receipt, then sets
   the source's schema version far above any the old runtime knows, so that
   runtime refuses the file as newer. It refuses while a runtime holds the
   source, a source with a rollback journal, a missing source or receipt
   directory, and a receipt that records a different fence. If it dies, rerun
   it with the same arguments; a rerun finishes the fence or returns the
   recorded one.
4. **Copy and reconcile.**

   ```sh
   import $OWNER --snapshot /var/lib/seeon-migration/edge-schema19.sqlite3
   reconcile $OWNER --snapshot /var/lib/seeon-migration/edge-schema19.sqlite3 \
     --report /var/lib/seeon-migration/reconcile-before-transfer.json \
     --source /var/lib/seeon-state/edge.sqlite3 \
     --fence-receipt /var/lib/seeon-migration/fence-receipt.json \
     --worker-state-dir /var/lib/seeon-worker-state \
     --expect-delivery-queue-sha256 <queue sha>
   ```

   `import` runs in one transaction; a failed copy leaves the target as
   provisioned and can be rerun. A second import is refused. Expect
   `EDGE_PG_MIGRATION_IMPORT_OK tables=15 rows=N ...` and
   `EDGE_PG_MIGRATION_RECONCILE_OK result=PASS report=...
   diagnostics_schema=seeon_edge_diagnostics`.

   The report compares every table by key and row hash, checks status
   distributions, identity floors, the audit tail, that the live source still
   equals the snapshot, that the authority is fenced, that no delivery rows
   exist and that the queue digest is unchanged. With `--fence-receipt` it
   also requires the source to be byte-equal to the receipt with no WAL, `-shm`
   or journal content (`sqlite:<reason>`), and the receipt to name this
   snapshot (`sqlite:snapshot_mismatch`). `--fence-receipt` needs `--source`;
   without it the command is a usage error (exit 2). Any failure prints
   `EDGE_PG_MIGRATION_RECONCILE_FAILED: result=FAIL` and lists its reasons in
   the report. Do not transfer on FAIL.

   The report lists the diagnostics schema apart, under `diagnostics`, with
   `mode=live` and `reconciled=false`. It checks that schema's ledger and
   table set and records its row counts, but compares nothing against the
   snapshot. A missing, foreign or drifted diagnostics schema fails the report
   as `diagnostics:schema`.
5. **Transfer, once.**

   ```sh
   transfer $OWNER --authority-file /run/seeon-authority/authority.json \
     --worker-state-dir /var/lib/seeon-worker-state
   ```

   Expect `EDGE_PG_MIGRATION_TRANSFER_OK generation=2 accepting=true
   egress_enabled=true`. It requires exactly one imported snapshot, empty
   delivery tables, a stopped worker and an authority file equal to the live
   row. The old generation-1 token is fenced in the same transaction.

   If the command dies, loses its connection or reports an unknown commit
   outcome, rerun it with the same arguments. `.authority.json.pending` records
   the staged token, and the rerun either publishes it (the commit happened)
   or discards it and transfers (it did not). A transfer that already
   succeeded is refused, not repeated. Start no runtime until you have the OK
   line. If a rerun cannot give it, follow `Torn transfer`.
6. **Reconcile after transfer**, before any runtime starts (see `Activation`).

   ```sh
   reconcile $OWNER --snapshot /var/lib/seeon-migration/edge-schema19.sqlite3 \
     --report /var/lib/seeon-migration/reconcile-after-transfer.json \
     --source /var/lib/seeon-state/edge.sqlite3 \
     --fence-receipt /var/lib/seeon-migration/fence-receipt.json \
     --worker-state-dir /var/lib/seeon-worker-state \
     --expect-delivery-queue-sha256 <queue sha> \
     --after-transfer
   ```

   Expect `EDGE_PG_MIGRATION_RECONCILE_OK result=PASS`. The report records
   `mode=after_transfer` and runs every check of step 4. Only the authority
   check changes: it requires a generation above 1 and fails
   `authority:not_transferred` on a generation-1 target. It does not check
   fencing, so a transferred target that was later frozen also passes. It
   tolerates one difference, the `edge_site` row that `transfer` seeds: the
   report shows `activation_seed=true` and `edge_site` goes from 0 rows to 1.
   A site that holds anything beyond its column defaults fails as
   `table:edge_site`.

   Without `--after-transfer` the transferred target fails
   `authority:not_fenced`, by design. Start no runtime on FAIL.
7. **Retain** until the rollback decision is closed: the `edge-migration`
   volume (the snapshot, both reconcile reports, the fence receipt and its
   `fence-receipt.pre-fence.sqlite3` copy), the queue digest line,
   the `edge-state` volume (`edge.sqlite3`, `edge-diagnostics.sqlite3`,
   `edge.sqlite3.schema18-backup.sqlite3`), the `worker-local-state` volume,
   the clip store, and the old `ML_API_IMAGE` and `ML_WORKER_IMAGE` digests.
   `edge-diagnostics.sqlite3` is not migrated; it stays where it is.
   Opening the source read-only can leave `edge.sqlite3-wal` and
   `edge.sqlite3-shm` beside it. Do not delete them while the old stack could
   still resume.

## Fresh install

Use this instead of `Cutover` only on an edge whose old stack never ran, so
there is no `edge.sqlite3` to export. The target opens empty and nothing is
imported. An edge that holds legacy data goes through `Cutover`.

1. Provision as in `Before the window`, step 2:

   ```sh
   docker compose up -d postgres edge-db-migrator
   ```

   Expect `EDGE_PG_MIGRATION_PROVISION_OK ... generation=1`.
2. **Transfer, once.**

   ```sh
   docker compose run --rm edge-db-cutover transfer \
     --owner-dsn-file /run/secrets/pg_owner_dsn --schema seeon_edge \
     --authority-file /run/seeon-authority/authority.json --fresh-install
   ```

   Expect `EDGE_PG_MIGRATION_TRANSFER_OK generation=2 accepting=true
   egress_enabled=true`. The target must be provisioned, empty and never
   imported, so a fresh install after `import` is refused.

   `--source` names the legacy SQLite path that must be absent. It defaults
   to `/var/lib/seeon-state/edge.sqlite3` and is accepted only with
   `--fresh-install`; on its own it is a usage error (exit 2). Before it
   touches PostgreSQL, the tool refuses when the default path or `--source`
   exists, or its `-wal` or `-journal` file does, or any of them is a
   symlink: `EDGE_PG_MIGRATION_TRANSFER_FAILED: legacy SQLite source exists
   ...; export and import it instead`. The authority stays fenced at
   generation 1. Move that data through `Cutover`.

   Keep the `edge-state` volume mounted, and do not remove it before this
   step. The refusal sees only the mounted volume: a missing or recreated
   volume hides legacy data that `export` and both rollbacks need.

   Rerun an unknown outcome with the same arguments, as in `Cutover`, step 5.
3. **Fence the absent source**, after `transfer`: `--fresh-install` refuses a
   source path that holds any file, a fenced one included.

   ```sh
   docker compose run --rm edge-db-cutover fence-sqlite \
     --source /var/lib/seeon-state/edge.sqlite3 \
     --authority-file /run/seeon-authority/authority.json \
     --fence-receipt /var/lib/seeon-migration/fence-receipt.json
   ```

   Expect `EDGE_PG_MIGRATION_FENCE_SQLITE_OK generation=2 user_version=1000002
   source_present=false sha256=<fenced sha>`. With no source and no
   `--snapshot`, the fence creates a stamped empty file, so an old runtime
   started on this volume refuses it instead of creating a new database.
   Such a site has no SQLite rollback: `rollback-check` denies
   `sqlite:source_absent` and `unfence-sqlite` refuses the receipt.
4. **Start the API.**

   ```sh
   docker compose up -d ml-api
   ```

   `edge-diagnostics.sqlite3` is abandoned. It is never imported, and live
   diagnostics start empty in `seeon_edge_diagnostics`. If the volume holds
   one, leave it in place.

## Activation

Pick one option and record it in the deployment record.

| Option | After `transfer` | Worker rollback |
| --- | --- | --- |
| A | Start the new `ml-api` and the Rust worker together. | Return to the old worker digest; keep PostgreSQL. |
| B | Start the new `ml-api` with the old worker digest. Switch to the Rust worker as a separate step. | Return to the old worker digest; keep PostgreSQL. |

The old worker's queue survives on `worker-local-state` and is replayed by
whichever worker starts. Option A therefore needs proof that the Rust worker
drains the Python queue format. Option B needs proof that the old worker
works against the new API.

## Table mapping

Migrated in load order, one-to-one by column: `credentials`, `edge_site`,
`locations` (floors first), `cameras`, `policies`, `clips`, `incidents`,
`artifacts`, `audit_events`, `execution_provenance`, `execution_segments`,
`execution_units`, `execution_records`, `execution_coverage`,
`execution_batches`. `cameras.incarnation` exists only in PostgreSQL and
takes its default. Identity sequences restart above both the highest used id
and the SQLite `AUTOINCREMENT` high-water.

| SQLite table | Decision |
| --- | --- |
| `schema_migrations` | Not copied. Verified as schema 19. PostgreSQL keeps its own ledger; the import stamps the source schema and snapshot hash on it. |
| `sqlite_sequence` | Carried as identity floors. |
| `sqlite_stat1`, `sqlite_stat4` | Planner statistics. Not copied. |
| any other table | `export` refuses. |

| PostgreSQL-only table | Initial state |
| --- | --- |
| `schema_migrations` | Provision row plus the import stamp. |
| `deployment_authority` | Provisioned fenced at generation 1, transferred once to generation 2. |
| `event_outbox` | Empty. The old API had no outbox; pending work is the worker queue. |
| `event_delivery_attempts`, `event_delivery_results`, `event_delivery_observations` | Empty. |

## Live diagnostics schema

Live execution diagnostics are written to `<API_POSTGRES_SCHEMA>_diagnostics`,
by default `seeon_edge_diagnostics`. The runtime derives the name from
`API_POSTGRES_SCHEMA`; there is no separate setting. The runtime only verifies
this schema and never creates it.

`provision` creates it beside the product schema from the same
`postgres_diagnostics.sql` DDL, with its own `schema_migrations` ledger. The
runtime role gets `USAGE` on it, `SELECT, INSERT, UPDATE, DELETE` on its
`execution_*` tables, `SELECT` on its ledger, and no DDL.

Migrated `execution_*` history lands in the product schema. The diagnostics
schema starts empty and holds only rows the new runtime writes. `reconcile`
does not compare it with the snapshot. `rollback-check` denies any row in
it, because SQLite never saw one (see `Rollback after transfer`).

`provision` refuses, and changes nothing, when the diagnostics schema:

- is owned by another role;
- has objects but no ledger;
- has a ledger that is newer than, or differs from, this tool's;
- has tables beyond the provisioned set;
- would not fit 63 bytes. The product schema name must leave room for the
  `_diagnostics` suffix.

## Torn transfer

A transfer that stopped after its commit but before it replaced
`authority.json` leaves the database at generation 2 and the file at
generation 1. `edge-db-migrator` then fails `authority file does not match
the database authority` on every `up`, so `ml-api` never starts. That is
fail-safe: no generation can write.

1. Keep the stack down. Do not delete `.authority.json.pending`, and do not
   edit `authority.json` or the `deployment_authority` row by hand. The
   pending file holds the only copy of the new writer token.
2. Rerun the `transfer` you ran, with the same arguments (`Cutover`, step 5,
   or `Fresh install`, step 2). When the database holds the staged token, the
   rerun publishes the pending file and prints
   `EDGE_PG_MIGRATION_TRANSFER_OK generation=2 accepting=true
   egress_enabled=true`. Continue from the step after that transfer.
3. Any other outcome ends the cutover with runtimes stopped:

| Rerun failure | Meaning |
| --- | --- |
| `authority was already transferred` | The file already matches the database at generation 2; the transfer completed. Continue from the step after it. |
| `cannot transfer a different persistence authority` | The database holds a token that neither the file nor a pending file names. No command rebuilds the file from the database. Escalate. |
| `pending authority matches neither the database nor the file` | The three copies disagree. Escalate. |

A rerun refused with `the old worker holds its runtime lease; stop it first`
or `the delivery queue is locked by another process` has changed nothing.
Stop that process and rerun.

## Rollback before transfer

Until `fence-sqlite` runs, the source is never modified.

1. Stop any running `edge-db-cutover` container.
2. If `fence-sqlite` ran, run `rollback-check` and `unfence-sqlite` as in
   `Rollback after transfer`, step 3. Skip `freeze`: before `transfer` the
   authority is still fenced at generation 1.
3. Start the old stack on `edge.sqlite3`.

A retry needs a new snapshot path and a fresh target, because `import`
refuses a target that already holds a snapshot. Provision a new schema name,
changing `API_POSTGRES_SCHEMA` and the `edge-db-migrator` `--schema` together,
or drop the abandoned schema by hand. The tool
never drops anything. Nothing has written to the diagnostics schema yet, and a
rerun of `provision` reuses it as it is.

## Rollback after transfer

1. Stop the new `ml-api` and worker.
2. Fence PostgreSQL:

   ```sh
   freeze $OWNER --authority-file /run/seeon-authority/authority.json
   ```

   Expect `EDGE_PG_MIGRATION_FREEZE_OK generation=2 accepting=false
   egress_enabled=false`. A frozen authority refuses `ml-api` startup.
3. Decide:

   ```sh
   rollback-check $OWNER --snapshot /var/lib/seeon-migration/edge-schema19.sqlite3 \
     --source /var/lib/seeon-state/edge.sqlite3 \
     --fence-receipt /var/lib/seeon-migration/fence-receipt.json \
     --report /var/lib/seeon-migration/rollback-check.json
   ```

   Do not open the fenced source with any SQLite tool, read-only included.

   `result=ALLOW reasons=none` means PostgreSQL holds nothing the snapshot
   lacks and the fenced source is byte-equal to its receipt. Restore the
   pre-fence source, then start the old stack on it:

   ```sh
   unfence-sqlite --source /var/lib/seeon-state/edge.sqlite3 \
     --fence-receipt /var/lib/seeon-migration/fence-receipt.json \
     --rollback-report /var/lib/seeon-migration/rollback-check.json
   ```

   Expect `EDGE_PG_MIGRATION_UNFENCE_SQLITE_OK result=RESTORED generation=1
   sha256=<pre-fence sha>`. It refuses a report that did not ALLOW, or that
   checked a different source, snapshot or fence, and it checks the live file
   against the receipt again before it restores the preserved copy. Optionally,
   before `unfence-sqlite`, rerun the reconcile of `Cutover`, step 6, as
   evidence, with a new `--report` path. It needs `--after-transfer`: the
   frozen target is still at generation 2 and holds the `edge_site` seed,
   which the default mode fails as `table:edge_site`.

   `result=DENY` means PostgreSQL has state that a restore would lose. Keep
   PostgreSQL, and either return to the old worker digest or fix forward.
   Nothing copies PostgreSQL back into SQLite, so a denied rollback cannot be
   forced. The frozen authority still keeps `ml-api` down, and this tool has
   no command that lifts a freeze, so escalate rather than edit the authority
   row or file.

| DENY reason | Meaning |
| --- | --- |
| `authority_not_fenced` | Run `freeze` first. |
| `delivery_history:<table>` | The new API recorded outbox or delivery rows. |
| `ledger:entries`, `ledger:snapshot_mismatch` | The ledger is not exactly the import of this snapshot. |
| `unimported_rows:<table>` | Rows exist without an import stamp. |
| `target_history:<table>` | A migrated table changed after import. |
| `diagnostics:schema` | The diagnostics schema is missing, foreign or drifted. |
| `diagnostics_history:<table>` | The new runtime wrote diagnostics rows SQLite never saw. |
| `sqlite:source_absent` | A fresh install: there is no SQLite file to restore. |
| `sqlite:receipt_snapshot_mismatch` | The receipt fenced a different snapshot. |
| `sqlite:live_changed`, `sqlite:source_missing`, `sqlite:source_not_regular` | The source is not the file the fence left. |
| `sqlite:wal_content`, `sqlite:shm_content`, `sqlite:journal` | Something opened the fenced source, even read-only, or wrote to it. If nothing wrote, rerun `fence-sqlite` with the same arguments and check again. |

Never restore the SQLite snapshot only because no new event arrived. Target
history without a new event still denies.

`rollback-check` inspects the product schema, the diagnostics schema and the
fenced source. Any diagnostics row denies, as `diagnostics_history:<table>`,
and a write into the product schema's `execution_*` tables denies as
`target_history:<table>`. On rollback
the diagnostics schema is abandoned or retained for inspection. It is never
copied back into SQLite. The old stack resumes on its own
`edge-diagnostics.sqlite3`, which never moved. Before a later cutover, drop the
abandoned diagnostics schema by hand if its rows must not carry over;
`provision` accepts it as it is.

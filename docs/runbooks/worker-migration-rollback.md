# Migrate or roll back edge state safely

The current edge release has one writable SQLite database:
`/var/lib/seeon-state/edge.sqlite3` in the `edge-state` volume. The backend API
is its sole application writer; the worker has no database mount and interacts
with the backend only over HTTP. The worker never opens, validates, migrates, or
repairs this database.

The stopped-runtime `edge-db-migrator` is the only DDL owner. It checkpoints,
backs up, integrity-checks, and imports the released `catalog.sqlite3`,
`connection-settings.sqlite3`, and `worker-state.sqlite3`. A legacy worker
source at schema 6 or 7 is migrated through outbox schema 8 in a secured working
copy before its rows are imported; schema 8 and 9 sources import unchanged. The
original legacy database is not upgraded.

Rollback is digest-based and decided per database file and traffic boundary.
Never open any database with a binary whose maximum supported schema is lower
than that database's schema. Never use a mutable image tag, direct SQL, or an
environment-derived facility identity.

## Before migration

Record the exact running and target API/worker image digests and retain the sealed
previous Compose deployment artifact. Stop API and worker, acquire the edge
deployment lock, and reject a running updater.

For each legacy database independently:

1. record its schema version
2. checkpoint WAL with `PRAGMA wal_checkpoint(TRUNCATE)`
3. use the SQLite online backup API into a root-only `0700` directory and `0600`
   timestamped file
4. run `PRAGMA integrity_check`
5. fsync the backup file and parent directory
6. record only filename, SHA-256, size, schema, table counts, and row counts

Require healthy volumes, drained queues, at least 20 GiB free clip capacity, and
verified backups before changing containers. Retain the untouched
`ml-api-state` and `ml-worker-state` volumes throughout the rollback window.
They are mounted only by the stopped-runtime migrator during import; the API and
worker must never mount or write them after cutover.

## Migrate and cut over

Obtain the deployment command once:

```sh
cd /opt/eldercare-fall-ml
DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'
```

Run inventory, then the one-shot schema-18 candidate cutover. This is the only
command allowed to execute DDL:

```sh
$DC up --pull always edge-filesystem-inventory
$DC up --pull always edge-db-migrator
```

Stop unless it exits zero and reports `EDGE_DB_COMPACT_CUTOVER_OK`. Confirm its
source receipts, counts, digests, and integrity result before admitting traffic.
Then preserve the required dependency order:

```sh
$DC up -d --wait ml-api
$DC up -d --wait ml-worker
```

Compose enforces inventory completed successfully -> candidate cutover completed
successfully -> `ml-api` healthy -> `ml-worker`. Do not start either runtime
around the dependency gates. Record
the first successful API write to `edge.sqlite3` as the **central cutover traffic boundary**. From that point, legacy snapshots are stale.

## Rollback decision

### Before central cutover traffic

If the migrator or readiness checks fail before the API writes
central state, a rollback to the untouched legacy volumes is permitted. Decide
compatibility separately for `catalog.sqlite3`,
`connection-settings.sqlite3`, and `worker-state.sqlite3`; retain their recorded
schemas with the sealed deployment artifact. Restore the sealed
previous Compose artifact and recorded image digests, remount the untouched
legacy volumes, and start the previous API before the previous worker.

Do not restore a backup over an untouched legacy volume. Do not mix a central
runtime with a legacy runtime, and do not copy imported central rows back into a
legacy database.

### After central cutover traffic

Once central traffic exists, never restore or restart a legacy database: doing so
would discard enrollment, topology, heartbeat, outbox, config, or fault writes
made after import. Preserve `edge-state` and fix forward, or roll back only to
recorded API/worker digests that explicitly support the current central schema.
A binary-only rollback must not restore database bytes.

For a central-compatible binary rollback:

1. stop new work and drain relay/evidence queues
2. verify the previous API image supports the current `edge.sqlite3` schema
3. restore the recorded immutable image digests
4. recreate `ml-api` and wait for `/health/ready`
5. recreate `ml-worker` only after the API is healthy
6. verify `/api/v1/system`, normal heartbeat, topology, and event idempotency

Never use `down -v`, delete `edge-state`, edit SQLite manually, or promise rollback
to an outbox schema-6/7 binary after it would open schema-8-compatible state.

## Roll forward

Deploy the sealed central-schema-compatible digests through the same migrator ->
API healthy -> worker sequence. On an already imported database the migrator is
idempotent and verifies the recorded source digests/counts rather than duplicating
rows. Enrollment and camera topology remain database/UI/Hub-owned; do not stamp a
facility ID into worker config or reconstruct it from a backup filename.

Run the lifecycle smoke only after the content-addressed execution receipt is
green. Boolean declarations are not lifecycle evidence: the receipt must contain
sequenced successful executions for enrollment, topology, heartbeat, event/clip,
rotation, timeout retry, ML-first rollback, roll-forward, and restart.

```sh
sh scripts/ops/cloud-enrollment-smoke.sh \
  --host happy-nursing-home-raw --full-lifecycle --rollback-drill
```

Success evidence contains only plan/SHA/digest bindings, backup metadata,
versioned API statuses, counts, and redacted lifecycle results. Never put SQLite
bytes or secret values in `.omo` evidence.

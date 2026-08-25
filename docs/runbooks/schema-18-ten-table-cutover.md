# Cutover: schema 18 ten-table candidate replace

This runbook takes a stopped edge unit from schema 17 to the compact ten-table
schema 18. The one-shot command copies the live database to a read-only v17
archive, builds a same-filesystem candidate, applies schema 18 only to that
candidate, then atomically replaces the live file.

Read this whole file before starting. The migration is **forward-only after its
first v18 write**. The v17 archive is permanent and read-only. There is no
archive deletion command.

## Preconditions

1. **Images built from the intended revision, all carrying schema identity 18.**

   Backend, baked frontend, and worker must advertise
   `edge_database_schema_version=18`. Mixed 17/18 images refuse to start.

   ```sh
   docker inspect "$ML_API_IMAGE" --format '{{index .Config.Labels "seeon.edge.database.schema-version"}}'
   docker inspect "$ML_WORKER_IMAGE" --format '{{index .Config.Labels "seeon.edge.database.schema-version"}}'
   docker run --rm --entrypoint cat "$ML_API_IMAGE" /opt/seeon/edge-database-schema-version
   docker run --rm --entrypoint cat "$ML_WORKER_IMAGE" /opt/seeon/edge-database-schema-version
   ```

   Both must print `18`. A 17/18 pair is a refused startup, not a dual-write.

2. **Confirm you are pointed at the live volume.**

   ```sh
   docker inspect <api-container> \
     --format '{{index .Config.Labels "com.docker.compose.project"}}'
   docker compose -f compose.edge.yaml config | grep '^name:'
   ```

   `compose.edge.yaml` requires `COMPOSE_PROJECT_NAME`. Set it to the value the
   first command prints. `docker compose down -v` is forbidden.

3. **Stop API and worker.** The cutover holds the deployment lock and refuses
   while a runtime is running.

4. **The schema-17 backlog must be drained first.** Check what is still in flight:

   ```sh
   docker exec <api-container> python3 -c "
   import sqlite3
   c = sqlite3.connect('file:/var/lib/seeon-state/edge.sqlite3?mode=ro', uri=True)
   print('events        ', c.execute(\"SELECT state, COUNT(*) FROM evidence_events GROUP BY 1\").fetchall())
   print('clips local   ', c.execute('SELECT local_state, COUNT(*) FROM evidence_clips GROUP BY 1').fetchall())
   print('clips publish ', c.execute('SELECT publish_state, COUNT(*) FROM evidence_clips GROUP BY 1').fetchall())
   print('GATE BLOCKS   ', bool(c.execute(\"\"\"
       SELECT EXISTS(SELECT 1 FROM evidence_events WHERE state IN ('STAGED','READY','IN_FLIGHT'))
           OR EXISTS(SELECT 1 FROM evidence_clips WHERE local_state = 'AWAITING_FINALIZE')
           OR EXISTS(SELECT 1 FROM evidence_clips WHERE publish_state = 'IN_FLIGHT')
           OR EXISTS(SELECT 1 FROM derivative_jobs WHERE state IN ('PENDING', 'RUNNING'))
           OR EXISTS(SELECT 1 FROM derivative_evidence_slots WHERE state = 'PENDING')
           OR EXISTS(SELECT 1 FROM evidence_retention_states WHERE state = 'PENDING')
   \"\"\").fetchone()[0]))
   "
   ```

   Proceed only when `GATE BLOCKS` prints `False`. That predicate is the
   cutover's own drain gate. An in-flight source leaves the live file
   byte-identical at schema 17.

5. **Packaged SQLite is 3.51.3 or newer.** An older injected runtime refuses
   before it creates a candidate.

## Backups

Checkpoint WAL, then keep four independent backups before the candidate is
written:

- live `edge.sqlite3` plus any leftover `-wal`/`-shm` after
  `PRAGMA wal_checkpoint(TRUNCATE)`
- the read-only v17 archive the cutover writes at
  `/var/lib/seeon-state/edge-v17-archive.sqlite3` (`0400`)
- the host clip-store directory
- the worker `delivery-queue` and `delivery-queue-dead-letter`

Record filename, SHA-256, size, and schema version. Do not open the archive
with a serving binary.

## Candidate validation

Compose already encodes inventory → candidate cutover → API → worker:

```sh
cd /opt/eldercare-fall-ml
DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'
$DC up --pull always edge-filesystem-inventory
$DC up --pull always edge-db-migrator
```

The migrator is this command, and nothing else may execute schema-18 DDL:

```sh
python -m backend.app.edge_db.compact_cutover \
  --source /var/lib/seeon-state/edge.sqlite3 \
  --live /var/lib/seeon-state/edge.sqlite3 \
  --archive /var/lib/seeon-state/edge-v17-archive.sqlite3 \
  --candidate /var/lib/seeon-state/edge-v18-candidate.sqlite3 \
  --receipt /var/lib/seeon-state/schema18-cutover-receipts.jsonl \
  --clip-store /var/lib/clip-store \
  --worker-state /var/lib/seeon-worker-state
```

Stop unless it prints `EDGE_DB_COMPACT_CUTOVER_OK`. Confirm:

- candidate `PRAGMA user_version` is 18 before rename
- application tables are exactly the ten compact names
- archive SHA-256 still equals the pre-cutover live digest
- a changed archive digest prints `EDGE_DB_CUTOVER_STALE_ARCHIVE` and leaves
  live at 17

## Same-filesystem replace

The command copies archive → candidate on the `edge-state` volume, applies
schema 18 only to the candidate, fsyncs, then `os.replace`s onto
`edge.sqlite3`. A symlink or cross-filesystem target is refused. The live file
is untouched until that replace.

## Post-cutover probes

```sh
$DC up -d --wait ml-api
$DC up -d --wait ml-worker
```

Then:

- `GET /health/live` is 200
- `GET /health/ready` is 200
- `GET /health/release-identity` returns `edge_database_schema_version: 18`
- `PRAGMA user_version` is 18
- worker boot against a 17 API identity exits 3

## Rollback before the first v18 write

If the API has not yet written compact application rows, restore the archive:

```sh
python -m backend.app.edge_db.compact_cutover \
  --source /var/lib/seeon-state/edge.sqlite3 \
  --live /var/lib/seeon-state/edge.sqlite3 \
  --archive /var/lib/seeon-state/edge-v17-archive.sqlite3 \
  --candidate /var/lib/seeon-state/edge-v18-candidate.sqlite3 \
  --receipt /var/lib/seeon-state/schema18-cutover-receipts.jsonl \
  --clip-store /var/lib/clip-store \
  --worker-state /var/lib/seeon-worker-state \
  --rollback
```

Live returns to the archived v17 bytes. The archive stays `0400`.

## Forward-only recovery after the first v18 write

After the first compact-table write, the same `--rollback` request prints
`EDGE_DB_CUTOVER_FORWARD_ONLY` and leaves live at 18. Recovery is a
schema-18-capable binary plus the live file, never a downgrade and never a
restore that would discard post-cutover writes.

## Permanent read-only v17 archive

`/var/lib/seeon-state/edge-v17-archive.sqlite3` remains on the volume. Serving
runtimes never mount it, ATTACH it, or open its legacy tables. Do not delete it.
A later legal-hold policy, not this command, owns archive retention.

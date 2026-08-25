# Cutover: backend-only SQLite ownership (schema 17)

The inference-runtime slot no longer holds a database. This runbook takes a live
edge deployment from schema 16 to 17 and swaps in images that carry no
`edge-state` mount for the worker.

Read this whole file before starting. The migration is **forward-only after its
first write**, and the drain gate exists to stop you migrating over evidence that
has not been delivered.

## Schema 18 staging boundary

This runbook lands the lossless schema-17 staging database only. The legacy
three-database importer now stops at v17; ordinary `migrate_database` is not
authorized to compact that file. Schema 18 must use the stopped-runtime
candidate command and its immutable source/archive/reconciliation artifacts:

```sh
python -m backend.app.edge_db.compact_cutover --help
```

The source clone and live v17 file are distinct inputs. Keep the source and
archive permanently read-only; only the verified candidate may atomically
replace `--live`. A complete schema-18 deployment procedure remains owned by
the schema-18 release runbook task.

## Preconditions

1. **Images built from the intended revision.** Verified for `fa3811a`:

   ```
   local/fall-ml-worker:fa3811a-boundary
   local/fall-ml-api:fa3811a-boundary  # gitleaks:allow
   ```

   Both were smoke-checked: the worker boots `python -m worker --check-config`
   with exit 0, the API constructs its app with 74 routes, and the worker image
   contains neither `shared.edge_db` nor `backend.app.edge_db`.

2. **Confirm the running worker writes its queue to a volume, not the container
   layer.** The worker's default state directory is home-relative, which inside
   a container is the writable layer. A deployment that mounts
   `worker-local-state` but does not pass `--state-dir` loses every pending
   evidence envelope on any container replacement, and the filesystem gate
   below then scans an empty volume and reports clear.

   ```sh
   # must print a --state-dir matching the worker-local-state mount
   docker inspect <worker-container> --format '{{json .Config.Cmd}}'
   docker inspect <worker-container> \
     --format '{{range .Mounts}}{{.Name}} -> {{.Destination}}{{"\n"}}{{end}}'

   # and the queue must actually be there
   docker exec <worker-container> ls /var/lib/seeon-state/delivery-queue
   ```

   If the queue directory only exists under `/root/.local/state/ml-worker`, the
   currently queued evidence is **not** on a volume. Drain it before replacing
   the container, or it is lost.

3. **Check whether the backend refused any evidence.** A 422 retains the entry
   instead of deleting it, so it survives but has not been delivered:

   ```sh
   docker compose --profile ops run --rm edge-refused-evidence
   ```

   The worker image carries no `scripts/ops` and the backend image has no
   writable worker-state mount, so this dedicated one-shot service is the only
   place the command can run.

   Exit 1 means evidence is held. Fix the cause the reason code names, then
   `docker compose --profile ops run --rm edge-refused-evidence --requeue` to return it to the live queue. Do not migrate while evidence is
   still held: it was never delivered.

4. **Confirm you are pointed at the live volume, before anything else.** This
   host carries `edge_edge-state`, `seeon-edge-wt-alert-api_edge-state` and
   `seeon-prod-edge-state`. Binding the wrong one migrates an empty database and
   reports success while the live one keeps its undelivered evidence.

   ```sh
   # what the running stack actually uses
   docker inspect <api-container> \
     --format '{{index .Config.Labels "com.docker.compose.project"}}'

   # what your invocation will use -- these must match
   docker compose -f compose.edge.yaml config | grep '^name:'
   ```

   `compose.edge.yaml` requires `COMPOSE_PROJECT_NAME` and refuses to render
   without it, so a forgotten value is a config-time error rather than a silent
   bind. Set it to the value the first command prints.

5. **The schema-16 backlog must be drained first.** Check what is still undelivered:

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

   The final line evaluates the migrator's own gate predicate, so it answers the
   question the rest of the output only hints at. Proceed when it prints
   `False`.

   Any `STAGED`, `READY` or `IN_FLIGHT` event, any clip with
   `local_state='AWAITING_FINALIZE'`, or any clip with
   `publish_state='IN_FLIGHT'` makes the migrator refuse with
   `EDGE_DB_DRAIN_INCOMPLETE`. `publish_state='WAITING'` deliberately does not
   block the gate: it is the default state, not proof of an active delivery.
   **That refusal is correct** — migrating would strand unresolved evidence.

6. **`edge-state` volume and compose project name are preserved.** The volume
   holds the camera registry. `docker compose down -v` is forbidden.

## Legacy schema-16 backlog

The live schema-16 database contains 1143 undelivered `evidence_events` and
1053 `evidence_clips` at `local_state='AWAITING_FINALIZE'`. A repaired worker cannot drain
these legacy rows: the worker now owns no SQLite and reads only filesystem
`DeliveryQueue` entries. The migrator runs before API and worker, and refuses
while pending event rows remain.

Use the backend-owned drain while the schema remains at 16. It delivers each
legacy event through the running backend relay, preserving the relay's existing
idempotent cloud delivery and durable acceptance receipt. Permanent and
compatibility failures remain recorded in `evidence_events`; they are never
deleted.

## Step 1 — run the backend legacy drain, migrate nothing

Keep the current API serving the schema-16 database and stop the legacy worker
writer. Run the operator command from the API image or the matching release
checkout:

```sh
python scripts/ops/drain-legacy-evidence.py \
  --database /var/lib/seeon-state/edge.sqlite3 \
  --relay-url http://ml-api:8000 \
  --relay-token "$API_EDGE_RELAY_TOKEN"
```

**Exit 0 means every legacy event actually reached the backend.** Any other exit
status means undelivered evidence remains, and the migrator will keep refusing.

Repeat the command while it reports `retryable` rows; those are transient and
will clear once the backend is reachable.

A `permanent` count is different and does **not** clear by repeating. Those rows
were rejected by the backend, most often for a payload the current contract will
not accept. They stay in `READY` with `last_error_code` recording why, so they
are still counted as pending and still block migration. That is deliberate: the
alternative is marking undelivered evidence as delivered and letting the
migration proceed over it. Resolving them is an explicit decision, never a side
effect of running the drain. The command never deletes evidence.

## Step 2 — recover legacy clips and verify both drains

Run the backend-owned clip recovery against the same database and the clip-store
root. It verifies a `READY` manifest's owned, regular `clip.mp4` against its
recorded size and SHA-256 before setting `local_state='VERIFIED'`. A missing
clip-store directory becomes `UNAVAILABLE/MISSING`; a malformed manifest or
media mismatch becomes `CORRUPT/CORRUPT`. The command never deletes a row.

```sh
python scripts/ops/recover-legacy-clips.py \
  --database /var/lib/seeon-state/edge.sqlite3 \
  --clip-store /var/lib/clip-store
```

**Exit 0 means no `AWAITING_FINALIZE` or `IN_FLIGHT` clip, active derivative,
or pending retention intent remains.** Exit 2 means work remains and must be
allowed to settle or investigated before migration.

**Exit 3 means the command refused before clip recovery and changed nothing.** It reports
`{"error": "clip_store_unavailable", ...}` when `--clip-store` does not point at
a mounted finalized clip store, or when the store cannot be read part way
through. Do not retry until the mount is fixed, and do not treat it as exit 2.

**Exit 4 means clip recovery completed but transitional recovery then refused.**
It reports `{"error":"clip_store_unavailable_after_clip_recovery",...}` with
the committed clip counts. Fix the mount and rerun the command; do not claim
that no rows changed.

The exit-3 distinction is the whole safety property of the command. Without it a
mistyped path or a dropped mount is indistinguishable from a store whose media
is genuinely gone: every clip would be classified `UNAVAILABLE` or `CORRUPT`,
their publication terminalized, the command would report success, and the
forward-only migration would proceed over evidence nobody ever looked at. A
mounted store that is genuinely empty still proceeds normally, because that is a
fact about the evidence rather than a fact about the mount.

There is no schema-16 clip publisher in the backend. Accordingly this command
classifies every terminal clip still at `publish_state='WAITING'` as
`publish_state='PERMANENT'`, increments its publish attempt count, and records
`last_error_code='LEGACY_CLIP_PUBLICATION_UNSUPPORTED'`. This is intentionally
not a claim that upstream received the clip; it records that no backend actor
can publish it, rather than leaving its upstream outcome undecided forever.
It also resolves a stopped-runtime `publish_state='IN_FLIGHT'` row to
`PERMANENT`, clears its abandoned lease, and records
`last_error_code='LEGACY_CLIP_PUBLISHER_RETIRED'`; it never records
`PUBLISHED`, because no upstream receipt exists.

The same command is the resolver for every remaining schema-17 gate condition:
`derivative_jobs.state IN ('PENDING','RUNNING')` becomes
`CANCELLED/LEGACY_DERIVATIVE_EXECUTOR_RETIRED`, because the retired worker
cannot truthfully produce a derivative; `evidence_retention_states.state='PENDING'`
becomes `PURGED` only after the command verifies that the recorded clip media is
absent, otherwise `FAILED/LEGACY_RETENTION_MEDIA_STILL_PRESENT`; and a pending
`derivative_evidence_slots.state='PENDING'` becomes
`UNAVAILABLE/LEGACY_DERIVATIVE_EXECUTOR_RETIRED`. A pending `PRIMARY_CLIP`
artifact slot is reconciled atomically with its clip verdict.
For an intact clip it creates the verified media object and primary projection,
marks the slot `AVAILABLE`, and advances the incident to `MEDIA_READY`. For a
missing or corrupt clip it records a terminal slot and primary projection and
advances the incident to `FAILED`. It never claims that a derivative was
rendered, media was purged, or a clip exists without verifying that fact.

Re-run the precondition query. Proceed only when no `STAGED`, `READY` or
`IN_FLIGHT` event, no `local_state='AWAITING_FINALIZE'` clip, and no
`publish_state='IN_FLIGHT'` clip remains. Inspect retained permanent or
compatibility outcomes before continuing.

## Step 3 — stopped-runtime cutover

Stop the runtime, run the migrator, then start backend and runtime in that order.
Compose already encodes the dependency: migrator completes, `ml-api` becomes
healthy, then `ml-worker` starts.

The migrator **refuses** with `EDGE_DB_DRAIN_INCOMPLETE` until the legacy drain
is complete. On refusal it leaves the database at 16 and byte-identical; return
to step 1 or step 2.

## Step 4 — verify

- `PRAGMA user_version` is 17.
- Every schema-16 application table and row survives. `tests/test_edge_db_schema17_migration.py`
  proves this with an independent pre-mutation manifest and a loss-detection
  negative test.
- The worker container has no `edge-state` mount and opens no SQLite. The
  boundary suite plus a `strace` run over `--check-config` showed 4066 open
  syscalls and zero SQLite opens.
- A fall detected during a backend outage still reaches the dashboard.
  `tests/test_acceptance_outage_fall_replay.py` covers this end to end.

## Rollback

Before the first v17 write, rollback is simply "do not migrate": the database is
untouched at 16.

After the first v17 write the migration is forward-only. Recovery is the
file-based schema-17 recovery receipt, not a downgrade. Do not attempt to hand-edit
`schema_table_families`.

## Open capacity finding

Two inputs, obtained differently on purpose.

**Incidence** is observation: 1143 live events over 34.5 hours across 13 cameras,
**2.55 events per camera-hour**. Nothing can synthesise this.

**Capacity** is measured through the running system, not calculated on paper.
`tests/test_outage_capacity_from_status_path.py` drives real entries into a real
`DeliveryQueue`, publishes through the worker's runtime-status sender, POSTs to
the relay route, stores, and then reads `GET /api/v1/status` — and derives the
budgets from the values **that endpoint returns**. An operator can reproduce the
same numbers from a running deployment by reading the same field.

As reported by `GET /api/v1/status` for eight falls:

| field | value |
| --- | --- |
| `accepted_count` | 24 |
| `accepted_bytes` | 137,520 |
| `max_accepted_entries` | 4,096 |
| `max_accepted_bytes` | 268,435,456 |
| `by_kind` | EVENT 8, SNAPSHOT_ATTACHMENT 8, SNAPSHOT_DISPOSITION 8 |

That is **3 entries and 17,190 bytes per fall**. Three, not two: a snapshot can
be admitted as an attachment and then followed by a disposition when its commit
fails, so a two-entry model would overstate the horizon.

Outage survival against the configured 4096 entries / 256 MiB:

| roster | survives | entries bound | bytes bound | binding limit |
| --- | --- | --- | --- | --- |
| 13 cameras | **41.2 h** | 41.2 h | 471.1 h | entries |
| 50 cameras | **10.7 h** | 10.7 h | 122.5 h | entries |

**The entry count binds at both rosters**, by a factor of more than ten, so
raising only the byte ceiling would buy nothing at all.

**The 72-hour target is not met at either roster**, including the current 13
cameras. Reaching it needs roughly **7,160 entries** at 13 cameras and **27,540**
at 50, against a configured 4,096.

The byte ceiling is not the problem and never was: at 13 cameras it would last
471 hours. If the horizon is to be extended, `MAX_ACCEPTED_ENTRIES` is the number
that matters, and its resource tests encode the current value deliberately.

One related observation, recorded because it contradicts the apparent design
intent: 256 MiB ÷ 4096 is exactly 64 KiB, which looks chosen so that a
maximum-size entry makes both bounds expire together. A maximum-envelope entry
actually serializes to 65,780 B, 244 B over that, because the JSON envelope keys
and identifiers were not counted. The practical effect is small — the byte
ceiling would refuse at about 4081 rather than 4096 max-size entries — but the
two bounds do not meet exactly.

Re-measure incidence after step 1, because delivery was entirely stalled by the
lease-backpressure defect during the observation window. `GET /api/v1/status`
now reports accepted entries, accepted bytes, the configured bounds, and the
EVENT/ATTACHMENT/DISPOSITION mix, so this no longer requires reading the
database by hand.

Whether to raise both bounds or accept a shorter horizon is deliberately left
open: the bounds were fixed by an approved plan, and their resource tests encode
them.

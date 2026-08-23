# Edge clip consistency repair

Use this runbook only for the backend-owned evidence relation repair shipped as
`scripts/ops/repair-clip-consistency.py`. It reconciles `clip_events` from
validated final READY/UNAVAILABLE manifests. It does not modify event payloads,
clip records, final manifests, or final media.

The repair runs against the backend database connection. Never run it in the
inference-runtime slot and never supply a database path: the runtime slot has
no database.

## 1. Prepare the backend maintenance command

Run from the deployed repository directory. The command is part of the backend
image at `/app/scripts/ops/repair-clip-consistency.py`; use the `ml-api` image,
not `ml-worker`. The backend has read-only access to the clip store, which is
sufficient because manifests are the authority.

```sh
cd /opt/eldercare-fall-ml
export EDGE_DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'
DC="$EDGE_DC"
```

The paths are:

```text
backend database: /var/lib/seeon-state/edge.sqlite3
clip store:       /var/lib/clip-store
maintenance root: /var/lib/seeon-state/clip-consistency-maintenance
```

Do not substitute the clip-store path for the maintenance root. Create the
maintenance root as a trusted directory owned by the operator account with mode
`0700` before proceeding.

## 2. Quiesce the inference runtime and record proof

Stop `ml-worker` so no event delivery can have an active lease. `--no-deps` on
the one-off command prevents Compose from starting services.

```sh
$DC stop ml-worker
test -z "$( $DC ps --status running -q ml-worker )"
```

Create `/var/lib/seeon-state/clip-consistency-maintenance/quiescence.json` on
the backend state volume with mode `0600`. Its JSON object must contain exactly
these values; `issued_at` and `expires_at` are Unix timestamps and their gap
must be no more than 3600 seconds.

```json
{
  "format_version": 1,
  "state_db": "/var/lib/seeon-state/edge.sqlite3",
  "clip_store": "/var/lib/clip-store",
  "stopped_service": "ml-worker",
  "stopped_db_writers": ["event", "config", "fault"],
  "operator_uid": 0,
  "issued_at": 0,
  "expires_at": 0
}
```

Use the UID of the account running the one-off backend command, not necessarily
`0`. The tool rejects a receipt outside the maintenance root, with unexpected
keys, unsafe ownership or permissions, expired timestamps, or mismatched paths.

## 3. Dry-run from the backend image

A plain invocation is a dry-run and does not mutate relations.

```sh
$DC run --rm --no-deps ml-api \
  python scripts/ops/repair-clip-consistency.py \
  --clip-store /var/lib/clip-store \
  --maintenance-root /var/lib/seeon-state/clip-consistency-maintenance \
  --quiescence-receipt /var/lib/seeon-state/clip-consistency-maintenance/quiescence.json
```

Record the JSON result. `state` must be `DRY_RUN`; investigate a refusal rather
than bypassing it. The command validates database integrity and foreign keys,
requires the evidence relation tables, rejects an active lease, and checks every
manifest and referenced database record.

## 4. Apply the reviewed plan

Only after the dry-run counters and manifest scope have been reviewed, repeat
with `--apply` while the quiescence receipt remains valid.

```sh
$DC run --rm --no-deps ml-api \
  python scripts/ops/repair-clip-consistency.py \
  --clip-store /var/lib/clip-store \
  --maintenance-root /var/lib/seeon-state/clip-consistency-maintenance \
  --quiescence-receipt /var/lib/seeon-state/clip-consistency-maintenance/quiescence.json \
  --apply
```

A successful apply returns `state: "DONE"` and writes a content-addressed
receipt under the maintenance root. Retrying the same plan returns that receipt;
do not delete it to force a second mutation.

## 5. Restore delivery and verify through the backend

Start the worker only after the command has completed and retain the dry-run,
apply result, and receipt with the incident record.

```sh
$DC up -d --wait ml-worker
$DC logs --since 3m ml-api
```

Verify incidents and clip availability in the dashboard or through the backend
API. Do not inspect or repair SQLite from the worker.

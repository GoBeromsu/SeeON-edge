# Happy Nursing Home SYSTEM_TEST activation

This runbook activates one privacy-safe facility-level `SYSTEM_TEST` from the
approved target **`happy nursing home`**. The configured OpenSSH alias is
`happy-nursing-home`: `ssh -G 'happy nursing home'` was checked during QA and
OpenSSH rejected the spaces, while the read-only command below confirmed the
configured alias without opening a connection.

The command does not start a worker, camera, model, inference loop, clip recorder,
or media upload. Its enable gate exists only in each one-shot process and is absent
from Compose by default.

> [!WARNING]
> Keep merge and live SSH blocked until the Hub capability, this Edge image, and a
> separately approved schema-8-compatible fix-forward image are retained. No live
> SSH, deployment, enrollment change, or Edge command was run while preparing this
> change.
>
> Never print or copy an `eft_v1` credential, dashboard password, camera address,
> RTSP URL, resident identifier, private host, account, or key path. ml-api uses the
> credential already enrolled in its connection-settings database. Operators must
> not read, export, replace, or pass that credential on a command line.

## 1. Pin the only remote target

Run this read-only local check. Stop unless it exits zero:

```sh
ssh -G happy-nursing-home >/dev/null
```

Open the approved target only by that configured alias:

```sh
ssh happy-nursing-home
```

All remaining commands run inside that SSH session. Do not substitute a host/user/key
triple or another alias.

## 2. Required read-only stop checks

Obtain these non-secret values from the approved release/change records. The
fix-forward image must be a retained immutable digest whose release explicitly
supports worker state schema 8 and upgrades schema 6 or 7 without queue loss.

```sh
EXPECTED_EDGE_INSTALLATION_ID='<approved non-secret installation UUID>'
EXPECTED_FACILITY_CODE='<approved non-secret facility code>'
APPROVED_EDGE_REVISION='<approved Edge git SHA>'
FIX_FORWARD_WORKER_IMAGE='<approved schema-8-compatible image@sha256 digest>'

cd /opt/eldercare-fall-ml
DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'
```

Every check in this section is read-only. Abort on the first mismatch; do not run an
operator action.

### Revision, retained fix-forward artifact, service, and disk

```sh
test "$(git rev-parse HEAD)" = "$APPROVED_EDGE_REVISION"
docker image inspect "$FIX_FORWARD_WORKER_IMAGE" >/dev/null
$DC ps ml-api ml-worker

test "$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$($DC ps -q ml-api)")" = healthy
test "$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$($DC ps -q ml-worker)")" = healthy

df -Pk /var/lib/docker
$DC exec -T ml-worker python -m worker --help | grep -- '--system-test'
```

Stop if the retained fix-forward digest is absent, either service is unhealthy, disk
headroom is insufficient under the site policy, the revision differs, or the CLI flag
is absent. Do not pull/rebuild/restart to repair a failed preflight.

### Safe installation/facility identity

Read only the non-secret identity fields and compare inside the container. The query
never selects the facility token.

```sh
$DC exec -T \
  -e EXPECTED_EDGE_INSTALLATION_ID="$EXPECTED_EDGE_INSTALLATION_ID" \
  -e EXPECTED_FACILITY_CODE="$EXPECTED_FACILITY_CODE" \
  ml-api python - <<'PY'
import os
from backend.app.features.connection.store import ConnectionSettingsStore

settings = ConnectionSettingsStore.from_env().load()
row = (
    settings.facility_code,
    settings.edge_installation_id,
    settings.enrollment_generation,
)
expected = (
    os.environ['EXPECTED_FACILITY_CODE'],
    os.environ['EXPECTED_EDGE_INSTALLATION_ID'],
)
if row[:2] != expected or not isinstance(row[2], int) or row[2] < 1:
    raise SystemExit('STOP: enrolled non-secret identity mismatch')
print({'facility_code': row[0], 'edge_installation_id': row[1],
       'enrollment_generation': row[2]})
PY
```

### Current worker state schema

This opens SQLite in read-only mode and does not invoke the worker migration path:

```sh
WORKER_SCHEMA="$($DC exec -T ml-worker python - <<'PY'
import sqlite3
from pathlib import Path

path = Path.home() / '.local' / 'state' / 'ml-worker' / 'worker-state.sqlite3'
with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
    print(db.execute('PRAGMA user_version').fetchone()[0])
PY
)"
case "$WORKER_SCHEMA" in
  6|7|8) printf 'worker_state_schema=%s\n' "$WORKER_SCHEMA" ;;
  *) echo 'STOP: unsupported worker state schema' >&2; false ;;
esac
```

Schema 6 is the last approved base. Schema 7 is the first PR's intermediate
operator-row layout; this image must fix-forward it to schema 8 while preserving all
rows. Schema 8 is current. Activation is prohibited in every case unless the retained
schema-8 fix-forward digest check above passed.

## 3. Hub precondition

An authorized Hub operator must create an active validation run with capability
`SYSTEM_TEST` for the exact installation/generation checked above. Record its
canonical `validationRunId`; it is a non-secret correlation ID. Keep the run active
through emit/retry/replay and close it immediately after verification.

## 4. One-shot command wrapper

The exact typed gate and confirmation are both required. The gate is scoped to the
child created by `exec`; it does not alter the running container or Compose
configuration.

```sh
run_system_test() {
  $DC exec -T \
    -e ML_WORKER_SYSTEM_TEST_GATE=SYSTEM_TEST_OPERATOR_ENABLED \
    ml-worker python -m worker "$@"
}
```

### Classify invalid authentication locally

This sends no event payload, does not open the worker outbox, and does not mutate the
enrolled credential. ml-api clones its backend client in memory with a known-invalid
diagnostic value and expects an actual backend 401 or 403.

```sh
run_system_test \
  --system-test auth-check \
  --confirm-system-test SYSTEM_TEST
```

Proceed only for `status=AUTH_CLASSIFIED` with `error_code=HTTP_401` or `HTTP_403`.
Any other result is a hard stop; do not change enrollment to make it pass.

## 5. First emit and forward-only boundary

`emit` is the first command here that opens the worker outbox. If the preflight read
schema 6 or 7, this open atomically migrates it to schema 8 before network delivery.

```sh
VALIDATION_RUN_ID='<canonical validation-run UUID>'
RESULT="$(run_system_test \
  --system-test emit \
  --system-test-validation-run-id "$VALIDATION_RUN_ID" \
  --confirm-system-test SYSTEM_TEST)"
printf '%s\n' "$RESULT"
EDGE_EVENT_ID="$(printf '%s' "$RESULT" | python -c \
  'import json,sys; print(json.load(sys.stdin)["edge_event_id"])')"
printf 'edge_event_id=%s\n' "$EDGE_EVENT_ID"
```

Expected first success is `ACKED`. If terminal output was lost, run the same `emit`
command with the same validation run ID: durable create-or-load returns
`PREVIOUSLY_ACKED` with the original `edge_event_id`/`backend_event_id` and performs
no network request. A due pending or failed row is retried through the same outbox row;
a duplicate invocation never creates a second event identity.

After this first outbox open, **do not deploy a schema-6 or schema-7 image**. Binary
rollback is not supported because those images fail closed with
`NewerSchemaVersionError` on the schema-8 volume.

## 6. Inspect and retry the exact durable ID

If the result is `RETRY_SCHEDULED`, inspect only safe delivery metadata:

```sh
$DC exec -T ml-worker python - "$EDGE_EVENT_ID" <<'PY'
import sqlite3
import sys
from pathlib import Path

path = Path.home() / '.local' / 'state' / 'ml-worker' / 'worker-state.sqlite3'
with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
    row = db.execute(
        '''SELECT delivery_state, attempt_count, next_attempt_at, last_error_code,
                  operator_only
           FROM evidence_events WHERE edge_event_id = ?''',
        (sys.argv[1],),
    ).fetchone()
print(row)
PY
```

Stop if the row is absent or `operator_only` is not `1`. Once `next_attempt_at` is due
and the underlying condition is healthy, retry only that ID:

```sh
run_system_test \
  --system-test retry \
  --system-test-edge-event-id "$EDGE_EVENT_ID" \
  --confirm-system-test SYSTEM_TEST
```

## 7. Explicit exact replay

After ACK, request one backend idempotency replay. Only this explicit action sends an
already-ACKed payload again:

```sh
run_system_test \
  --system-test replay \
  --system-test-edge-event-id "$EDGE_EVENT_ID" \
  --confirm-system-test SYSTEM_TEST
```

Expected status is `REPLAY_ACKED` with the original Edge and backend event IDs. Replay
does not reset or mutate terminal outbox state.

The Hub operator confirms exactly one facility-level SYSTEM_TEST event, the explicit
system-test label, the same replay identity, and no camera, room, resident, person,
snapshot, clip, evidence, or media data.

## 8. Deactivation and forward-only recovery

1. Close the Hub validation run immediately.
2. Exit the SSH session. No persistent Edge gate exists to disable.
3. Preserve the ACKed row and the entire worker-state volume.

If activation aborts **before the first schema-8 open** and the read-only schema check
still reports 6, follow the ordinary release process; no SYSTEM_TEST migration
occurred. If it reports 7, rollback is already prohibited and the retained fix-forward
path is mandatory. If any SYSTEM_TEST emit/retry/replay process opened the database,
recover only by deploying the retained `FIX_FORWARD_WORKER_IMAGE` (or a newer
approved schema-8-compatible digest) while preserving the volume and queue.

Never restore a schema-6/schema-7 binary over schema-8 state, delete or replace the volume,
run `down -v`, drop queue rows/tables, edit SQLite manually, rotate enrollment, or
edit Python/Compose on the Edge.

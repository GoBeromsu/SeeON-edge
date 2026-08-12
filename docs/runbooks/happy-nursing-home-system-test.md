# Happy Nursing Home SYSTEM_TEST activation

This runbook activates one privacy-safe facility-level `SYSTEM_TEST` from the Happy
Nursing Home Edge. It does not start a worker, camera, model, inference loop, clip
recorder, or media upload. The enable gate exists only in each one-shot `docker
compose exec` process and is disabled by default everywhere else.

> [!WARNING]
> Do not run this until both the Hub SYSTEM_TEST capability and the matching Edge
> image are deployed. This repository change does not authorize touching the live
> Edge, and no live command was run while preparing it.
>
> Never paste an `eft_v1` credential, dashboard password, camera address, RTSP URL,
> resident identifier, or SSH endpoint into this public repository, an issue, or an
> evidence log. The command uses the credential already enrolled in ml-api's
> connection-settings database; operators must not read, export, replace, or pass it
> on the command line.

## Preconditions

An authorized Hub operator must:

1. Confirm the Edge installation and enrollment generation for Happy Nursing Home.
2. Create an active validation run with capability `SYSTEM_TEST` and record its
   canonical `validationRunId`. The run ID is a correlation identifier, not a token.
3. Keep the run active through emit/retry/replay, then close it immediately after
   verification.

An Edge operator must confirm the deployed image contains this change, ml-api is
healthy, and the existing connection status is enrolled. Do not rotate or re-enroll
for this test.

Set SSH values only in the operator's local shell from the approved credential note:

```sh
export EDGE_HOST='<approved Happy Nursing Home SSH host>'
export EDGE_USER='<approved SSH account>'
export EDGE_KEY='<approved private-key path>'
ssh -i "$EDGE_KEY" "$EDGE_USER@$EDGE_HOST"
```

On the Edge host, use the deployed Compose definition without editing its env file:

```sh
cd /opt/eldercare-fall-ml
DC='docker compose --env-file .env.edge.prod -f compose.edge.yaml'
$DC ps ml-api ml-worker
$DC exec -T ml-worker python -m worker --help | grep -- '--system-test'
```

Stop if either service is unhealthy or the flag is absent. Do not pull, rebuild,
restart, or redeploy as part of this procedure.

## One-shot command wrapper

The exact typed gate and confirmation are both required. The gate below is scoped to
the child process created by `exec`; it does not alter the running container or its
Compose environment.

```sh
run_system_test() {
  $DC exec -T \
    -e ML_WORKER_SYSTEM_TEST_GATE=SYSTEM_TEST_OPERATOR_ENABLED \
    ml-worker python -m worker "$@"
}
```

## 1. Classify invalid authentication locally

This sends no event payload and does not mutate the enrolled credential. ml-api
clones its backend client in memory with a known-invalid diagnostic value and expects
an actual backend 401 or 403.

```sh
run_system_test \
  --system-test auth-check \
  --confirm-system-test SYSTEM_TEST
```

Proceed only when stdout is JSON with `status` equal to `AUTH_CLASSIFIED` and
`error_code` equal to `HTTP_401` or `HTTP_403`. Any other status means stop; do not
change enrollment to make the diagnostic pass.

## 2. Emit one event

Copy the active validation run ID supplied by the Hub operator:

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

Expected success is `status=ACKED` with the same value in `edge_event_id` and
`correlation_id`, plus a non-empty `backend_event_id`. The payload contains only the
SYSTEM_TEST sentinels, timestamp, validation run ID, generated event ID, and a safe
non-resident label on the local hop. ml-api forwards only the Hub's five accepted
wire fields. Facility scope comes from the enrolled `eft_v1` principal; it is never
client-declared.

## 3. Retry only the same durable ID

If emit returns `RETRY_SCHEDULED`, do not emit another event. Inspect the existing
row's due time without opening cameras or changing state:

```sh
$DC exec -T ml-worker python - "$EDGE_EVENT_ID" <<'PY'
import sqlite3
import sys
from pathlib import Path

path = Path.home() / ".local" / "state" / "ml-worker" / "worker-state.sqlite3"
with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
    row = db.execute(
        """SELECT delivery_state, attempt_count, next_attempt_at, last_error_code,
                  operator_only
           FROM evidence_events WHERE edge_event_id = ?""",
        (sys.argv[1],),
    ).fetchone()
print(row)
PY
```

Stop if the row is missing or `operator_only` is not `1`. After
`next_attempt_at` is due and the underlying network/backend condition is healthy,
retry that exact ID:

```sh
run_system_test \
  --system-test retry \
  --system-test-edge-event-id "$EDGE_EVENT_ID" \
  --confirm-system-test SYSTEM_TEST
```

Retries use the production outbox lease, attempt counter, backoff, receipt, and ACK
transitions. Operator-only rows are excluded from the periodic sender.

## 4. Verify exact-ID replay

After ACK, verify backend idempotency once with the same immutable event ID and
payload:

```sh
run_system_test \
  --system-test replay \
  --system-test-edge-event-id "$EDGE_EVENT_ID" \
  --confirm-system-test SYSTEM_TEST
```

Expected status is `REPLAY_ACKED`, with the same `edge_event_id` and
`backend_event_id` as the original ACK. Replay does not reset or modify the terminal
outbox row.

The authorized Hub operator then reads the validation run's event list and confirms:

- exactly one facility-level SYSTEM_TEST event for `EDGE_EVENT_ID`;
- it is visibly labeled as a system test, not a resident alert;
- there is no camera, room, resident, person, snapshot, clip, evidence, or media data;
- replay resolved to the original backend event identity.

## Deactivation and rollback

1. Close the Hub validation run immediately; this removes server-side SYSTEM_TEST
   capability from the already-enrolled credential.
2. Exit the SSH session and unset local SSH variables.
3. Do not delete the ACKed outbox row; it is the durable audit/correlation record.

There is no persistent Edge gate to turn off: each `exec` process exits after one
command, and the normal worker container never receives
`ML_WORKER_SYSTEM_TEST_GATE`. If code rollback is required, redeploy the previously
approved image through the normal deployment process. Do not edit Python on the
Edge, change Compose, remove volumes, rotate enrollment, or run `down -v`.

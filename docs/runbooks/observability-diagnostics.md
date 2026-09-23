# Execution-record diagnostics (original-run observability)

Answer "why was there no alert at 14:03 on camera 7?" from what actually ran,
not from re-analysing the clip. Issue #545; schema 19.

This runbook has no numbers in it on purpose. Retention budget, lane depth,
batch size and flush interval come from the measurement harness
(`tests/AGENTS.md`, "Observability load (Gate M/V)"), never from a code default.
Both sides refuse to boot when enabled without their values.

## What is recorded

One immutable record per producer observation, keyed by
`(camera_id, worker_boot_id, source_generation, stream_epoch, producer,
producer_sequence)`. Producers never wait on each other or on the Backend.

| kind | who writes it | what it says | what it does NOT say |
|---|---|---|---|
| `sdk.frame` | Flow metadata slot, after an accepted publish | SDK frame/source ids, whether an inference tensor was present, raw/eligible/matched row counts | that the policy consumed the frame |
| `policy.consume` | policy pump, per processed frame | slot counter delta (accepted/overwritten/late) | that a decision was made |
| `model.score` | policy pump, only when the classifier actually scored on this call | raw logit, applied temperature, class origins, calibrated score, track, window facts | a native three-class output (`fallen` is a synthetic zero) |
| `policy.decision` | policy pump, per snapshot | reason, previous/current state, triggered, values, missing reasons (`classifier-warmup`, `classifier-stride-not-due`, ...), `decision_trace_id` | that the event was delivered |
| `event.delivery` | Flow evidence binding (admission) and the evidence sender (every attempt outcome) | admitted/refused; retry-transient / retry-counted / refused-retained / exhausted / ack-removal-deferred, with attempt counts and failure class | that the Hub accepted anything |
| `backend.acceptance` | evidence sender, when a relay receipt is observed | `accepted_local` (persisted on the edge Backend) **or** `hub-accepted` (Hub receipt), never both | delivery to a phone or pager |

`accepted_local` is terminal local persistence. It is **not** Hub acceptance
and the alert is never forwarded later.

### Joining a decision to its delivery

- `policy.decision.payload.decision_trace_id` equals the relayed alert's
  `audit.decision_trace_id`. One function computes both
  (`worker.types.trace.decision_trace_id`).
- `event.delivery` and `backend.acceptance` share `causal_unit_id ==
  edge_event_id`. The decision → delivery hop is correlated by frame identity
  (`camera_id, worker_boot_id, stream_epoch, frame_seq`) plus that id; there
  is no foreign key, and the runbook does not claim one.
- Records written by the durable-queue drainer are **process-scoped**: they
  stamp the boot that observed the outcome and `PROCESS_SCOPE` (0) for
  generation/epoch, because the queue outlives the boot that staged the event.
  That is declared vocabulary, not a missing value.

## Querying

```
GET /api/v1/diagnostics/executions?camera_id=<id>&from_ns=<ns>&to_ns=<ns>&limit=<1..500>&cursor=<opaque>
```

Dashboard session required. The response carries `records`, `units` (each with
`causal_state` and `terminal`), `coverage` rows, `availability` ranges,
`queryable_range` and `next_cursor`.

Read `availability` first. It is the only honest answer to "what can this
query see":

| word | meaning | what to do |
|---|---|---|
| `AVAILABLE` | retained records exist for this range | read them |
| `MISSING_NOT_RECORDED` | the worker itself reported a drop (`cause`: `lane-overflow`, `export-failed`, `record-invalid`) or the Backend rejected it (`oversize`) | the run happened; the record did not survive the worker → Backend hop. Check lane depth and export health |
| `DELETED_BY_CAPACITY` | an exact tombstone or a proven contiguous prefix says these sequences were pruned | evidence existed and was removed by the retention budget; raise the budget if this range matters |
| `UNKNOWN_COARSENED` | summary rows were merged past the per-epoch bound; the per-range deleted-vs-unknown distinction is lost | do not infer deletion from this |
| `UNKNOWN` | before the earliest evidence, after the last observation, or a crash tail with no record either way | nothing can be concluded. Do **not** read this as "no person" or "no fall" |

Absence of evidence is never rendered as a negative observation.

### Reading a non-event

1. Find `policy.decision` rows for the track around the time in question.
   `reason` and `missing_values` say why nothing fired: `below-threshold`,
   `outside-detection-window`, `classifier-warmup`, `classifier-stride-not-due`,
   `score-missing`, ...
2. If there is a `model.score` for that frame, compare `raw_logit` /
   `applied_temperature` / calibrated `fall_transition` against
   `transition_threshold` in the decision values.
3. If there is no `model.score` for a frame where you expected one, the
   `policy.decision.missing_values` row says whether the classifier skipped it
   (warmup / stride) or the score was missing for another reason.
4. If a decision *did* trigger, follow `causal_unit_id == edge_event_id`
   into `event.delivery` (was it admitted? how many attempts? retained for an
   operator?) and `backend.acceptance` (`accepted_local` vs `hub-accepted`).
5. If the range is `MISSING_NOT_RECORDED` / `DELETED_BY_CAPACITY` / `UNKNOWN`,
   stop and say so. The system cannot tell you more.

## Retention

Total-capacity, not per-kind TTL. Records belong to a logical causal unit
(one fall decision and its inputs; one event and its delivery/acceptance).
Units become terminal by horizon, by a newer boot/epoch, or when forced by
capacity. Pruning removes a whole unit atomically and writes an exact
`DELETED_BY_CAPACITY` row; it never leaves a child with a deleted parent.
The budget envelope includes coverage/receipt rows and SQLite WAL headroom,
not just payload bytes.

## Restart and rollback

- Backend restart: records committed before the restart are queryable after
  it. That is the D0 guarantee, and the whole guarantee.
- Worker restart / crash / power loss: records still in the in-memory lanes
  are lost. The worker reports the drop as a gap **if** it gets to flush; a
  hard kill leaves an `UNKNOWN` tail. There is no persistent worker spool.
- Schema rollback: see `edge-database-schema-19.md`. A schema-18 image refuses
  a schema-19 database; restoring the backup discards every application write
  made after the extension.

## Enabling (seams)

Both sides default OFF and refuse to boot when enabled without their values.

| side | key | when enabled |
|---|---|---|
| ml-api | `ML_API_EXECUTION_RECORDS_ENABLED` | `ML_API_EXECUTION_RECORDS_BUDGET_BYTES` required; `ML_API_BUILD_REVISION` required |
| ml-worker | `ML_WORKER_EXECUTION_RECORDS_ENABLED` | `ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY`, `_BATCH_MAX`, `_FLUSH_MS` required; relay URL + token required; every provenance identity (build revision, image digest, model/calibration digests, preprocessing identity, config digest, policy id) must resolve — a missing one refuses to start by name |

Keys are inventoried in `edge-env-inventory.json` and passed through
`compose.edge.yaml`. Disabled: no store on the API, both endpoints answer 503,
the worker composes no lanes and the hot path reads nothing extra.

## Measuring before enabling

`uv run pytest -m real_stack -k observability` on the DeepStream host with
`OBS_STREAM_PATH` set writes `obs-<N>.json` (offered fps, records/s, gap
rows/s, lane high-water, backlog slope, exporter batch p50/p95, CPU delta).
Those numbers, and only those, become the four keys above. A backlog slope
that is not ≤ 0 over the second half of the run means the export path cannot
keep up at that load; do not enable at that load.

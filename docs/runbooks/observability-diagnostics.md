# Execution-record diagnostics (original-run observability)

Answer "why was there no alert at 14:03 on camera 7?" from what actually ran,
not from re-analysing the clip. Issue #545; schema 19.

This runbook carries no *deployment* numbers on purpose (the API contract
facts it mentions - the query `limit` range and `PROCESS_SCOPE` - are wire
constants, not budgets). The four operator knobs - retention budget, lane
depth, batch size and flush interval - come from the measurement harness
(`tests/AGENTS.md`, "Observability load (Gate M/V)"), never from a code
default, and both sides refuse to boot when enabled without them.

Seven pre-measurement *shape* constants do ship in
`backend/app/features/diagnostics/retention.py` and are design values, not
measured budgets; Gate M replaces them: `unit_horizon_ns` (unit terminal
horizon), `coverage_rows_per_epoch` (coarsening bound), and the derived
fractions of `total_bytes` - `control_reserve` (1/16), `high_water`
(total minus reserve), `low_water` (7/8 of high-water), `segment_bytes`
(1/64), `max_record_bytes` (1/256). The three per-row control-envelope
estimates are gone; the budget is measured on disk with `dbstat` (execution_*
tables and their indexes). Until Gate M they bound behaviour; they are not
deployment numbers.

## What is recorded

One immutable record per producer observation, keyed by
`(camera_id, worker_boot_id, source_generation, stream_epoch, producer,
producer_sequence)`. Producers never wait on each other or on the Backend.

| kind | who writes it | what it says | what it does NOT say |
|---|---|---|---|
| `sdk.frame` | Flow metadata slot, after an accepted publish | SDK frame/source ids, whether an inference tensor was present, raw/eligible/matched row counts | that the policy consumed the frame |
| `policy.consume` | policy pump, per processed frame | slot counter delta (accepted/overwritten/late) | that a decision was made |
| `model.score` | policy pump, only when the classifier actually scored on this call | raw logit, applied temperature, class origins, calibrated score, track, window facts | a native three-class output (`fallen` is a synthetic zero) |
| `policy.decision` | policy pump, one per snapshot from **every** decider | `module_qualified_id` (`fall.v2`, `bed_exit.v1`, or `null` when composition gave that decider no policy), `authority_role` (`authoritative` or `shadow` - a shadow evaluation is never a cause), reason (incl. the suppression reasons `episode-already-open`, `episode-reassociated`, `episode-resolved-hold`, `episode-candidate`, and `outside-detection-window`), previous/current state, triggered, values, missing reasons (`classifier-warmup`, `classifier-stride-not-due`, `resample-gap`, ...), `decision_trace_id` computed with that module's own identity. A frame the module did not evaluate (resampler yielded no row) is one row with `outcome: coasted` and `missing_values.decision_state: resample-gap` - never the previous frame's rows re-stamped | that the event was delivered; that a `shadow` row influenced anything; that a `coasted` row saw a person |
| `event.delivery` | Flow evidence binding (admission) and the evidence sender (every non-success attempt outcome) | `admitted` / `refused`; then `retry-transient`, `retry-counted`, `refused-retained`, `refused-retention-full`, `exhausted-retained`, `exhausted-retention-full`, `ack-removal-deferred`, with attempt counts and failure class. A successful delivery adds no *sender* `event.delivery` row - the admission row from the Flow binding is the only `event.delivery` for that event, and `backend.acceptance` records the result | that the Hub accepted anything |
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
  For `backend.acceptance` the wire contract enforces this
  (`PROCESS_SCOPED_KINDS`); for the sender's `event.delivery` rows it is a
  documented convention, because the same kind is also produced stream-scoped
  by the Flow binding at admission. Either way it is not a missing value.

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

Filter on `module_qualified_id` first. A fall question reads `fall.v2` rows; a
bed-exit question reads `bed_exit.v1` rows. Ignore `authority_role == shadow`
rows when asking *why* something did or did not fire - they are evaluations
that never trigger. A row with `module_qualified_id: null` came from a decider
that was composed without an effective policy; it has no `decision_trace_id`.

1. Find `policy.decision` rows for the module and track around the time in question.
   `reason` and `missing_values` say why nothing fired: `below-threshold`,
   `outside-detection-window`, `classifier-warmup`, `classifier-stride-not-due`,
   `score-missing`, ...
2. If there is a `model.score` for that frame, compare `raw_logit` /
   `applied_temperature` / calibrated `fall_transition` against
   `transition_threshold` in the decision values.
3. If there is no `model.score` for a frame where you expected one, the
   `policy.decision.missing_values` row says whether the classifier skipped it
   (warmup / stride), the module coasted on that frame (`coasted` /
   `resample-gap`), or the score was missing for another reason.
   A `triggered: false` row whose reason starts with `episode-` means the
   onset fired but the episode authority declined it (already open,
   re-associated, on hold, or still a candidate) - it is a suppressed onset,
   not a missed one. `outside-detection-window` means the same for the
   detection window.
4. If a decision *did* trigger, follow `causal_unit_id == edge_event_id`
   into `event.delivery` (was it admitted? how many attempts? retained for an
   operator?) and `backend.acceptance` (`accepted_local` vs `hub-accepted`).
5. If the range is `MISSING_NOT_RECORDED` / `DELETED_BY_CAPACITY` / `UNKNOWN`,
   stop and say so. The system cannot tell you more.

## Retention

Total-capacity, not per-kind TTL. `ML_API_EXECUTION_RECORDS_BUDGET_BYTES` is
the byte envelope of the logical live rows of the six `execution_*` tables:
the summed `pg_column_size` of every visible row. It excludes indexes, page
overhead, dead tuples and WAL, so the physical size of the
`seeon_edge_diagnostics` schema is larger than the budget; never size the
disk by the budget alone.

Records belong to a logical causal unit (one fall decision and its inputs;
one event and its delivery/acceptance). Units become terminal by horizon, by
a newer boot/epoch, or when forced by capacity. Pruning removes whole units
atomically, oldest-terminal first, and writes an exact `DELETED_BY_CAPACITY`
row; it never leaves a child with a deleted parent.

How the limit is enforced: `high_water` (total minus a 1/16 control reserve)
is the prune trigger; each ingest prunes at most a bounded number of whole
units toward `low_water` (7/8 of high-water) so a backlog drains across
requests instead of one request blocking. Usage is measured at most once per
second; between measurements the store accrues what it wrote, scaled by the
ratio it measured. That estimate only decides *when* to measure - pruning
happens only on an exact measurement. Usage can therefore sit above
`high_water` by at most one second of writes, which is what the control
reserve absorbs; the total envelope is the hard limit. Measured on the
reference edge with 8 cameras: a 2 GB backlog drained at ~2 MB/s with
`/health` unaffected.

## Restart and rollback

- Backend restart: records committed before the restart are queryable after
  it. That is the D0 guarantee, and the whole guarantee.
- Worker restart / crash / power loss: records still in the in-memory lanes
  are lost. The worker reports the drop as a gap **if** it gets to flush; a
  hard kill leaves an `UNKNOWN` tail. There is no persistent worker spool.
- Rollback: see `postgresql-cutover.md`. It says when the diagnostics schema
  is kept, dropped, or denies a rollback.

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

`uv run pytest -m real_stack -k observability_real_stack` on the DeepStream host with
`OBS_STREAM_PATH` set writes `obs-<N>.json` (offered fps, records/s, gap
rows/s, lane high-water, backlog slope, exporter batch p50/p95, CPU delta).
Those numbers, and only those, become the four keys above. A backlog slope
that is not ≤ 0 over the second half of the run means the export path cannot
keep up at that load; do not enable at that load.

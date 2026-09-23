# worker/pipeline/diagnostics

Own the Worker-side execution-record buffer and export drain. Producers only
read existing values and call `try_emit`. The Backend SQLite store is out of
scope.

## Ownership

- `lanes.py`: per-(camera, producer) bounded deques. Overflow is counted and
  reported as `WireGap` cause `lane-overflow`. A contract failure after the
  sequence was consumed is a one-item `WireGap` cause `record-invalid`.
  `try_emit` is a short-lock append-or-drop and never raises.
- `exporter.py`: runtime-owned drain thread. Batches per (camera, boot) up to
  configured N records or T ms. Export failure drops the batch and reports
  `WireGap` cause `export-failed` on the next successful batch. No Worker DB
  and no persistent spool (D0).
- `provenance.py`: build `WireProvenance` from identities the composition root
  already resolved. Missing identities refuse to start by name; never stamp
  `unknown`.
- `record_builder.py`: shared `WireRecord` construction and `try_emit`.
- `emit_policy.py`: sdk.frame, policy.consume, model.score, policy.decision
  payload builders. They never change control flow.
- `emit_delivery.py`: event.delivery and backend.acceptance payload builders.
  They never change control flow.

`producer_sequence` is assigned by the lane. `causal_unit_id` for sdk records
uses `seq // 30` as a pre-Gate-R frame bucket; no `cpu.projection` producer
exists yet. Fall units use explicit `NO_TRACK` / `NO_GENERATION` tokens when
track or generation is absent.

Seam default is `None` (feature off). No stub sink.

Focused tests: `tests/test_execution_record_lanes.py`,
`tests/test_execution_record_exporter.py`. Boundary:
`uv run --group lint lint-imports`.

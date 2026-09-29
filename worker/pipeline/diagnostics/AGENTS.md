# worker/pipeline/diagnostics

Own the Worker-side execution-record buffer and export drain. Producers only
read existing values and call `try_emit`. The backend PostgreSQL store is out
of scope.

## Ownership

- `lanes.py`: per-(camera, producer) bounded deques. Overflow is counted and
  reported as `WireGap` cause `lane-overflow`. A contract failure after the
  sequence was consumed is a one-item `WireGap` cause `record-invalid`.
  `try_emit` is a short-lock append-or-drop and never raises. Encoding and
  HTTP stay outside the lane lock. `account_unsendable_records` turns drained
  records that cannot be exported into one `record-invalid` gap each,
  preserving neighbor order and existing gap counts. New gaps carry the
  dropped record's generation/epoch; runs never bridge scope or sequence holes.
  Legacy gaps without scope retain their wire hash but are UNKNOWN at ingest,
  never assigned a surviving neighbour's scope. This supports old Worker to
  new API, not scoped new Worker to old API.
- `exporter.py`: runtime-owned drain thread. One drain takes up to configured
  N records or waits T ms, then splits that drain's records and gaps into
  batches whose UTF-8 bodies — provenance and envelope included — are at most
  `MAX_EXECUTION_RECORD_BODY_BYTES` (1MiB). Camera, boot, and producer
  sequence stay in order; each gap `record_count` is preserved. A record that
  cannot fit alone becomes one `record-invalid` gap plus an operator-visible
  log line (camera id and the reason are in the message). A failed chunk is
  `export-failed` for that attempted chunk only; the same camera/boot drain
  stops there. Never-attempted records return to the fronts of their original
  lanes without new identities or sequences; unattempted gaps are restored
  unchanged. Restoration preserves older work before concurrent arrivals and
  reports any bounded-capacity tail eviction as `lane-overflow`. Flushes are
  serialized without holding the producer lock across encoding or HTTP.
  Already-known gaps must commit before later watermark-advancing records;
  independent cameras can still progress. Delayed loss can lower Backend
  terminal certainty without reopening the unit.
  `STORAGE_UNAVAILABLE` is a failed delivery, not a committed receipt.
  If the provenance/gap envelope itself cannot fit, the exporter logs
  that and retains its loss accounting; the drain backs off interruptibly
  rather than spinning or silently deleting gaps. Receipt and failure histories keep at most
  `EXPORT_HISTORY_LIMIT` (256) newest entries. Transport and encoding
  exceptions are logged and do not kill the drain thread. No Worker DB and
  no persistent spool (D0).
- `provenance.py`: build `WireProvenance` from identities the composition root
  already resolved. Missing identities refuse to start by name; never stamp
  `unknown`.
- `record_builder.py`: shared `WireRecord` construction and `try_emit`.
- `emit_policy.py`: sdk.frame, policy.consume, model.score, policy.decision
  payload builders. They never change control flow.
- `emit_delivery.py`: event.delivery and backend.acceptance payload builders.
  They never change control flow.

## event.delivery outcomes

Two producers share the kind and join on `causal_unit_id == edge_event_id`:

- **Admission** (`FlowEvidenceBinding`): stream-scoped. Stamps the triggering
  frame (`worker_boot_id`, `source_generation`, `stream_epoch`, `frame_seq`).
  Outcome is `admitted` or `refused` from the durable queue's `try_admit`
  result (or `stage()` raising). Never emit `admitted` without that proof.
- **Sender dispositions** (`EvidenceSender.run_once`): process-scoped by
  convention, like `backend.acceptance`. Stamps the observing boot and
  `PROCESS_SCOPE` for generation/epoch. The wire `PROCESS_SCOPED_KINDS` set
  still lists only `backend.acceptance`; stream-scoped admission records of
  the same kind must keep the real frame identity. Closed vocabulary in
  `DELIVERY_ATTEMPT_OUTCOMES`:
  `retry-transient` (RETRY / 5xx / unreachable; attempt budget not consumed),
  `retry-counted` (attempt consumed: send exception, PERMANENT non-4xx,
  receipt `edge_event_id` mismatch),
  `refused-retained` (PERMANENT 4xx kept in the dead-letter directory),
  `refused-retention-full` (PERMANENT 4xx, retention area full, still queued),
  `exhausted-retained`, `exhausted-retention-full`,
  `ack-removal-deferred` (delivered but `queue.acknowledge` failed).

These sender outcomes are the sender's own dispositions, not Hub acceptance.
`backend.acceptance` is the only record that says `accepted_local` /
`hub-accepted`.

Focused tests also include `tests/test_execution_record_delivery.py`,
`tests/test_evidence_sender.py`, `tests/test_worker_flow_evidence_binding.py`,
`tests/test_observability_end_to_end.py`.

`producer_sequence` is assigned by the lane. `causal_unit_id` for sdk records
uses `seq // FALL_WINDOW_FRAMES` (imported from `worker.domains.fall.classifier`, the deployed 30-frame window) as a pre-Gate-R frame bucket; `cpu.projection`, `handoff.slot`, and `coverage.gap` do not exist. Fall units use explicit `NO_TRACK` / `NO_GENERATION` tokens when
track or generation is absent.

Seam default is `None` (feature off). No stub sink.

Focused tests: `tests/test_execution_record_lanes.py`,
`tests/test_execution_record_exporter.py`,
`tests/test_execution_record_gap_integration.py`,
`tests/test_execution_records_wire.py`. Boundary:
`uv run --group lint lint-imports`.

## Attribution (every policy.decision names its producer)

`EventAggregator.attributed_trace_snapshots()` tags each snapshot with the
producing decider's `DecisionIdentity` (from composition, per module) and an
`authority_role` (`authoritative`, or `shadow` for the trailing
`last_shadow_trace_count` snapshots of a `ShadowTraceProvider` such as the
bed-exit monitor). `policy.decision` writes `module_qualified_id` and
`authority_role` into the payload, computes `decision_trace_id` with that
module's identity only, and picks the causal unit by module: the fall
track/generation unit for `fall.v2`, a module-scoped frame unit otherwise
(`NO_MODULE` when unattributed). `model.score` is emitted only for the fall
decider's own authoritative snapshots. The alert audit's `decision_trace_id`
is computed with the identity of the decider that produced that event. No
snapshot ever borrows another module's identity or unit.

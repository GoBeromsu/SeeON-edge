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
  `operator-blocked` (`CAMERA_MAPPING_MISSING`; `run_once` applies that wait
  only to CLIP entries, so EVENT never takes this outcome today),
  `ack-removal-deferred` (delivered but `queue.acknowledge` failed).

These sender outcomes are the sender's own dispositions, not Hub acceptance.
`backend.acceptance` is the only record that says `accepted_local` /
`hub-accepted`.

Focused tests also include `tests/test_execution_record_delivery.py`,
`tests/test_evidence_sender.py`, `tests/test_worker_flow_evidence_binding.py`,
`tests/test_observability_end_to_end.py`.

`producer_sequence` is assigned by the lane. `causal_unit_id` for sdk records
uses `seq // FALL_WINDOW_FRAMES` (imported from `worker.domains.fall.classifier`, the deployed 30-frame window) as a pre-Gate-R frame bucket; no `cpu.projection` producer
exists yet. Fall units use explicit `NO_TRACK` / `NO_GENERATION` tokens when
track or generation is absent.

Seam default is `None` (feature off). No stub sink.

Focused tests: `tests/test_execution_record_lanes.py`,
`tests/test_execution_record_exporter.py`. Boundary:
`uv run --group lint lint-imports`.

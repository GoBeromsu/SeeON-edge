# worker/pipeline: decision and output coordination

Pipeline coordinates post-Flow policy and output. DeepStream Flow owns capture,
decode, inference, and tracking; pipeline code does not recreate those stages.
`runtime` composes it; `domains` owns business decisions; adapters own vendor
objects.

## Ownership

- `decision/`: `IncidentManager` (cooldown, admission, persisted identity, and
enrichment) and `EventAggregator`.
- `output/`: event publication, evidence handoff, live observation, and
snapshot coordination. Clip, snapshot, and relay side effects run after
admission, here only.
- `analytics/`: post-Flow observation handling that feeds domain decisions.
- `perception/`: worker-owned conversion and feature helpers; do not duplicate
SDK tracking or inference.
- `trace/`: bounded diagnostic traces.

## Evidence and ownership

`AlertEvidenceAttacher` adds audit and a bounded JPEG after admission.
Attachment failure must not block the alert. `EvidenceEventSink.emit_for_frame`
stages the event with its trigger frame; legacy event-only `emit` is rejected.
The smart record actor owns primary clips. Decoded frames are analysis and
snapshot taps, never a replacement clip path.

A queue owns every accepted item until take, eviction, or close. After `take()`,
the taker owns it and must release it. Keep admission, staging, delivery, and
shutdown paths balanced.

## Focused Tests

- `tests/test_worker_incident_manager.py`, `tests/test_pipeline_bootstrap.py`

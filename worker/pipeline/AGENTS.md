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
admission, here only. See `output/AGENTS.md` for the HTTP surface.
- `analytics/`: post-Flow observation handling that feeds domain decisions.
  `provision_extractors`; `merge.result_merger_names` is also read by the
  domain registry.
- `perception/`: worker-owned conversion and feature helpers; do not duplicate
SDK tracking or inference. Pure numeric: import-linter contract "worker
  perception features stay pure" bans adapters, decision, output, domains,
  runtime, and `shared.events`. Numpy is allowed in `features/`; results leave
  as plain Python scalars.
- `trace/`: replay-trace models and the bounded JSONL writer used by
  policy pumps; decision-trace ids are stamped in runtime flow, not here.
- `diagnostics/`: bounded in-memory execution-record lanes and export drain.
- `inference_telemetry.py`: pose geometry and physical-batch histograms.
- `decision/event_identity.py`: `EventIdentityStore`, the persisted event
  identity behind admission.

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
- `tests/test_event_identity_bounds.py`, `tests/test_features_window.py`,
  `tests/test_perception_observation_builder.py`, `tests/test_pts_resample.py`,
  `tests/test_bed_pose_features.py`

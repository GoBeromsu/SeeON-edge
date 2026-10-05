# EVENTS KNOWLEDGE BASE

Backend↔worker wire: outbound event shape, Event API client, evidence HTTP, relay failure logs.

## Ownership

- `schemas.py`: `EmittedEvent`, `build_emitted_event`, `build_audit_envelope`. Re-exports `AlertEventType` and `EventApiPayload`.
- `edge_ingest_client.py`: no-HMAC Event API client for alerts and heartbeats. Shared by the API relay and the worker.
- `evidence_export_client.py`: `RelayEvidenceClient` (worker to ml-api) and `BackendEvidenceClient` (ml-api to Hub).
- `evidence_export_contract.py`: `DeliveryFailure`, receipts, capabilities, `RETRY` / `PERMANENT` / `COMPATIBILITY`.
- `evidence_http_transport.py`: bounded HTTP, receipt parse, status classification.
- `relay_failure_log.py`: rate-limited, classified relay failure reporter.
- `execution_records.py`: Worker -> Backend diagnostics wire contract.
- `execution_records_client.py`: relay HTTP client posting `WireBatch`.
- `delivery_queue.py`: file-backed publish-once `DeliveryQueue` (fcntl lock, fsync-ordered moves, bounded admission, dead-letter retention). Used by worker evidence staging and `scripts/ops/review-refused-evidence.py`.
- `clip_analysis_wire.py`: `ClipAnalysisResult` encode/decode, `MAX_CLIP_ANALYSIS_FRAMES`.
- `replay_wire.py`: `decode_replay_trace` and derived replay body bounds.
- `envelope_limits.py`: named `*_MAX_*` limits; worst-case envelope size is computed from them.
- `clip_identity.py`: `is_clip_id`.

Ids are derived, never caller-supplied: `EventEntry.entry_id`, `WireRecord.record_id`, `WireBatch.batch_id` are stamped in `__post_init__`. Size caps are products of named maxima, not hand-picked numbers.

## Schemas

ML emits typed events. Backend owns severity, channel, policy, and final dedup. Incident management owns only idempotency and cooldown.
`EmittedEvent` carries facility, camera, domain, event_type, lifecycle, severity, front_event_type, and evidence. Empty `event_type` raises.
`build_audit_envelope` stamps `clock_source=edge_wall_clock` plus optional model, detector, and threshold. Don't invent Hub fields here.

## EdgeIngestClient and one-way egress

Worker to backend command/event traffic is one-way relay HTTP. This package is the egress client, not a command bus back into the worker.
Post alerts to the single `API_BACKEND_EVENTS_URL`. Heartbeats go to that URL plus `/heartbeat`. JSON bodies only. No HMAC headers.
Optional Bearer token. Blank token omits `Authorization`. `for_camera` clones the client; it does not open a camera.
`send_alert` and `send_heartbeat` return bool. `send_heartbeat_result` classifies `auth`, `timeout`, or `unreachable`. `send_alert_receipt` is the idempotent `edge_event_id` path.
`emit` and `publish` send only `fall` and `bed-exit`. `detection-lost` is dropped with no POST and no failure count.
A successful alert may PUT a JPEG snapshot to `{id}/snapshot`. Snapshot failure prints to stderr and does not fail the alert.

## Failure visibility

`failure_count` on the ingest client is lock-protected. Heartbeat and alert POST failures increment it.
`classify_http_failure` treats 401/403 as `RETRY`, not dead-letter. 404/405 are `COMPATIBILITY`. Payload 4xx stay `PERMANENT`.
`RelayFailureLog` is one instance per logical channel. First failure and class change log full detail. Repeats fold into a 60s summary. Recovery logs once.
Never log response bodies, request headers, or tokens. Status, transport class, static hint, and path only. Client errors log at ERROR; transport and 5xx at WARNING.

## Delivery queue

`acknowledge_backend` accepts only `{200, 201, 202, 204, 409}`. A refused (422) entry is `dead_letter`ed, never deleted and never reported as acknowledged; deleting refused entries once lost real bed-exit events.
Retention refuses when full; it does not evict. `requeue_dead_lettered` goes through the queue lock, never direct file writes.

## Imports

Allowed: `contracts` and local `shared.events`.
Forbidden: `backend`, `worker`, camera I/O, model load, RTSP, decode, serving.
`contracts` and `worker.pipeline.perception` must not import this package. Import-linter owns that.

## Forbidden ownership

This package doesn't own cameras, runtimes, detectors, or clip stores.
Don't open RTSP, construct `WorkerRuntime`, seed a camera registry, or write `runtime_*` / `evidence_*` tables.
Publishers and outboxes stay network-free except `EdgeIngestClient` and the evidence HTTP clients.
Camera id is a string field. A live session belongs in `Flow media plane`. Durable evidence staging belongs in `worker/pipeline/output/evidence`.

## Focused tests

```bash
uv run pytest -q tests/test_events_schema.py tests/test_dead_shared_surfaces.py tests/test_events_ingest_client.py tests/test_evidence_export_client.py tests/test_evidence_http_transport.py
uv run pytest -q tests/test_delivery_queue.py tests/test_clip_analysis_wire.py tests/test_execution_records_wire.py tests/test_replay_transfer_bound.py tests/test_relay_failure_log.py
uv run --group lint lint-imports
```

Schema tests lock `EmittedEvent` fields and the Event API payload. Ingest tests hit a local HTTP server: no HMAC, optional Bearer, omitted `clip_id`, failure count, dropped `detection-lost`, and receipt `on_accepted`. `test_dead_shared_surfaces.py` keeps the retired `outbox.py` / `local_publisher.py` and their publisher names out of this package.

# worker/pipeline

Staged data path: ingest, bus, perception/analytics, decision, output.
Stages hold per-camera state and call ports. Process lifecycle stays in runtime.

## Ownership rule

`worker.pipeline` must not import `worker.runtime`. Runtime composes and injects.
`perception/` is stricter: pure numeric math. Forbidden: `backend`,
`shared.events`, `worker.adapters`, `worker.pipeline.{bus,ingest,decision,output}`,
`worker.domains`, `worker.runtime`. Feature math needs no camera, model, or filesystem.

## Five-stage ownership

- `ingest/`: registry, descriptor checks, capture loop, reconnect, pacing, RTSP credential masking. `DecodeAdapter` port only. Open or iterate failure is camera `DEGRADED` plus `camera.offline`. Later processing failure is not offline.
- `bus/`: named bounded subscriptions, `scheduler.py`, publish/take/drop counters, queue age.
- `perception/` + `analytics/`: observation builder, greedy tracker, `SceneState`, window buffer, `DecisionInput`, `features/`. `CompositeExtractor` owns one camera's tracker/scene and reuses shared named extractors. Scheduler picks due modules; every packet still coasts or advances tracker/scene.
- `decision/`: `IncidentManager` (cooldown, admission, persisted identity, enrichment) and `EventAggregator`.
- `output/`: `EventSink`, evidence (encoder, finalizer, manifests, outbox, retention, reconciliation), overlay, live view, snapshot store. Clip, snapshot, and relay side effects run after admission, here only.
- `camera_pipeline.py`: per-camera wiring. No business math.
- `inference_coordinator.py`: shared pose owner between `bus.inference` and the pump. Not a sixth stage.

## Bounded bus, coordinator, leases

No unbounded queue. A full subscription drops by its documented policy and increments its drop counter once. `inference` and `live`: latest-only, capacity 1; new publish evicts the queued packet and releases its lease. `evidence`: FIFO, default 128; a full queue rejects the incoming packet and releases it. Close drains every queued lease. `BoundedFrameBus.publish` precharges one child lease per subscription, then releases the publisher handle. Packets stay immutable on the bus. Copy the image before draw or mutate.

Who takes what: coordinator drains `bus.inference` at timeout 0; `LiveViewPump` drains `bus.live` on its own thread; evidence/clip feeder drains `bus.evidence`; `CameraPipelinePump` takes `InferenceResultSlot`, never the inference bus.

`CapabilityInferenceCoordinator` owns every pose forward. It drains one latest frame per ready camera, batches up to 16, and publishes `CoordinatedInference` into a capacity-1 `InferenceResultSlot` per camera. The slot owns the queued packet lease. Overwrite or close releases the replaced packet. A failed forward releases every selected packet before re-raise. `stop()` drains leftover inference subscriptions and closes every result slot. After that handoff, `CameraPipelinePump` is the sole owner of per-camera analytics, tracker, scene, and domain mutation. Pose is never forwarded from a per-camera loop.

A queue owns every accepted packet until take, eviction, or close. After `take()`, the taker owns the lease and must release it. Keep these paths balanced: publish fanout, latest-only eviction, FIFO reject, subscription close, coordinator forward failure, `stop()` drain, result-slot overwrite/close, `CameraPipelinePump.run` `finally`, `LiveViewPump` after each live frame, evidence clip admission drops. Don't publish a released packet or take and forget.

## Pixel / numeric, per-camera, output

Pixels stop at model extract, derivative evidence, overlay/live view, and the alert snapshot. Decision sees `DecisionInput` only: observation, frame size, live track ids, time, frame index, bed region. No array, no buffer, no frame handle. Need a pixel-derived number? Extract it in `perception/` first.

Pipeline-owned per camera: tracker, `SceneState`, window buffer, scheduler, `IncidentManager`, bus slot, encoder ring, ingest backoff, result slot, live observation cache row. Shared once: model objects, named extractors, serving client, coordinator. Don't hoist a per-camera row to save memory.

`AlertEvidenceAttacher` adds audit and a bounded JPEG after admission. Attachment failure must not block the alert. `EvidenceEventSink.emit_for_frame` stages the event with its trigger packet. Legacy event-only `emit` is rejected. `LiveViewPump` is a tap, not a stage. It draws the latest cached observation against `bus.live`. Stale pose is dropped, not drawn as current. Zero viewers means no JPEG encode. Primary clips remux source packets from camera-local rings. Decoded frames are analysis and snapshot taps.

## Focused Tests

- `tests/test_worker_frame_bus.py`, `tests/test_frame_lease.py`, `tests/test_capability_inference_coordinator.py`
- `tests/test_worker_camera_pipeline_pump.py`, `tests/test_perception_observation_builder.py`
- `tests/test_worker_incident_manager.py`, `tests/test_live_view_pump.py`
- `tests/test_pipeline_bootstrap.py`, `tests/test_import_dependency_ladder.py`
- Boundary: `uv run --group lint lint-imports`

Keep new pure-code modules at or below 250 logical LOC. Split by stage, not into a new monolith. Preserve lease balance and per-camera isolation whenever a consumer or handoff changes shape.

# worker/pipeline/output/evidence

Durable clip, snapshot, and filesystem delivery path after admission. Pixels stop here as remuxed source packets and JPEG snapshots. Decision never writes this tree.

`worker.pipeline` must not import `worker.runtime`. Runtime injects the store dir, delivery queue, lock, and sender. Missing production wiring or a locked store refuses to start. Remote failures do not block startup: retryable classes retry, compatibility failures reprobe, and payload-invalid failures become permanent. Two workers must not share one delivery queue.

## Packet FIFO and feed

`bus.evidence` is FIFO, default 128. A full subscription rejects the incoming packet and releases it. `ClipFrameFeeder` is the only drain: take, `ClipRecorder.on_frame`, then `packet.release()`. Admission retains before enqueue. A full recorder queue (also 128) drops the frame or event, releases the retained handle, and increments the drop counter.

`PacketRingRepository` owns one `SourcePacketRing` per camera under a process-wide byte ceiling. Primary clips remux those source packets. Decoded frames are analysis and snapshot taps only. Under `nvidia`, `NativeAuReceiver` feeds this repository from the native pre-decode encoded-AU tee; no decode, OSD, or encoder enters the primary path. `select()` holds a `PacketSelection` lease; `close()` must pair it. Epoch roll retires leased packets and drops the rest. Global pressure evicts the oldest unleased packet from the fattest ring. Don't remux across a discontinuity, a closed ring, or a stale stream epoch. Don't close the repository from a routine encoder flush. Rings live until ingest stops.

## Clip recording

Shared `ClipRecorder`, one actor thread, camera-local rings. `ClipAdmission` allocates a collision-free `ClipId` and a `.staging` dir. `PacketClipRecordingCoordinator` window-selects around the trigger PTS, stream-copies to `clip.mp4`, then the publisher atomically replaces into `clips/<clip_id>/`. Unavailable is a published outcome, not a silent skip. Thumbnail miss logs and still publishes. `ClipStoreLock` takes a non-blocking flock on `.worker.lock` for the recorder lifetime.

The `nvidia` runtime owns the policy/attach handoff; this package receives its
`NativeEvidenceTrigger` and binds source identity to the packet-ring selection.
See `worker/runtime/deepstream/AGENTS.md`.

## Durable staging and delivery queue

`EvidenceEventSink.emit_for_frame` stages the alert first, then binds an optional clip. Legacy event-only `emit` is rejected. `DurableEvidenceStager` writes a canonical payload into the publish-once filesystem `DeliveryQueue`. Same `edge_event_id` must replay identical bytes.

## Runtime and sender

`EvidenceExportRuntime.initialize_under_lock` marks the queue ready, then the sender may start. `EvidenceSender` is event-first. Clips send only when the live export switch and backend capability both allow it.

## Snapshots and manifests

`SnapshotStore` is two-phase: stage under `.snapshot-staging`, publish bytes after the event commit, then commit identity metadata. Crash leaves a named transition. Ready manifests pin sha256, size, duration, codec, and event refs. Verify those facts on recovery. Snapshot capacity drops are sink backpressure, not alert failure.

`SceneRing` retains bounded, image-free scene records per camera for accepted evidence windows. Clip publication may write `scene-index.json`: canonical compact JSON in source-pixel coordinates, bound by a content-addressed manifest claim. It is metadata beside the immutable source clip, never burned pixels or a derivative video.

## Retention and deletion

Hold-aware. Unpublished or incomplete incidents stay HELD. Missing hold hook treats every clip as held. `EvidenceRetention` verifies the manifest and containment, writes PENDING, deletes the dir, then PURGED. Fail closed on verify, lock, or leftover path. Operator delete and age/pressure rotate share that purge. Floor is 60 days. Disk high watermark defaults to 0.80. Stale staging sweep skips held ids. Snapshot retention uses the same tombstone-before-unlink rule.

## Leases and who owns what

A queue owns every accepted `FramePacket` until take, reject, or close. After take, the feeder owns it through `on_frame` and must release. Admission's retain is a second handle the actor releases. Packet-ring selections are a third lease family. Process-wide: delivery queue, clip-store lock, sender owner id, snapshot store. Per camera: packet ring, feeder thread, clip reservation. Don't hoist a camera ring or reservation.

## SQLite vs files

The worker keeps only the publish-once queue, media files, integrity sidecars,
zero-payload locks, and startup-purged scratch. The backend owns delivery
metadata, claims, retention intent, and incident lifecycle. Do not add a
database or database connection to worker evidence. Files own immutable bytes:
`clips/<id>/clip.mp4`, `thumbnail.jpg`, `manifest.json`, and staged snapshots.
Media never lives in rows. Path is not identity; hash and size are.

## Focused Tests

- `tests/test_packet_ring.py`, `tests/test_packet_remux_clip.py`, `tests/test_worker_clip_frame_feeder.py`
- `tests/test_worker_clip_admission.py`, `tests/test_worker_clip_actor.py`, `tests/test_worker_clip_recorder.py`
- `tests/test_worker_clip_publication.py`, `tests/test_worker_clip_recording.py`, `tests/test_clip_store_lock.py`
- `tests/test_evidence_stager.py`, `tests/test_evidence_sender.py`, `tests/test_evidence_delivery_queue_restart.py`
- `tests/test_retired_worker_dead_surfaces.py`, `tests/test_evidence_retention.py`, `tests/test_worker_clip_maintenance.py`
- `tests/test_snapshot_store.py`
- `tests/test_scene_ring.py`, `tests/test_scene_index_sidecar.py`, `tests/test_scene_index_claim.py`
- Boundary: `uv run --group lint lint-imports`

Keep new pure-code modules at or below 250 logical LOC. Preserve lease balance, hold-before-delete, and the SQLite-vs-bytes split whenever a handoff changes shape.

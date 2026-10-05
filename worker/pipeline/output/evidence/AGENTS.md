# worker/pipeline/output/evidence

Own the evidence path after alert admission: smart record actor, clip
publication, sealed sidecar, durable stager, delivery queue, and snapshot store.
The Flow media plane supplies approved evidence inputs; this package does not
own capture, decode, inference, tracking, or vendor SDK objects.

## Clip recording

The smart record actor owns primary clip creation. Clip publication binds the
artifact to its event and sealed sidecar. Decoded frames are analysis and
snapshot taps only. A trigger failure must not block the alert; an incomplete
artifact is recorded through the durable path rather than silently discarded.

## Durable staging and delivery queue

The stager writes durable event/clip work before relay delivery. The delivery
queue is publish-once and owned by one worker process. Retry classes retry,
compatibility failures reprobe, and payload-invalid failures are permanent.
Never use the backend database or a JSON state store for queue state.

## Snapshot store

Snapshots are bounded content-addressed evidence. Keep their lifetime,
retention, and sidecar identity aligned with the published clip. Snapshot work
must not delay alert admission or relay delivery.

## Playback renditions

`clip.mp4` is immutable primary evidence. `playback_rendition.py` may asynchronously create a
view-only H.264 rendition and digest sidecar beside a sealed clip; this work never blocks manifest
publication or relay delivery, and renditions are never relay or export inputs.

## Where to look

| Task | File |
| --- | --- |
| Smart Record extension policy | `smart_record_actor.py`: one serialized `SmartRecordActor` per camera |
| Flow sealed clips | `flow_clip_publication.py`, `flow_sealed_sidecar.py` (crash recovery records) |
| Re-encoded clip publication | `clip_publication.py` (crash-resumable `ClipPublisher`), `clip_publication_types.py` |
| Clip ids and roots | `clip_identity.py`: `ClipIdAllocator`, `bounded_clip_roots` |
| Terminal outcome | `terminal_outcome.py`: exactly one terminal clip outcome per event identity |
| Manifests (schema v2) | `evidence_manifest.py`, `manifest_models.py`, `manifest_media_models.py` |
| Media inspection | `evidence_media.py`: strict MP4 inspection from one immutable fd |
| Staging and delivery | `evidence_stager.py`, `evidence_sender.py` (570 LOC), `evidence_runtime.py`, `evidence_outbox_types.py` |
| Snapshots | `snapshot_store.py` (615 LOC), `snapshot_files.py` (descriptor-relative atomic ops) |
| Store ownership | `clip_store_lock.py`: lifetime advisory lock, `ClipStoreLockedError` |
| Clip re-analysis artifacts | `clip_analysis_artifact.py`: immutable and identity-bound |
| Env, retention, fsync | `clip_config.py`, `durability.py` |

Package `__init__` exports nothing; import from the concrete module.

Focused tests: `tests/test_worker_clip_publication.py`,
`tests/test_flow_clip_publication.py`, `tests/test_flow_sealed_recovery.py`,
`tests/test_evidence_sender.py`, `tests/test_clip_receipt_delivery.py`,
`tests/test_playback_rendition.py`, `tests/test_evidence_trust_boundaries.py`.

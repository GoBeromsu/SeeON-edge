# worker/types: internal envelopes

Own the worker's frozen pipeline vocabulary. Bottom of the import ladder.
Every higher layer imports this package. None of it runs a camera, a model, or a socket.

## Ownership rule

`worker.types` imports only the standard library and `contracts`: no I/O,
model, or framework import, not even as a convenience. import-linter contract
"worker.types imports only contracts from the internal graph" bans `backend`,
`shared`, and every higher `worker` layer. Host lease checks inspect ndarray
shape; that is not a license for cv2 or torch. Device handles stay `object`.

## Local Ownership

- `frame_packet.py`: `FramePacket` is the only envelope allowed to carry an
  image. Identity is `FrameKey(worker_boot_id, camera_id, stream_epoch, seq,
  pts, source_pts?, source_time_base?)`. Storage is a `FrameLease`. `_frame`
  and `lease` are compare=False, hash=False, repr=False. `retain()` returns a
  new packet with a retained lease. `release()` drops this handle.
- `frame_memory.py`: `FrameLease` is one independently releasable handle over
  refcounted host or device storage. `retain()` and `precharge()` fan out; last
  release recycles. A second release or borrow after recycle raises `FrameLeaseReleasedError`.
- `module_result.py`: `ModuleResult(module_name, result, elapsed_ms, output_adapter?)`
  wraps `contracts.runner.RunnerResult`. `module_name` is component identity, not merger routing.
- `decision_input.py`: `DecisionInput` keeps the original seven fields
  (`observation`, `frame_width`, `frame_height`, `live_track_ids`, `time_sec`,
  `frame_index`, `bed_region`) plus additive `bed_pose_features` (default empty
  `FrameBedPoseFeatures`). Numeric only: no image, buffer, or frame handle.
- `temporal_profile.py`: `TemporalProfile` owns ingest fps, pose fps, and
  per-domain decision Hz. `CURRENT_TEMPORAL_PROFILE` is 30fps ingest, pose
  pinned at 15fps (fall conformance checks that exact value), bed every 180
  frames (1/6 Hz, 6.0s). Raising `_CURRENT_INGEST_FPS` must re-denominate
  `_CURRENT_BED_INTERVAL_FRAMES` in the same edit: Hz is the invariant. A relay
  `CameraRuntimeConfig.fps` is a recorded hint, never the owner.
- `business_event.py`: `BusinessEvent(domain, event_type, identity, camera_id,
  facility_id, time_sec, probability, person_id?, bed_id?, audit?,
  snapshot_jpeg?)`. Domains emit; pipeline admits, records, and relays.
- `perception_frame.py`: image-free `PerceptionFrameV1` is the worker-internal
  native contract. `PerceptionFrameIdentity` carries boot, camera, stream epoch,
  sequence, and optional source PTS. Person-box, human-pose, and bed-region
  channels independently use `ChannelState.INFERRED`, `INFERRED_EMPTY`, or
  `SKIPPED`; optional `AssociationResult` binds tracks to person cues.
- `evidence_trigger.py`: image-free `NativeEvidenceTrigger` binds a native decision to
  camera, boot, stream epoch, source generation, sequence, source PTS, and source time.
- `metadata.py`, `trace.py`, `source_packet.py`: media-plane `MetadataFrame` /
  `SourceBinding` / `MetadataCounters`; `DecisionTraceSnapshot`,
  `DecisionIdentity`, `decision_trace_id`; `SourcePacket`, `StreamEpoch`.
- Small envelopes: `bed_pose_features.py`, `capabilities.py`, `copy_metrics.py`,
  `preview.py` (`OverlaySelection`, `FallPreviewState`), `fall_model_input.py`.

## Conventions

`@dataclass(frozen=True, slots=True)`, `from __future__ import annotations`,
explicit `__all__`. Prefer a new envelope over widening an existing one.

`contracts.frame.Frame`, `contracts.runner.RunnerResult`,
`contracts.observation.FrameObservation`, `BedRegionDebugSnapshot`, and
`contracts.event.EventPayload` stay authoritative. No class named `DetectionResult`.

`Frame.image` is a mutable NumPy array, so hashes skip payload fields; keep that
when adding a field. Publish packets immutable; copy the image before draw or mutate.

Pixel sinks are model extract, overlay/MJPEG, and alert snapshot only. Domains
take `DecisionInput` and return `BusinessEvent` tuples. A detector that needs
pixels is a design error; extract the number in `pipeline/perception` first.
`FallModelInput` is nested float tuples, never an ndarray.

## Forbidden runtime behavior

No decode, encode, inference, file open, or HTTP here. Mutating a published
packet is a bug. `host_frame` is illegal on a device lease. Recycle lives in the
lease callback. Live frames do not belong on `DecisionInput` or `BusinessEvent`;
`snapshot_jpeg` is optional bytes, not a frame handle. An envelope that needs
I/O or a model type belongs in `worker/interfaces` or `worker/adapters`.

Focused tests: `tests/test_worker_types.py`, `tests/test_frame_lease.py`,
`tests/test_perception_frame_v1.py`, `tests/test_import_dependency_ladder.py`.
Boundary: `uv run --group lint lint-imports`.

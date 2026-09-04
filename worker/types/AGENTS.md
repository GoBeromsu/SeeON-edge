# worker/types: internal envelopes

Own the worker's frozen pipeline vocabulary. Bottom of the import ladder.
Every higher layer imports this package. None of it runs a camera, a model, or a socket.

## Ownership rule

`worker.types` imports only the standard library and `contracts`. No I/O,
model, framework, or higher-worker-layer import, not even as a convenience.
import-linter contract "worker.types imports only contracts from the internal
graph" forbids `backend`, `shared`, `worker.interfaces`, `worker.adapters`,
`worker.pipeline`, `worker.domains`, and `worker.runtime`. Host lease checks
already inspect ndarray shape. That is not a license for cv2 or torch. Device
handles stay `object`.

## Local Ownership

- `frame_packet.py`: `FramePacket` is the only envelope allowed to carry an
  image. Identity is `FrameKey(worker_boot_id, camera_id, stream_epoch, seq,
  pts, source_pts?, source_time_base?)`. Storage is a `FrameLease`. `_frame`
  and `lease` are compare=False, hash=False, repr=False. `retain()` returns a
  new packet with a retained lease. `release()` drops this handle.
- `frame_memory.py`: `FrameLease` is one independently releasable handle over
  refcounted host or device storage. `retain()` and `precharge()` fan out.
  Last release recycles. A second release or a borrow after recycle raises
  `FrameLeaseReleasedError`.
- `module_result.py`: `ModuleResult(module_name, result, elapsed_ms,
  output_adapter?)` wraps `contracts.runner.RunnerResult`. `module_name` is
  component identity, not merger routing.
- `decision_input.py`: `DecisionInput` fields start with the original seven:
  `observation`, `frame_width`, `frame_height`, `live_track_ids`, `time_sec`,
  `frame_index`, `bed_region`. `bed_pose_features` is additive with a default
  empty `FrameBedPoseFeatures`. Numeric only. Never an image, buffer, or frame
  handle.
- `temporal_profile.py`: `TemporalProfile` owns ingest fps, pose fps, and
  per-domain decision Hz. Pipeline defaults and the composition-root schedule
  compute from it. `CURRENT_TEMPORAL_PROFILE` is today's 15fps identity
  (bed every 90 frames, i.e. 1/6 Hz and 6.0s wall clock). Raising
  `_CURRENT_INGEST_FPS` must re-denominate `_CURRENT_BED_INTERVAL_FRAMES` in
  the same edit: the Hz is the invariant, the frame count is derived.
- `business_event.py`: `BusinessEvent(domain, event_type, identity,
  camera_id, facility_id, time_sec, probability, person_id?, bed_id?, audit?,
  snapshot_jpeg?)`. Domains emit these. Pipeline admits, records, and relays
  them.
- `perception_frame.py`: image-free `PerceptionFrameV1` is the worker-internal
  native contract. `PerceptionFrameIdentity` carries boot, camera, stream epoch,
  sequence, and optional source PTS. Person-box, human-pose, and bed-region
  channels independently use `ChannelState.INFERRED`, `INFERRED_EMPTY`, or
  `SKIPPED`; optional `AssociationResult` binds tracks to person cues.
- `evidence_trigger.py`: `NativeEvidenceTrigger` binds a native decision to
  camera, boot, stream epoch, source generation, sequence, source PTS, and
  source time without carrying or retaining an image.

## Conventions

`@dataclass(frozen=True, slots=True)`, `from __future__ import annotations`,
explicit `__all__`. Prefer a new envelope over widening an existing one.

`contracts.frame.Frame`, `contracts.runner.RunnerResult`,
`contracts.observation.FrameObservation`, `BedRegionDebugSnapshot`, and
`contracts.event.EventPayload` stay authoritative. Don't duplicate or shadow a
vendored type, and don't add a class named `DetectionResult`.

`Frame.image` is a mutable NumPy array, so hashes skip payload fields. Keep
that when you add a field. Publish packets immutable. Copy the image before
draw or mutate.

Four sinks may see pixels: model extract, overlay/MJPEG, and
alert snapshot. Domains take `DecisionInput` and return `BusinessEvent`
tuples. A detector that needs pixels is a design error. Extract the number in
`pipeline/perception` first.

`FallModelInput` is nested float tuples, never an ndarray.

## Focused Tests

- `tests/test_worker_types.py`
- `tests/test_frame_lease.py`
- `tests/test_perception_frame_v1.py`
- `tests/test_import_dependency_ladder.py`
- Boundary: `uv run --group lint lint-imports`

## Forbidden runtime behavior

This package does not decode, encode, infer, open files, or talk HTTP.
Mutating a published packet is a bug. `host_frame` is illegal on a device
lease. Recycle lives in the lease callback, not here. A second release is an
error. Live frames do not belong on `DecisionInput` or `BusinessEvent`.
`snapshot_jpeg` is optional bytes, not a frame handle. An envelope that needs
I/O or a model type belongs in `worker/interfaces` (port) or
`worker/adapters` (impl), not here.

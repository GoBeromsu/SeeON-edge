# worker/adapters — concrete implementations

Own the concrete implementations behind the `worker/interfaces` ports: video
decode, model inference, and clip encoding. This is where OpenCV, FFmpeg, CUDA,
Ultralytics, sklearn, and torch are allowed to appear.

## Ownership rule

**`worker.adapters` must not import `worker.pipeline`, `worker.domains`, or
`worker.runtime`.** An adapter is constructed and injected by the composition
root; it never reaches back up to a stage, a domain, or the runtime that owns it.
If an adapter needs a scheduling decision, camera identity policy, or config
resolution, take it as a constructor argument instead of importing the owner.

Enforced by import-linter contracts *"worker adapters do not depend on pipeline,
domains, or runtime"* and *"worker runtime is the sole composition root"*.

## Local Ownership

- `decode/cpu_av/`: explicit OpenCV/CPU RTSP decode. Selected only by the `cpu`
  and `mps` profiles; never a CUDA fallback.
- `decode/nvdec_cuvid/`: fail-loud NVDEC/CUVID decode — FFprobe metadata, CUVID
  codec selection, bounded read queue, child-process reaping.
- `model/`: `ModelRegistry`, the YOLO pose/person/bed-seg adapters, the sklearn
  and torch LSTM fall adapters, real warmup, and `in_process.py`
  (`InProcessServingClient`).
- `encode/`: `FFmpegSegmentEncoder` — one long-lived segment-muxer process per
  camera, with start/recreate/failure counters.

## Conventions

- Both decode adapters emit the same `FramePacket` contract and the same decode
  metrics: measured non-negative `decode_time_ms`, monotonically increasing `seq`,
  correct width/height, and stream time.
- The `cpu` profile is available only when OpenCV's own `videoio_registry`
  reports `CAP_FFMPEG`. Import success alone is not capability: a `cv2` build
  without the FFMPEG video I/O backend fails `decode_capability` preflight
  instead of failing every camera open later. A missing registry, a missing
  `CAP_FFMPEG` constant, or a raising query are all reported as unavailable
  with an explicit reason. There is no alternate-backend fallback (ADR-0003).
- The profile selects the decoder and encoder. An adapter never probes its way to
  a different backend, and NVENC failure never starts `libx264`.
- Model objects are provisioned through the serving client, one per task per
  process. Preserve object identity across cameras.
- Warmup is required, not optional: one real forward on a correctly shaped
  synthetic input, with CUDA synchronize only on CUDA.
- `close()` is idempotent and reaps every child process. Errors are typed and
  sanitized — never echo a credentialed RTSP URL.
- Do not ship placeholder or `NotImplementedError` adapters. An unused decoder
  skeleton is deleted, not exposed as a registry option.

## Focused Tests

- `tests/test_worker_interfaces.py` (port conformance)
- `tests/test_profile_boot.py` (profile-to-backend policy)
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Change Boundary

A new backend is a new adapter package behind the existing port, registered by
`worker/runtime` — not a branch inside an existing adapter and not a new
`if profile == ...` in a pipeline stage.

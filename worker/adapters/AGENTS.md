# worker/adapters: concrete port implementations

Own replaceable vendor-backed implementations behind `worker/interfaces`.
Source/live-view helpers and runtime provenance remain in their owning scopes.

## Ownership rule

`worker.adapters` must not import `worker.pipeline`, `worker.domains`, or
`worker.runtime`. Runtime constructs and injects. Need schedule, identity,
or config? Constructor argument. Need a higher-layer type? Local Protocol
(`FallModelConfigLike`). Stay structurally compatible.

Allowed: stdlib, `contracts`, `worker.types`, `worker.interfaces`.
Forbidden: `backend`, `worker.pipeline`, `worker.domains`, `worker.runtime`.
Enforced by "worker adapters do not depend on pipeline, domains, or
runtime" and "worker runtime is the sole composition root".

## Local ownership

- `decode/`: concrete CPU, VAAPI, NVDEC/CUVID, PyAV, and experimental device-resident backends. Backend-specific ownership, packet, epoch, and cleanup rules live in `decode/AGENTS.md`.
- `encode/nvenc_device/`: experimental device-input encoder. Only `nvidia-device-experimental`; production profiles never construct it.
- `model/`: `ModelRegistry` (pose/person/bed), fall-family registry, YOLO and LSTM runners, required warmup, `InProcessServingClient`.
- `encode/`: `FFmpegSegmentEncoder` (one long-lived muxer per camera), concat finalizer, thumbnail.
- `device/`: honest probes. `cuda/` / `mps/` answer whether this process can build `device="cuda"` / `device="mps"`. `nvml/` fills GPU telemetry. Import success is not capability.
- `frame/`: `HostFrameMaterializer`. `view` is zero-copy host-only. `materialize` copies and counts.

## Interface implementation

Implement the port. Don't invent a parallel API.

- `DecodeAdapter.open(config) -> DecodeSession`. Same `FramePacket` metrics on every decoder: non-negative `decode_time_ms`, monotonic `seq`, real width/height, stream time.
- `ClipEncoder.open(camera, profile, geometry)`. Profile arrives resolved. Don't re-parse `ML_WORKER_PROFILE`.
- `ServingClient.create(task, ...)`. One runner per task/options per process. Preserve object identity across cameras.
- `FrameMaterializer` / `HostFrameView`. Non-host `view` fails. No silent host<->device transfer.
- `DeviceResidentPool` / `DeviceResidentBatcher` stay experimental.

A new backend is a new package behind the existing port, registered by `worker/runtime`. Not a branch inside an adapter. Not `if profile` in a pipeline stage. Delete unused skeletons. Don't leak FFmpeg, cv2, or torch through the port signature.

## Explicit backend selection

Runtime profile owns `(device, decode, encode)`. Adapters execute that choice. They do not probe onto another backend.

Fail closed: OpenCV needs `videoio_registry` `CAP_FFMPEG` (ADR-0003, no `VideoCapture` retry). NVDEC preflight fails loud, no silent OpenCV path. Unknown fall `type` raises `UnknownFallModelTypeError`. Unknown task raises. `"fall"` is not in `DEFAULT_REGISTRY`.

Loud named exceptions only: VAAPI may demote to OpenCV at boot (`resolve_decode_or_fallback`). NVENC may demote to `libx264` once per camera (WARNING + `EncodeSelection`). Same H.264 content, cost only.

A probe answers this process. `available=True` does not mean the camera is reachable or the artifact loads. Warmup and ingest own those.

## Resource ownership

`close()` is idempotent and reaps every child. Encoder sessions are per-camera. Model objects are shared once per task per process. Device pools refuse past capacity. Decoded frames are analysis and snapshot taps. Primary clips remux source packets. Sanitize errors. Never echo a credentialed RTSP URL. Warmup: one real forward on a correctly shaped synthetic input. CUDA synchronize only on CUDA.

## Focused tests

`tests/test_worker_interfaces.py`, `tests/test_profile_boot.py`, `tests/test_worker_decode_cpu.py`, `tests/test_worker_nvdec_adapter.py`, `tests/test_worker_vaapi_adapter.py`, `tests/test_worker_opencv_decode_probe.py`, `tests/test_worker_segment_encoder.py`, `tests/test_worker_model_serving.py`, `tests/test_worker_yolo_adapters.py`, `tests/test_worker_fall_adapters.py`, `tests/test_fall_model_family_registry.py`, `tests/test_worker_real_warmup_no_stub.py`, `tests/test_worker_cuda_device_probe.py`, `tests/test_worker_mps_device_probe.py`, `tests/test_worker_nvml_device_probe.py`, `tests/test_frame_lease.py`. Boundary: `uv run --group lint lint-imports`.

Default tests stay hardware-free. Assert probe invariants, not "this machine has no GPU".

# worker/runtime: composition root

Own process lifecycle and wiring: config, profile, staged bootstrap, GPU
containment, camera supervision, telemetry, CLI. Business math and model parse
stay out. `worker.runtime` is the only package that may import everything
(`types`, `interfaces`, `adapters`, `pipeline`, `domains`, `contracts`,
`shared.events`). Nothing imports it. Import-linter: "worker runtime is the
sole composition root". Runtime constructs, injects, and owns lifecycle.
Perception math, domain judgment, and vendor kits live elsewhere. `backend`
is forbidden. Relay HTTP only.

## Ownership rule

- `worker.py`: `WorkerRuntime`. Shared component graph once, then per-camera
  ingest, tracker, `SceneState`, scheduler, domain deciders, bus, pump. Wires
  evidence, MJPEG, coordinator, watchdog, shutdown. `TemporalProfile` is the
  fps contract; `_preflight_camera_graph` passes it into registry activation
  so `Scheduler(dict(resolved_plan.schedule))` is computed, not hardcoded.
- `bootstrap.py`: `gpu_lease -> profile/device -> decode_capability ->
  model_backend_init -> real_warmup -> camera_activation`. Mutable
  `BootstrapContext` owns the lease.
- `model_composition.py`: `SharedComponentPool` by artifact identity.
- `ingest_composition.py`: profile decode token plus optional camera override. Unknown token or `nvdec` without an nvdec profile raises.
- `profile/`: `ML_WORKER_PROFILE` to `(device, decode, encode)`. Unset or blank becomes `cpu`. Unknown names fail closed.
- `config/`: models, resolver, relay pull, LKG, restart epoch. YAML is a developer hatch. No env roster.
- `provenance/`: applied runtime manifest in `edge.sqlite3`.
- `telemetry/`: `StatusStore`, `WorkerDiagnostics`, `RuntimeStatusSender`. Local snapshot can grow. Relay wire stays frozen.
- `faults/`: one first-fault record, stop every loop, `os._exit(4)`.
- `lease.py`: advisory flock at `~/.local/state/ml-worker/.gpu.lease`. Acquire before CUDA, NVDEC, or model construction. No env override.
- `watchdog.py`: hung forward drives the same `FaultHandler` path.

## Failures, allocation, CLI, tests

Global stages are fatal. Failed named stage starts zero cameras and exits
non-zero. Lease, profile/device, and decode miss exit 3. Ordinary stage
failure exits 1. `FatalAcceleratorError` exits 4 and outranks the stage code.
Lease is released before exit. `run_camera_stage` degrades only that camera.
`FatalAcceleratorError` still re-raises: GPU context is process-wide. RTSP
fail retries with backoff. Decode stall reopens the source. Source open
failure is camera `DEGRADED` plus `camera.offline`. Later processing failure
is not offline. Missing evidence wiring or a locked local store refuses to
start. Remote delivery is classified: retry classes retry, compatibility
failures reprobe, and payload-invalid failures become permanent. Two workers
must not share one outbox. NVENC may demote to `libx264` during preflight and once per camera
after session-open failure, logged. VAAPI may demote to `opencv` at boot.
NVDEC and OpenCV decode stay fail-fast. Mid-run device loss writes one first-fault
record, stops every camera, hard-exits 4. CUDA context is never recreated
in-process. Worker never queries Xid state. Zero configured cameras is a
valid boot.

Shared once: models, extractors, serving client, coordinator, profile/device,
GPU lease, config/LKG, evidence outbox, clip-store lock, audit overlay,
snapshot store, diagnostics. Per camera: tracker, `SceneState`, window
buffer, fall latch, bed assignment plus grace/hold, `IncidentManager`, bus,
encoder ring, ingest backoff, result slot, live observation cache, evidence
attacher. Hoisting a per-camera row leaks one resident into another.
`tests/test_worker_per_camera_fall_state.py` asserts both halves.

Seam defaults are `None`. Missing wiring refuses to start (ADR-0002).
Always-fail stubs belong in tests only. Pass the real MJPEG probe,
`serving_client`, ingest loop, and evidence path, or refuse boot. Never omit
an argument and let `_unavailable_probe` ship a dead feature while boot
succeeds. Root anti-pattern "조립 루트 스텁 배선". `serving_client` has no
default. Fall model must be explicit.

`worker/__main__.py` owns argparse and exit codes and constructs
`WorkerRuntime` from `worker.runtime.worker`
directly. Don't add `worker.runtime.edge_worker` or an `edge` alias. Keep
`--config`, `--check-config`, `--heartbeat-on-start`,
`--max-frames-per-camera`, `--backfill-thumbnails`, and signal shutdown.
Exit codes: 0 clean, 1 generic, 2 config, 3 refuse-to-start, 4 fatal
accelerator. `--check-config` has no model, camera, or relay side effect.
Production pulls from the baked relay plus `RELAY_TOKEN`. `RELAY_URL` is
retired. `--backfill-thumbnails` writes missing `thumbnail.jpg` and exits
non-zero while any playable clip lacks one.

Focused tests: `tests/test_profile_boot.py`, `tests/test_pipeline_bootstrap.py`,
`tests/test_worker_per_camera_fall_state.py`, `tests/test_worker_composition.py`,
`tests/test_worker_cli_residue.py`, `tests/test_runtime_manifest.py`,
`tests/test_worker_gpu_lease.py`, `tests/test_worker_fatal_accelerator.py`.
Boundary: `uv run --group lint lint-imports`. New observability lands in
structured logs and the local snapshot. Strict backend `RelayRuntimeStatusRequest` stays byte-for-byte compatible. Operator-visible fields belong in the message string.

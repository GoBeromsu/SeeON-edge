# worker/runtime: composition root

Own process lifecycle and wiring: config, staged bootstrap, Flow lifecycle,
camera supervision, telemetry, and CLI. Business math and vendor parsing stay
out. `worker.runtime` is the only package that may import everything (`types`,
`interfaces`, `adapters`, `pipeline`, `domains`, `contracts`, `shared.events`).
Nothing imports it. `backend` is forbidden. Relay HTTP only.

## Ownership rule

- `worker.py`: `WorkerRuntime`. Composes one Flow media plane, per-camera CPU
  policy state, evidence, telemetry, relay, and shutdown.
- `flow/`: Flow media-plane lifecycle, metadata admission, policy pump, and
  evidence handoff. The SDK owns capture, decode, inference, and tracking.
- `bootstrap.py`: named boot stages. It refuses activation until the deployed
  nvinfer engine matches the configured batch.
- `profile/`: `flow` is the only production profile; unknown or absent profile
  selection fails closed.
- `config/`: models, resolver, relay pull, LKG, and restart epoch. YAML is a
  developer hatch. No env roster.
- `provenance/`: applied runtime manifest delivered through the backend relay.
- `telemetry/`: `StatusStore`, `WorkerDiagnostics`, and `RuntimeStatusSender`.
  Local snapshot can grow; relay wire stays frozen.
- `faults/`: one first-fault record, stop every loop, `os._exit(4)`.
- `lease.py`: advisory flock at `~/.local/state/ml-worker/.gpu.lease`.
- `watchdog.py`: hung Flow work drives the same `FaultHandler` path.

`worker/tools/edge_engine_build.py` builds the engine before source activation.
Boot never performs a lazy build. Source-open faults are camera-local; media
plane, engine, parser, or deployed-batch identity faults are process-fatal.

## Failures, allocation, CLI, tests

Global stages are fatal. Failed named stage starts zero cameras and exits
non-zero. `FatalAcceleratorError` exits 4 and outranks the stage code. Lease is
released before exit. `run_camera_stage` degrades only that camera;
`FatalAcceleratorError` still re-raises because the GPU context is process-wide.
RTSP failure retries with backoff. Source open failure is camera `DEGRADED` plus
`camera.offline`; later processing failure is not offline.

Missing evidence wiring or a locked local store refuses to start. Remote
delivery is classified: retry classes retry, compatibility failures reprobe,
and payload-invalid failures become permanent. Two workers must not share one
delivery queue. Zero configured cameras is a valid boot.

The Flow plane is process-shared. Per-camera ownership includes domain policy
state, evidence attachers, and camera-local observation state. All cameras share
config/LKG, GPU lease, evidence delivery queue, clip-store lock, snapshot store,
and diagnostics. Hoisting a per-camera row leaks one resident into another.

Seam defaults are `None`. Missing wiring refuses to start (ADR-0002).
Always-fail stubs belong in tests only. Pass the real media plane and evidence
path or refuse boot.

`worker/__main__.py` owns argparse and exit codes and constructs `WorkerRuntime`
from `worker.runtime.worker` directly. Keep `--config`, `--check-config`,
`--heartbeat-on-start`, `--max-frames-per-camera`, `--backfill-thumbnails`, and
signal shutdown. Exit codes: 0 clean, 1 generic, 2 config, 3 refuse-to-start,
4 fatal accelerator. `--check-config` has no model, camera, or relay side effect.
Production pulls from the baked relay plus `RELAY_TOKEN`. `RELAY_URL` is retired.

Focused tests: `tests/test_profile_boot.py`, `tests/test_pipeline_bootstrap.py`,
`tests/test_worker_flow_activation.py`,
`tests/test_runtime_manifest.py`, `tests/test_worker_gpu_lease.py`, and
`tests/test_worker_fatal_accelerator.py`. Boundary:
`uv run --group lint lint-imports`.

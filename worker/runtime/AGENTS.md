# worker/runtime — composition root

Own process lifecycle and wiring: configuration, profile and device policy,
staged bootstrap, GPU containment, camera supervision, telemetry, and the CLI.

## Ownership rule

**`worker.runtime` is the only composition root, and the only package that may
import everything** (`worker.types`, `worker.interfaces`, `worker.adapters`,
`worker.pipeline`, `worker.domains`, `contracts`, `shared.events`). Nothing
imports `worker.runtime` — import-linter contract *"worker runtime is the sole
composition root"* forbids it from every other worker package.

The privilege comes with a restriction: **no business math and no concrete model
parsing here.** Runtime constructs objects, injects them, and manages lifecycle.
Feature math belongs in `worker.pipeline.perception`, interpretation in
`worker.domains`, and vendor specifics in `worker.adapters`.

`backend` stays forbidden. Worker and backend talk only over relay HTTP.

## Local Ownership

- `worker.py`: the composition root — build the shared model/extractor bundle
  once, then per-camera ingest subscription, tracker, `SceneState`, fall
  classifier and latch, bed assignments, scheduler, `IncidentManager`, and
  encoder state; wire output and status; supervise shutdown and restart.
- `profile/`: `ML_WORKER_PROFILE` to the exact `(device, decode, encode)` triple,
  fail-fast device verification via an injected `CudaProbe`, decode preflight.
  This package holds policy, not hardware access.
- `config/`: worker config models, resolver, backend config pull, LKG store,
  restart epoch. Backend-primary with offline YAML fallback.
- `bootstrap.py`: the named stages `profile/device → decode capability → model
  backend init → real warmup → camera activation`.
- `supervisor.py`: per-camera thread/session supervision and failure isolation.
- `telemetry/`: status store, runtime diagnostics, runtime-status sender.

## Conventions

- Global stages are fatal: any named stage failure starts zero cameras and exits
  non-zero. Per-camera failures degrade only that camera.
- `auto`, blank, missing, and unknown profiles fail loudly. No silent CPU or
  OpenCV fallback, and a failed CUDA verifier never resolves to the software
  encoder policy.
- A `FatalAcceleratorError` persists exactly one bounded first-fault record under
  `ML_WORKER_STATE_DIR`, stops scheduling every camera, and exits with the
  documented code. The CUDA context is never recreated or reused in-process.
- One advisory GPU lease under `ML_WORKER_STATE_DIR` prevents overlap between the
  worker and this repo's smoke/replay commands. Acquire it before any
  CUDA/NVDEC/model construction.
- Preserve every legacy env key and YAML field: `ML_WORKER_*`, `WORKER_*`,
  `API_*`, `RELAY_*`, and the legacy decode compatibility keys.
- New observability lands in structured logs and the local snapshot. The strict
  backend `RelayRuntimeStatusRequest` payload stays byte-for-byte compatible and
  receives only its existing fields.
- The worker never queries or claims GPU Xid state. That is a host runbook
  concern.
- No stub wiring at this composition root: when a collaborator seam defaults to
  an always-failing/no-op stub (e.g. `mjpeg_server`'s `_unavailable_probe`),
  runtime must pass the real implementation or refuse to boot — never omit the
  argument and let the feature ship dead while boot succeeds. See the root
  `AGENTS.md` anti-pattern "조립 루트(runtime)의 스텁 배선" (#81).

## CLI

`python -m worker` is the only supported entrypoint; `worker/__main__.py`
owns argparse and exit codes and constructs `WorkerRuntime` from
`worker.runtime.worker` directly. Preserve `--config`,
`--check-config`, `--heartbeat-on-start`, signal shutdown, and the documented
exit codes. Do not add `worker.runtime.edge_worker` or an `edge` alias.
`--check-config` performs no model, camera, or relay side effect.

## Focused Tests

- `tests/test_profile_boot.py`, `tests/test_pipeline_bootstrap.py`
- Boundary enforced by import-linter (`uv run --group lint lint-imports`)

## Change Boundary

Model objects are shared once per task per process; every per-camera mutable
object is constructed once per camera. When changing the bundle or the
supervisor, preserve both properties — they are asserted by
`tests/test_worker_per_camera_fall_state.py`.

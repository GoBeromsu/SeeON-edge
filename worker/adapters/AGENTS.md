# worker/adapters: concrete port implementations

Own vendor-backed implementations behind `worker/interfaces`.
`worker.adapters` must not import `worker.pipeline`, `worker.domains`, or
`worker.runtime`. Runtime constructs and injects. Need schedule, identity, or
config? Constructor argument. Need a higher-layer type? Local Protocol.

Allowed: stdlib, `contracts`, `worker.types`, `worker.interfaces`.
Forbidden: `backend`, `worker.pipeline`, `worker.domains`, `worker.runtime`.
Enforced by "worker adapters do not depend on pipeline, domains, or runtime"
and "worker runtime is the sole composition root".

## Local ownership

- `deepstream/`: the only worker package allowed to import `pyservicemaker` or
  `pyds`. Imports are lazy; convert vendor metadata immediately to worker
  envelopes. Owns Flow sources and Service Maker composition helpers.
- `model/`: model registry and CPU model helpers used by domain policy.
- `device/`: honest probes. Import success is not capability.
- `frame/`: host-frame materialization. `view` is zero-copy host-only;
  `materialize` copies and counts.

## Interface implementation

Implement the port. Don't invent a parallel API.

- `MediaPlane` implementations keep SDK objects behind the interface boundary.
- `ServingClient.create(task, ...)` preserves object identity across cameras.
- `FrameMaterializer` / `HostFrameView`: non-host `view` fails. No silent
  host↔device transfer.

A new backend is a new package behind the existing port, registered by
`worker/runtime`. Not a branch in a pipeline stage. Delete unused skeletons.
Don't leak vendor objects through a port signature.

## Explicit backend selection

The Flow runtime owns production media-plane selection. Adapters execute that
choice and do not probe onto another backend. A probe answers this process:
`available=True` does not mean the camera is reachable or the artifact loads.
Warmup and source activation own those checks.

## Resource ownership

Vendor objects stay in the adapter that created them. Release SDK resources on
its lifecycle path; higher layers own only worker envelopes. Default tests stay
hardware-free. Assert lazy-import and metadata-conversion invariants, not host
GPU availability.

Focused tests: `tests/test_deepstream_adapter_plane.py`,
`tests/test_deepstream_adapter_metadata.py`.
Boundary: `uv run --group lint lint-imports`.

# WORKER INSTANCE KNOWLEDGE BASE

Own the deployable inference instance (`ml-worker` image): RTSP ingest and decode,
a bounded frame bus, composable model extraction, numeric decision-making, and
event/evidence egress to `ml-api` over one-way relay HTTP.

Canonical entrypoint, and the only supported one:

```sh
python -m worker
```

`worker/__main__.py` owns the CLI (argparse, exit codes) and constructs
`WorkerRuntime` from `worker.runtime.worker` directly. Do not add a
second entrypoint, a `worker.runtime.edge_worker` module, or an `edge` alias.

Deployment identity is frozen: image `ml-worker`, `Dockerfile.edge`,
`compose.edge.yaml`, `.env.edge.prod*`, dependency group `worker`, and the
`ML_WORKER_*` / `WORKER_*` env prefixes. Renaming any of them breaks the edge
host contract.

## Layers (each with its own AGENTS.md)

| Package | Role | May import |
| --- | --- | --- |
| `types/` | worker-internal envelopes (`FramePacket`, `ModuleResult`, `DecisionInput`, `BusinessEvent`) | stdlib, `contracts` |
| `interfaces/` | one `typing.Protocol` per replaceable seam | stdlib, `contracts`, `worker.types` |
| `adapters/` | concrete decode / model / encode implementations | stdlib, `contracts`, `worker.types`, `worker.interfaces` |
| `pipeline/` | ingest, bus, perception, decision, output stages | everything except `worker.runtime` |
| `domains/` | fall and bed-exit interpretation and latching | `contracts`, `worker.types`, `worker.interfaces`, `worker.pipeline.perception` |
| `runtime/` | composition root: config, profile, bootstrap, supervisor, telemetry | everything |

The ladder is `runtime → pipeline → domains → adapters → interfaces → types →
contracts`, enforced by import-linter (`uv run --group lint lint-imports`), not by
a hand-rolled walker.

## Boundary with vendored `contracts`

`contracts` holds cross-instance L0 data only — the shapes `backend`, `worker`,
and `shared` must all agree on. Worker-internal ports and envelopes live under
`worker/`: `worker/types` owns `FramePacket`, `ModuleResult`, `DecisionInput`, and
`BusinessEvent`; `worker/interfaces` owns the decode/bus/extract/decision/encode/
output/serving Protocols. They import `contracts` and keep `Frame`,
`RunnerResult`, `FrameObservation`, `BedRegionDebugSnapshot`, and `EventPayload`
authoritative — never duplicate or shadow a vendored type.

Do not add a worker-internal shape to `contracts`. That tree is ADR-0004 vendored
byte-for-byte against `eldercare-dataset-ops`, and
`tests/test_vendor_drift.py` compares **every** file under `contracts/`,
including `contracts/AGENTS.md` — not only the `*.py` modules. Editing anything
there, documentation included, breaks the drift firewall until the sibling repo is
re-synced in the same change. State worker-scoped rules here instead.

## Imports

Allowed: `contracts`, `shared.events`, and local `worker.*` modules.

Forbidden (enforced by import-linter): `backend`. The worker and the backend are
import-independent and communicate only over relay HTTP
(`/api/v1/relay/{config,restart,alerts,heartbeat,runtime-status}`).

## How to contribute

1. Read [`docs/architecture.md`](../docs/architecture.md) first. It holds the
   five-layer diagram, the raw/numeric fan-out rules, the per-camera vs shared
   state table, and the failure matrix.
2. Read the `AGENTS.md` of the package you are changing. Each one states an
   ownership rule that import-linter enforces.
3. Put new code in the layer that owns the concern. If it needs an import the
   layer forbids, the design is wrong — introduce a Protocol in `interfaces/` and
   let `runtime/` inject the concrete object.
4. New replaceable behavior arrives as a Protocol plus at least two
   implementations, or one implementation plus a test double.
5. Keep new pure-code modules at or below 250 logical LOC. Split by port or stage
   rather than reproducing a monolith under a new name.
6. Run the standing gates from the repository root:

   ```bash
   uv run pytest -q
   uv run --group lint lint-imports
   uvx ruff check .
   ```

## Gotchas

Model objects are shared once per task per process; every temporal thing
(tracker, `SceneState`, fall windows and latch, bed assignments and grace/hold,
`IncidentManager`, scheduler, encoder ring) is per camera and must never be
shared. `tests/test_worker_per_camera_fall_state.py` guards both halves.

`FramePacket` is the only envelope allowed to carry an image. Raw frames reach
model extraction, derivative evidence, overlay/MJPEG, and the alert snapshot —
nothing else. The decision layer receives `DecisionInput` only.

Global bootstrap stages (profile/device, decode capability, model backend, real
warmup) are fatal and exit non-zero. Per-camera failures degrade only that
camera. There is no silent CPU or OpenCV fallback, and `auto` is a loud failure.

Primary clean clips use ADR-0001 source-packet stream-copy/remux through bounded
camera-local packet rings. Decoded frames remain analysis/snapshot taps only;
never add a silent re-encode fallback or describe a transformed derivative as
the preserved source clip.

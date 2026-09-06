# worker

DeepStream Flow worker: SDK-owned capture, decode, inference, and tracking; CPU-owned fall and bed-exit decisions; evidence and one-way relay egress.
Image `ml-worker`. Sole production command: `python -m worker`.
Replay is backend-owned; the production worker has no replay CLI.

## Layers

Run `uv run --group lint lint-imports` for configured contracts. A forbidden import means the design is wrong: add a Protocol in `interfaces/` and inject from `runtime/`. Focused dependency tests and manual path review own native/vendor ceilings.

| Package | Role | Worker-layer ceiling |
| --- | --- | --- |
| `types/` | internal envelopes | no other `worker` layer |
| `interfaces/` | one Protocol per seam | `types` |
| `adapters/` | DeepStream vendor integration and model helpers | `types`, `interfaces` |
| `pipeline/` | decision and output coordination | everything except `runtime` |
| `domains/` | fall and bed-exit decisions | `types`, `interfaces`, `pipeline` |
| `runtime/` | sole composition root | everything |
| `tools/edge_engine_build.py` | nvinfer engine build before source activation | out of the production import graph |
| `tools/fetch_models/` | pinned model provisioning (`edge-model-fetch`) | stdlib only; out of the production import graph |

Order: `runtime -> pipeline -> domains -> adapters -> interfaces -> types -> contracts`.
`tools/` is out-of-band; import-linter forbids every worker layer from importing it.
`contracts` contains cross-instance L0 data only. Worker-internal ports and envelopes live under `worker/`; never duplicate or shadow a vendored type, including `contracts/AGENTS.md`.
Shared leaves are scope-owned: `detection_policies`, `events`, and `rtsp_url_policy`. Worker never imports `backend` or database modules.

## Data and lifetime boundaries

`types/AGENTS.md` owns the pixel/numeric envelope contract. `runtime/AGENTS.md` owns process-shared versus per-camera allocation. Keep both boundaries intact; details stay in those scoped guides.

## Hardware failure policy

`runtime/AGENTS.md` owns boot exit codes, Flow lifecycle, and camera-local degradation. `adapters/deepstream/AGENTS.md` owns lazy vendor imports. `pipeline/AGENTS.md` owns output coordination, and `pipeline/output/evidence/AGENTS.md` owns delivery durability. Read those scoped guides before changing failure behavior.

## Package navigation

Read the nearest `AGENTS.md` before changing that package.

| Path | Go here for |
| --- | --- |
| `types/` | `FramePacket`, `DecisionInput`, `ModuleResult`, `BusinessEvent` |
| `interfaces/` | media-plane, association, output, and serving seams |
| `adapters/deepstream/` | lazy `pyservicemaker`/`pyds` integration, sources, and metadata conversion |
| `adapters/model/` | model registry and CPU model helpers |
| `pipeline/decision/` | `IncidentManager`, admission |
| `pipeline/output/` | event publication and evidence handoff |
| `pipeline/output/evidence/` | smart record actor, clip publication, sealed sidecar, durable stager, delivery queue, snapshot store |
| `domains/fall/` | window classifier and rising-edge latch |
| `domains/bed_exit/` | assignment, grace, and hold |
| `runtime/worker.py` | composition root |
| `runtime/flow/` | Flow media plane, policy pump, lifecycle, and evidence handoff |
| `runtime/bootstrap.py` | named stages and boot gate |
| `tools/edge_engine_build.py` | nvinfer engine build and deployed-batch identity |
| `tools/fetch_models/` | manifest-pinned model download + SHA-256 verification into `/app/models` |

New seam: Protocol plus two implementations, or one plus a test double.
Keep new pure-code modules at or below 250 logical LOC. Split by port or stage.

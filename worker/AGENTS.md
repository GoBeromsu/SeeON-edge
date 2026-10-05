# worker

DeepStream Flow worker: SDK-owned capture, decode, inference, and tracking; CPU-owned fall and bed-exit decisions; evidence and one-way relay egress.
Image `ml-worker`. Production command:
`/usr/local/bin/ml-worker run --auxiliary-runtime=onnxruntime-cpu`.
Live pose remains on the GPU; auxiliary models use the existing ORT CPU owner.
The shipping worker contains native artifacts and configuration, not the Python
worker package/venv. The vendor DeepStream base may contain Python tooling.
Python worker modules remain source-level reference/verification code, not the
shipping entrypoint. The isolated model-fetch operator runs in `ml-api`'s image.
Replay is backend-owned; the production worker has no replay CLI.

Rust binaries capture `ML_WORKER_BUILD_REVISION` at compilation.
Only a trusted build of frozen, clean source may supply that declaration; a
runtime environment value cannot attribute an undeclared binary. Image markers
and runtime revision values must agree with the compiled declaration. Runtime
Git is not a source of identity for an already-compiled executable. This is
caller-declared attribution, not final-image attestation or a production cutover.

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
| `adapters/media/` | bounded native RTSP frame capture for one-off CPU recognition |
| `pipeline/decision/` | `IncidentManager`, admission |
| `pipeline/output/` | event publication and evidence handoff |
| `pipeline/output/evidence/` | smart record actor, clip publication, sealed sidecar, durable stager, delivery queue, snapshot store |
| `domains/fall/` | window classifier and rising-edge latch |
| `domains/bed_exit/` | assignment, grace, and hold |
| `runtime/worker.py` | composition root |
| `runtime/flow/onnx_shape.py` | shared ONNX input-shape inspection for engine build and Flow boot gates |
| `tools/export_pose_onnx.py` | owned dynamic-batch pose export; imports ultralytics only inside tool functions |
| `runtime/flow/` | Flow media plane, policy pump, lifecycle, and evidence handoff |
| `runtime/bootstrap.py` | named stages and boot gate |
| `tools/edge_engine_build.py` | nvinfer engine build and deployed-batch identity |
| `tools/fetch_models/` | manifest-pinned model download + SHA-256 verification into `/app/models` |

New seam: Protocol plus two implementations, or one plus a test double.
Keep new pure-code modules at or below 250 logical LOC. Split by port or stage.

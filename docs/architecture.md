# Architecture

The worker is one process that turns RTSP frames into business events. It is
organised as five layers with one direction of flow, a single canonical
entrypoint, and an explicit rule about which envelopes may carry pixels. This
document is the decision record for that shape: what each layer owns, what state
is per camera, what happens on each class of failure, and which legacy module
became which worker module.

Read the scoped `AGENTS.md` next to the code you are changing for the
per-package import ceiling. Boundaries are enforced by import-linter
(`uv run --group lint lint-imports`), not by convention.

The runtime slot produces media and sends delivery records to the backend. The
backend owns metadata, database access, and media access. The frontend consumes
only the backend HTTP API.

## Co-located persistence boundary

The one supported edge deployment is one local Linux host, one Compose release
unit, one API process, and one worker process. Those processes remain
import-independent and HTTP remains their command/event notification boundary.
The backend alone owns `/var/lib/seeon-state/edge.sqlite3`; the database,
`edge.sqlite3-wal`, and `edge.sqlite3-shm` stay together in the same private
`0700` local directory and the database is `0600`. The runtime slot has no
database mount and never opens, migrates, or repairs a SQLite database.

Only the one-shot `python -m backend.app.edge_db.compact_cutover` migrator executes DDL
or advances `PRAGMA user_version`. Backend connections verify the machine-readable
migration ledger and ownership map, enable foreign keys, use WAL with
`synchronous=FULL` and a fixed 5000 ms busy timeout, and are guarded by a SQLite
authorizer that rejects DDL and unauthorized writes. Transactions are short:
never hold one across hash, fsync, HTTP, or other external work.

| Table | Sole writer |
| --- | --- |
| `schema_migrations` | one-shot migrator |
| compact application tables | backend API |

Schema compatibility is an explicit inclusive range, not an optimistic open.
Schema 18 is the compact ten-table contract. Registration refuses to apply
schema 18 while undrained schema-17 evidence remains, emits
`EDGE_DB_DRAIN_INCOMPLETE`, and leaves the schema-17 database byte-identical.

| Database version relative to binary range | Runtime behavior |
| --- | --- |
| below minimum | refuse; migrator required |
| minimum through maximum | open read/write with ownership guard, no DDL |
| above maximum | refuse; binary is too old |

The central database is forward-only and requires the complete cutover schema.
Temporary SYSTEM_TEST operator-only rows and their `system_test_runs` mapping are
retired on the same forward-only path as released main outbox schema 9: legacy
worker snapshots may still contain them, but the central migrator and stopped-
runtime importer purge operator-only events (and dependent central projections)
while preserving ordinary evidence, clips, config/fault rows, and clip-deletion
reasons. No runtime CLI flag, relay route, or sender API retains executable
SYSTEM_TEST authority.

## Layers

Flow is one-way. A layer may depend on the layer above it and on
`worker/types` and `worker/interfaces`; it may never reach back down.

```text
        ┌─────────────────────────────────────────────────────────────┐
        │ 1. INGEST            worker/pipeline/ingest/                │
        │    RTSP/file/webcam source -> decode adapter -> FramePacket │
        │    per-camera capture loop, reconnect/backoff, probe        │
        └─────────────────────────────────────────────────────────────┘
                                   │  FramePacket  (carries an image)
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 2. FRAME BUS         worker/pipeline/bus/                   │
        │    bounded, drop-oldest fan-out; per-camera scheduler;      │
        │    depth/drop metrics. Back-pressure is visible, not silent │
        └─────────────────────────────────────────────────────────────┘
                                   │  FramePacket  (fan-out, see below)
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 3. ANALYTICS         worker/pipeline/perception/            │
        │                      worker/pipeline/analytics/             │
        │    extractors (person/pose/bed-seg) -> FrameObservation;    │
        │    tracker, SceneState, window buffer -> DecisionInput      │
        └─────────────────────────────────────────────────────────────┘
                                   │  DecisionInput  (numeric only)
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 4. DECISION          worker/domains/                        │
        │                      worker/pipeline/decision/              │
        │    fall + bed-exit detectors, latches, incident manager,    │
        │    event aggregation -> BusinessEvent                       │
        └─────────────────────────────────────────────────────────────┘
                                   │  BusinessEvent  (+ explicit snapshot)
                                   ▼
        ┌──────────────────────────────┐  ┌──────────────────────────┐
        │ 5a. OUTPUT                   │  │ 5b. TELEMETRY            │
        │  worker/pipeline/output/     │  │  worker/runtime/telemetry│
        │  relay event sink, evidence  │  │  heartbeat, runtime      │
        │  clips + outbox, overlay,    │  │  status, diagnostics,    │
        │  MJPEG live view             │  │  local metrics           │
        └──────────────────────────────┘  └──────────────────────────┘
```

`worker/runtime/` is the composition root: it builds the layers, owns the
bootstrap gates, the GPU lease, the watchdog, and the fault handler. It is the
only package permitted to import everything.

## Types and the contracts boundary

Worker-internal ports and envelopes live under `worker/`; cross-instance L0 data
stays in `contracts`. `contracts/` is ADR-0006 vendored byte-for-byte from
`eldercare-dataset-ops` and is snapshotted by `tests/test_vendor_drift.py`,
including `contracts/AGENTS.md` — never edit anything under it as part of worker
work, and never duplicate or shadow a vendored type inside `worker/`.

| Envelope | Module | Carries pixels |
| --- | --- | --- |
| `FramePacket` | `worker/types/frame_packet.py` | yes |
| `ModuleResult` | `worker/types/module_result.py` | no |
| `DecisionInput` | `worker/types/decision_input.py` | no |
| `BusinessEvent` | `worker/types/business_event.py` | no |

## Raw-image vs numeric fan-out

`FramePacket` is the only envelope permitted to carry an image. Four
subscribers may receive it, and no others:

| Subscriber | Why it needs pixels |
| --- | --- |
| Model extraction | runs the person/pose/bed-seg adapters |
| Derivative evidence | feeds the per-camera segment encoder for clips |
| Overlay / MJPEG live view | draws debug output for operators |
| Alert snapshot | encodes the single bounded JPEG attached to an alert |

Everything downstream of analytics is numeric.
`DecisionInput` carries exactly the seven fields `observation`, `frame_width`,
`frame_height`, `live_track_ids`, `time_sec`, `frame_index`, and `bed_region` —
no array, no buffer, no handle from which a frame could be recovered. A domain
detector that needs a pixel is a design error: add an extractor in layer 3 and
pass the number it produced.

## Per-camera state vs shared state

Anything temporal belongs to exactly one camera. Anything expensive and
stateless is built once per process. `tests/test_worker_per_camera_fall_state.py`
guards both halves.

| State | Scope | Owner |
| --- | --- | --- |
| Tracker | per camera | `worker/pipeline/perception/tracker.py` |
| SceneState | per camera | `worker/pipeline/perception/scene_state.py` |
| Window buffer and fall probabilities | per camera | `worker/pipeline/perception/window_buffer.py` |
| Fall latch | per camera | `worker/domains/fall/detector.py` |
| Bed assignments, grace/hold, night window | per camera | `worker/domains/bed_exit/` |
| IncidentManager | per camera | `worker/pipeline/decision/incident_manager.py` |
| Frame-bus subscription and scheduler slot | per camera | `worker/pipeline/bus/` |
| Encoder session and segment ring | per camera | `worker/adapters/encode/session.py` |
| Ingest loop and reconnect backoff | per camera | `worker/pipeline/ingest/lifecycle.py` |
| Model objects and extractor instances | shared, one per task per process | `worker/runtime/model_composition.py` |
| Resolved profile and device selection | shared, one per process | `worker/runtime/profile/` |
| GPU lease | shared, one per process | `worker/runtime/lease.py` |
| Config / LKG store | shared, one per process | `worker/runtime/config/` |
| Evidence outbox database | shared, one per process | `worker/pipeline/output/evidence/evidence_outbox_database.py` |
| Clip store lock | shared, one per host directory | `worker/pipeline/output/evidence/clip_store_lock.py` |

Sharing a per-camera row across cameras is a correctness bug, not an
optimisation: it leaks one resident's motion history into another's fall
decision.

## Entrypoint

The canonical and only command is:

```sh
python -m worker
```

`worker/__main__.py` owns the CLI: it parses argv, loads config, constructs
`WorkerRuntime` directly, and maps outcomes to exit codes. `worker/runtime/worker.py`
stays composition-only and exports exactly three classes —
`CameraRuntimeContext`, `HeartbeatReporter`, `WorkerRuntime`. There is no
module-level `main` there and no delegate indirection; earlier drafts of this
document claimed one, and that claim was wrong. Do not add an alias module, a
console script, or a per-submodule `python -m` target: one entrypoint is a plan
requirement.

| Exit code | Meaning |
| --- | --- |
| 0 | clean shutdown |
| 1 | generic runtime error |
| 2 | config or resolution error |
| 3 | refuse-to-start (a bootstrap gate failed) |
| 4 | fatal accelerator fault (`worker/runtime/faults/handler.py`, hard exit) |

## Failure matrix

Global faults kill the process loudly. Per-camera faults degrade exactly one
camera. There is no silent CPU or OpenCV fallback, and `auto` device selection
is a loud failure, per ADR-0002.

| Fault | Scope | Behaviour |
| --- | --- | --- |
| Profile unresolvable / `auto` requested | global | refuse to start, exit 3 |
| Requested accelerator unavailable at boot | global | refuse to start, exit 3, reason logged |
| Decode capability probe fails for the profile | global | refuse to start, exit 3 |
| Model artifact missing or unloadable | global | refuse to start, exit 3 |
| Real warmup inference fails | global | refuse to start, exit 3 |
| GPU lease already held by another process | global | refuse to start, exit 3 |
| Config invalid and no LKG available | global | exit 2 |
| Config invalid but LKG present | global | run from LKG, report degraded status |
| Accelerator fault mid-run (device lost, unrecoverable) | global | record first fault, hard exit 4 for supervised restart |
| Inference deadline exceeded | global | watchdog records and escalates to the fault path |
| RTSP connect/auth failure | per camera | that camera retries with backoff; others unaffected |
| Decode stall or stream EOF | per camera | reopen the source; camera reports unhealthy meanwhile |
| Frame bus full | per camera | drop oldest, increment drop metric; never block ingest |
| Extractor raises on one frame | per camera | frame dropped, camera continues |
| Clip encode failure | per camera | alert still relays; evidence marked incomplete |
| Relay POST failure | per camera | durable outbox retries; no event is lost in memory |
| Clip store locked by another process | global | worker refuses to start: two workers must not share one outbox (ADR-0003) |
| Evidence delivery enabled but misconfigured or unable to initialise | global | worker refuses to start rather than run with alerts stranded in the local outbox (ADR-0003) |

Rollback of a bad worker image is image-digest based; see
[`docs/runbooks/worker-migration-rollback.md`](runbooks/worker-migration-rollback.md).

## Source-to-target ownership

Every non-`__init__` source file in the legacy tree has exactly one owner below.
These rows are historical citations of a migration, not operator instructions.
`tests/test_worker_architecture_docs.py` asserts the map stays complete and
unambiguous while the legacy tree exists, and stops constraining it once the
tree is deleted.

| Current source | Final owner |
| --- | --- |
| `edge/AGENTS.md` | `worker/AGENTS.md` |
| `edge/__main__.py` | `worker/__main__.py` (argparse, exit codes, `python -m worker`) |
| `edge/pyproject.toml` | `worker/pyproject.toml` (`project.name = "eldercare-worker"`) |
| `edge/ml-worker.example.yaml` | `worker/ml-worker.example.yaml` |
| `edge/domains/AGENTS.md` | `worker/domains/AGENTS.md` |
| `edge/domains/base.py` | `worker/domains/base.py` |
| `edge/domains/bed_exit/AGENTS.md` | folded into `worker/domains/AGENTS.md` |
| `edge/domains/bed_exit/detector.py` | `worker/domains/bed_exit/detector.py` |
| `edge/domains/bed_exit/latch.py` | `worker/domains/bed_exit/latch.py` |
| `edge/domains/bed_exit/schema.py` | `worker/domains/bed_exit/schema.py` |
| `edge/domains/fall/AGENTS.md` | folded into `worker/domains/AGENTS.md` |
| `edge/domains/fall/detector.py` | `worker/domains/fall/detector.py` |
| `edge/domains/fall/schema.py` | `worker/domains/fall/schema.py` |
| `edge/evidence/clip_recorder.py` | `worker/pipeline/output/evidence/clip_recorder.py` plus the `clip_*` split modules beside it |
| `edge/evidence/clip_store_lock.py` | `worker/pipeline/output/evidence/clip_store_lock.py` |
| `edge/evidence/event_identity.py` | `worker/pipeline/output/evidence/event_identity.py` |
| `edge/evidence/evidence_manifest.py` | `worker/pipeline/output/evidence/evidence_manifest.py` |
| `edge/evidence/evidence_media.py` | `worker/pipeline/output/evidence/evidence_media.py` |
| `edge/evidence/evidence_outbox.py` | `worker/pipeline/output/evidence/evidence_outbox.py` |
| `edge/evidence/evidence_outbox_clips.py` | `worker/pipeline/output/evidence/evidence_outbox_clips.py` |
| `edge/evidence/evidence_outbox_delivery.py` | `worker/pipeline/output/evidence/evidence_outbox_delivery.py` |
| `edge/evidence/evidence_outbox_schema.py` | `worker/pipeline/output/evidence/evidence_outbox_schema.py` |
| `edge/evidence/evidence_outbox_stage.py` | `worker/pipeline/output/evidence/evidence_outbox_stage.py` |
| `edge/evidence/evidence_outbox_types.py` | `worker/pipeline/output/evidence/evidence_outbox_types.py` |
| `edge/evidence/evidence_reconciliation.py` | `worker/pipeline/output/evidence/evidence_reconciliation.py` |
| `edge/evidence/evidence_retention.py` | `worker/pipeline/output/evidence/evidence_retention.py` |
| `edge/evidence/evidence_runtime.py` | `worker/pipeline/output/evidence/evidence_runtime.py` |
| `edge/evidence/evidence_sender.py` | `worker/pipeline/output/evidence/evidence_sender.py` |
| `edge/evidence/evidence_stager.py` | `worker/pipeline/output/evidence/evidence_stager.py` |
| `edge/evidence/snapshot_store.py` | `worker/pipeline/output/evidence/snapshot_store.py` |
| `edge/features/AGENTS.md` | folded into `worker/pipeline/AGENTS.md` |
| `edge/features/geometry.py` | `worker/pipeline/perception/features/geometry.py` |
| `edge/features/pose_normalization.py` | `worker/pipeline/perception/features/pose_normalization.py` |
| `edge/features/window_features.py` | `worker/pipeline/perception/features/window_features.py` |
| `edge/perception/AGENTS.md` | folded into `worker/pipeline/AGENTS.md` |
| `edge/perception/domain_input.py` | `worker/pipeline/perception/decision_input.py` |
| `edge/perception/fall_window_classifier.py` | `worker/domains/fall/classifier.py` |
| `edge/perception/observation_builder.py` | `worker/pipeline/perception/observation_builder.py` |
| `edge/perception/overlay_renderer.py` | `worker/pipeline/output/_overlay_primitives.py` (drawing primitives only) |
| `edge/perception/scene_state.py` | `worker/pipeline/perception/scene_state.py` |
| `edge/perception/tracker.py` | `worker/pipeline/perception/tracker.py` |
| `edge/perception/window_buffer.py` | `worker/pipeline/perception/window_buffer.py` |
| `edge/runners/AGENTS.md` | folded into `worker/adapters/AGENTS.md` |
| `edge/runners/device.py` | `worker/runtime/profile/device.py` |
| `edge/runners/registry.py` | `worker/adapters/model/registry.py` |
| `edge/runners/sklearn_fall.py` | `worker/adapters/model/sklearn_fall.py` plus `sklearn_metadata.py` |
| `edge/runners/torch_lstm_fall.py` | `worker/adapters/model/torch_lstm_fall.py` plus `lstm_manifest.py` |
| `edge/runners/warmup.py` | `worker/adapters/model/warmup.py` |
| `edge/runners/yolo_bed_seg.py` | `worker/adapters/model/yolo_bed_seg.py` |
| `edge/runners/yolo_person.py` | `worker/adapters/model/yolo_person.py` |
| `edge/runners/yolo_pose.py` | `worker/adapters/model/yolo_pose.py` |
| `edge/serving_client/base.py` | `worker/interfaces/serving.py` |
| `edge/serving_client/in_process.py` | `worker/adapters/model/in_process.py` |
| `edge/sources/AGENTS.md` | folded into `worker/pipeline/AGENTS.md` |
| `edge/sources/camera_probe.py` | `worker/pipeline/ingest/camera_probe.py` |
| `edge/sources/frame_source.py` | `worker/pipeline/ingest/frame_source.py` |
| `edge/sources/probe.py` | `worker/pipeline/ingest/probe.py` |
| `edge/sources/registry.py` | `worker/pipeline/ingest/registry.py` |
| `edge/sources/rtsp.py` | `worker/pipeline/ingest/rtsp.py` |
| `edge/sources/rtsp_backend.py` | `worker/adapters/decode/cpu_av/` and `worker/adapters/decode/nvdec_cuvid/` |
| `edge/sources/rtsp_url.py` | `worker/pipeline/ingest/rtsp_url.py` |
| `edge/sources/video_file.py` | `worker/pipeline/ingest/video_file.py` |
| `edge/sources/webcam.py` | `worker/pipeline/ingest/webcam.py` |
| `edge/runtime/camera_worker.py` | split, no single owner: orchestration is `worker/pipeline/ingest/lifecycle.py` (`CameraIngestLoop`), `worker/pipeline/analytics/composite.py` (`CompositeExtractor`), and `worker/runtime/worker.py` (`CameraRuntimeContext`) — see the open gap below |
| `edge/runtime/config_pull.py` | `worker/runtime/config/config_pull.py` plus `http_transport.py`, `pull_models.py` |
| `edge/runtime/config_resolver.py` | `worker/runtime/config/config_resolver.py` |
| `edge/runtime/edge_worker.py` | `worker/runtime/worker.py` (`WorkerRuntime`); its CLI half is `worker/__main__.py` |
| `edge/runtime/edge_worker_config.py` | `worker/runtime/config/worker_models.py` plus `camera_models.py`, `domain_models.py`, `loader.py` |
| `edge/runtime/edge_worker_supervisor.py` | `worker/pipeline/ingest/lifecycle.py` (`IngestSupervisor`); restart policy folded into `worker/runtime/worker.py` |
| `edge/runtime/incident_manager.py` | `worker/pipeline/decision/incident_manager.py` |
| `edge/runtime/latest_frame.py` | `worker/pipeline/output/live_view.py` (`LatestFrameStore`) |
| `edge/runtime/lkg_store.py` | `worker/runtime/config/lkg_store.py` |
| `edge/runtime/mjpeg_server.py` | `worker/pipeline/output/mjpeg_server.py` plus `_mjpeg_http.py` |
| `edge/runtime/overlay_renderer.py` | `worker/pipeline/output/overlay.py` (`OverlayRenderer`) |
| `edge/runtime/pipeline_bootstrap.py` | `worker/runtime/bootstrap.py` |
| `edge/runtime/profile/AGENTS.md` | folded into `worker/runtime/AGENTS.md` |
| `edge/runtime/profile/boot.py` | `worker/runtime/profile/boot.py` |
| `edge/runtime/profile/registry.py` | `worker/runtime/profile/registry.py` |
| `edge/runtime/runtime_diagnostics.py` | `worker/runtime/telemetry/runtime_diagnostics.py` |
| `edge/runtime/runtime_status_sender.py` | `worker/runtime/telemetry/runtime_status_sender.py` plus `wire.py` |
| `edge/runtime/scheduler.py` | `worker/pipeline/bus/scheduler.py` |
| `edge/runtime/status_store.py` | `worker/runtime/telemetry/status_store.py` |

Deployment identity is deliberately unchanged by this map: the image and service
stay `ml-worker`, and `Dockerfile.edge`, `compose.edge.yaml`, `.env.edge.prod*`,
and the `ML_WORKER_*` / `WORKER_*` / `ML_API_*` / `API_*` prefixes keep their
legacy names. Only the Python package and the entrypoint change.

## Feature parity ledger

The ownership table above maps *files*. This ledger maps *user-observable
behaviour* from the original repository onto its v2 owner, so `edge/` can be
deleted without silently dropping a capability. It is the parity criterion for
the v2 cutover.

**Baseline: `eldercare-fall-ml` at committed `aeed6a8`.** Uncommitted work in
that checkout is out of scope for parity except where a row says otherwise.

Disposition vocabulary:

- `ported` — behaviour lives in the v2 owner and is covered by a behaviour test.
- `tracked-deferred` — intentionally not ported yet; a GitHub issue tracks it.
- `out-of-scope (uncommitted)` — present only as uncommitted work in the
  baseline checkout, so it is not part of the committed parity baseline.

Missing-capability rule: a missing **runtime feature** is reported to the user
and then restored; a missing **script or tool** is filed as a GitHub issue and
deferred; a missing **behaviour-coverage test** is restored as part of the
feature it proves. Developer-convenience harnesses are deferred with the tools.

| Capability | v2 owner | Behaviour test | Disposition |
| --- | --- | --- | --- |
| RTSP ingest and reconnect policy | `worker/pipeline/ingest/rtsp.py`, `worker/pipeline/ingest/lifecycle.py` | `tests/test_worker_ingest_rtsp.py`, `tests/test_worker_ingest_lifecycle.py` | ported |
| CPU decode adapter and capability probe | `worker/adapters/decode/cpu_av/adapter.py`, `worker/adapters/decode/cpu_av/probe.py` | `tests/test_worker_decode_cpu.py`, `tests/test_worker_opencv_decode_probe.py` | ported |
| NVDEC decode probe | `worker/adapters/decode/nvdec_cuvid/probe.py` | `tests/test_worker_nvdec_probe.py` | ported |
| CUDA device selection and verification | `worker/adapters/device/cuda/probe.py` | `tests/test_worker_cuda_device_probe.py` | ported |
| Model registry, warmup, inference | `worker/adapters/model/registry.py`, `worker/interfaces/serving.py` | `tests/test_worker_production_boot_dependencies.py` | ported |
| Fall interpretation and latching | `worker/domains/fall/` | `tests/test_domains_fall.py`, `tests/test_worker_fall_decider.py` | ported |
| Bed-exit interpretation and latching | `worker/domains/bed_exit/` | `tests/test_domains_bed_exit.py`, `tests/test_worker_domains_bed_exit.py` | ported |
| Incident cooldown and duplicate suppression | `worker/pipeline/decision/incident_manager.py` | `tests/test_worker_incident_manager.py` | ported |
| Relay heartbeat and alert egress | `shared/events/edge_ingest_client.py` | `tests/test_e2e_night_bed_exit_relay.py` | ported |
| Evidence clip recording and finalisation | `worker/pipeline/output/evidence/clip_recorder.py` | `tests/test_clip_recorder.py` | ported |
| Snapshot store | `worker/pipeline/output/evidence/snapshot_store.py` | `tests/test_snapshot_store.py` | ported |
| Evidence outbox and export delivery | `worker/pipeline/output/evidence/evidence_runtime.py` | `tests/test_worker_evidence_export_composition.py` | ported |
| Worker config load and LKG fallback | `worker/runtime/config/loader.py` | `tests/test_ml_worker_yaml_config.py` | ported |
| Runtime status and diagnostics | `worker/runtime/telemetry/status_store.py`, `worker/runtime/telemetry/runtime_status_sender.py` | `tests/test_worker_runtime_status_sender_composition.py` | ported |
| CLI entrypoint and bounded-run cap | `worker/__main__.py` | `tests/test_worker_entrypoint.py`, `tests/test_worker_max_frames_per_camera_composition.py` | ported |
| Per-frame perception: tracking, scene state, window buffering | `worker/pipeline/perception/`, `worker/pipeline/camera_pipeline.py` | `tests/test_perception_observation_builder.py`, `tests/test_demo_tracking.py`, `tests/test_worker_camera_pipeline_pump.py` | ported |
| Debug overlay rendering | `worker/pipeline/output/overlay.py`, `worker/pipeline/output/_overlay_primitives.py` | `tests/test_worker_overlay_renderer.py`, `tests/test_worker_overlay_primitives.py` | ported |
| Operator MJPEG live view | `worker/pipeline/output/mjpeg_server.py`, `worker/pipeline/output/live_view.py`, composed in `worker/runtime/worker.py` | `tests/test_worker_live_view_composition.py` | ported |
| GPU stability preflight installer | — | — | tracked-deferred (`scripts/edge-preflight/gpu-stability-install.sh`, untracked at baseline; [#6](https://github.com/SeniorAILab/eldercare-fall-ml-v2/issues/6)) |
| GPU telemetry preflight | — | — | tracked-deferred (`scripts/edge-preflight/gpu-telemetry.sh`, untracked at baseline; [#7](https://github.com/SeniorAILab/eldercare-fall-ml-v2/issues/7)) |

### Baseline uncommitted work

`eldercare-fall-ml@aeed6a8` carries ten uncommitted entries in its checkout.
Two are tracked above; the remaining eight are `out-of-scope (uncommitted)`:

Listed as a bullet list rather than a table: a two-column table row whose first
cell is a backticked `edge/` path is reserved for the ownership map above.

- tracked-modified: `compose.edge.yaml`
- untracked: `docs/research/blackwell-gsp-halt-edge-gpu.md`
- untracked: `backend/app/features/AGENTS.md`
- untracked: `edge/evidence/AGENTS.md`
- untracked: `edge/runtime/AGENTS.md`
- untracked: `.codex/`
- untracked: `auto.crt`
- untracked: `auto.key`

## Open gaps

These are known divergences between the plan and the tree. They are recorded
here so nobody documents an intention as a fact.

**One parity row cannot be fully proven on macOS.** The ledger calls a row
`ported` when a behaviour test covers it. Running the cited tests for every
`ported` row on this host gives 214 passed, 2 failed, and both failures belong
to one row:

| Cited test | Row it is evidence for | Why it cannot run |
| --- | --- | --- |
| `tests/test_clip_recorder.py::test_clip_recorder_finalizes_atomic_manifest_with_pre_and_post_window` | Evidence clip recording and finalisation | resolving `/proc/self/fd/N` fails, so no manifest is published |
| `tests/test_clip_recorder.py::test_clip_recorder_fsyncs_media_and_manifest_before_staging_cleanup` | Evidence clip recording and finalisation | reads `/proc/self/fd`, which macOS does not have |

This row is genuinely Linux-only, and not by accident:
`worker/pipeline/output/evidence/evidence_media.py` hands `ffprobe` a
`/proc/self/fd/{descriptor}` reference to an already-open inode so the probe
cannot be TOCTOU-swapped for a different file. That is production code, so the
capability itself requires `/proc` — the deploy target is a Linux container, so
this is a deliberate floor rather than a portability bug. `tests/test_clip_recorder.py`
documents it in its module docstring and keeps these two cases specifically to
pin that floor.

The row stays `ported`: the evidence exists and CI runs on Ubuntu. What is
missing is *local* proof, which matters because this branch was developed and
gated on macOS. Treat it as CI-verified rather than dev-verified, and do not
read its local failures as a parity gap.

All eight local failures outside the vendor-drift one trace to this single
design decision, not to a missing tool: five in
`tests/test_clip_export_reconciliation.py`, two in `tests/test_clip_recorder.py`,
one in `tests/test_evidence_trust_boundaries.py`. `ffprobe` **is** installed on
this machine, so "install ffprobe" does not clear any of them. Note that the
production error text says otherwise — it reports `ffprobe unavailable` for an
unresolvable `/proc/self/fd/N` too, which is
[#11](https://github.com/SeniorAILab/eldercare-fall-ml-v2/issues/11) and is a
misreported reason rather than a second cause.

**Snapshot store used to be listed here too, and no longer is.** Three of its
cited tests also failed on macOS, but for an entirely different reason: they
read `/proc/self/fd` purely as *test instrumentation* to resolve a descriptor
back to a path and to count open descriptors. `snapshot_store.py` itself never
touches `/proc`, so nothing about the capability was Linux-only. Those tests now
use `fcntl(F_GETPATH)` and `/dev/fd` on macOS and the same `/proc` reads on
Linux, so the row is dev-verified on both. The distinction worth keeping: a
cited test failing on your machine may be pinning a real runtime floor, or may
just be instrumentation that was never written to be portable, and the two look
identical from the test report.

**Resolved: the shipped example config pinned a fall contract nothing produces yet.**
`worker/ml-worker.example.yaml` used to pin `models.fall.schema_version: 2`
and the current coco17 `preprocessing_identity`, while
`models/fall/lstm/metadata.yaml` declares neither — so it loads as
`LEGACY_SCHEMA_VERSION` (1) with the legacy identity, and the pinned pair was
refused. Neither side was malformed: the loader supports both generations as
first-class cases (`worker/adapters/model/lstm_manifest.py`,
`SUPPORTED_PREPROCESSING_IDENTITIES`), but `eldercare-dataset-ops` currently
emits `schema_version: 1` for fall and no preprocessing identity
(`ml/training/model_artifacts.py::build_fall_lstm_metadata`, which never
writes either field), so schema_version 2 is not a contract any export path
produces today — the example was documenting an aspirational target, not the
artifact it ships with.

The example was corrected to pin the legacy contract
(`schema_version: 1`, `legacy-coco17-xyc-frame-normalized-zero-fill-v1`) that
the shipped artifact and current training pipeline actually satisfy, so
copying the example boots the fall model it ships with. When
`eldercare-dataset-ops` starts exporting fall artifacts with
`schema_version: 2` and the current coco17 identity, re-export
`models/fall/lstm` from that pipeline and bump the example's pins back to the
v2 values at the same time — the fail-closed validation in
`_validate_expected_identity` (`worker/adapters/model/torch_lstm_fall.py`)
stays unchanged either way; only the pinned values move. The regression is
covered by
`tests/test_worker_real_warmup_no_stub.py::test_example_config_fall_contract_matches_the_local_artifact`.
Tracked in [#8](https://github.com/SeniorAILab/eldercare-fall-ml-v2/issues/8).

**Operator scripts hang on heredocs larger than `PIPE_BUF`.**
Bash 5.3.15 writes a heredoc body into a pipe before exec'ing the reader, so a
body over `PIPE_BUF` blocks forever against a pipe nobody is draining — bash
never execs the command. The boundary is exact: on macOS (`PIPE_BUF` 512) a
512-byte body passes and 513 hangs; bash 3.2.57 stages heredocs in a temp file
and is unaffected at any size. Four bash scripts under `scripts/` pin
`#!/bin/bash` for this reason, which on macOS resolves to 3.2.57.

**That pin does not help on Linux.** There `PIPE_BUF` is 4096 and `/bin/bash` is
itself a modern bash, so only the threshold moves. Measured at 4096, one script
is exposed on a Linux edge host: `ml-worker-single-rtsp-bedexit-e2e.sh:139`
carries a 7162-byte heredoc. Moving that body into a file is the durable fix.

`tests/test_shell_script_heredoc_contract.py` enforces the rule at the 512-byte
threshold. CI is unaffected: the only script a test executes is
`ml-worker-real-rtsp-bedexit-e2e.sh --render-config`, whose two heredocs on that
path are 527 B and 797 B, both under the Linux threshold. Tracked in
[#9](https://github.com/SeniorAILab/eldercare-fall-ml-v2/issues/9), which
carries the reproduction and the measured size census.

**`worker/pipeline/camera_pipeline.py` exists, but does not own what the plan
said it would.** This entry previously asserted that the file was absent, which
was false: it is tracked, 163 lines, and was added in `6ce0bbc`. It holds
`CameraPipelinePump`, which drives one camera's frame → extraction → decision
path.

What is still true is the ownership claim. The migration plan names this file as
the owner of `edge/runtime/camera_worker.py`'s orchestration, and that
orchestration remains split across `worker/pipeline/ingest/lifecycle.py`
(supervision and reconnect), `worker/pipeline/analytics/composite.py`
(extraction fan-out), and `worker/runtime/worker.py` (composition and restart
policy). The ownership row above describes that split because that is what is on
disk. Either move the orchestration into the pump, or amend the plan's Scope
table — do not describe the file as owning something it does not.

**No `worker/runtime/supervisor.py` and no `worker/pipeline/bus/latest_frame.py`.**
The plan named both. Supervision landed in `worker/pipeline/ingest/lifecycle.py`
as `IngestSupervisor` with restart policy in `worker/runtime/worker.py`, and the
latest-frame store landed in `worker/pipeline/output/live_view.py` as
`LatestFrameStore` (a non-consuming latest-value store, not a queue). The rows
above reflect the real locations.

**ADR-0001 source-packet preservation is complete for primary clean clips.**
The worker keeps bounded encoded-packet history per camera and remuxes one
keyframe-aligned stream epoch/configuration without transcoding. Decoded frames
remain analysis and snapshot taps; transformed clips are separate derivatives
and never replace the preserved clean clip.

**Live overlays and annotated evidence share one canonical scene.**
`worker/types/overlay_scene.py` owns the versioned, hardware-neutral primitives
and explicit present/stale/missing/not-evaluated semantics.
`OverlaySceneBuilder` creates those primitives from either frozen live
observations or persisted analysis/decision traces; renderers do not run policy
or inference. The OpenCV CPU still renderer and event-only FFmpeg CPU MP4 path
consume that same scene contract with fixed colors, layout, transforms, and
host-independent CJK raster cells.

Annotated MP4s are bounded, content-addressed, immutable derivative media.
Publication verifies the clean source identity, fsyncs and atomically renames
media before a short central-DB linkage transaction, and startup reconciliation
converges pending/mutated/orphan state without changing clean evidence. The
authenticated clip artifact projection exposes clean/analysis/annotated state;
an unavailable or invalid annotation falls back to descriptor-pinned clean
playback. Queue count/source bytes, frame memory, scene count, duration, render
time, output bytes, and aggregate derivative disk usage all have explicit
limits. No continuous second encoder or second inference pass is involved.

**Encoder-lifecycle work is risk reduction, not a GPU fix.** The per-camera
encoder session, segment ring, and their instrumentation reduce the window in
which a hardware encoder is left in a bad state and make faults observable.
They do not diagnose or repair any Xid GPU fault, and nothing in this migration
may be presented as fixing one. Host driver and CUDA faults are handled by
[`docs/runbooks/driver-cuda-alignment.md`](runbooks/driver-cuda-alignment.md).

## Related

- [`docs/decisions/`](decisions/) — decision records, index in
  [`decisions/README.md`](decisions/README.md)
- [`docs/runbooks/worker-migration-rollback.md`](runbooks/worker-migration-rollback.md)
  — image-digest rollback with volume preservation
- [`worker/AGENTS.md`](../worker/AGENTS.md) — worker package rules and the
  per-layer import ceilings

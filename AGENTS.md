# PROJECT KNOWLEDGE BASE

**Generated:** 2026-10-06
**Commit:** 30d9ce5
**Branch:** main

Python/uv + React monorepo for fall and bed-exit detection. Three deployable
instances (`front`, `backend`, `worker`) plus `shared` and the canonical
cross-repo `contracts` leaf mirrored into `eldercare-dataset-ops`. Import-linter
(`[tool.importlinter]` in `pyproject.toml`) owns the boundaries. Training lives
in `eldercare-dataset-ops`. Local weights stay under `models/` and are never
committed.

## Package Boundaries

| Package | Ownership |
| --- | --- |
| `front` | React/Vite SPA. Feature-sliced: `src/features/*`, `src/shared/{ui,api}`, `src/app`. Talks to the backend only over same-origin `/api/v1`. |
| `backend` | FastAPI gateway. Eleven vertical slices under `app/features/*` (router + store); `app/lifespan.py` is its composition root. |
| `worker` | DeepStream Flow worker. SDK media plane, CPU domain decisions, evidence, and relay egress. |
| `shared` | `shared.events` (backend↔worker wire, publish-once delivery queue), `detection_policies` (the one fall/bed-exit policy parser for both sides), `release_identity`. Imports stdlib and `contracts` only. |
| `contracts` | ADR-0006 canonical typed-vocabulary leaf; mirrored byte-for-byte into `eldercare-dataset-ops`. Imports no other repo package; `__init__` re-exports 34 names pinned by `tests/test_contract_symbol_exports.py`. |
| `tests` | Flat pytest tree: contracts, boundary coverage, and workflow-pinning tests. `tests_support` is the test-only helper package, the one place `backend` and `worker` are imported together. |

`backend` and `worker` do not import each other. HTTP relay is the command/event
boundary, one-way from worker to backend. The backend alone opens
`/var/lib/seeon-state/edge.sqlite3` through `backend.app.edge_db`; the worker has
no database. That file is persistence, never worker state or polling IPC.

Directory names are `front`/`backend`/`worker`. Deployment images keep the
legacy identity: `ml-api` (`Dockerfile.backend`, `ML_API_`/`API_*`) and
`ml-worker` (`Dockerfile.edge`, `WORKER_*`/`ML_WORKER_*`). Do not rename those
prefixes. `front` is built into the backend image and served at `/`.

`flow` is the only production worker profile. The DeepStream Flow owns capture,
decode, inference, and tracking inside the SDK. Only
`worker/adapters/deepstream/` may import `pyservicemaker` or `pyds`, and those
imports stay lazy so host modules import without the SDK. `worker.runtime` is
the sole composition root. The worker is an RTSP client only.

## Code Map

| Surface | Path | Role |
| --- | --- | --- |
| Backend factory | `backend/app/main.py` | `create_app()` registers 19 feature routers under `/api/v1`, seeds `app.state.edge_relay_token` (handlers never re-read the env), mounts `front` dist. Unversioned probes stay at `/health/live`, `/health/ready`, and `/health/release-identity`. |
| Worker CLI | `worker/__main__.py` | `python -m worker`, the sole production command. Parses flags, loads config, constructs `WorkerRuntime`. Exit codes: 0 clean, 1 generic, 2 config, 3 refuse-to-start, 4 fatal accelerator (never caught to retry). Containers pass the mounted volume as `--state-dir`. |
| Composition root | `worker/runtime/worker.py` | Composes the Flow plane, CPU policy, evidence, and relay. |
| Flow runtime | `worker/runtime/flow/` | Media-plane lifecycle, metadata admission, policy pump, and evidence handoff. The Flow process never imports torch or ultralytics. |
| Vendor adapter | `worker/adapters/deepstream/` | Lazy DeepStream Service Maker integration and vendor-metadata conversion. |
| Domain decisions | `worker/domains/` | CPU fall and bed-exit decisions. `DETECTION_MODULE_REGISTRY` (`fall.v2`, `bed_exit.v1`) is the extension surface; `EpisodeAuthority` alone mints event identity. |
| Engine build and gate | `worker/tools/` | Flow engines are image-owned artifacts: `edge_engine_build.py` builds the nvinfer engine ahead of source activation, never at boot; boot verifies digests and deployed-batch identity before accepting sources and rejects a mismatch. `export_pose_onnx.py` exports the dynamic-batch pose ONNX used by the engine build gate. `fetch_models` (compose service `edge-model-fetch`) pulls weights. `worker.tools` is banned from every production layer. |
| Evidence | `worker/pipeline/output/evidence/` | Smart record actor, clip publication, sealed sidecar, durable stager, delivery queue, and snapshot store. |
| Worker HTTP | `worker/pipeline/output/` | `/probe`, `/replay`, `/stream/`, `/snapshot/`, `/overlay/` on `127.0.0.1:8090` by default; `ml-api` is the only consumer, server-side. |
| Event wire | `shared/events/` | Schemas and `edge_ingest_client.py` (events and clip receipts to the backend over relay HTTP), and the worker's `delivery_queue.py`. |
| Release identity | `shared/release_identity.py` | `EDGE_DATABASE_SCHEMA_VERSION = 19`, read by backend bootstrap and health and by the worker pair check. A bump moves the DDL and `front/src/shared/releaseIdentity.ts` with it. |
| Dashboard | `front/src/app/App.tsx` | `AuthGate` + `Dashboard`. Pages: events, operations, settings. URL state in `dashboardLocation.ts`. |
| SQLite foundation | `backend/app/edge_db/` | Schema 19 (compact ten-table plus six execution-record tables), the create-or-extend bootstrap, and ownership. The backend writes the fifteen application tables; the bootstrap alone writes `schema_migrations`. `python -m backend.app.edge_db` is the sole DDL owner (create 19 or extend exact 18) and also bootstraps the telemetry file `edge-diagnostics.sqlite3`. |
| CI and release | `.github/workflows/`, `scripts/` | `ci.yml` (`secrets`, `lint`, four `test` shards; `ci-ok` is the single required check) downloads no model weights. Release is an annotated `seeon-edge-v<semver>` tag checked by `scripts/release_guard.py`; `scripts/edge_image_plan.py` owns build-vs-reuse. Workflow YAML is pinned by tests, so a workflow change needs its test change. |
| Docs | `docs/` | `architecture.md`, `decisions/` (ADR 0001-0008), `runbooks/`. |

Read the nearest scoped `AGENTS.md` before changing a package: `worker/`,
`backend/app/`, `shared/`, `contracts/`, `front/`, `tests/`, `tests_support/`,
`scripts/`, and `docs/` each have one, most with deeper files beneath them.

## Commands

From the repo root:

```bash
uv sync
uv run pytest -q
uv run pytest -q -m "not real_stack and not heavy and not integration"  # CI filter
uvx ruff check .
uv run --group lint lint-imports  # architecture boundaries
uv run --group lint mypy contracts shared
docker build -f Dockerfile.backend .
docker build -f Dockerfile.edge .
pnpm --dir front install --frozen-lockfile && pnpm --dir front test
pnpm --dir front build && pnpm --dir front lint
```

## Conventions

- Keep `contracts` in sync with `eldercare-dataset-ops` (ADR-0006). Domain
  decision math stays worker-internal.
- Cameras are registered at runtime through the dashboard registry. Do not seed
  them from env, YAML, or a backend `cameras` pull.
- Use `uv`. Re-run `lint-imports` after any import-boundary change.
- Dashboard UI: icons and buttons over text; status is an icon, secondary flows
  are popups, overlay subjects are per-subject toggles that persist. See
  `front/AGENTS.md` "Owner UI preferences".

## Anti-patterns

- No `backend`↔`worker` imports. Relay HTTP only.
- No RTSP publisher, MediaMTX, or FFmpeg stream server on the worker.
- No committed model artifacts or training code.
- No real-stack E2E in CI. Mark those tests `real_stack`; `ci.yml` deselects them.
- No required CI check behind `if:` or `paths:` (a skipped check reads green); no
  secrets, RTSP URLs, or camera IPs in docs, issues, or shell history.
- No `docker compose down -v`, `edge-state` deletion, or direct SQL repair of
  `edge.sqlite3`; no mutable image tags (`latest`, `:dev`) or hand-written digests.
- 기사님한테 회사 숙제 시키기: baking company-known deploy values (backend URL)
  into a field-tech form. Company-known values go in env/image; only site-local
  values (facility id, token) stay in the UI.
- 조립 루트 스텁 배선: production seam defaults that are always-fail stubs so
  `worker/runtime` boots with a dead feature. Seam defaults are `None`; missing
  wiring must refuse to start (ADR-0002). Always-fail stubs belong in tests only.
- 암묵 정책: model choice, Flow parser behavior, extract schedule, or result
  priority hidden in branch fall-through or dict insertion order. Lift the
  decision to an explicit owner (registry, config, declaration).
- No JSON state stores for application data. Mutable application state belongs in
  the backend-owned SQLite database (`backend/app/edge_db`); the inference-runtime
  slot holds no database at all (ADR-0005) and uses only its approved bounded
  file surfaces: the publish-once delivery queue, a verified bounded config read
  cache, media-integrity sidecars, zero-payload lock inodes, and startup-purged
  scratch. Content-addressed evidence files (`manifest.json`, `scene-index.json`,
  snapshots) are the exception.
- 침묵하는 `extra=` 로그: `worker/__main__.py` `basicConfig` renders
  `%(message)s` only. Operator-visible fields (camera_id) must live in the
  message string. Assert `record.getMessage()`, not LogRecord extras.

## Notes

- Image names and `ML_*`/`API_*`/`WORKER_*` prefixes are frozen. Renaming
  breaks `.env.edge.prod`, GHCR, and `contracts/worker_config.py`.

<!-- BEGIN CRAFT-SKILLS INIT DEVELOPMENT FLOW -->
## Development Flow Recipe

Use an issue-driven loop for all repository work:

1. Open or select one GitHub issue describing the change.
2. Before starting new work, confirm the base worktree has no tracked local changes, then run `git fetch origin` and `git pull --ff-only` on the default branch so the work starts from the latest remote commit. Never pull across uncommitted work or discard user changes; create/reuse the task worktree only after the base is current.
3. Never commit directly on `main`. Use a worktree when you need isolation: `git wt <name>` creates (or reuses) a named worktree off the updated default branch. Reuse a small fixed pool (e.g. `lane-1`~`lane-3`) rather than making a new one per issue.
4. Plan first for non-trivial work: write the intended change, affected files, verification, and rollback note before editing.
5. Fan out into small PRs when a change spans unrelated domains, mixes assets with logic, or needs independent review lanes.
6. Attach review evidence to each PR: tests or checks run, screenshots/transcripts for user-facing behavior, and the issue or planning links that justify the change.
7. Merge only after review. If the user explicitly asks to record a durable decision, hand off to the `document` skill and use `docs/decisions/` as the destination.

Conventions agents must follow:

- Keep each change scoped to its issue. When work — planning, a requirements interview, or implementation — surfaces an out-of-scope problem (a new topic, unrelated bug, or follow-up idea beyond the current issue), open a new GitHub issue for it with one Type label instead of expanding the current change.
- Plan before editing non-trivial code.
- Prefer fan-out PRs over broad mixed-purpose PRs.
- Include review evidence before requesting/performing review.
- Do not merge before review.
- Do not create or require ADRs unless the user explicitly asks for ADRs.
<!-- END CRAFT-SKILLS INIT DEVELOPMENT FLOW -->

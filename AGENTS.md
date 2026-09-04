# PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-25
**Commit:** 1d79e2f
**Branch:** seeon-edge-ds-09

Python/uv + React monorepo for fall and bed-exit detection. Three deployable
instances (`front`, `backend`, `worker`) plus `shared` and the canonical
cross-repo `contracts` leaf mirrored into `eldercare-dataset-ops`. Import-linter
(`[tool.importlinter]` in `pyproject.toml`) owns the
boundaries. Training lives in `eldercare-dataset-ops`. Local weights stay under
`models/` and are never committed.

## Package Boundaries

| Package | Ownership |
| --- | --- |
| `front` | React/Vite SPA. Feature-sliced: `src/features/*`, `src/shared/{ui,api}`, `src/app`. |
| `backend` | FastAPI gateway. Vertical slices under `app/features/*` (router + store). |
| `worker` | RTSP inference client. Layers: types / interfaces / adapters / pipeline / domains / runtime. |
| `shared` | `shared.events` (backend↔worker wire). |
| `contracts` | ADR-0006 canonical typed-vocabulary leaf; mirrored byte-for-byte into `eldercare-dataset-ops`. |
| `tests` | pytest contracts and boundary coverage. |

`backend` and `worker` do not import each other. HTTP relay is the command/event
boundary. The backend alone opens `/var/lib/seeon-state/edge.sqlite3` through
`backend.app.edge_db`; the worker has no database. That file is persistence,
never polling IPC.

Directory names are `front`/`backend`/`worker`. Deployment images keep the
legacy identity: `ml-api` (`Dockerfile.backend`, `ML_API_`/`API_*`) and
`ml-worker` (`Dockerfile.edge`, `WORKER_*`/`ML_WORKER_*`). Do not rename those
prefixes. `front` is built into the backend image and served at `/`.

GPU serving is in-process behind `worker/interfaces/serving.py`
(`worker/adapters/model/in_process.py`) except under `nvidia`, where the native
DeepStream child owns media/inference and the Python parent owns supervision
infra fails fast (ADR-0002). The worker is an RTSP client only.

## Code Map

| Surface | Path | Role |
| --- | --- | --- |
| Backend factory | `backend/app/main.py` | `create_app()` registers feature routers under `/api/v1`, seeds `app.state.edge_relay_token`, mounts `front` dist. Health stays at `/health/live` and `/health/ready`. |
| Worker CLI | `worker/__main__.py` | `python -m worker`. Parses flags, loads config, constructs `WorkerRuntime`. |
| Composition root | `worker/runtime/worker.py` | Selects the host coordinator/camera pipeline or the `nvidia` native media plane/policy pumps; wires evidence, MJPEG, and relay. |
| Worker profile routing | `worker/runtime/profile/` | `ML_WORKER_PROFILE` selects the canonical stage/memory descriptor; `nvidia` routes production construction to the native child instead of host adapters. |
| Host cross-cam inference | `worker/pipeline/inference_coordinator.py` | Non-`nvidia` only: `CapabilityInferenceCoordinator` drains latest-only frames and owns host pose forwards. The `nvidia` path never constructs it. |
| Evidence | `worker/pipeline/output/evidence/` | Clip recorder, packet ring, snapshot store, durable stager, filesystem delivery queue. |
| Dashboard | `front/src/app/App.tsx` | `AuthGate` + `Dashboard`. Pages: events, operations, settings. |
| Event wire | `shared/events/` | Schemas and `edge_ingest_client.py` (facts + heartbeats to the Event API). |
| SQLite foundation | `backend/app/edge_db/` | Schema 18 (the only schema), the create-only bootstrap, and ownership. The backend writes the nine application tables; the bootstrap alone writes `schema_migrations`. |

On non-`nvidia` profiles, the per-frame path after the coordinator is
`worker/pipeline/camera_pipeline.py` and `worker/pipeline/perception/` into
`worker/domains/` (fall, bed-exit). Under `ML_WORKER_PROFILE=nvidia`, the native
child owns decode/inference/parser/association and `NativePolicyPump` feeds the
Python-owned policy state. Read the nearest scoped `AGENTS.md` before changing
a package.

## Commands

From the repo root:

```bash
uv sync
uv run pytest -q
uvx ruff check .
uv run --group lint lint-imports
docker build -f Dockerfile.backend .
docker build -f Dockerfile.edge --target runtime .
pnpm --dir front install --frozen-lockfile && pnpm --dir front test
```

Architecture boundaries: `uv run --group lint lint-imports`. Contract-symbol
exports: `tests/test_contract_symbol_exports.py`. Docs live in
`docs/architecture.md`, `docs/decisions/`, `docs/runbooks/`.

## Conventions

- Keep `contracts` in sync with `eldercare-dataset-ops` (ADR-0006). Perception
  math under `worker/pipeline/perception/features` stays worker-internal.
- Worker→backend command/event traffic is one-way over relay HTTP. The backend's
  local `edge.sqlite3` is never worker persistence or polling IPC.
- Cameras are registered at runtime through the dashboard registry. Do not seed
  them from env, YAML, or a backend `cameras` pull.
- Use `uv`. Re-run `lint-imports` after any import-boundary change.

## Anti-patterns

- No `backend`↔`worker` imports. Relay HTTP only.
- No RTSP publisher, MediaMTX, or FFmpeg stream server on the worker.
- No committed model artifacts or training code.
- No real-stack E2E in CI. Mark those tests `real_stack`; `ci.yml` deselects them.
- 기사님한테 회사 숙제 시키기: baking company-known deploy values (backend URL)
  into a field-tech form. Company-known values go in env/image; only site-local
  values (facility id, token) stay in the UI.
- 조립 루트 스텁 배선: production seam defaults that are always-fail stubs so
  `worker/runtime` boots with a dead feature. Seam defaults are `None`; missing
  wiring must refuse to start (ADR-0002). Always-fail stubs belong in tests only.
- 암묵 정책: model choice, encoder fallback, extract schedule, or result
  priority hidden in branch fall-through or dict insertion order. Lift the
  decision to an explicit owner (registry, config, declaration).
- No JSON state stores for application data. Mutable application state belongs in
  the backend-owned SQLite database (`backend/app/edge_db`); the inference-runtime
  slot holds no database at all (ADR-0005) and uses only its approved bounded
  file surfaces: the publish-once delivery queue, a verified bounded config read
  cache, media-integrity sidecars, zero-payload lock inodes, and startup-purged
  scratch. Content-addressed
  evidence files (`manifest.json`, `scene-index.json`, snapshots) are the exception.
- 침묵하는 `extra=` 로그: `worker/__main__.py` `basicConfig` renders
  `%(message)s` only. Operator-visible fields (camera_id) must live in the
  message string. Assert `record.getMessage()`, not LogRecord extras.

## Notes

- Image names and `ML_*`/`API_*`/`WORKER_*` prefixes are frozen. Renaming
  breaks `.env.edge.prod`, GHCR, and `contracts/worker_config.py`.
- Serving seam is a batch-input contract so a future networked serving service
  can swap in. That swap is deferred (ADR-0002).

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

# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-10
**Commit:** 0ea0269
**Branch:** main

Python/uv + React monorepo for fall and bed-exit detection, organized as three
deployable instances (`front`, `backend`, `worker`) plus a `shared` library, each
owning its subtree to 2 levels. Boundaries are enforced by import-linter
(`[tool.importlinter]` in `pyproject.toml`) — not a hand-rolled walker.

## Package Boundaries

| Instance / Package | Ownership |
| --- | --- |
| `front` | React/Vite SPA, feature-sliced: `src/features/*`, `src/shared/{ui,api}`, `src/app` |
| `backend` | FastAPI gateway, vertical-slice: `app/features/{cameras,clips,evidence,relay,status}` (router+store), `app/core`, `app/shared` |
| `worker` | RTSP inference worker, layered: `types/interfaces/adapters/pipeline/domains/runtime` (see `docs/architecture.md` "Layers") |
| `shared` | `shared.events` (backend↔worker wire code) and dependency-light `shared.edge_db` persistence contracts |
| `contracts` | ADR-0004 vendored from `eldercare-dataset-ops` (top-level shared leaf; `test_vendor_drift` firewall) |
| `tests` | pytest contracts and boundary coverage |

`edge/` was the pre-migration legacy tree that `worker/` replaced. It has been
deleted (todo 34 "atomic edge deletion"). See `docs/architecture.md`
"Source-to-target ownership" for the historical file-by-file mapping and
"Feature parity ledger" for the capability-level disposition — those rows are
migration citations, not operator instructions.

`backend` and `worker` are import-independent. HTTP remains their only
command/event notification boundary; on one co-located Linux edge release unit
they may also open `/var/lib/seeon-state/edge.sqlite3` through
`shared.edge_db`, with disjoint table-family writers. The database is
persistence, never polling IPC. Both may import `contracts` and `shared`. Training belongs to
`eldercare-dataset-ops`; local runtime artifacts belong under `models/` and are
never committed. GPU serving is in-process behind `worker/interfaces/serving.py`
(`worker/adapters/model/in_process.py`); the seam exposes a **batch-input
contract** so a future networked batched serving service (50-camera scale) can
swap in without a rewrite (deferred; see ADR-0002). Required GPU/NVDEC infra
fails fast (no silent CPU/OpenCV fallback) per ADR-0002.

## Runtime Topology

- `backend` is the FastAPI control/status/relay gateway (`ml-api` image). Product
  routes live under `/api/v1`; health probes stay at `/health/live` / `/health/ready`.
- `worker` consumes configured RTSP streams, runs model/domain logic in-process
  via `worker/interfaces/serving.py`, and relays facts to `backend` at
  `/api/v1/relay/*`.
- `worker` is not an RTSP server. Do not add a stream publisher, MediaMTX,
  FFmpeg publisher, or synthetic RTSP runtime surface.

## Deployment identity (dir ↔ image)

Directory/instance names are `front`/`backend`/`edge`. The deployment **images**
and the `ML_API_`/`ML_WORKER_` and `API_*`/`WORKER_*` env prefixes intentionally
keep their **legacy identity** — renaming would break `.env.edge.prod`, the GHCR
image names, and the vendored `contracts/worker_config.py` (ADR-0004):

| Instance | Image / service | Dockerfile | Env identity |
| --- | --- | --- | --- |
| `front` | served by `backend` at `/` | `Dockerfile.backend` (build stage) | — |
| `backend` | `ml-api` | `Dockerfile.backend` | `Settings(env_prefix="ML_API_")`, `API_*` |
| `edge` | `ml-worker` | `Dockerfile.edge` | `WORKER_*`, `ML_WORKER_*` |

## CODE MAP

| Surface | Entry / Owner | Role |
| --- | --- | --- |
| Backend bootstrap | `backend/app/main.py`, `backend/app/lifespan.py` | FastAPI control/status/relay gateway. |
| Backend slices | `backend/app/features/*` | Per-capability router + service + store. |
| Worker bootstrap | `worker/__main__.py`, `worker/runtime/worker.py` | CLI (`python -m worker`), `WorkerRuntime` composition root, bootstrap stages. |
| Per-frame path | `worker/pipeline/camera_pipeline.py`, `worker/pipeline/perception/` | Frame → observation → domain signal. |
| Domains | `worker/domains/` | Fall and bed-exit interpretation/latching. |
| Serving seam | `worker/interfaces/serving.py`, `worker/adapters/model/in_process.py` | ServingClient interface + in-process runner provisioning. |
| Model adapters | `worker/adapters/model/` | Registry, device selection, warmup, inference adapters. |
| Event egress | `shared/events/edge_ingest_client.py` | Backend Event API facts and heartbeats. |
| Frontend | `front/src/` | Feature-sliced React SPA. |

## Commands

Run commands from the repository root:

```bash
uv sync
uv run pytest -q
uvx ruff check .
docker build -f Dockerfile.backend .
docker build -f Dockerfile.edge .
pnpm --dir front install --frozen-lockfile && pnpm --dir front test
```

Architecture boundaries are enforced by import-linter (`uv run --group lint lint-imports`,
also a pre-commit hook and CI step); contract-symbol exports by `tests/test_contract_symbol_exports.py`.
## DOCS & DECISIONS

- Architecture map: [`docs/architecture.md`](docs/architecture.md)
- Decision records (explicit-only): [`docs/decisions/`](docs/decisions/) — index in [`docs/decisions/README.md`](docs/decisions/README.md)
- Active plans: [`docs/exec-plan/active/`](docs/exec-plan/active/) · Research: [`docs/research/`](docs/research/) · Rules: [`docs/rules/`](docs/rules/)
- Operator runbooks: [`docs/runbooks/`](docs/runbooks/) — 엣지 호스트에서 손으로 실행하는 절차. 현재 4건: `local-e2e-rtsp-source.md`(핀된 클립으로 로컬 RTSP 소스 구성), `worker-migration-rollback.md`(`ML_WORKER_IMAGE` digest 롤백), `driver-cuda-alignment.md`, `edge-image-publish.md`(엣지 이미지 발행 — `main` 병합만으로는 이미지가 생기지 않는다).

## CONVENTIONS

- Keep `contracts` dependency-light and in sync with `eldercare-dataset-ops` (ADR-0004 vendoring); `worker/pipeline/perception/features` is worker-internal pure math.
- Keep the worker→backend command/event boundary one-way over relay HTTP. The
  approved persistence exception is one local `edge.sqlite3` for the single API
  and worker in one release unit: API writes only `control_*`/`qa_*`; worker
  writes only `runtime_*`/`evidence_*`/`derivative_*`; the one-shot migrator
  alone writes `schema_*` and executes DDL. Never poll SQLite as IPC or place it
  on NFS/NAS.
- Use `uv` and run `lint-imports` after any import-boundary change.

## ANTI-PATTERNS (THIS PROJECT)

- No `backend` import from `worker`, no `worker` import from `backend`; instances communicate only over relay HTTP.
- No RTSP publishing/runtime server surface; `worker` is an RTSP client only.
- No committed local model artifacts or training code.
- No expensive tests in CI: real-stack E2E (external binaries like mediamtx, live RTSP, GPU) must be marked (`real_stack`) and deselected from `ci.yml`; they run locally only. CI stays fast, deterministic, and free of external binary fetches.
- No pre-provisioned camera rosters: cameras are registered at runtime through the edge camera registry (dashboard CRUD, the SSOT). Do not seed or accept cameras from the backend ml-config `cameras` pull, env vars, or static YAML lists — that inbound path is legacy; do not extend it.
- 기사님한테 회사 숙제 시키기: 패키징(배포) 때 회사가 이미 아는 값(예: 백엔드 주소)을 현장 입력폼에 올려 기사님이 타이핑하게 만드는 것. 판별법: "이 값은 누가·언제 아는 값인가?" 처방: 회사가 배포 때 아는 값은 env/이미지에 굽고, 요양원 현장에서만 정해지는 값(시설 ID·토큰)만 UI 폼에 남긴다. 사례: 연결 설정 폼이 env 변수 4개(이벤트 URL/설정 URL/시설 ID/토큰)를 텍스트박스로 옮겨 놓았던 것 → 주소는 API_BACKEND_BASE_URL로 굽고 폼은 2칸으로 축소.
- 조립 루트(runtime)의 스텁 배선: 프로덕션 seam(주입 파라미터)의 기본값이 "항상 실패/no-op 스텁"인데, composition root(worker/runtime)가 그 파라미터를 생략해도 부팅이 멀쩡히 성공하는 것. 기능이 죽은 채로 배포되고, 증상은 진짜 장애처럼 위장된다. 판별법: "이 배선을 빼먹으면 부팅이 거부되는가, 아니면 조용히 해당 기능만 죽는가?" 처방: seam 기본값은 스텁이 아니라 `None`으로 두고 미배선이면 부팅 거부(fail-closed, ADR-0002) 또는 최소한 boot 로그 CRITICAL로 노출; 항상-실패 스텁이 필요하면 테스트 코드에만 둔다. 사례: dev MJPEG `/probe`가 `_unavailable_probe` 기본값으로 남아 스트림이 정상이어도 모든 카메라 등록이 `error_class=decode`로 실패 — 스트림/코덱 문제로 보여 원인 추적이 한참 걸렸다(#81에서 실제 probe 배선으로 수정).
- 코드가 몰래 정한 정책(암묵 정책): 시스템이 무엇을 할지 정하는 결정(어떤 모델을 쓸지, 어떤 인코더로 폴백할지, 무엇을 추출할지, 어느 결과가 우선인지)이 명시적 선언 없이 분기 낙하·dict 삽입 순서·하드코딩 리터럴에 숨어, 소유자가 모르는 동작이 생기는 것. 판별법: "이 동작이 존재한다는 걸 코드를 안 읽고도 알 수 있는가? 이 결정의 소유자가 이 결정이 내려졌다는 사실을 아는가?" 처방: 핵심 의사결정은 소유자가 찾고 바꿀 수 있는 명시적 자리로 끌어올린다 — 형태는 자유다(레지스트리·설정·선언 등; PROFILE_REGISTRY·DOMAIN_REGISTRY는 예시일 뿐 강제 아님). 금지되는 것은 특정 패턴의 부재가 아니라, 결정이 부수효과로 내려지는 것 자체다. 사례: 낙상 모델 미설정 시 분기 낙하로 random-forest가 조용히 실행(#43); person 박스가 스케줄 dict 삽입 순서 때문에 pose 박스를 암묵 덮어쓰기(#44); 추출 스케줄이 활성 도메인의 선언에서 유도되지 않고 하드코딩(#47); 인코더 폴백을 if-else로 박는 대신 프로파일별 후보 체인 선언으로 설계(#53).
- No JSON state stores: SQLite is the single home for mutable runtime state. The
  target edge topology has one physical `edge.sqlite3`; legacy per-runtime DBs
  remain temporarily only until the separately verified import/cutover. Credentials,
  the camera registry, pulled-config last-known-good, fault records, and latency
  samples belong in tables, not in hand-rolled atomic-write JSON files — do not
  add another one. Content-addressed evidence artifacts are the deliberate
  exception and stay as files: clip `manifest.json` and snapshot identity
  metadata travel with their media and are byte-verified, and image-baked model
  artifacts are read-only. The test is mutability — read-modify-written at
  runtime means a table; written once and hash-verified means a file. Several
  JSON stores predate this rule; migrating them is tracked in #35, so do not
  treat their existence as precedent.
- 침묵하는 `extra=` 로그: 이 리포의 콘솔 로깅은 `worker/__main__.py`의 `logging.basicConfig` 포맷(`"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`)이 `%(message)s`만 렌더링한다. 운영자가 콘솔에서 봐야 하는 진단값(camera_id 등)을 `extra=`에만 담으면 LogRecord 속성으로는 존재해도 실제 콘솔 출력에서는 완전히 침묵한다. 판별법: "이 값이 실제로 콘솔에 찍히는가, 아니면 `extra=`를 통해 LogRecord 속성으로만 존재하는가?" 처방: 진단값은 message 문자열에 %-style lazy interpolation으로 직접 포함하고, `extra=`는 구조적 로깅 소비자를 위해 병행 유지한다. 관련 테스트는 LogRecord 속성이 아니라 `record.message`/`record.getMessage()`(실제로 렌더링되는 값) 기준으로 assert한다. 사례: 재연결 경고가 camera_id 없이 콘솔에 찍힌 것(#115, PR #116으로 수정); 같은 결함의 ingest pacing 관측 로그가 리뷰 단계에서 지적된 것(PR #120, #122).

## NOTES

- Training belongs to `eldercare-dataset-ops`; `models/` is local and ignored.
- Read the nearest scoped `AGENTS.md` before changing a package.

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

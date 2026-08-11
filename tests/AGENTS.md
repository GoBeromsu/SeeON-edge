# TESTS KNOWLEDGE BASE

Own pytest coverage for the ML uv project, including dependency-boundary guards and fixtures.

## Local Ownership

- `test_contract_symbol_exports.py`: contract-symbol export checks (runner/tracker/worker_config). Import boundaries are enforced by import-linter (`uv run --group lint lint-imports`), not a pytest walker.
- `edge_worker_fixtures.py`, `demo_app_control_helpers.py`, `e2e_worker_relay_fixtures.py`: shared test helpers.
- `test_*`: package-specific and cross-boundary tests.
- `test_e2e_night_bed_exit_relay.py`: real-stack E2E (marked `real_stack`) -- synthetic RTSP via `mediamtx` + `ffmpeg` into the real worker/backend composition. Deselected in CI (`-m "not real_stack and not heavy and not integration"`) and skipped locally when `mediamtx` is not on PATH; see "Real-stack E2E" below.

## Imports

Allowed: any production package needed by the test under coverage, pytest helpers, and local fixtures.

Forbidden: importing private generated data/model artifacts as required test inputs, relying on camera hardware, network services, or uncommitted local files for default tests.

## Commands

```bash
uv run --group lint lint-imports
```

## Real-stack E2E

`test_e2e_night_bed_exit_relay.py` is marked `real_stack` and requires the `mediamtx`
RTSP server binary (plus `ffmpeg`, already a default dev dependency) on PATH. CI
deselects it (`uv run pytest -q -m "not real_stack and not heavy and not integration"`) rather than fetching an
external binary, per `tests/test_public_repository_privacy.py`'s untrusted-CI
contract -- it runs locally only. To run it:

```bash
# install mediamtx and put it on PATH, e.g.:
#   brew install mediamtx           # macOS
#   see https://github.com/bluenviron/mediamtx for other platforms
uv run pytest -m real_stack
```

Without `mediamtx` on PATH, the tests are skipped (not errored) with an explicit reason.

## Live-stack integration (`integration`)

`test_cloud_edge_provisioning_integration.py` is marked `integration` and drives a
**running, already-enrolled ml-api** — it is not a mock-backed test. CI deselects it
(the `not integration` above); the marker always said so in `pyproject.toml`, but the
CI argument only started honouring it once the test began failing on `main` with
`RuntimeError: CLOUD_EDGE_ML_URL is required`. Unlike `real_stack` it needs no RTSP
tooling, so `real_stack` would be the wrong marker for it.

It fails loudly rather than skipping, because every one of its inputs is a deliberate
pointer at live state that must not be guessed. Run it against a real Edge:

```bash
CLOUD_EDGE_ML_URL=http://<edge-host>:8000 \
CLOUD_EDGE_RELAY_TOKEN=... \
CLOUD_EDGE_ML_CATALOG_PATH=/var/lib/ml-api/catalog.sqlite3 \
CLOUD_EDGE_PRE_V1_BACKUP_PATH=... \
CLOUD_EDGE_SECRET_HANDOFF_PATH=... \
uv run pytest -m integration
```

It writes to the catalog sqlite file it is pointed at, so never aim it at a production
volume.

## Gotchas

Keep boundary tests small and explicit. When import policy changes, update the `[tool.importlinter]` contracts in `pyproject.toml` and the relevant AGENTS files in the same change.
## Test Boundary

- Keep default tests deterministic and hardware-free.
- Exercise both allowed and forbidden imports when changing the dependency ladder.
- Use package-focused tests before the full suite.

## `heavy` — CI에서 돌리지 않는 테스트

`heavy`로 표시한 테스트는 **실제 인터프리터 서브프로세스를 띄우고, 그
프로세스가 벽시계 감시(워치독 deadline, hard-exit 경로)로 끝나기를
기다린다.** 한가한 호스트에서는 정확하지만 부하가 걸린 CI 러너에서는
**프로세스 기동만으로 deadline을 넘겨** 깨진다.

실제로 겪었다 — 전체 스위트가 112초에서 318초로 늘어난 실행에서
`test_watchdog_subprocess_hard_exits_with_fatal_accelerator_code`만
`TimeoutExpired`로 두 번 연속 실패했다. 로컬에서는 2.1초에 통과한다.

그래서 CI는 `-m "not real_stack and not heavy and not integration"`로 제외한다. 타임아웃 여유를
늘리는 것으로 덮지 않는다 — 그러면 CI 시간만 늘고 같은 종류의 불안정이
남는다.

`worker/runtime/`(워치독, 폴트 핸들러, 가속기 경로)을 건드리면 **로컬에서
직접 돌린다.**

```bash
uv run pytest -q -m heavy
```

# tests

Flat pytest tree. Contracts, slice coverage, and boundary guards live here as
`test_*.py` beside shared helpers. No nested suite dirs, no `__init__.py`.
319 test modules; `fixtures/` holds data only (golden episodes, replay v2 traces,
nvinfer letterbox capture).

## Ownership

- `conftest.py`: autouse hermetic isolation.
- `edge_worker_fixtures.py`: typed worker config payloads.
- `observability_stack_fixtures.py`: in-process Backend over real uvicorn, `wait_until`, honest `mediamtx`/`ffmpeg`/`pyservicemaker` probes.
- `observability_load_harness.py` + `fanout_benchmark_metrics.py`: recorded-stream load measurements. Product code stays unpatched.
- `fanout_benchmark_metrics.py` (`RunMetrics`, `take_sample`, `capacity_verdict`, `write_document`) emits `null` for unobserved fields so a bench JSON never looks healthy on defaults. Orphaned today: no `test_fanout_benchmark` module or `fanout_benchmark_harness.py` exists, nothing imports it, and `-k fanout_benchmark` selects nothing. Restore the harness before quoting a bench command or `BENCH_*` knobs.
- `receipt_helpers.py`, `replay_fixtures.py`, `compact_cutover_sensitive_fixture.py`: one schema-valid payload shared across suites. Hand-built variants drift from the CHECK constraints.
- `_yolo_load_timeout_exit_repro.py`, `_yolo_offline_guard_repro.py`: not collected. `test_worker_yolo_adapters.py` runs them in a fresh interpreter with a bounded `subprocess.run(timeout=...)`.
- `sqlite_ownership_baseline.txt`: ADR-0005 ratchet read by `test_sqlite_ownership_boundary.py`. Entries are removed, never added.
- `test_public_repository_privacy.py` (2449 lines): pins tracked-file privacy, the `ci.yml` steps, the shard cover, and `ci-ok` as the single required check. `test_release_workflow_contract.py`, `test_edge_topology_contract.py`, `test_backend_image_build_contract.py` pin the other workflows and compose. Change workflow and test together.
- Cross-suite builders with real importers live in `tests_support/`, not here.
- `test_contract_symbol_exports.py`: contracts keep exporting runner, tracker, and worker_config symbols. Import direction is `lint-imports`, not a walker.
- `test_*.py`: named after the slice or seam under test.

Allowed: the package under test, pytest, local helpers. Forbidden as default inputs: private artifacts, cameras, live network, uncommitted local files.

## Hermetic fixtures

`conftest.py` pins the host so a fail means code, not the machine.

- Dashboard bootstrap is explicit `API_DASHBOARD_*`. Unconfigured-path tests must `delenv`.
- `API_BACKEND_ALLOW_INSECURE_HTTP=1` is a fixture opt-in. HTTPS-policy tests unset it.
- Dashboard credentials persist only through `PostgresDashboardCredentialsStore` in the per-test PostgreSQL sandbox. Never `~/.local/state/ml-api` or `/var/lib/ml-api`.
- PostgreSQL connection settings require an injected owner and authority; use
  `tests_support.postgres_sandbox.postgres_product_sandbox`. There is no
  connection-settings path/from-env override. The sandbox owns only its unique
  test namespace, closes its pool before dropping it, and requires an explicit
  `SEEON_TEST_POSTGRES_DSN`; a missing DSN fails, never skips.
- `Path.home()` redirects to `tmp_path`. Don't rewrite `HOME`.
- RTSP DNS is stubbed. Real `getaddrinfo` lives in `test_rtsp_url_policy.py`.
- Private fall bundle: `_PRIVATE_BUNDLE_MODULES` (15 module names) skip with a reason when `models/fall/pose-bbox56-gru/model.onnx` is absent. CI downloads no weights. A new bundle-reading module joins that set.
- Process umask is `0o022`. Insecure-mode tests `chmod` the path. `mkdir(..., mode=...)` sets the leaf only. Create each parent with an explicit mode when a validator walks the tree.

## Naming and async

Keep the tree flat. Name `test_<capability>.py` after the slice or seam (`test_api_clips.py`, `test_capability_inference_coordinator.py`). Don't add `unit/`, `e2e/`, or package-mirroring folders.

Async tests must not pass by sleep. Subscribe to the event or state, act, then await with a bound timeout. `wait_until(predicate, timeout=..., what=...)` is the shared helper. A bare `time.sleep` is not an assertion.

Modules keep their own `_login`, `_app`, `clip_env`, `_record` helpers (13 files define `_login`). Typed fakes stay module-local and underscore-prefixed.

## Markers

CI runs `uv run pytest -q -m "not real_stack and not heavy and not integration and not private_bundle"`. `test_public_repository_privacy.py` pins that filter. Deselect only in that `-m`, never in pytest `addopts`, so an unfiltered run selects everything. Don't widen timeouts to hide load flakiness.

- default: hermetic, hardware-free. `uv run pytest -q tests/test_<file>.py`
- `real_stack`: real composition plus `mediamtx`/`ffmpeg` on PATH. Skip if missing, don't error. `uv run pytest -m real_stack`
- `integration`: live enrolled ml-api. Needs explicit `CLOUD_EDGE_*`. Writes the catalog it is pointed at. Never a production volume. `uv run pytest -m integration`
- `heavy`: real interpreter subprocess whose exit is a wall-clock watchdog or hard-exit path. Idle-host correct, CI-load flaky. `uv run pytest -q -m heavy`
- `private_bundle`: reads the private fall bundle under `models/fall/pose-bbox56-gru` (`scripts/fetch-models.sh`). `tests_support/private_bundle.py` marks it at collection and fails a selected test when the bundle is missing, naming the path. `uv run pytest -q -m private_bundle`

`private_bundle` fails rather than skips: an unprovisioned bundle is deselected with `-m`, never skipped by the test. `real_stack` is RTSP tooling, not "any live service". `integration` is a live enrolled API, not RTSP. `heavy` is subprocess deadline supervision, not "slow".

CI shards the same filter 4 ways by sorted test file; a module that matches no shard pattern never runs. After an import boundary change, run `uv run --group lint lint-imports` and update `[tool.importlinter]` and the matching AGENTS files in the same commit.

## Observability load (Gate M/V)

`test_observability_real_stack.py` is `real_stack` and operator-gated. `observability_load_harness.py` starts a local `mediamtx` serving N looping copies of `OBS_STREAM_PATH`, a Backend via `serve_backend()`, and a real `WorkerRuntime` with `ML_WORKER_EXECUTION_RECORDS_ENABLED=1`. Output is `obs-<N>.json` under `OBS_OUTPUT_DIR` (default `.omo/evidence/observability`). The harness records measurements only: offered fps, records/sec accepted, gap rows/sec, lane high-water, backlog slope, p50/p95 exporter batch latency, CPU delta. It never asserts a numeric threshold. Those numbers are the ONLY source for deployment budgets (Gate M/V); never bake them as defaults in product code. Skip, don't error, when `mediamtx`/`ffmpeg`/`pyservicemaker` or `OBS_STREAM_PATH` are missing. Knobs: `OBS_STREAMS` (default `1`), `OBS_DURATION_SEC` (default `30`), `OBS_CAMERA_FPS` (default `15`), `OBS_OUTPUT_DIR`, `OBS_STREAM_PATH`.

```bash
uv run pytest -m real_stack -k observability_real_stack
OBS_STREAMS=1 OBS_DURATION_SEC=30 OBS_CAMERA_FPS=15 \
  OBS_STREAM_PATH=/path/to/recorded.ts uv run pytest -m real_stack -k observability_real_stack
```

## Commands

```bash
uv run pytest -q tests/test_<file>.py
uv run pytest -q -m "not real_stack and not heavy and not integration and not private_bundle"
uv run pytest -m real_stack
uv run pytest -m integration
uv run pytest -q -m heavy
uv run pytest -q -m private_bundle
uv run --group lint lint-imports
```

Every run needs `SEEON_TEST_POSTGRES_DSN` pointing at a disposable PostgreSQL 18 database (CI starts one per shard). A missing DSN fails the PostgreSQL tests.

Need `mediamtx` on PATH for `real_stack`. A missing tool skips. After an import boundary change, update `[tool.importlinter]` and the matching AGENTS files in the same commit.

## Anti-patterns

- Local Hero: outcome decided by umask, GPU, PATH, locale, timezone, or core count. Assert code invariants. Guard or skip on missing env. Never assert "this machine has no GPU".
- Host-state probes named `*_on_this_dev_machine`. If `available=True`, assert the honest-probe contract (reason present, metadata rules), not the inventory.
- Required inputs from uncommitted weights, live cameras, or the developer's state directory.
- Sleep-as-assert, unbounded polls, or "wait a bit and hope".
- Nested test packages that fake a scope the tree doesn't have.
- Baking always-fail stubs into runtime so the suite boots. Stubs stay here.
- Stretching CI deadlines so `heavy` looks green.

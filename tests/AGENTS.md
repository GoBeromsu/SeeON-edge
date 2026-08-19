# tests

Flat pytest tree. Contracts, slice coverage, and boundary guards live here as
`test_*.py` beside shared helpers. No nested suite dirs.

## Ownership

- `conftest.py`: autouse hermetic isolation.
- `edge_worker_fixtures.py`: typed worker config payloads.
- `e2e_worker_relay_fixtures.py`: MediaMTX, ffmpeg, in-process backend, scripted serving, `wait_until`.
- `fanout_benchmark_harness.py` + `fanout_benchmark_metrics.py`: recorded-stream fan-out. Product code stays unpatched.
- `clip_listing_reader_concurrency_fixtures.py`: listing-reader concurrency.
- `test_contract_symbol_exports.py`: contracts keep exporting runner, tracker, and worker_config symbols. Import direction is `lint-imports`, not a walker.
- `test_*.py`: named after the slice or seam under test.

Allowed: the package under test, pytest, local helpers. Forbidden as default inputs: private artifacts, cameras, live network, uncommitted local files.

## Hermetic fixtures

`conftest.py` pins the host so a fail means code, not the machine.

- Central `edge.sqlite3` is a per-test tmp file. `EDGE_DATABASE_PATH` is monkeypatched on every module that reads it.
- Dashboard bootstrap is explicit `API_DASHBOARD_*`. Unconfigured-path tests must `delenv`.
- `API_BACKEND_ALLOW_INSECURE_HTTP=1` is a fixture opt-in. HTTPS-policy tests unset it.
- `DashboardCredentialsStore.from_env` and `ConnectionSettingsStore.from_env` resolve under `tmp_path`. Never `~/.local/state/ml-api` or `/var/lib/ml-api`.
- `Path.home()` redirects to `tmp_path`. Don't rewrite `HOME`.
- RTSP DNS is stubbed. Real `getaddrinfo` lives in `test_rtsp_url_policy.py`.
- Process umask is `0o022`. Insecure-mode tests `chmod` the path. `mkdir(..., mode=...)` sets the leaf only. Create each parent with an explicit mode when a validator walks the tree.

## Naming and async

Keep the tree flat. Name `test_<capability>.py` after the slice or seam (`test_api_clips.py`, `test_capability_inference_coordinator.py`). Don't add `unit/`, `e2e/`, or package-mirroring folders.

Async tests must not pass by sleep. Subscribe to the event or state, act, then await with a bound timeout. `wait_until(predicate, timeout=..., what=...)` is the shared helper. A bare `time.sleep` is not an assertion.

## Markers

CI runs `uv run pytest -q -m "not real_stack and not heavy and not integration"`. `test_public_repository_privacy.py` pins that filter. Don't widen timeouts to hide load flakiness.

- default: hermetic, hardware-free. `uv run pytest -q tests/test_<file>.py`
- `real_stack`: real composition plus `mediamtx`/`ffmpeg` on PATH. Skip if missing, don't error. `uv run pytest -m real_stack`
- `integration`: live enrolled ml-api. Needs explicit `CLOUD_EDGE_*`. Writes the catalog it is pointed at. Never a production volume. `uv run pytest -m integration`
- `heavy`: real interpreter subprocess whose exit is a wall-clock watchdog or hard-exit path. Idle-host correct, CI-load flaky. `uv run pytest -q -m heavy`

`real_stack` is RTSP tooling, not "any live service". `integration` is a live enrolled API, not RTSP. `heavy` is subprocess deadline supervision, not "slow".

## Fan-out benchmark

`test_fanout_benchmark.py` is `real_stack` and operator-gated. A local `mediamtx` serves N looping recorded streams. A real `WorkerRuntime` loads `models/`. Output is `bench-<N>.json` under `BENCH_OUTPUT_DIR` (default `.omo/evidence/bench`). N is in `{1,2,4,8,13}`. `BENCH_STREAMS` selects which run (default `1,2`). Leave extra N unset so a bare suite doesn't burn minutes. Other knobs: `BENCH_DURATION_SEC`, `BENCH_PROFILE`, `BENCH_LABEL`, `BENCH_VIEWERS`, `BENCH_CAMERA_FPS`. Timing and relay stubs wrap test-side only.

```bash
uv run pytest -m real_stack -k fanout_benchmark
BENCH_STREAMS=1,2,4,8,13 uv run pytest -m real_stack -k fanout_benchmark
# 13 cameras at 15fps (todo 13). TemporalProfile is not on origin/main yet
# (PR #356); BENCH_CAMERA_FPS is the fps owner until that contract lands.
BENCH_STREAMS=13 BENCH_CAMERA_FPS=15 BENCH_DURATION_SEC=300 BENCH_VIEWERS=0 \
  BENCH_LABEL=13x15 uv run pytest -m real_stack -k 'test_fanout_benchmark['
```

## Commands

```bash
uv run pytest -q tests/test_<file>.py
uv run pytest -q -m "not real_stack and not heavy and not integration"
uv run pytest -m real_stack
uv run pytest -m integration
uv run pytest -q -m heavy
uv run --group lint lint-imports
```

Need `mediamtx` on PATH for `real_stack`. A missing tool skips. After an import boundary change, update `[tool.importlinter]` and the matching AGENTS files in the same commit.

## Anti-patterns

- Local Hero: outcome decided by umask, GPU, PATH, locale, timezone, or core count. Assert code invariants. Guard or skip on missing env. Never assert "this machine has no GPU".
- Host-state probes named `*_on_this_dev_machine`. If `available=True`, assert the honest-probe contract (reason present, metadata rules), not the inventory.
- Required inputs from uncommitted weights, live cameras, or the developer's `catalog.sqlite3`.
- Sleep-as-assert, unbounded polls, or "wait a bit and hope".
- Nested test packages that fake a scope the tree doesn't have.
- Baking always-fail stubs into runtime so the suite boots. Stubs stay here.
- Stretching CI deadlines so `heavy` looks green.
- Aiming `CLOUD_EDGE_ML_CATALOG_PATH` at a production sqlite.

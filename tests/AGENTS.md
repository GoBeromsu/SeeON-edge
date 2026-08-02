# TESTS KNOWLEDGE BASE

Own pytest coverage for the ML uv project, including dependency-boundary guards and fixtures.

## Local Ownership

- `test_contract_symbol_exports.py`: contract-symbol export checks (runner/tracker/worker_config). Import boundaries are enforced by import-linter (`uv run --group lint lint-imports`), not a pytest walker.
- `edge_worker_fixtures.py`, `demo_app_control_helpers.py`, `e2e_worker_relay_fixtures.py`: shared test helpers.
- `test_*`: package-specific and cross-boundary tests.
- `test_e2e_night_bed_exit_relay.py`: real-stack E2E (marked `real_stack`) -- synthetic RTSP via `mediamtx` + `ffmpeg` into the real worker/backend composition. Deselected in CI (`-m "not real_stack"`) and skipped locally when `mediamtx` is not on PATH; see "Real-stack E2E" below.

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
deselects it (`uv run pytest -q -m "not real_stack"`) rather than fetching an
external binary, per `tests/test_public_repository_privacy.py`'s untrusted-CI
contract -- it runs locally only. To run it:

```bash
# install mediamtx and put it on PATH, e.g.:
#   brew install mediamtx           # macOS
#   see https://github.com/bluenviron/mediamtx for other platforms
uv run pytest -m real_stack
```

Without `mediamtx` on PATH, the tests are skipped (not errored) with an explicit reason.

## Gotchas

Keep boundary tests small and explicit. When import policy changes, update the `[tool.importlinter]` contracts in `pyproject.toml` and the relevant AGENTS files in the same change.
## Test Boundary

- Keep default tests deterministic and hardware-free.
- Exercise both allowed and forbidden imports when changing the dependency ladder.
- Use package-focused tests before the full suite.

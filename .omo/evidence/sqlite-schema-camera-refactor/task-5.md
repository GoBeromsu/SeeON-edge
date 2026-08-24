# Task 5 evidence

- Issue: #377
- Branch: `refactor/377-remove-camera-mapping`
- Baseline: `uv run pytest -q tests/test_camera_roster_sync.py` -> `3 passed`.
- Unused-symbol proof before deletion: LSP references for `BackendCameraMapper.put_mapping` returned declaration only; structural search found no Python callers.
- Change: deleted `BackendCameraMapper.put_mapping`; retained `_mapping_request`, `push_camera`, `put_roster`, `MappingResult`, and exports. Updated stale request-builder and `push_camera` docstrings.
- Green verification: `uv run pytest -q tests/test_camera_roster_sync.py` -> `3 passed`; `uv run ruff check backend/app/shared/backend_mapping.py` -> `All checks passed!`.
- Structural proof after deletion: `rg -n "put_mapping" --glob '*.py' backend tests` returned no matches. LSP references returned `No references found`.
- LSP diagnostics: only pre-existing basedpyright warnings remain (unannotated mapper attributes, urllib `Any`, unknown dict access, and unused `_timeout_sec`); no new errors.
- Manual driver: a real `ThreadingHTTPServer` and `BackendCameraMapper.push_camera` invocation produced `success: PASS classification=ok`, `auth: PASS classification=auth`, `timeout: PASS classification=timeout`, and `unreachable: PASS classification=unreachable`; timeout completed at `0.050s`. Server shutdown, socket close, thread join, and port-release cleanup passed.
- Probe notes: dirty worktree checked before change; malformed response remains covered by existing `ValueError` classification path; timeout used a bounded `0.05s` client timeout; no sleeps were used for assertions (the server delay only simulated the timeout response).

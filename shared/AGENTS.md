# SHARED KNOWLEDGE BASE

Backend and worker common library. Wire, SQLite, versioned detection
policies, RTSP admission. Not a runtime.

## Package map
- `events/`: backend↔worker egress. Read `events/AGENTS.md`.
- `edge_db/`: one local `edge.sqlite3`. Read `edge_db/AGENTS.md`.
- `detection_policies.py`: closed typed policy parser and bundle.
- `rtsp_url_policy.py`: RTSP/RTSPS destination admission and IP pin.
- `pyproject.toml`: `eldercare-shared`. Empty deps.

Import-linter owns the graph. Allowed: stdlib, `contracts`, local `shared.*`.
Forbidden: `backend`, `worker`, pydantic, cv2, torch, model load, camera I/O,
decode, serving. `contracts` must not import this package. mypy is `--strict`
on `shared.*`. Relative imports are banned.

## detection_policies.py
Single owner of numeric fall and bed-exit documents. API store and worker
pull both parse here. Drift never degrades into image defaults. Closed parse:
module/schema identity, exact fields, ranges, cross-field caps, content hash.
Extra or missing fields raise `PolicyDocumentError`. `parse_policy_bundle`
requires every latest module. `resolve` picks camera, then default. Version
mismatch raises. `make_effective_policy` stamps `effective_policy_id` from
canonical JSON. Identity mismatch is an error. Persistence lives in
`backend/app/features/detection_settings`. Worker YAML `detection_policies`
is retired. Pull the versioned bundle from SQLite.

## rtsp_url_policy.py
Shared by API camera admission and worker open/probe. Only absolute `rtsp` /
`rtsps`. Userinfo is allowed and never classifies the host. Static check:
`reject_rtsp_url_reason` / `assert_rtsp_url_allowed`. Connect, probe, store:
`resolve_rtsp_endpoint` / `assert_rtsp_endpoint_allowed`. Check every A/AAAA
answer. Open `pinned_url` so the decoder cannot re-resolve past the gate.
`ML_RTSP_ALLOW_PRIVATE_DESTINATIONS=1` admits RFC1918/CGNAT.
`ML_RTSP_ALLOW_LOCAL_DESTINATIONS=1` admits loopback + link-local + private
for QA only. Metadata and link-local stay denied under PRIVATE-only.

## Where to look / tests
Event wire: `events/`. SQLite: `edge_db/`. Types live in this file. Rows live in
`backend/app/features/detection_settings/`. Worker pull:
`worker/runtime/config/pull_models.py`. Camera admit:
`backend/app/features/cameras/router.py`. Worker open:
`worker/runtime/ingest_composition.py`. Child suites stay in those guides.
`uv run pytest -q tests/test_detection_policy_models.py tests/test_rtsp_url_policy.py tests/test_worker_policy_resolution.py tests/test_worker_static_detection_policy_authority.py tests/test_api_detection_policy.py tests/test_camera_rtsp_destination_api.py`
Then `uv run --group lint lint-imports`. Real `getaddrinfo` lives in
`test_rtsp_url_policy.py`. Other tests stub DNS.

## Anti-patterns
- Importing `backend` or `worker` from here.
- Adding a dependency. This member stays empty-deps.
- Soft-parsing a drifted policy into defaults.
- Reimplementing policy ranges in a store, detector, or form.
- Static YAML policy authority on the worker.
- Opening RTSP or writing table families here.
- Using `assert_rtsp_url_allowed` at connect time. Pin the IP.
- Treating `edge.sqlite3` as a mailbox.
- A third sibling package without import-linter, mypy, and this map.

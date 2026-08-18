# FEATURE SLICES

Vertical cut: one capability, one package. Router and store live together.
Sibling `*_router.py` files stay in that package. `create_app` is the only
mounter. `qa/` is store-only until a router lands here.
## Layout

Export `router`. End the router module with `__all__` that includes it.
Do not `include_router` a sibling slice from inside this tree.
Owner constructs (`from_env()` or lifespan) and exposes a getter that
writes `app.state` once. Dependents call the getter. They never
`from_env()` a second copy of someone else's store.
Lifespan pre-builds `camera_registry`, `heartbeat_store`,
`runtime_status_store` and starts `maintain_clip_listing`. Other stores
may lazy-open so `no_lifespan` tests still boot. Catalog is optional:
`get_catalog_store` returns `None` and sets `catalog_error` when the file
cannot open. Relay still accepts the alert.
## Cross-slice graph

Read or call. Do not construct the other slice's store.
- `relay` consumes cameras (`worker_config_snapshot`, registry), clips
  catalog, and status stores. No store of its own.
- `evidence` reuses `relay.auth` plus `_camera_binding`, clip-dir constants,
  and the runtime-settings export gate. Worker ingest stays under `/relay`.
  Operator incidents are the second router in this same slice.
- `cameras` merges detection, connection, clip storage location, runtime
  settings, and heartbeat age. `worker_config_snapshot` is the only place
  local detection overrides meet pulled config.
- `connection` drives cameras roster/topology helpers and heartbeat-relay
  state. After enroll it calls `lifespan.apply_connection_settings` and
  `refresh_backend_config`. That callback is the approved lifespan import.
- `status` reads the cameras expected set and runtime settings. `/status`
  merges heartbeat plus runtime snapshot. `/system` is disk and image
  metadata, not camera liveness.
- `detection_settings` reads cameras (Hub id only) and connection. It never
  writes `app.state.pulled_config`.
- `streams` lives in cameras. It proxies worker MJPEG and duplicates the
  relay header constant so it does not import cameras-router privates.
Hard edges: no `backend.app.main` imports. Connection is the only slice
that imports `lifespan`. `qa` imports nothing under `features/`; nothing
here imports `qa` until its router exists.
## HTTP, stores, tests

Parent locks BaseModel shape. Schemas sit next to the router or in slice
`schemas.py`. No package-wide `models.py`. Query objects are frozen models
via `Annotated[..., Query()]`. Optimistic writes carry `expected_version`
and return 409 with the current row. Dashboard routes call
`authorize_dashboard`. Worker routes call `relay.auth.authorize_relay`.
New slices do not borrow `cameras.router._authorize`. Body caps stay on
relay `BoundedBodyRoute`; add a suffix entry, do not copy the class.
API actor writes `control_*` and `qa_*` only. Never INSERT `runtime_*`,
`evidence_*`, or `derivative_*`. `QaStore` opens as `RuntimeActor.API` and
never imports worker replay. Auth has no SQLite row; sessions live in
`shared/dashboard_auth.py`. Incomplete enrollment deletes ingest, evidence,
and mapper attrs and sets `backend_configured=False`. Handlers do not
build `EdgeIngestClient`. Drive the slice through
`create_app(lifespan=no_lifespan)` plus an injected store, or full lifespan
when listing or refresh is the subject. Slice tests: `tests/test_api_*.py`,
`tests/test_connection_*.py`, `tests/test_clip_listing_*.py`. New
`features.*` import: `uv run --group lint lint-imports`.
## Anti-patterns

New product verb in `routes/` (probes and compat GETs only). Second
top-level folder for the same capability. Operator UX growing inside
`relay/router.py` (belongs in `evidence/operator_router.py`). Emitting a
local registry id on a Hub-bound payload. Worker-config keeps unmapped cameras
and uses `backend_camera_id` or the local id so ingestion never stops. Raw path joins into the clip store (use
`clips/descriptor_files.py` or evidence `_verified_media`, O_NOFOLLOW).
Teaching `qa/` HTTP from `create_app` without adding the router here.
Polling `edge.sqlite3` for worker progress; HTTP relay is the signal.

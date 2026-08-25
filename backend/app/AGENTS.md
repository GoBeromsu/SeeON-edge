# BACKEND APP KNOWLEDGE BASE

Own `create_app`, lifespan, and `features/*`. Thin ml-api gateway: worker relay
in, Event API out, dashboard HTTP in between. No camera loops. No shared memory
with the worker.

## create_app / lifespan

- `create_app()` seeds `app.state.edge_relay_token` from `API_EDGE_RELAY_TOKEN`, mounts unversioned `probe_router`, then registers product routers under `Settings.api_v1_prefix` (`/api/v1`). Front dist mounts at `/` from `API_FRONT_DIST` when that dir exists.
- Auth reads only `app.state.edge_relay_token`. Handlers never re-read env. Lifespan seeding is `hasattr`-guarded so the factory (and tests) win. `no_lifespan` is the test hook.
- `lifespan.py` rejects retired env keys, resolves the ml-api state dir, then assembles in-memory `heartbeat_store` / `runtime_status_store`, `camera_registry`, and the connection-derived ingest / evidence bundle. Clip listing is compact-authority on request.
- Config refresh and heartbeat relay each get a dedicated 1-worker executor. `API_BACKEND_HEARTBEAT_RELAY_SEC=0` kills the relay loop.
- Clip listing is compact-authority on request. Shutdown cancels both loops, then `catalog_store.close()`.
- Lifespan does not build detectors or RTSP. Pulled ml-config `cameras` are not admission; the dashboard registry is the camera SSOT.

## Feature slices

One capability per `features/<slice>/`. The slice owns its router and store. Extra routers stay in the same slice.

- `auth`: dashboard session + credential rotation (`shared/dashboard_auth.py`).
- `cameras`: registry, topology, bed zones, worker-config, MJPEG proxy. Owns `camera_registry`, `bed_zone_store`.
- `clips`: listing, media, storage, catalog. Owns compact `clips` / `artifacts` access, `catalog_store`, and `clip_storage_location_store`.
- `connection`: enrollment, Hub URL, roster sync, topology confirm. Owns `ConnectionSettingsStore`.
- `detection_settings`: settings + policy apply/rollback. Owns `detection_settings_store`, `detection_policy_store`.
- `evidence`: worker clip ingest under `/relay` plus operator incidents.
- `relay`: `/relay/{config,restart,alerts,heartbeat,runtime-status}`. No store; consumes cameras / status / clips via deps.
- `runtime_settings`: operator knobs. Owns `RuntimeSettingsStore`.
- `status`: `/status` and `/system` from relay-derived liveness. Owns `heartbeat_store`, `runtime_status_store`.
- `qa`: retired. No QA/replay table is a runtime owner.
- `routes/`: app-level health + models. See `routes/AGENTS.md`.
- `core/` is `Settings` (`ML_API_`). `shared/` is infra (mapping, sessions, sqlite bootstrap, state dir), never feature state.

## Wire models

HTTP schemas are Pydantic `BaseModel`, never `dataclass`. They live in the slice or a slice `schemas.py`. Keep them out of `contracts`.

- Requests `<Action>Request`, responses `<Action>Response`. A trivial ack may return `dict[str, str]`.
- `model_config = ConfigDict(extra="forbid")`. Every field uses `Field(...)`. Routes declare `response_model=...`. Header/Query deps are `Annotated`.
- Settings: `BaseSettings` + `SettingsConfigDict(env_prefix="ML_API_", extra="ignore")`.
- One `APIRouter(prefix=..., tags=[...])` per slice. Thin handlers. Product routes under `/api/v1`. `/health/live` and `/health/ready` stay unversioned. End with `__all__ = ["router"]`.
- Injected collaborators are `typing.Protocol`s bound in `lifespan.py`.

## Store ownership

`app.state` is the injection board. The owning slice constructs the store (`from_env()` or lifespan) and exposes a getter. Other slices depend; they do not build a second copy.

- cameras: `camera_registry`, `bed_zone_store`
- clips: compact clip/artifact stores, `catalog_store`, `clip_storage_location_store`
- status: `heartbeat_store`, `runtime_status_store`
- detection_settings: `detection_settings_store`, `detection_policy_store`

Connection and runtime settings load through their slice `from_env()` helpers. API writes remaining compact authorities as `RuntimeActor.API`. Do not open worker table families. `shared/sqlite_bootstrap.py` may connect; it must not import feature stores.

## Focused tests

- Factory: `tests/test_serving_health.py`, `tests/test_serving_status.py`, `tests/test_serving_models.py`, `tests/test_serving_boundary_contract.py`
- Relay: `tests/test_api_ingest_relay.py`, `tests/test_relay_body_auth_ordering.py`, `tests/test_api_heartbeat_store.py`, `tests/test_api_runtime_status.py`, `tests/test_backend_heartbeat_relay.py`, `tests/test_ml_api_config_pull.py`
- Slices: `tests/test_api_*.py`, `tests/test_connection_*.py`, `tests/test_auth_login_throttle.py`, `tests/test_dashboard_auth.py`
- Import direction: `uv run --group lint lint-imports`

## Anti-patterns

- Do not assemble camera loops, detectors, or worker runtime in `lifespan.py`.
- Do not seed cameras from env, YAML, or a pulled backend roster.
- Do not re-read `API_EDGE_RELAY_TOKEN`. `app.state` is the only source.
- Hub ingest needs a Hub-issued `backend_camera_id`. Skip egress until that mapping exists; keep the local catalog / heartbeat write anyway.
- Stamp `/relay/heartbeat` `received_at` after auth, before binding and before egress. Local liveness is edge truth even when registry or Hub fails.
- Wire models stay out of `contracts`. Feature state stays out of `shared/`.
- `core/` and `shared/` cannot import `features`, `routes`, `main`, or `lifespan`. `worker` is import-forbidden. Relay HTTP only.
- Retired env keys fail boot via `reject_retired_backend_environment`.
- Config-refresh and heartbeat-relay keep separate executors.
- Relay bodies stay inside the per-route caps on `BoundedBodyRoute`.

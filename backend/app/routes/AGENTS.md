# APP-LEVEL ROUTES

Own the leftover compatibility/health HTTP surface: two k8s-style probes plus two read-only GETs. Product capabilities do not land here.

## Local Ownership

- `health.py` exports two routers, not one.
  - `probe_router`: unversioned `/health/live` (always `{"status":"ok"}`) and `/health/ready` (reads `app.state.readiness`; 503 while booting).
  - `router`: versioned `GET /api/v1/health`. Legacy pairing dump: gateway ready/booting, relay token present, ingest client present, registry camera count.
- `models.py`: `GET /api/v1/models`. Static gateway metadata (`service=ml-api`, `role=gateway`, `ml=external-worker`) plus the same relay configured/count snapshot.
- `__init__.py`: re-exports `health` and `models` only.

`main.py` mounts `probe_router` on the app root and the other two under `settings.api_v1_prefix`. There is no `status.py` in this folder. `/api/v1/status` lives in `features/status`.

`/models` is not a weight registry and does not report loaded-model state. Worker serving owns that.

## What belongs in `backend/app/features` instead

A new product verb, write path, store, or worker-facing command goes under `features/<slice>/` (`router.py` + `store.py`):
- `status/`: heartbeat + runtime-status stores and `/status`
- `relay/`: worker ingest, heartbeat, restart, config
- `cameras/`, `clips/`, `evidence/`, `auth/`, `connection/`
- `detection_settings/`, `runtime_settings/`

Keep this package tiny. A frozen compatibility GET or a process probe can stay. Auth, mutation, and snapshot merge cannot.

## Imports

Allowed: FastAPI, Pydantic, `backend.app.features.cameras.store` (read-only `CameraRegistryStore` + `registry_expected_cameras`), `app.state` fields lifespan already set.

Forbidden: `worker`, `training`, camera open, model load/train, store writes, collaborator construction.

Handlers only `getattr` and count. Lifespan owns construction. Keep the cameras-store import lazy inside the handler, as the modules do today. Don't reach into status/heartbeat/runtime stores to "enrich" these responses.

## Focused Tests

- `tests/test_serving_health.py`: live 200, ready 503 while booting, ready 200 after lifespan, unwritable catalog still ready, no ML runtime on `app.state`, legacy `/api/v1/health` `camera_count`.
- `tests/test_serving_models.py`: `/api/v1/models` is gateway metadata only.
- `tests/test_route_version_contract.py`: probes stay unversioned; product paths stay under `/api/v1`.
- `tests/test_serving_boundary_contract.py`: both paths stay on the public allowlist.
- `tests/test_serving_status.py` covers `/api/v1/status`, not this package.
- Boundary: `uv run --group lint lint-imports`

## Gotchas

`/health/ready` is process readiness, not catalog writability and not camera liveness. Catalog opens lazily on first relay use.

`/api/v1/health` is a pairing/compat dump, not a probe. Orchestrators hit `/health/live` and `/health/ready`.

`registry_expected_cameras` is inbound lookup only. Use it to count. Never emit local registry ids on outbound Hub payloads (FACILITY_BINDING_MISMATCH).

These handlers have no request body. If you add query or header validation, cover the error status codes. Don't invent POST or PATCH here.

## Change Boundary

Probes stay unversioned. `/health` and `/models` stay under `/api/v1`. Handlers stay read-only and thin. New product routes go to a feature slice.

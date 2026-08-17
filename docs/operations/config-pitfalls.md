# Edge configuration authority and migration pitfalls

`edge-env-inventory.json` is the machine-consumed source of truth. Tests compare
it with `compose.edge.yaml` and `.env.edge.prod.example`; adding or removing a
deployment key without an inventory disposition fails CI.

## Supported deployment environment

Only these values cross the production deployment boundary:

- release image digests: `ML_API_IMAGE`, `ML_WORKER_IMAGE`
- runtime profile alias: `ML_WORKER_PROFILE`
- host clip root: `CLIP_STORE_HOST_DIR`
- public Hub origin and timeout: `API_BACKEND_BASE_URL`,
  `API_BACKEND_INGEST_TIMEOUT_SEC` (default `10` seconds)
- dashboard bootstrap pair: `API_DASHBOARD_USERNAME`,
  `API_DASHBOARD_PASSWORD`
- internal relay pair secret: `API_EDGE_RELAY_TOKEN` (projected to worker
  `RELAY_TOKEN`)

Facility identity/token and camera roster have no env or static-YAML fallback.
Enrollment and camera registration in the dashboard persist them in the
connection-settings and camera-registry tables.

## Baked topology

The API loopback port is `8000`; the private worker relay is
`http://ml-api:8000`; worker MJPEG/probe is `http://ml-worker:8090`; container
model and clip roots are `/app/models` and `/var/lib/clip-store`; runtime state
paths are code-owned. These values are deliberately not deployment knobs.

## Mutable authority boundary

Domain enablement and detection windows are already DB-backed and connected to
the typed versioned worker-config response. Camera cadence/decode choices remain
camera-registry fields. Clip storage subdirectory remains a DB-backed config
field.

Model selection/numeric model policy and clip recording/delivery policy do not
yet have the immutable revision schema required by Todo 9. Until Todo 9 creates
that DB schema, API projection, apply/rollback state, and typed worker fields,
the worker-config parser explicitly rejects `models` and `clip` payload members;
image defaults are fixed rather than mutable. This is a deliberate migration
boundary, not a silent placeholder. Todo 9 must add the backend fields and worker
consumption atomically before either becomes configurable.

Static YAML `models`, `domains`, or `clip` policy is rejected, as is any
non-empty static `cameras` list. Production always pulls roster/domain state from
the API and cannot acquire a second mutable authority through YAML.

## Retired keys

Runtime startup rejects retired keys named in `edge-env-inventory.json`; it does
not warn-and-ignore or let them override DB/config. For the old
`API_BACKEND_EVENTS_URL` / `API_BACKEND_CONFIG_URL` pair, migrate their common
origin to `API_BACKEND_BASE_URL` before upgrading. Divergent per-field origins
are no longer supported.

`API_ALLOW_LEGACY_DASHBOARD_AUTH` has been retired. Dashboard routes accept only
the server-issued HttpOnly session cookie; worker relay credentials are never
operator credentials. Backend startup rejects the retired variable; remove it
from stale deployment environments before upgrading.

The one-shot `SYSTEM_TEST` operator CLI action, its `ML_WORKER_SYSTEM_TEST_GATE`
environment gate, and its dedicated relay HTTP routes are retired and removed;
no runtime surface accepts them any more.

# 0004 — Camera Roster Sync Contract Assumptions

- Status: Accepted
- Date: 2026-08-02
- Relates to: story G004 (camera roster sync), `backend/app/shared/backend_mapping.py`,
  `backend/app/features/cameras/roster_sync.py`, `backend/app/features/connection/store.py`.

## Context

Story G004 requires pushing the local camera registry to the external
eldercare-fall-ai backend via `PUT /v1/edge/cameras` so a facility's camera
roster appears there, with per-camera sync state tracked and queryable. Two
parts of the external contract needed to be pinned down:

1. **Roster payload shape.** The existing per-camera mapping call
   (`BackendCameraMapper.put_mapping`) sends one camera at a time as
   `{"edge_camera_ref", "label", "spaceId"}`. The first draft of this ADR
   assumed a bulk `PUT /v1/edge/cameras` variant (`{"cameras": [...]}}`,
   `spaceId` optional per entry) for pushing the whole roster in one call.
   That assumption was **verified against the real fall-ai backend source**
   (read-only reference at `../eldercare-fall-ai/backend/src/cameras/
   cameras.controller.ts` and `dto/camera.dto.ts`) and found wrong on both
   counts — see Decision §1 below.
2. **Token source for the roster path specifically.** `BackendCameraMapper`'s
   existing per-camera path (`from_env()`) reads raw process env with four
   accepted token aliases (`EDGE_FACILITY_TOKEN`, `API_FACILITY_TOKEN`,
   `API_BACKEND_FACILITY_TOKEN`, `API_EDGE_FACILITY_TOKEN`). The task brief for
   G004 explicitly asked for the roster sync's endpoint/token resolution to go
   through `ConnectionSettingsStore` (the G001/G003 dashboard-configurable
   settings store) instead, so a technician can point the roster push at a
   relinked backend without an env change or restart.

## Decision

**1. Payload shape (verified, not assumed)**: `PUT /v1/edge/cameras`
(`EdgeCamerasController.upsert` in `cameras.controller.ts`) takes exactly
**one** `EdgeCameraMappingRequestDto` per call:

```json
{"edge_camera_ref": "<id>", "label": "<label>", "spaceId": "<space-id>"}
```

`dto/camera.dto.ts`'s `EdgeCameraMappingRequestDto` declares all three fields
`@IsString()` with no `@IsOptional()`, so **`spaceId` is required** — a body
missing it, or a bulk `{"cameras": [...]}}` wrapper, fails class-validator
with 400. The response is `{cameraId, spaceId, facilityId}`
(`EdgeCameraMappingResponseDto`).

There is no bulk variant of this endpoint. `BackendCameraMapper.put_roster`
therefore issues one `PUT` per eligible camera (via the same request builder
`put_mapping` uses — `_mapping_request`) and aggregates the per-camera
outcomes into a `RosterPushResult` with a `cameras: dict[str, CameraPushResult]`
field, rather than sending a single bulk request. Only cameras that already
carry a non-empty `space_id` in the local registry are included in a push
attempt — see `roster_sync.py`'s `_split_roster_payload` — mirroring the same
space_id-required precedent already applied at
`cameras/router.py:442-451`'s `retry_pending_backend_mappings`. Cameras
without a `space_id` are never sent (a guaranteed 400 otherwise); they read
as sync status `pending` with `error_class: null` and a Korean detail
explaining they're awaiting a space assignment, not as a failure.

**2. Token/endpoint resolution via `ConnectionSettingsStore`**:
`roster_sync.py`'s `_build_mapper` resolves `events_url`/`facility_token`/
`facility_id` from `ConnectionSettingsStore.from_env().load()` (file-over-env,
single `EDGE_FACILITY_TOKEN` env fallback — see
`connection/store.py:75-85`), constructing its own `BackendCameraMapper`
instance via `derive_edge_cameras_endpoint(settings.events_url)` rather than
calling `BackendCameraMapper.from_env()`. This means the roster-sync path does
**not** recognize the three legacy token aliases
(`API_FACILITY_TOKEN`/`API_BACKEND_FACILITY_TOKEN`/`API_EDGE_FACILITY_TOKEN`)
that the older per-camera mapping path still does — only `EDGE_FACILITY_TOKEN`
(directly, or as the store's env seed) or a token explicitly saved to the
store.

## Consequences

- A facility relying on one of the three legacy token env aliases (without
  ever having saved a token through the dashboard, and without
  `EDGE_FACILITY_TOKEN` also set) will see the per-camera mapping path
  (`put_mapping`, used by camera create/update) succeed while the roster sync
  path (`put_roster`) reports `disabled`/`unconfigured` — a real, visible
  behavioral split between the two HTTP paths against the same backend.
  Operators should standardize on `EDGE_FACILITY_TOKEN` (or the dashboard
  settings UI) going forward; the legacy aliases remain supported only for the
  older per-camera path, unchanged by this story.
- **Gap: a brand-new camera cannot appear on the external backend until it
  has a space.** The intended onboarding flow (per the project brief) is
  "edge registers a camera → external service assigns it a room/floor" —
  `contracts/worker_config.py`'s `PulledCameraConfig` already has
  `space_name`/`floor_name` fields anticipating the backend filling these in
  *after* registration. But `EdgeCameraMappingRequestDto.spaceId` is
  required *from the edge on the same call that creates the mapping*, so a
  freshly registered camera with no `space_id` yet is permanently excluded
  from every roster push (reads as `pending`, never even attempted) until an
  operator assigns it a space locally. There is currently no way for the
  external backend to assign a room to a camera it has never seen. This ADR
  does **not** implement a fix (out of scope: eldercare-fall-ml-v2 only, per
  the brief's scope decision) — it records the gap as a proposed follow-up
  for the fall-ai repo: make `spaceId` optional on `EdgeCameraMappingRequestDto`
  (`@IsOptional()`), allowing an edge to register a camera with no space and
  have the backend (or an operator there) assign one later, at which point a
  subsequent `sync_camera_roster` attempt — triggered once the local registry
  also picks up a `space_id`, e.g. via the config-refresh loop pulling
  `space_name` back down — would push the update.
- `ConnectionSettingsStore` was deliberately left untouched (not widened to
  recognize the three legacy aliases) — that store is owned by stories
  G001/G003 and widening its fallback chain was out of scope for G004.

## Alternatives considered

**Route the roster push through `BackendCameraMapper.from_env()` (raw env,
all four token aliases) instead of `ConnectionSettingsStore`.** Rejected: the
G004 brief explicitly required dashboard-configurable, no-restart endpoint/
token resolution, which only the store provides; `from_env()` cannot pick up
a token saved live via the connection settings UI.

**Widen `ConnectionSettingsStore`'s token fallback to all four aliases to
match `from_env()`.** Deferred, not rejected: this would remove the
behavioral split above, but touches a module owned by a different story and
was not requested by the G004 brief. Left as a follow-up if the split proves
to be a real operational problem.

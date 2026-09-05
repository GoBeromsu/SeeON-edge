# shared/api

HTTP + DTO boundary. Features call this. Screens stay out.

## Ownership
- `client.ts`: dashboard JSON verbs. Re-exports types and session URL helpers.
- `topologyClient.ts`: floors, rooms, sync, preview, confirm.
- `clipPagination.ts`: paged `/clips` plus one-clip metadata.
- `types.ts`, `topologyTypes.ts`, `clipPaginationTypes.ts`: DTOs only.
- `*Normalizer.ts` + `normalizerFields.ts`: parse or throw. Barrel is `normalizers.ts`.
- `http.ts` + `session.ts`: `requestJson`, `HttpError`, 401/403 bus, `getApiBase()`, media URLs.
- `usePollingResource.ts`: shared poller plus named resource hooks.
- `useMjpegStream.ts`: `fetch` + canvas MJPEG hook (Content-Length framed parts, 3s stall reconnect, backoff). Used by operations `LiveStreamPanel` and `shared/ui/BedZoneRecognitionPanel`.

## client.ts
All traffic goes through `requestJson`. Paths are `/api/v1`-relative. Encode ids.
`cameraBody` and `connectionBody` omit unset keys. Included `null` clears. Omitted means leave it.
Connection writes send facility code, token, and installation ref. Hub address stays env/image.
`saveDetectionSettings` is a full replace. Send every domain.

Narrow failures here: `cameraProbeFailureDetail` (422), `cameraDuplicateDetail` (409), `bedZoneRecognitionFailureDetail` (422). Features don't parse `HttpError.body`.
New 200 envelopes get a normalizer. Don't add another `as Type` on `requestJson`. Policies, incidents, and artifacts still cast. Shrink that set.

## types
Projected API shapes. `Camera.rtsp_url_masked` only, never a live RTSP secret.
`floor` is the user override. `floor_name` is roster. Display is `floor ?? floor_name`.
`Clip.duration_s` and `size_bytes` may be absent on old backends. Absent means unknown. Don't invent a gauge.
`ClipStorageInfo.mount_label` is a label, not a host path. Incident and clip views show API ids and states only.

## normalizers
Reject unknown success shapes. Throw. Don't render the raw payload.
Camera list/response is strict on required fields. On-camera `bed_zone` is defensive and becomes null. `normalizeBedZoneRecognitionResponse` is strict.
Clip `video_path` is rewritten with `getClipVideoUrl`. The browser never sees a store path.
Legacy clip-store `root` becomes a basename label. `toEventFacet` maps to `fall` / `bed-exit` / `other`.
Missing or malformed connection `heartbeat_relay` becomes the disabled default.

## session / http
`credentials: 'same-origin'`. Session is an HttpOnly cookie. No bearer, no `localStorage`, no `document.cookie`, no token in a query string.
`VITE_ML_API_BASE_URL` is cross-origin QA only. Production stays same-origin `/api/v1`.
401/403 notify `subscribeUnauthorized` then throw. `AuthGate` listens. This package does not paint the gate.
Media builders live here: stream, snapshot, clip video, clip thumbnail.

## polling
`usePollingResource` owns abort, request id, interval, `retry`, and `replace`.
Call `replace` with the mutation response. Don't wait for the next tick to show an accepted write.
`useRuntimeSettingsResource` drops a read when `incoming.version < current.version`.
Named hooks own their intervals. Don't invent a second poller in a feature. `enabled: false` stops the work.

## runtime parsing
`normalizeStatusSnapshot` requires `cameras` plus `runtime.cameras`. Bad rows drop. The envelope still must be a record.
`stale` on a runtime camera means the worker may be dead while `measured_fps` is last-known.
Device `backend` names the live decode/inference path, not CUDA-only. `normalizeRuntimeSettings` requires boolean `clip_export_enabled` and an integer `version`.

## credentials
Login `POST /auth/session`. Probe `GET`. Logout `DELETE`. Rotate `PUT /auth/credentials` on an already-authed session. No current-password field.
Worker relay tokens, RTSP secrets, clip-store paths, and Hub credentials stay out of kept fields and UI copy.
Connection test may send unsaved form values. Persist is a separate `PUT`.

## tests
Colocate: `client.test.ts`, `normalizers.test.ts`, `usePollingResource.test.tsx`, `useMjpegStream.test.ts`, `topologyClient.test.ts`, `clipPagination.test.ts`, `clipThumbnail.test.ts`.
Client tests reject contract-invalid 200 envelopes. Normalizer tests stay table-style.
Polling tests use `createRoot` + `act`. Subscribe, act, then await. No `sleep`.
Assert URLs, statuses, and DTO fields. Don't pin Korean copy unless it's a sentinel.

## Forbidden
Widgets, CSS, wizard steps, and page copy live in `features/*` and `shared/ui`.
This package must not import `@/features/*` or mount `AuthGate`, MJPEG, or clip players.
A second HTTP client, or a `fetch` that skips credentials or the 401/403 bus, is out.
Camera seed, YAML import, and silent roster pull stay out of the browser.
Never poll `edge.sqlite3`, worker ports, or the clip-store directory.
Omitted size, duration, or storage used is unknown. Show guidance. Don't guess.

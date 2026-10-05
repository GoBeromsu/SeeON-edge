# shared/: cross-slice leaf

Building blocks used by 2+ slices or by `app/`. No barrel: import the exact module (`@/shared/ui/Toast`), never `@/shared` or `@/shared/ui`. `api/` has its own `AGENTS.md`; this file covers `ui/`, `format/`, and `releaseIdentity.ts`. Earned its file as the highest-fan-in area of the SPA (score 12, distinct domain).

## Where to look

| Need | Module | Notes |
| --- | --- | --- |
| Modal or sheet | `ui/AccessibleDialog.tsx` | Portal, focus trap, `variant` `dialog`/`sheet`, `initialFocus` `first-control`/`heading`. |
| Toast | `ui/Toast.tsx` | Imperative `toast.success` / `toast.error`, callable outside React. |
| Status tone | `ui/StatusBadge.tsx` | Four variants: `approved`, `rejected`, `pending`, `closed`. |
| Login gate | `ui/AuthGate.tsx` | Also exports `useAuthSession`. |
| Clip video | `ui/AutoplayVideo.tsx` | Reports `ready` / `blocked` / `failed` through `onPlaybackState`. |
| Clip poster | `ui/ClipThumbnail.tsx` | URL from clip id via `getClipThumbnailUrl`. |
| Bed region | `ui/BedZoneRecognitionPanel.tsx` | Mounts `BedZonePolygonEditor`. |
| Overlay subject icon | `ui/OverlayTargetIcon.tsx` | Keyed by `keyof OverlaySelection`. |
| Byte size | `format/bytes.ts` | `formatBytes`. |
| UUID | `format/uuid.ts` | `generateUuidV4`. |

## ui contracts

- `AccessibleDialog` sizes are the design-handoff modal widths: `xs` 380, `sm` 420, `md` 440, `default` 520, `lg` 640, `xl` 720. Pick a size, don't add a width class in a feature.
- `ToastViewport` mounts once, in `app/App.tsx`. A feature calls `toast.*`; it never mounts a second viewport.
- `StatusBadge` tones map to the four semantic tokens in `styles/tokens-base.css`. `getBackendStatus` and `getConnectionStatus` share one `{configured, reachable}` contract with different wording. Add a helper here instead of a per-feature class map.
- `AuthGate` is the only caller of `loginDashboard` / `fetchDashboardSession` / `logoutDashboard` and the only `subscribeUnauthorized` listener. `NavBar` reads the session through `useAuthSession`.
- `BedZoneRecognitionPanel` props: `cameraId`, `bedZone`, `onSaved`, `onCancel`. It owns `recognizeBedZone` + `saveBedZone` and one snapshot via `getCameraSnapshotUrl`. No live stream. Consumers: settings `CameraEditModal`, operations `RoomDetail`.
- `BedZonePolygonEditor` is internal to that panel. Region ids come from `generateUuidV4`. It reports draft validity through `onDraftValidityChange`; the panel gates save on it.
- `OverlayTargetIcon` is the only component here without a sibling test.

## format contracts

- `formatBytes` is SI base-1000 on purpose: the design spec shows 8,400,000 bytes as "8.4 MB". MB/GB keep one decimal with no trailing `.0`; KB/B are whole numbers. Don't switch to 1024.
- `generateUuidV4` draws from `crypto.getRandomValues`. Output must satisfy `contracts/edge_provisioning_validation.py::require_uuid`: lowercase, version nibble `[1-8]`, variant nibble `[89ab]`.

## releaseIdentity

`EDGE_DATABASE_FORMAT_IDENTITY = 'seeon-edge-v1'`, `EDGE_DATABASE_SCHEMA_VERSION = 19`. Mirrors repo-root `shared/release_identity.py` (the backend `edge_db` schema number). A schema bump changes this constant and `releaseIdentity.test.ts` in the same PR. Only `main.tsx` reads it.

## Anti-patterns

- A component here that fetches a resource a feature already polls. Take the data as props. `AuthGate` and the bed-zone panel are the two deliberate exceptions.
- Feature-specific copy or layout in `ui/`. A piece moves here when a second slice needs it, not before.
- An `index.ts` barrel. Import paths stay exact so the lint boundary rules can read them.
- A second dialog, toast stack, or status-colour map in a feature.
- A live MJPEG stream inside the bed-zone panel. It edits against one still snapshot.

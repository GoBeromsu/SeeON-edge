# features/: domain slices

One folder, one capability. `app/pages/` composes these. Parent already names the folders and the three frozen seams. This file is the in-slice contract.

## Ownership

- `operations/`: live wall + room detail. Owns MJPEG (`useMjpegStream`), snapshot queue, overlay fetch/set, floor/liveness sort (`operationsModel`), room clip history. `ClipPlayerModal` plays that history. No delete, no artifacts, no analysis.
- `events/`: clip catalog + incident review. Owns filters, pager, `useEventsPage` / `useClipMetadata`, `eventTypes` facets (`bed-exit` / `fall` / `other`). `ClipPlaybackModal` is the evidence surface: delete, artifacts, analysis, derivative control.
- `settings/`: registry writes and site policy. `CameraSection` orchestrates table + register + edit + delete. Its `DetectionSettingsCard` is the global schedule editor (`saveDetectionSettings`). Also clip storage/export, policy evidence, processing status, bed-zone panel.
- `connection/`: 3-step wizard + `ConnectionSettingsPanel`. `wizardSteps.ts` is pure and server-state only (`enrolled`, camera total, `dirty_registry_version`, `readiness_error`, `preview.confirmed`). SettingsPage mounts the wizard. No page of its own.
- `cameras/`: topology widgets only. Pairing list, structure editor, confirm dialog. No wizard progress, no registry table. Wizard steps 2 and 3 consume them.
- `account-settings/`: single-admin username/password rotate. `App` mounts the modal. Already-authed `PUT` credentials. No current-password field.

## Allowed deps

- `@/shared/api/*`: client, types, http, polling hooks, topology client, session URL builders.
- `@/shared/ui/*`: AccessibleDialog, Toast, StatusBadge, ClipThumbnail, AutoplayVideo, AuthGate.
- `@/shared/format/*`: bytes, uuid. Use `generateUuidV4`, never `crypto.randomUUID`.
- `@/app/dashboardLocation` types and helpers, only through each slice's `use*Location`.
- Own-slice modules. React. vitest in tests.

Forbidden: sibling slices except the frozen seams below. `App.tsx`. A second fetch client. `localStorage` for wizard or nav.

## Frozen crossings

Do not add a fourth. Lift a new shared widget to `shared/ui` or `shared/api`.

`connection` -> `cameras` (one way):
- Step 2 `CameraSyncStep` mounts `CameraPairingList` + `TopologyStructureEditor`, then `syncTopology()`.
- Step 3 `ServerSyncStep` mounts `TopologyConfirmationDialog`. `preview.cameras/rooms/floors` counts are deactivations, not creates.
- `cameras` never imports `connection`. Pairing is not the settings registry. Don't fold `CameraPairingList` into `CameraSection`.

`operations` -> `settings`:
- `RoomDetail` reuses `CameraEditModal` / `DeleteCameraDialog`. Same sibling-dialog rule as `CameraSection`: edit never confirms its own delete. Nested `AccessibleDialog` inerts the first dialog.
- Room `DetectionSettingsCard` is read-only. It imports `DOMAIN_LABELS`, `DOMAIN_ORDER`, `formatDomainSchedule` from `settings/detectionSettingsForm`. Gear calls `navigateToPage('settings')`. Global `GET /detection-settings` is not per-camera. Don't import the settings editor card.

`settings` -> `operations`:
- `BedZoneRecognitionPanel` reuses `useMjpegStream` (live canvas, not a still). Recognition is `POST /cameras/{id}/bed-zone/recognize` on a 2s interval while the session is active. Polygon overlay is a sibling `<svg>`. Drawing on the MJPEG canvas loses the next frame.

Same-named cards stay separate files. operations card = status + overlay + navigate. settings card = editor. Don't merge by import.

Clip modals stay separate. operations `ClipPlayerModal` = room-history playback. events `ClipPlaybackModal` = delete / artifacts / analysis. Don't grow delete into operations.

`operations/crossPageNavigation.ts` fakes `?page=` + `popstate` because App owns the controller. Only operations uses it. Don't invent a second navigator. Don't import App.

## Tests

Colocate: `Foo.tsx` next to `Foo.test.tsx` (or `.ts`) in the same slice.
Pure helpers stay table-driven beside the module: `operationsModel`, `wizardSteps`, `connectionSettingsForm`, `detectionSettingsForm`, `eventTypes`, `formatters`, `SnapshotQueue`, `rtspSubstreamGuidance`.
Facet splits stay next to the owner (`ClipPlaybackModal.delete.test.tsx`).
Page suites live under `app/pages/`, not here.
`cameras` unit-tests the confirm dialog. Pairing and editor coverage rides `connection` step tests.
`src/test/` is setup only.

## Anti-patterns

- A new feature->feature import "just this once."
- Client-remembered wizard step. Refresh must land on server-known `enrolled` / topology / preview.
- Treating zero cameras + a clean `dirty_registry_version` as step-2 success.
- Copying `useMjpegStream` into settings, or an `<img multipart>` live tile.
- Seeding or importing cameras in the browser. Registry writes go through the API.
- Showing clip-store paths, RTSP secrets, or Hub credentials.
- Pinning Korean prose in a test unless it is a sentinel or `data-testid`.

# Dashboard src (feature-sliced)

Vite + React SPA. `@/*` maps to `src/*` (vite.config + tsconfig `paths`).
Import that way in app, features, shared, and tests. Never `../../`.

## Ownership

- `main.tsx`: Vite entry only. Stays at `src/`. Stamps `data-seeon-edge-format` / `data-seeon-edge-schema` on `<html>` from `shared/releaseIdentity`.
- `app/`: shell and URL state. `App.tsx` wraps `AuthGate` + `Dashboard` (NavBar, page switch, account modal, toasts). `dashboardLocation.ts` owns query keys (`page`, `floor`, `camera`, `event`, `clip`) and canonicalize/restore. `app/pages/` holds the three route pages: Operations, Events, Settings. Pages compose features; they do not own view widgets.
- `features/<slice>/`: components + view logic for one capability. A slice may add its own `AGENTS.md`.
  - `operations/`: camera wall, room detail, live tiles (via `shared/api/useMjpegStream`), overlay, snapshot queue, room event history.
  - `events/`: clip grid, filters, keyset pager, clip playback of unmodified source bytes with an optional client-drawn overlay canvas; snapshot JPEG carries burned boxes.
  - `settings/`: camera registry table/modals, detection form, clip storage/export, policy evidence, processing status, bed-zone panel.
  - `connection/`: edge setup wizard + connection settings panel.
  - `cameras/`: topology editor, pairing list, confirm dialog. Consumed by the connection wizard.
  - `account-settings/`: single-admin username/password modal. Mounted from `App`, not a page.
- `shared/`: cross-slice building blocks only.
  - `shared/api/`: HTTP (`http.ts`), session/base URL (`session.ts`), `client.ts` + per-resource normalizers, DTO types, polling hooks (`usePollingResource.ts`), MJPEG live-stream hook (`useMjpegStream.ts`, sole consumer is operations `LiveStreamPanel`), topology client.
  - `shared/ui/`: shared interface pieces: NavBar, AuthGate, AccessibleDialog, Toast, StatusBadge, ClipThumbnail, AutoplayVideo, OverlayTargetIcon, BedZoneRecognitionPanel + BedZonePolygonEditor. Contracts in `shared/AGENTS.md`.
  - `shared/format/`: tiny formatters (`bytes`, `uuid`).
  - `shared/releaseIdentity.ts`: edge DB format identity + schema version constants.
- `styles/`: token + shell CSS imported from `styles.css` in fixed order: tokens-base, application-surfaces, shell-auth-dialog, dashboard-shell-session, responsive-layout. Semantic `:root` custom properties live in `tokens-base.css`; Tailwind does layout. Don't dump feature layout here.
- `test/setup.ts`: jsdom act flag + `HTMLMediaElement.play` stub. Vitest `setupFiles` points here.
- `test/testConfig.test.ts`, `test/importStructure.test.ts`: machine guards for the Vitest determinism config and the `client` module import structure. The only tests allowed in `test/`.

## Imports

Allowed: `@/app/*`, `@/features/<own-slice>/*`, `@/shared/*`.
Features may read `dashboardLocation` types/helpers for their `use*Location` hooks.
`shared/` must not import `@/features/*` or `@/app/*`.
`shared/ui/NavBar` already types against `@/app/dashboardLocation`. Don't spread that pattern.
These rules are lint errors: `front/eslint.config.js` (boundary rules only, `pnpm --dir front lint`). NavBar is the single file-level exception there.

## Cross-slice seams (keep them few)

Existing feature→feature imports, do not grow new ones:

- `connection` → `cameras` (wizard pairing + topology confirm).
- `operations` → `settings` (`RoomDetail` reuses `CameraEditModal` / `DeleteCameraDialog`; room `DetectionSettingsCard` reads `detectionSettingsForm` labels).

Need a new shared widget or hook? Lift it to `shared/ui` or `shared/api`. Don't import a sibling slice "just this once."
Two cards share a name (`DetectionSettingsCard` in operations and settings). They are different files. Don't merge them by import.

`operations/crossPageNavigation.ts` fakes a `?page=` + `popstate` because `App` owns `DashboardLocationController`. Don't reach into `App` from a feature. Don't invent a second navigator.

## Tests

Colocate: `Foo.tsx` + `Foo.test.tsx` (or `.ts`). Page suites may split as `EventsPage.<facet>.test.tsx` next to the page, with `EventsPage.testSupport.tsx` for a shared harness.
Import `describe` / `it` / `expect` / `vi` from `vitest`. Use `@/` in tests too.
Component tests: `createRoot` + `act`, wipe `document.body` and restore mocks in `afterEach`.
Pure helpers get a `.test.ts` with table-style cases (see `operationsModel.test.ts`).
`src/test/` is setup plus the two config guards, not a test dump.

## Where to look

- Route or query key → `app/dashboardLocation.ts`, then the page under `app/pages/`.
- Screen widget → owning `features/<slice>/`. Page file stays a composer.
- HTTP / DTO / normalizer → `shared/api/` (`client.ts`, `types.ts`, a `*Normalizer.ts`).
- Presentational primitive used by 2+ slices → `shared/ui/`.
- URL / media builders → `shared/api/session.ts`.
- Visual copy/layout contracts → `front/design-handoff/README.md` (pages cite section numbers).
- Alias / Vitest env → `front/vite.config.ts`, `front/tsconfig.json`.

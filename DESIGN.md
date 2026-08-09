# ElderCare ML Control Design Contract

## Authority and provenance

This document is the design contract for the completed `front` redesign — global shell, tokens, accessibility, interaction primitives, and the three implemented pages. Its design source is the Claude Design prototype bundled at `front/design-handoff/`: `front/design-handoff/README.md` is the authoritative spec for page-level visual and interaction detail (관제/이벤트/설정 screens and their modals), and `Eldercare Prototype.dc.html` in the same directory is the interactive visual reference the spec describes. The bundle is reference-only — never imported, copied, or edited at runtime — and where this document and the handoff spec disagree on page content, the handoff spec wins and this document must be updated in the same change. Section-by-section, this document also records where the shipped implementation deliberately deviates from the handoff spec (a field the API doesn't provide, a UI simplification); those deviations are the current design contract, not bugs.

The product is a single-facility edge console: one eldercare facility's on-site edge device, operated by installers and facility staff, not a multi-tenant SaaS product. There is no facility switcher, no backend-status ribbon, and no multi-facility concept anywhere in the shell.

The local implementation sources are `front/src/styles.css` (and its `front/src/styles/*.css` imports), `front/tailwind.config.js`, and feature-sliced React components under `front/src`. A new component or token must reuse a semantic contract here or update this document in the same change.

## Product identity and goals

The product is `Senior AI Lab Edge`, a calm operational safety console for checking camera availability, reviewing fall and bed-exit evidence, and managing the single admin account and device connection. It should feel quiet, exact, and trustworthy during repeated daily use.

The interface must reveal what is known, when it was last known, and what the operator can do next. It must never imply live data, camera health, or history that the current APIs do not provide.

## Information architecture

There are exactly three primary destinations, all reachable from the sticky top `NavBar`. At `768px` and wider it is a single 56px row; below `768px` it reflows into two rows without changing navigation content. There is no left rail and no bottom tab bar.

| Query value | Navigation label | Owns |
| --- | --- | --- |
| `operations` | 관제 | Camera view/selection (default destination) |
| `events` | 이벤트 | Historical event/clip evidence |
| `settings` | 설정 | Camera, connection, and system settings |

The `NavBar` also carries the brand mark (text, left), the three nav destinations (grid/clock/gear inline SVG icons, `front/src/shared/ui/NavBar.tsx`), an account-settings icon button (opens `AccountSettingsModal`), and — only while a session exists — a logout button. It never renders a backend-status pill or facility name; connection status is surfaced only inside the 설정 page. The active destination uses a `bg-muted` rounded chip and `aria-current="page"`; the room-detail view under 관제 keeps the 관제 destination active.

All three destinations plus the login gate are implemented against `front/design-handoff/README.md`; there is no placeholder-page state left. Page-level content (filters, camera wall, room detail, clip review, settings cards) is owned by the handoff spec; the sections below (`관제`, `이벤트`, `설정`, `모달`) record the implemented reality and its deliberate deviations from that spec.

## 관제 (Operations)

`front/src/app/pages/OperationsPage.tsx` composes the camera wall and, once a `camera` query value resolves against the loaded camera list, swaps to room detail — both under the single `관제` destination (`useOperationsLocation`, `front/src/features/operations/useOperationsLocation.ts`). There is no page-level `h1` in `OperationsPage` itself: the wall and room-detail views each own their own primary heading (matching the handoff spec's per-view header rather than a duplicated "관제" title above a second, view-specific one), and each still exposes exactly one `[data-dialog-focus-fallback]` heading so `AccessibleDialog`'s close-focus fallback keeps working.

**Camera wall** (`CameraWall.tsx`, `CameraWallTile.tsx`): the header row is `h1` "관제" plus two count badges — `온라인 {n}` (approved/green) and `오프라인 {n}` (rejected/red), counts computed client-side (`countCamerasByLiveness`) with a non-breaking space so a badge never wraps mid-count — with a right-aligned floor `<select>` (전체 + one `<option>` per distinct `floor_name`, same control styling as the 이벤트 page's `CameraFilterSelect`) replacing the floor-chip row. Below that, a responsive grid (1 col → 2 at `sm` → 4 at `lg`) of 16:9 tiles: each online identity requests one queued snapshot when it becomes active in the filtered wall (`useSnapshotQueue.ts`/`SnapshotQueue.ts`), with no recurring refresh and no MJPEG connection; deactivation followed by reactivation permits one new snapshot. The tile overlays `linear-gradient(transparent, rgba(0,0,0,.72))` with the camera label and a green status dot; an offline tile is an `aspect-video bg-muted` panel with centered "오프라인" text (no dark media frame), plus a separate white bottom bar with the label and a red dot. The whole tile is the click target to open room detail.

**Room detail** (`RoomDetail.tsx`): a breadcrumb ("관제" link `/` camera name, the camera name is the `[data-dialog-focus-fallback]` `h1`) plus an online/offline status badge, then a 2/3-1/3 grid — `LiveStreamPanel` (live view) on the left, `CameraInfoCard` + `DetectionSettingsCard` stacked on the right — followed by `EventHistoryList` below. `RoomDetail` also owns the shared `CameraEditModal`/`DeleteCameraDialog` instances (imported from `front/src/features/settings/`, not duplicated) so 연결 관리 opens in place; a delete from here removes the camera being viewed, so it also navigates back to the wall (`onBack`) after refreshing (`onRetryConnection`, which is `OperationsPage`'s `camerasResource.retry`). Stacking two `AccessibleDialog`s is avoided the same way `CameraSection` avoids it: `CameraEditModal` never opens its own delete-confirm dialog, it bubbles the request up via `onRequestDelete` to a sibling `DeleteCameraDialog`.

- `LiveStreamPanel.tsx` is the only operations surface that opens MJPEG: it renders one selected camera's MJPEG-over-`fetch`+`<canvas>` live view (`getCameraStreamUrl`, no WebRTC/HLS) with a top-right diagnostics chip. Deviation from the handoff spec: the spec describes separate decode/inference FPS and a running frame counter; `RuntimeCameraDiagnostics` only ever reports one `measured_fps` plus a decode backend name, so the chip shows only `"{fps} FPS · {backend}"` (or `FPS 측정 중` while unmeasured) rather than fabricating the extra fields. An offline camera renders the spec's gray `bg-muted` panel — "카메라에 연결할 수 없습니다" / "탐지가 중단된 상태입니다" — with [재연결 시도] (calls `onRetryConnection`, re-polling the cameras resource) and [연결 관리] (opens the shared `CameraEditModal` via `onManageConnection`) buttons.
- `CameraInfoCard.tsx` is now header-title-as-camera-name with an xs outline "연결 관리" button in the header (opens the shared `CameraEditModal` in place via `onManageConnection`) and a `dl` of 층/RTSP 주소(masked, `rtsp_url_masked`) only — the 상태 row and the old full-width bottom button are removed, since status now shows as the breadcrumb-row badge and 연결 관리 no longer navigates away.
- `DetectionSettingsCard.tsx` (operations variant, distinct from the settings-page card of the same name, now takes the `camera` object rather than just its id) shows the global `GET /detection-settings` domains read-only in a 3-col grid (이벤트명 / schedule, tabular right-aligned / status pill), reusing the settings feature's `DOMAIN_LABELS`/`DOMAIN_ORDER`/`formatDomainSchedule`, and a gear-icon header button that navigates to 설정 (`navigateToPage('settings')`) rather than editing in place. The status pill has three states, not two: 탐지 중 (approved — domain `on` and the camera is online), 꺼짐 (closed — domain `off`), 중단됨 (pending — the camera is offline, which overrides the domain's own on/off flag regardless of its saved state, since an offline camera can't be detecting anything). `OverlayModeControl` still renders below a border inside this same card.
- `OverlayModeControl.tsx` is a three-chip `role="group"` (오버레이 없음/침대 이탈/낙상) backed by `GET`/`POST /streams/{camera_id}/pose`; the selected chip is `aria-pressed` + primary-filled, selection is optimistic-disabled (`pending`) during the POST, and a failed POST reverts with a toast (`오버레이 모드를 변경하지 못했습니다.`).
- `EventHistoryList.tsx` polls `GET /clips` scoped to the camera and offers a 최신순/종류별 sort select; 최신순 is a flat 4-col clip grid, 종류별 groups clips under `"{종류} · n건"` headings. Each tile is a muted video preview (`clip.video_available` gates a native `<video muted>` vs. an unavailable-copy fallback) plus type label and timestamp; selecting one opens `ClipPlayerModal` (`front/src/features/operations/ClipPlayerModal.tsx`, `size="xl"`, same shape as the 이벤트 page's `ClipPlaybackModal`).

## 이벤트 (Events)

`front/src/app/pages/EventsPage.tsx`: a filter row of `EventTypeFilterChips.tsx` (전체 n / 침대 이탈 n / 낙상 n, counts derived client-side from the loaded `GET /clips` page) on the left and `CameraFilterSelect.tsx` on the right (`ml-auto`), driving a 4-col `ClipGrid.tsx` (each card labels its camera in the top-left of the thumbnail via `resolveCameraLabel.ts`). The empty state reads `조건에 맞는 이벤트가 없습니다.`. Selecting a clip opens `ClipPlaybackModal.tsx`. TP/FP labeling (the old `PUT /clips/{id}/label` review workflow) is intentionally removed from this redesign — the modal shows the event-type chip in its place, and the label-mutation call is deleted from the frontend entirely (backend route removal is out of scope).

The `camera`-scoped `GET /clips` request is filter-driven, not URL-driven (`selectedCameraId` local state) and doesn't restart `usePollingResource`'s own poll loop on a camera-filter change, so `EventsPage` explicitly calls `clipsResource.retry()` on every `selectedCameraId` change to avoid up to an 8s stale window.

## 설정 (Settings)

`front/src/app/pages/SettingsPage.tsx`: a 7:5 two-column grid (`lg:grid-cols-12`, 7/5 span) — `CameraSection` on the left; `DetectionSettingsCard`, `ConnectionSettingsPanel`, `ProcessingStatusCard`, `ClipStorageCard` stacked on the right.

- **카메라** (`CameraSection.tsx` + `CameraTable.tsx`): header with a primary "카메라 등록" button (camera+plus SVG) opening `CameraRegisterModal`; a table of 카메라(label + masked RTSP)/층/**침대 영역**/상태/작업. The 침대 영역 column (인식 완료 approved-badge / 인식 필요 rejected-badge, from `camera.bed_zone`) is an addition beyond the handoff spec's 카메라/층/상태/작업 columns, reflecting the bed-zone feature. An offline row is tinted `bg-status-rejectedBg`. 수정 opens `CameraEditModal`; 삭제 opens `DeleteCameraDialog` (a plain confirm dialog, "되돌릴 수 없습니다", destructive-filled confirm), not an immediate delete. Below the table, a `font-mono` version line reads `v{version} · ml-api@{digest} · ml-worker@{digest}` (parts omitted if `GET /system` doesn't report them, never fabricated) — no separate technical-info card.
- **탐지 설정** (`DetectionSettingsCard.tsx`, settings variant): read mode is a 3-col grid per domain (이벤트명 / schedule, right-aligned tabular / status badge — 탐지 중 approved when `on`, 미탐지 closed when off). The pencil button enters edit mode: per-domain checkbox + 항상/시간 지정 select + (`mode=window`) two `type="time"` inputs. Every field change calls `PUT /detection-settings` immediately (no save button; "변경 즉시 모든 카메라에 적용됩니다." caption), and a failed PUT toasts and reverts the local draft. A ghost "완료" button exits edit mode.
- **서버 연결** (`ConnectionSettingsPanel.tsx`, `front/src/features/connection/`): read mode is a `dl` of 시설 ID(`font-mono`)/시설 토큰(masked)/마지막 동기화, with a connection-status badge and pencil in the header. The pencil opens an inline form below a border: 시설 ID + 시설 토큰 inputs, a "연결" test button (`testConnection`, shows `연결 성공 · {detail}` inline in green on success, the raw detail string on failure) and a primary "저장" submit (`saveConnection`, closes on success + toast). This is the only settings-card flow with an explicit save button — 탐지 설정 and 계정 are immediate/no-current-password respectively, but connection changes are deliberately not auto-applied.
- **처리 상태** (`ProcessingStatusCard.tsx`): a status badge (정상/중단됨/확인 중, from `runtime.worker.alive`) plus a `dl` of 실행 디바이스 (`device_name · backend`, e.g. `Apple M2 · MPS`), 디코드 (first camera's `decode.selected`/`decode.requested` as a representative sample — there is no single global decode field), 인코드 (`clip_recorder.encoder`), 전송 지연 (`최대 {max_sec}초` across cameras' `latency.max_sec`). No CUDA-only diagnostics (FPS averages, NVML) are rendered regardless of device — the card renders whatever the backend reports, without a CUDA-specific code path.
- **클립 저장 공간** (`ClipStorageCard.tsx`): usage line `"{used} / {total} GB"` (tabular, one decimal, `—` when unknown) over an 8px pill progress bar (`role="progressbar"`), then a `font-mono` "저장 위치 {root}/{selected_path}" line and a "변경" button opening `FolderBrowserModal`. A successful `PUT /clips/storage/location` toasts and refreshes; a failure toasts `저장 위치를 변경하지 못했습니다.` without closing the modal state.

`CameraRegisterModal`'s 1단계 in the shipped implementation deviates from the handoff spec: there is no floor-chip picker. `floor_name` is a read-only field populated by an external space-sync process, so step 1 collects only 이름 and RTSP 주소, and `POST /cameras` performs the RTSP probe and the save in a single call (no separate probe endpoint) — a probe failure (`timeout`/`auth`/`decode`) or a duplicate-stream conflict (offering a "그래도 등록" force-register retry) surfaces inline rather than blocking on a separate step. 2단계 is `BedZoneRecognitionPanel` (shared with `CameraEditModal`'s 다시 인식 flow): a live camera snapshot (`getCameraSnapshotUrl`, refetched by key after each recognize) with the persisted bed polygon drawn as an SVG overlay once known, an "인식 중..." status overlay while `POST /cameras/{camera_id}/bed-zone/recognize` is in flight, and an "인식 시작"/"다시 인식" button. This is a simpler implementation than the handoff spec's animated dashed-polygon-pulse preview — recognition is a discrete server round trip (one YOLO segmentation pass per click), not a live client-side animation, and "저장하고 완료" is blocked with a toast until a `bed_zone` exists.

## 모달과 AccessibleDialog 크기 (Modals)

All five modals use `AccessibleDialog` (`front/src/shared/ui/AccessibleDialog.tsx`) with a `DialogSize` variant matching the handoff spec's pixel widths:

| Size | Width | Modal | Component |
| --- | --- | --- | --- |
| `xs` | 380px | 계정 설정 | `front/src/features/account-settings/AccountSettingsModal.tsx` |
| `sm` | 420px | 폴더 탐색기 | `front/src/features/settings/FolderBrowserModal.tsx` |
| `md` | 440px | 카메라 등록 1단계 · 연결 관리(view mode) | `CameraRegisterModal.tsx` (step 1) · `CameraEditModal.tsx` (mode `view`) |
| `lg` | 640px | 카메라 등록 2단계 · 연결 관리 다시 인식 | `CameraRegisterModal.tsx` (step 2) · `CameraEditModal.tsx` (mode `reseg`) |
| `xl` | 720px | 클립 재생 | `front/src/features/events/ClipPlaybackModal.tsx` · `front/src/features/operations/ClipPlayerModal.tsx` |

`default` (520px, unchanged) remains for other callers (e.g. `DeleteCameraDialog`, which uses no explicit `size`).

Deviation from the handoff spec: 연결 관리 and 침대 영역 다시 인식 are one modal (`CameraEditModal`) that switches view via internal `mode` state (`view` ↔ `reseg`), not two separately-opened modals. `AccessibleDialog` makes everything outside the open dialog `inert`; stacking a second dialog on top of a first would make the first dialog's own container `inert` too (it isn't a descendant of the second), producing a focus trap. Widening the same dialog to `lg` for the 다시 인식 sub-view avoids that bug. The same reasoning routes camera deletion through `CameraSection`'s single shared `DeleteCameraDialog` via an `onRequestDelete` callback rather than a delete-confirm dialog nested inside `CameraEditModal`.

**클립 재생** (`ClipPlaybackModal.tsx`/`ClipPlayerModal.tsx`, xl): header is `"{종류} · {카메라명}"` + timestamp; body is a native `<video controls>` (or an unavailable-copy fallback) inside `.event-media-frame`, then a `dl` of 카메라/시간/길이/해상도/크기(only when `size_bytes` is non-null — the row is omitted, never shown as `0` or fabricated). 길이 prefers the clip manifest's `duration_s`, falling back to the loaded `<video>` element's metadata; 해상도 is read only from loaded `<video>` metadata (the API's `Clip` type carries no dimensions field). Deviation from the handoff spec: there is no "파일" row showing the full clip-store path — `AGENTS.md` prohibits exposing the clip-store filesystem path or camera credentials in the UI, so only `video_path` (an API URL, never a raw filesystem path) is used, and the path row is dropped entirely rather than shown redacted. Footer has one action, 다운로드 (`<a download>` on `video_path`), shown only when `video_available`.

**폴더 탐색기** (`FolderBrowserModal.tsx`, sm): a current-path row (↑ parent-navigate icon button, disabled at root, plus a `font-mono` path) over a directory list (`GET /clips/storage/browse?path=`) with drill-down on click; an empty directory reads "하위 폴더 없음". Footer: 취소 / "이 위치 사용" (disabled until a browse response has loaded), which calls `PUT /clips/storage/location`.

**계정 설정** (`AccountSettingsModal.tsx`, xs): 아이디 / 새 비밀번호 / 새 비밀번호 확인 — no current-password field, matching the single-admin-account, already-authenticated model. Client-side validates length (≥4) and confirm-match before calling `PUT /auth/credentials`; success shows an inline `dialog-success` message and resets the password fields (username stays populated).

**카메라 등록** and **연결 관리**: see the 설정 page section above for their content; both are described there rather than duplicated here since their behavior is inseparable from the 카메라 카드 flow they belong to.

## Behavioral contracts the UI depends on

- **Overlay mode** — `GET`/`POST /api/v1/streams/{camera_id}/pose`, body/response `{"mode": "none"|"bedexit"|"fall"}` (`backend/app/features/cameras/streams_router.py`, proxying the worker's `/overlay/{id}/pose`). Rendering is entirely server-side burn-in on the worker (`worker/pipeline/output/overlay.py`'s `OverlayRenderer`) — the frontend never draws boxes/skeletons itself, it only selects a mode and displays whatever JPEG the stream returns. `none` draws nothing. `bedexit` draws every person box + pose skeleton, then a teal dashed outline per bed polygon labeled `bed:{occupancy}` (occupancy from the bed-exit domain's per-frame debug snapshot, keyed positionally to `observation.bed_boxes`). `fall` draws every person box + skeleton, then a per-track FALL (danger-red) or NORMAL (neutral) label positioned under that track's box.
- **Bed-zone recognition** — `POST /api/v1/cameras/{camera_id}/bed-zone/recognize` (dashboard session auth) triggers one server-side YOLO segmentation pass against the camera's latest frame; success (200) persists `{polygon, image_width, image_height, recognized_at}` server-side (`BedZoneStore`) and is echoed back under `bed_zone`. No usable bed found → 422 `{"detail": {"error_class": "bed_not_found"}}`, surfaced as "침대를 찾지 못했습니다…"; any other upstream failure (worker down, no frame, decode failure) → 503, surfaced as a generic retry message. `GET /cameras` includes each camera's persisted `bed_zone: {...}|null`, which drives the 인식 완료/인식 필요 badge everywhere it appears (카메라 테이블, 연결 관리, 카메라 등록 2단계).
- **Detection settings** — `GET`/`PUT /api/v1/detection-settings`, shape `{"domains": {"fall": {"on": bool, "mode": "always"|"window", "start": "HH:MM"|null, "end": "HH:MM"|null}, "bed_exit": {...}}}`; `PUT` replaces both domains atomically and returns the saved shape. This is one global, all-cameras setting — there is no per-camera override anywhere in the UI. Precedence: once an operator has saved a domain through this UI, that local value (`DetectionSettingsStore`) takes over from whatever the backend externally pulls, merged in at `worker_config_snapshot()` build time; a domain never explicitly saved falls back to reflecting the live externally-pulled window, and only falls back further to `on=true, mode=always` if there's no external window either — so a first-time visitor sees the schedule actually in effect, not a fabricated default. A save is not instantaneous on the worker: the worker only picks it up on its next poll cycle (tens of seconds), so the UI promises only "저장됨," never live effect.
- **Clip storage location** — `GET /api/v1/clips/storage` (`{root, selected_path, total_bytes, used_bytes, used_pct}`), `GET /api/v1/clips/storage/browse?path=` (`{path, parent, directories:[{name, path}]}`), `PUT /api/v1/clips/storage/location` (body `{path}`, returns the same shape as `GET /clips/storage`). Scope: every path is a subdirectory selection *inside* the `CLIP_STORE_DIR` mount (default `/var/lib/clip-store`) only — there is no way to change the host mount itself from this UI, and every client-supplied path is walked with `O_NOFOLLOW` dir_fd chaining so a crafted `../../etc` or a symlink planted inside the store can never resolve outside the configured root (`backend/app/features/clips/storage_router.py`).
- **Credentials** — `PUT /api/v1/auth/credentials`, body `{username?: string, new_password: string}`. There is no `current_password` field in the request or the form; the single-admin-account model treats an already-authenticated dashboard session as sufficient authorization to rotate credentials.
- **Clip metadata** — a clip's `duration_s` and `size_bytes` are optional manifest fields; the UI never fabricates them. 길이 falls back to the loaded `<video>` element's metadata when `duration_s` is absent; 크기 has no client-side fallback — the entire 크기 `dl` row is omitted from `ClipPlaybackModal` when `size_bytes` is `null`, never rendered as `0` or hidden behind a fake value.

## Shared resource states

Every polled resource exposes `idle`, `loading`, `success`, and `error` without collapsing prior truth:

| State | Presentation contract |
| --- | --- |
| First loading | Preserve layout, show a named loading state, and do not render invented rows, counts, or health. |
| Empty success | State what is empty and offer the supported next action; empty is not an error. |
| Refreshing | Retain last-good content and timestamp; indicate background refresh without blocking the page. |
| Stale last-good | Keep the last successful data, label it as delayed, show `마지막 확인`, and offer retry where useful. |
| Partial data | Render known fields, label unknown fields `정보 없음`, and keep independent health dimensions separate. |
| Unavailable media | Keep identity and metadata visible inside a stable dark frame; show `영상을 불러올 수 없음`. |
| Request error | Use a concise Korean explanation, a retry action, and a polite live region; never expose raw backend text. |
| Recovery | Replace the error with current truth, preserve relevant selection when still valid, and avoid celebratory motion. |

`usePollingResource` (`front/src/shared/api/usePollingResource.ts`) is the shared implementation: abort-on-cleanup, no overlapping polls, stale/out-of-order completions ignored, last-good data retained through a later error.

## Authentication and shell states

- The login card (`AuthGate`, `.auth-card`) is a quiet single-column card (`min(380px, 100%)`, `min-height: 360px`) on the neutral shell background: brand heading, username/password fields, one primary submit. Border-first, no shadow.
- After the form mounts, focus moves to the username field. Submit disables controls and shows `확인 중…`. Error copy has a reserved line so the card does not jump.
- Do not display or prefill credentials. Invalid credentials make no dashboard request beyond the login attempt; valid credentials establish the server session cookie only.
- Checking and connection-failure states reuse the same card shell so the layout does not jump.
- Logout is a `NavBar` action. It resets to `?page=operations` via `history.replaceState`, never a new history entry.
- `AccountSettingsModal` (single admin account, already authenticated) collects username and new password only — there is no current-password field. It calls `PUT /auth/credentials` with `{username?, new_password}`.

## Native URL query contract

Use the browser `URLSearchParams` and History APIs; add no router. The only recognized keys, serialized once in canonical order, are `page`, `floor`, `camera`, `event`, `clip`. There is no `room` or `wallPage` key.

For repeated recognized keys, the last occurrence wins. Remove earlier duplicates and all unknown keys. Empty values are invalid.

| Key | Exact value/source | Applies to | Canonicalization |
| --- | --- | --- | --- |
| `page` | `operations\|events\|settings` | Global | Missing or invalid becomes `operations`; a destination change removes all keys inapplicable to that destination. |
| `floor` | Exact non-null camera floor name | `operations` | After successful cameras data, unknown removes `floor`, then incompatible `camera`. |
| `camera` | Exact local camera `id` | `operations` | Operations removes an unknown or filter-incompatible camera. |
| `event` | Exact non-empty clip `event_type` in the successfully loaded clip set | `events` | Unknown removes `event`, then an incompatible `clip`. |
| `clip` | Exact clip `id` | `events` | Clip event must match active filters, or `clip` is removed after upstream invalidation. |

Deliberate destination, filter, and selection changes call `pushState`. Default insertion, duplicate/unknown/invalid removal, canonical ordering, data-driven invalidation, and logout/login reset call `replaceState`. A `popstate` restoration calls no `pushState`.

## Responsive viewport contract

The shell uses one sticky top `NavBar` at every width. At `768px` and wider it remains a single 56px row and only the horizontal gutter steps up. Below `768px`, the same bar becomes a two-row grid: the brand owns the first row, while the destinations and account/logout actions share the second row in DOM and focus order. There is no separate mobile chrome, no left rail, and no bottom tab bar at any width. Content below the `NavBar` is centered with `max-width: 1280px`.

At every width, critical controls wrap rather than clip, Korean text remains legible, the page has no horizontal overflow, and touch targets are at least 44px effective (compact 36px controls keep a 44px hit area where practical).

## Dialog semantics

`AccessibleDialog` is the sole modal primitive: portal to `document.body`, initial focus, Tab/Shift+Tab containment, Escape, safe-backdrop close, `inert` background, body scroll lock, and invoker focus restoration on close. When the invoking control is gone, focus falls back to `#main-content [data-dialog-focus-fallback]` or `#main-content h1[tabindex="-1"]` — every page must expose a focusable heading under `#main-content` for this reason.

Forms inside a dialog use the shared `.accessible-dialog form/label/input` styling and `.dialog-actions`/`.dialog-secondary-action` for the cancel/confirm button pair, mirroring the login card's `.auth-card label/input` styling so every dialog looks consistent without new one-off classes.

## Toast notifications

`ToastViewport` (`front/src/shared/ui/Toast.tsx`), mounted once at the app root, renders a bottom-right stack (`.toast-stack`). `toast.success(message)` / `toast.error(message)` show a `.toast`/`.toast-success`/`.toast-error` entry that auto-dismisses after 2.2s. This is the only notification surface; do not build a second one.

## Design tokens

### Color

One light application shell with dark media frames. Tokens are CSS custom properties on `:root` (`front/src/styles/tokens-base.css`), mapped into Tailwind's `theme.extend.colors` (`front/tailwind.config.js`) under shadcn-style names. There is no `surface2`, `ink`, `ink-soft`, `brand`, or `status-danger`/`status-stable`/`status-caution` token — those names do not exist in the palette and must never appear as Tailwind utility classes (`bg-surface2`, `text-ink`, `ring-brand`, …); using them produces unstyled output.

| Role | Token | Value | Tailwind class |
| --- | --- | --- | --- |
| Page/app canvas | `--background` | `#ffffff` | `bg-background` |
| Primary text | `--foreground` | `#1a1a1a` | `text-foreground` |
| Card surface | `--card` | `#ffffff` | `bg-card` |
| Card text | `--card-foreground` | `#1a1a1a` | `text-card-foreground` |
| Border | `--border` | `#e4e4e7` | `border-border` |
| Input border | `--input` | `#e4e4e7` | `border-input` |
| Muted surface | `--muted` | `#f4f4f5` | `bg-muted` |
| Muted text | `--muted-foreground` | `#71717a` | `text-muted-foreground` |
| Primary action fill | `--primary` | `#2f6fb0` | `bg-primary` (or the `.brand-action` class) |
| Primary action text | `--primary-foreground` | `#ffffff` | `text-primary-foreground` |
| Destructive fill | `--destructive` | `#dc2626` | `bg-destructive` |
| Destructive text | `--destructive-foreground` | `#ffffff` | `text-destructive-foreground` |
| Overlay teal | `--overlay-teal` | `#2bb6a3` | `text-teal` |
| Status approved bg | `--status-approved-bg` | `#e7f7ee` | `bg-status-approvedBg` |
| Status approved text | `--status-approved-fg` | `#166e3d` | `text-status-approvedFg` |
| Status rejected bg | `--status-rejected-bg` | `#fdeceb` | `bg-status-rejectedBg` |
| Status rejected text | `--status-rejected-fg` | `#b8261b` | `text-status-rejectedFg` |
| Status pending bg | `--status-pending-bg` | `#fdf2df` | `bg-status-pendingBg` |
| Status pending text | `--status-pending-fg` | `#884e07` | `text-status-pendingFg` |
| Status closed bg | `--status-closed-bg` | `#f4f4f5` | `bg-status-closedBg` |
| Status closed text | `--status-closed-fg` | `#71717a` | `text-status-closedFg` |
| Media frame | `--media-bg` | `#0d0d0d` | n/a (CSS only, `.event-media-frame`) |
| Media overlay fill | `--media-label-bg` | `rgba(0,0,0,.72)` | n/a (`.media-status-overlay`) |
| Media overlay text | `--media-label-fg` | `#ffffff` | n/a (`.media-status-overlay`) |
| Backdrop | `--backdrop` | `rgba(0,0,0,.42)` | n/a (`.dialog-backdrop`) |

Radii: `--radius-card: 12px` (`rounded-card`), `--radius-control: 8px` (`rounded-control`). Modal shadow: `--modal-shadow: 0 12px 36px rgba(0,0,0,.2)` (`shadow-modal`).

Every brand-filled primary action uses the `.brand-action` class (`background: var(--primary); color: var(--primary-foreground)`); do not rebuild that pairing from raw Tailwind color utilities. `StatusBadge.tsx`'s camera/backend status mapping predates this token set and still emits the retired `bg-surface2`/`text-ink-soft`/`bg-status-dangerBg`/`bg-status-stableBg` names — it is not yet reconciled with this table; do not copy its className strings into new code, and treat fixing it as follow-up work for whichever wave next touches it.

### Typography

`Pretendard`, `-apple-system`, `BlinkMacSystemFont`, `system-ui`, `Apple SD Gothic Neo`, `Noto Sans KR`, `sans-serif`. Page titles (`.shell-page-title`, `.auth-card h1`, `.accessible-dialog h2`) are 20px/600. Body copy is 14px; field labels 13px/600; the `NavBar` brand text is 14px/600. Use tabular numerals (`.tabular-nums`, `time`, `[role="meter"]`) for dates, durations, and counts.

### Spacing and geometry

- `NavBar` is a sticky 56px row at `768px` and wider, and a content-sized two-row grid below `768px`. `.app-main` centers content at `max-width: 1280px` with 16-24px horizontal padding depending on viewport.
- Controls default to a 36px height (nav chips, icon buttons, inputs, secondary/primary dialog actions) — smaller than the historical 44px minimum, matching the design-handoff spec's compact control scale; keep effective touch targets reasonable on touch devices.
- Radii: 8px for controls/nav, 12px for cards/dialogs.
- Borders use `1px solid var(--border)`; only dialogs/sheets use a shadow (`--modal-shadow`).
- Layer order: content `0`, sticky `NavBar` `20`, dialog backdrop `40`, dialog/sheet `50`, skip link/transient notice `60`, toast stack `70`.

## Content voice

Concise Korean-first operational language: name the object, state current truth, offer one supported next action. Prefer `다시 시도`, `마지막 확인`, `영상을 불러올 수 없음`, and similar honest, non-fabricated copy.

Do not expose `/api/v1`, Authorization headers, worker/relay internals, schema names, raw enums, digests, stack traces, camera credentials, RTSP URLs outside an authenticated registration form, or raw JSON in primary copy.

Buttons use verbs (`변경하기`, `취소`, `로그아웃`, `다시 시도`). Errors do not blame the user and do not claim data was saved unless the mutation actually succeeded. Unknown values read `정보 없음`, never `0` or a fabricated healthy state.

## Accessibility

- A first-focusable skip link (`.skip-link`) targets `#main-content`. Semantic `header`/`nav`/`main` landmarks; one clear page heading per screen (`h1.shell-page-title`, `tabIndex={-1}` so it is a valid dialog-focus fallback target).
- The active `NavBar` destination uses `aria-current="page"`. Icon-only buttons (`.icon-button`) have `aria-label`.
- All functions are keyboard operable; focus order follows visual order; visible focus uses a 2px `--primary` outline plus offset (global `:focus-visible` rule).
- Do not rely on color alone; every status includes readable Korean text.
- New errors and success confirmations use a polite live region (`role="alert"`/`role="status"`); loading and background refresh avoid noisy announcements.
- `AccessibleDialog`/`AccessibleSheet` own the full modal focus lifecycle described above; background content is `inert` while a dialog is open, and body scroll is restored on every close/unmount path.

## Motion and reduced motion

CSS-only motion that explains state change, never decorates routine monitoring. Hover/focus color changes ~120ms; the `.skip-link` reveal and dialog/sheet transitions ~180ms. `prefers-reduced-motion: reduce` collapses transition/animation durations to near-zero and disables smooth scrolling (`front/src/styles/responsive-layout.css`); focus movement and state changes remain immediate.

## Visual QA capture contract

The prior 39-PNG capture packet described a four-destination camera-wall product that no longer exists and has been retired. A new capture contract, when one is needed, should be scoped to the three real destinations and their states as documented in this file: 관제 (camera wall — empty/loading/error/populated, floor-filtered; room detail — online/offline live panel, each overlay mode, populated/empty event history), 이벤트 (filtered/unfiltered clip grid, empty state, playback modal), and 설정 (all four right-column cards in read and edit mode, camera table with a mixed online/offline/bed-zone-pending set, all five modals). There is no active capture contract today — do not treat its absence as a regression.

## Deviations from the sibling product

- This is a single-facility edge console: no facility switcher, no multi-tenant concepts, no backend-status ribbon anywhere in the shell.
- Navigation uses the native query/history contract above (3 destinations). Do not add React Router, RBAC, facility selection, Zustand, Lucide, Recharts, or sibling components.
- The top `NavBar`, neutral tokens, Korean type hierarchy, border-first depth, and labelled semantic statuses provide family resemblance with the sibling product; page composition is purpose-built per `front/design-handoff/README.md`.

## Implementation constraints

- Preserve React, Vite, TypeScript, Tailwind, the feature-sliced directory boundaries, centralized API utilities (`front/src/shared/api`), native media controls, and current backend/edge separation.
- Add no router, global state manager, component library, icon package, animation library, or font package for this redesign.
- Keep polling page-local and active-page-only, retain last-good values, avoid overlapping requests, abort on cleanup, and reject stale/out-of-order completion (`usePollingResource`).
- Do not add backend/edge routes, retained alerts, WebSocket/SSE, RTSP publishing/emulation, or inference changes as part of shell/design work.
- All API calls go through `/api/v1`; never bundle worker/relay credentials into the frontend; clip media is fetched via the API only, never a direct worker/relay URL; no `playbackRate` compensation hacks on native video.

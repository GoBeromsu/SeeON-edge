# ElderCare ML Control Design Contract

## Authority and provenance

This root document is the design authority for the `front` application's global shell, tokens, accessibility, and interaction primitives — the parts shared by every page. `front/design-handoff/README.md` is the authoritative source for page-level visual and interaction detail (관제/이벤트/설정 screens and their modals); where the two disagree on page content, the handoff document wins and this document must be updated in the same change. `front/design-handoff/*.html` prototypes are visual reference only and are never imported or copied at runtime.

The product is a single-facility edge console: one eldercare facility's on-site edge device, operated by installers and facility staff, not a multi-tenant SaaS product. There is no facility switcher, no backend-status ribbon, and no multi-facility concept anywhere in the shell.

The local implementation sources are `front/src/styles.css` (and its `front/src/styles/*.css` imports), `front/tailwind.config.js`, and feature-sliced React components under `front/src`. A new component or token must reuse a semantic contract here or update this document in the same change.

## Product identity and goals

The product is `Senior AI Lab Edge`, a calm operational safety console for checking camera availability, reviewing fall and bed-exit evidence, and managing the single admin account and device connection. It should feel quiet, exact, and trustworthy during repeated daily use.

The interface must reveal what is known, when it was last known, and what the operator can do next. It must never imply live data, camera health, or history that the current APIs do not provide.

## Information architecture

There are exactly three primary destinations, all reachable from a single 56px top `NavBar` present at every viewport width. There is no left rail and no bottom tab bar.

| Query value | Navigation label | Owns |
| --- | --- | --- |
| `operations` | 관제 | Camera view/selection (default destination) |
| `events` | 이벤트 | Historical event/clip evidence |
| `settings` | 설정 | Camera, connection, and system settings |

The `NavBar` also carries the brand mark (text, left), the three nav destinations, an account-settings icon button (opens `AccountSettingsModal`), and — only while a session exists — a logout button. It never renders a backend-status pill or facility name; connection status is surfaced only inside the 설정 page.

Page-level content for 관제/이벤트/설정 (filters, camera wall, clip review, settings cards) is owned by `front/design-handoff/README.md` and is implemented incrementally; a page not yet implemented renders a minimal placeholder heading plus one line of Korean copy under `#main-content`, never a broken or empty screen.

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

The shell is a single 56px sticky top `NavBar` at every width; only the horizontal gutter changes with viewport (`.app-main`/`.app-navbar` padding steps up at `768px`). There is no separate mobile chrome, no left rail, and no bottom tab bar at any width. Content below the `NavBar` is centered with `max-width: 1280px`.

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

- `NavBar` is a fixed 56px sticky top bar. `.app-main` centers content at `max-width: 1280px` with 16-24px horizontal padding depending on viewport.
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

Deferred. The prior 39-PNG capture packet described a four-destination camera-wall product that no longer exists; a new capture contract belongs to the wave that implements the 관제/이벤트/설정 page content against `front/design-handoff/README.md`, once there is real page behavior worth capturing. Until then, this section intentionally has no active contract — do not treat its absence as a regression.

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

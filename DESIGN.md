# ElderCare ML Control Design Contract

## Authority and provenance

This root document is the single design authority for the `front` application. It governs product vocabulary, information architecture, URL state, responsive behavior, visual tokens, component states, accessibility, motion, and visual acceptance.

The visual family reference is the local sibling repository `../eldercare-fall-ai`: its `DESIGN.md`, neutral admin shell, status language, and Senior AI Lab mark were audited on 2026-07-20. Those sources establish family resemblance, not a pixel-clone requirement. This repository must remain independently buildable: do not import sibling source, components, CSS, assets, routes, stores, permissions, or dependencies at runtime.

The local implementation sources are `front/src/styles.css`, `front/tailwind.config.js`, and feature-sliced React components under `front/src`. A new component or token must reuse a semantic contract here or update this document in the same change.

## Product identity and goals

The product is `Senior AI Lab · ML Control`, a calm operational safety console for checking camera availability, reviewing fall and bed-exit evidence, managing camera registration, and diagnosing runtime health. It should feel quiet, exact, and trustworthy during repeated daily use.

The default experience is a camera wall, not a metric dashboard. The interface must reveal what is known, when it was last known, and what the operator can do next. It must never imply a live event feed, persistent label state, camera health, or system history that the current APIs do not provide.

## Product jobs

- An operator scans cameras by floor and room, distinguishes camera liveness from image availability, and selects one camera to enter its single focused live view in one click.
- A reviewer filters historical clips, opens evidence with labelled native video controls, sees unavailable media honestly, and applies only the supported review labels.
- An administrator creates, edits, tests, and deletes camera registrations without exposing secrets or confusing a successful save with a failed follow-up probe.
- An engineer checks backend reachability, heartbeat freshness, and structured runtime/facility status without reading raw object dumps.
- Any signed-in user can deep-link to a supported page and state, use browser back/forward, recover from failures, and complete core tasks with keyboard or touch.

## Information architecture

There are exactly four primary destinations. Floor and room are contextual filters, not destinations.

| Query value | Navigation label | Owns |
| --- | --- | --- |
| `operations` | 관제 | Snapshot camera wall, camera selection, and one focused live camera entered directly per selection |
| `events` | 이벤트 기록 | Historical clip evidence, filters, detail, playback, and supported label review |
| `cameras` | 카메라 관리 | Camera registry, create/edit/test/decode actions, and deletion |
| `system` | 시스템 | Backend reachability, heartbeat/runtime diagnostics, and explicit technical details |

Desktop uses a persistent left navigation rail and sticky top bar. Mobile and tablet use a compact top app bar and four persistent bottom tabs. Navigation is never duplicated inside page content.

## Page and state contracts

### Shared resource states

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

Camera heartbeat (`online|stale|never_seen`), snapshot lifecycle, focused-feed lifecycle, backend reachability, and runtime freshness are independent states. A successful snapshot does not prove a fresh heartbeat; a stale heartbeat does not rewrite registry transport status; malformed or missing values become `unknown`/`정보 없음`.

### 관제

- The default is the snapshot wall at `wallPage=1` after canonicalization; a valid `camera` value enters that camera's focused view directly.
- Group visible cameras by floor and room and render stable snapshot cards with camera label, location, labelled liveness, image state, and last-seen metadata.
- Selecting a visible card enters the focused view directly and mounts exactly one stream; there is no intermediate selection context.
- Focused view is page content at every width, never a modal. Back returns to the wall and restores the prior filters, wall page, and scroll position, moving focus to the originating card.
- If a successful registry refresh or filter change removes the selected camera, return to Wall, clear invalid selection, and announce why.

### 이벤트 기록

- Show historical clips only; there is no retained real-time alert rail or live resident alert feed in the current contract.
- Filters are floor, room, camera, and exact event type. Clip detail preserves event time, camera/location context when resolvable, media availability, reviewer metadata when confirmed, and labelled native video controls.
- A server clip without label fields is `라벨 정보 없음`, not `미검토`. In-memory confirmed label overlays may survive list polling but disappear on reload unless the server returns label truth.
- An unresolved/orphan clip is allowed only without active floor/room/camera filters and is described as camera information unavailable.

### 카메라 관리

- Present registry identity, display label, location, enabled/connection information, and only supported create/edit/test/decode/delete actions.
- Camera `id` is the local query, selection, and CRUD identity. `backend_camera_id`, when present, is only the snapshot/focused-stream/heartbeat alias and is never serialized as `camera`.
- RTSP terminology is permitted only inside the authenticated registration form and masked preview. Never show credentials, tokens, complete secrets, or storage paths.
- A successful create remains successful if the optional follow-up probe fails; communicate the saved registration and the separate probe outcome.
- Destructive delete always uses the shared confirmation dialog and names the camera being removed.

### 시스템

- Show actual backend reachability plus structured heartbeat/runtime facility fields only. Keep per-camera heartbeat, runtime freshness, and API availability visibly distinct.
- Place digests, config identifiers, and similarly technical values behind an explicitly labelled `기술 정보` disclosure.
- Do not render raw JSON, fake gauges, unsupported detection-setting saves, or fabricated update/rollback history.

### Authentication and shell states

- Configured builds show the existing local username/password gate without displaying or prefilling credentials and without exposing a token field or token value. Invalid credentials make no dashboard request; valid credentials copy the configured relay token into memory only.
- If the relay token is absent, replace the form with a non-interactive `서비스 설정 필요` state and make no dashboard request.
- Logout is a shell action. It resets authenticated page state through URL replacement, never by adding a history entry.

## Native URL query contract

Use the browser `URLSearchParams` and History APIs; add no router. The only recognized keys are serialized once in canonical order: `page`, `floor`, `room`, `camera`, `event`, `clip`, `wallPage`.

For repeated recognized keys, the last occurrence wins. Remove earlier duplicates and all unknown keys. Empty values are invalid. `URLSearchParams` owns UTF-8 percent decoding and encoding. Compare DTO-backed values as exact decoded Unicode without trimming, normalization, or case folding. Enum values are exact lowercase ASCII.

| Key | Exact value/source | Applies to | Canonicalization and dependent invalidation |
| --- | --- | --- | --- |
| `page` | `operations|events|cameras|system` | Global | Missing or invalid becomes `operations`; a destination change removes all keys inapplicable to that destination before later validation. |
| `floor` | Exact non-null `Camera.floor_name` | `operations`, `events` | After successful cameras data, unknown removes `floor`, then `room`, then incompatible `camera`/`clip`; remove off these pages. |
| `room` | Exact non-null `Camera.space_id`; when `floor` exists, the same camera must match it | `operations`, `events` | After successful cameras data, unknown/incompatible removes `room`, then incompatible `camera`/`clip`; remove off these pages. |
| `camera` | Exact local `Camera.id`; never `backend_camera_id` | `operations`, `events` | Operations removes an unknown or filter-incompatible camera, which yields the wall. Events requires a matching camera or removes it and then any incompatible clip. |
| `event` | Exact non-empty `Clip.event_type` in the successfully loaded clip set after active filters | `events` | Unknown removes `event`, then an incompatible `clip`; remove off events. |
| `clip` | Exact `Clip.id` | `events` | Clip event and resolvable camera must match active filters. An existing unresolved/orphan clip survives only when floor, room, and camera are all absent; otherwise remove only `clip` after upstream invalidation. |
| `wallPage` | Canonical ASCII decimal matching `[1-9][0-9]*`; parse as `BigInt`, bound to `1..9007199254740991`, then convert if needed | `operations` | Invalid or out of bounds becomes `1`; serialize the parsed decimal; after successful filtering clamp to the last non-empty page, with an empty wall at `1`; remove off operations. |

Run synchronous syntax and page-applicability canonicalization immediately. Retain DTO-backed `floor`, `room`, `camera`, `event`, `clip`, and data-derived `wallPage` while required cameras/clips data is idle, loading, or error. Validate them only after required successful responses: operations waits for cameras; events waits for both cameras and clips.

The full invalidation precedence is `page -> floor -> room -> camera -> event -> clip -> wallPage`. Rebuild the entire query in canonical order after validation.

Deliberate destination, filter, page, selection, and focus changes call `pushState`. Default insertion, duplicate/unknown/invalid removal, canonical ordering, data-driven invalidation, logout/login reset, and wall-page clamping call `replaceState`. A legacy `mode` key from a stored deep link is unrecognized; it is removed via `replaceState`, leaving `history.length` unchanged. A `popstate` restoration calls no `pushState`; on the restored entry it may perform at most one immediate syntax/applicability replacement and at most one later DTO-derived replacement. Both keep `history.length` unchanged and are omitted when the canonical query already matches.

## Responsive viewport contract

The named widths are mandatory acceptance viewports, not approximate device labels.

| Width | Shell and navigation | Content and wall | Focused view |
| --- | --- | --- | --- |
| 375 | Compact 56px top app bar; four fixed bottom tabs; 12-16px gutter; content clears safe areas and tab bar | One-column wall; controls stack; media uses full available width | Page content at full width |
| 768 | Compact top app bar and four bottom tabs; 20px gutter | Two-column wall; filters wrap without horizontal scroll | Page content at full width |
| 1024 | Desktop left rail and sticky 64px top bar; 20-24px gutter; no bottom tabs | Two-column wall; focused media remains page content | Page content at full width |
| 1440 | 256px left rail, sticky 64px top bar, 24px gutter, bounded readable content | Three/four-column wall as card minimum permits; stable 16:9 media | Page content at full width |

At every width, critical controls wrap rather than clip, Korean text remains legible, the page has no horizontal overflow, and touch targets are at least 44px. The 1024px boundary is desktop: it changes shell chrome (rail versus bottom tabs), not the underlying selection, URL, or focused-view presentation.

## Camera wall and media budget

- Sort by floor, then room, then camera label with Korean collation and stable `Camera.id` tie-breaking. Missing floor or room is grouped under `미분류`.
- Render 12 tiles per page. Only the active page loads images; changing filters resets or clamps paging deterministically.
- Permit at most six (6) concurrent snapshot loads across the wall. Queue the rest; do not let tile-level timers bypass the shared limit.
- Refresh snapshots about five seconds after completion plus deterministic jitter, so cameras do not synchronize into request spikes. Retain the last-good image and timestamp while refreshing; visibly distinguish initial loading, stale, and failed images.
- Wall cards use snapshots only and produce zero focused-stream/MJPEG requests. Focused mode mounts exactly one stream element app-wide; switching camera or page removes the prior element before mounting another.
- Snapshot, focused-live, and clip viewports use dark media frames (`#0D0D0D`) with stable 16:9 geometry, `object-fit: contain`, a labelled unavailable state, and no stretching. The surrounding shell remains light.
- Visual evidence may fulfill finite image responses. It proves DOM/request identity and concurrency, not transport-level MJPEG socket closure or real camera integration.

## Dialog and screen-transition semantics

Camera add and delete flows, plus evidence detail, use the shared accessible dialog surface; camera edits remain inline controls on the camera card. Destructive actions require an explicit labelled confirmation; safe cancel is initially reachable. Every modal handles initial focus, Tab/Shift+Tab containment, Escape, safe-backdrop close, body scroll lock, and invoker focus restoration. Focused live video is page content at all widths, never a modal.

Entering the focused view moves focus to its heading or its back control. Returning to the wall moves focus to the card that opened the focused view, or to the wall heading when that camera no longer exists. Wall cards are navigation, not toggles, and carry no `aria-pressed`.

## Design tokens

### Color

Ship one light application shell with dark media frames. Existing unused dark variables may remain in code for compatibility, but they are not an exposed theme or a release claim.

| Role | Token | Value | Use |
| --- | --- | --- | --- |
| Page | `--c-bg` | `#FAFAFA` | Application canvas |
| Primary surface | `--c-surface` | `#FFFFFF` | Rail, top bar, cards, sheets, dialogs |
| Secondary surface | `--c-surface-2` | `#F0F0F0` | Hover, selected-neutral, grouped detail |
| Border | `--c-border` | `#E0E0E0` | Dividers, cards, controls |
| Primary ink | `--c-ink` | `#0F0F0F` | Headings, body, values |
| Secondary ink | `--c-ink-soft` | `#595959` | Labels and supporting copy |
| Tertiary ink | `--c-ink-faint` | `#6B6B6B` | Timestamps and metadata on light application surfaces |
| Interactive | `--c-brand` | `#2F6FB0` | Primary action, active nav, link, focus |
| Interactive soft | `--c-brand-soft` | `#EAF2FB` | Active navigation/control background |
| Primary action foreground | `--c-action-foreground` | `#FFFFFF` | Text and icons on brand-filled primary actions |
| Brand teal | `--c-teal` | `#2BB6A3` | SA mark only; never generic decoration |
| Stable | `--c-stable` / `--c-stable-bg` | `#166E3D` / `#E7F7EE` | Online, success |
| Caution | `--c-caution` / `--c-caution-bg` | `#884E07` / `#FDF2DF` | Stale, delayed, partial |
| Danger | `--c-danger` / `--c-danger-bg` | `#B8261B` / `#FDECEB` | Failure, destructive, fall risk |
| Check | `--c-check` / `--c-check-bg` | `#1554E0` / `#E8F0FE` | Review/check-needed state |
| Media | `--c-media` / `--c-media-ink` | `#0D0D0D` / `#F5F5F5` | Snapshot, live, and clip frames only |
| Media supporting text | `--c-media-text` | `#B8B8B8` | Normal supporting text on dark media frames and the dark authentication brand panel |
| Media overlay | `--c-media-overlay` | `rgba(13,13,13,.72)` | Snapshot and focused-live status labels over imagery |
| Media overlay foreground | `--c-media-overlay-foreground` | `#F5F5F5` | Text and icons on media status overlays |
| Backdrop | `--c-backdrop` | `rgba(15,15,15,.42)` | Dialog and sheet backdrop |

Surfaces and text are achromatic. Brand blue is interactive only; semantic colors convey status/risk and always include a text label or icon. Every App-reachable brand-filled primary action uses `.brand-action`; component code must not rebuild that pairing from `bg-brand`, dark fill, or raw foreground utilities. Status copy over imagery uses `.media-status-overlay`. Reachable components must not use raw black/white utility aliases or literal foreground/overlay values. Prefer borders over shadows. Do not add tinted page canvases, glow shadows, decorative gradients, or one-off raw colors.

### Typography

Use the installed CSS stack: `Pretendard`, `-apple-system`, `BlinkMacSystemFont`, `system-ui`, `Apple SD Gothic Neo`, `Noto Sans KR`, `sans-serif`. Do not add a font dependency. Page titles are 20px/700; section titles 18px/700; body 16px/400-700; compact controls and labels 14px/500-700; metadata 12px/500-700. Use normal tracking, 1.45-1.6 line height for prose, and tabular numerals for dates, durations, counts, pages, and health ages.

### Spacing, geometry, and depth

- Spacing follows a 4px base: 4, 8, 12, 16, 20, 24, 32, and 40px. Default page/card gutters are 16-24px.
- Controls are at least 44px high; compact desktop controls may visually occupy 36-40px only when their interactive target remains 44px.
- Radii are 8px for controls/nav, 12px for cards/media, and 16px for elevated panels. Bottom sheets may use 20px only on their top corners.
- Borders use `1px solid var(--c-border)`. Resting cards use at most `0 1px 3px rgba(15,15,15,.08)`; sheets/dialogs use at most `0 12px 36px rgba(15,15,15,.18)`.
- Media keeps 16:9 aspect ratio. Status dots are 8px but never stand alone. Icons are simple local SVGs or text; no icon package or emoji controls. Primary navigation uses monoline local SVGs (20px, `currentColor`) beside labels — never first-character letter marks.
- Layer order is content `0`, sticky shell `20`, backdrop `40`, dialog/sheet `50`, and transient non-modal notice `60`.

## Content voice

Use concise Korean-first operational language: name the object, state current truth, and offer one supported next action. Prefer `다시 시도`, `마지막 확인`, `연결 지연`, `연결 이력 없음`, `영상 불러오는 중`, `영상을 불러올 수 없음`, `라벨 정보 없음`, and `서비스 설정 필요`.

Do not expose `/api/v1`, Authorization, worker/MJPEG, schema names, raw enums, query tokens, digests, stack traces, camera credentials, RTSP URLs outside the masked registration job, storage paths, resident identity, or raw JSON in primary copy. `시스템` may reveal sanitized identifiers under `기술 정보`, with a human label before the value.

Buttons use verbs (`카메라 추가`, `이 카메라 클립 보기`, `다시 시도`, `삭제`). Errors do not blame the user and do not claim data was saved unless the supported mutation succeeded. Dates and times use one locale-consistent Korean presentation; unknown values read `정보 없음`, never `0` or a fabricated healthy state.

## Accessibility

- Provide a first-focusable skip link to the main content target, then semantic header, navigation, main, section, and complementary landmarks with one clear page heading.
- Active destinations use `aria-current="page"`; toggles use native state or `aria-pressed`; icon-only buttons have accessible names.
- All functions are operable by keyboard. Focus order follows visual order; focus is never hidden behind sticky bars or bottom tabs; visible focus uses a 2px brand outline plus offset.
- Touch targets are at least 44x44px. Do not rely on color alone; every status includes readable Korean text and maintains 4.5:1 text contrast (3:1 for large text and non-text UI boundaries).
- Loading and background refresh avoid noisy announcements. New errors and selection-invalidated explanations use a polite live region; urgent destructive confirmation is conveyed by dialog name and copy, not animation.
- Images have contextual alternative text; decorative marks are hidden when adjacent wordmark text already names the brand. Media controls are native and keyboard accessible.
- Dialog and bottom-sheet focus lifecycle follows the semantics section. Background content is inert/unreachable while modal, and body scroll is restored on every close/unmount path.
- Validate CJK wrapping, 200% zoom, reduced motion, and no horizontal overflow at 375, 768, 1024, and 1440.

## Motion and reduced motion

Motion explains state change; it never decorates routine monitoring. Use CSS only. Hover/focus color changes take 120ms; panels/sheets use opacity and transform for 180ms; background refresh cross-fades may use 160ms. Use `cubic-bezier(.2,.8,.2,1)` for entry and standard ease for color. Never animate layout dimensions, camera card size, continuously pulse routine status, or autoplay attention effects.

Under `prefers-reduced-motion: reduce`, remove transforms and nonessential transitions, set durations to effectively zero, disable smooth scrolling, and replace any loading motion with a static labelled state. Focus movement, announcements, and state changes remain immediate and complete.

## Visual QA capture contract

Visual QA runs the real Vite/React/CSS surface on the host and fulfills only external API/media responses with synthetic, privacy-safe fixtures. No screenshot or static mock may replace the DOM. Cross-repo screenshots are qualitative family references; image diffs are directional evidence, not a pixel-clone threshold.

The exact capture packet is **31 state PNGs + 8 interaction PNGs = 39 PNGs**, each captured exactly once from one current configured build:

- 10 primary states: `operations` at 375/768/1024/1440, plus `events`, `cameras`, and `system` at 375/1440.
- 21 edge states: login default; login invalid; loading/empty/error for each of four pages (12); focused stream success; focused stream unavailable; clip detail available; clip detail unavailable; camera dialog validation; camera dialog server failure; failure-to-recovery.
- 8 interaction states: focus and activated frames for mobile navigation, desktop camera cards, focused-view back control, and desktop dialog.

Separate automated assertions run at all 375/768/1024/1440 widths and camera scales 0/1/12/13/50. Fixtures cover `mixed`, `loading`, `empty`, `error`, and `recovery`, including online/stale/never-seen cameras, missing location, snapshot failure, fall/bed-exit evidence, playable/unavailable clips, and partial system/runtime data.

The manifest records capture ID, canonical URL, fixture/failing endpoint, viewport, action trace, exact capture moment, assertion, PNG signature/dimensions/mtime, source/build identity, sanitized network trace, console result, privacy scan, and cleanup. Required network evidence is zero focused-stream requests on Wall, no off-page snapshot polling, no more than six unresolved snapshots, exactly one focused stream identity, and prior focused element removal on switch. Artifacts contain no real resident, credential, token, RTSP URL, storage path, or footage.

Acceptance requires keyboard/dialog flows, reduced-motion behavior, CJK/overflow checks, no uncaught console errors, fresh artifact validation, and two independent visual reviews of the same complete capture set.

## Deviations from the sibling product

- This product uses a camera wall and evidence workflows, not the sibling room-status treemap or staff monitor board.
- This release exposes only the neutral light operational shell; dark treatment is confined to media. The sibling's broader theme variants do not define ML release scope.
- Navigation uses the native query/history contract above. Do not copy React Router, RBAC, facility selection, Zustand, Lucide, Recharts, or sibling components.
- The left rail, top bar, neutral tokens, Korean type hierarchy, border-first depth, labelled semantic statuses, and SA geometry provide family resemblance; ML page composition remains purpose-built.
- Snapshot support already exists in this product. The retired notes' endpoint-gap assumptions and proposed simultaneous live wall are obsolete.
- No readable real-time alert source exists. Event history uses clips; status uses heartbeat/runtime truth without inventing retained alerts.

## Local Senior AI Lab mark geometry

Provenance: the following geometry was copied from the local sibling `../eldercare-fall-ai/front/src/components/Logo.tsx` during the 2026-07-20 audit. Only SVG geometry and brand colors are reproduced here. The ML implementation must copy it into a local component/asset with its own accessible labelling; it must not import or resolve the sibling path at runtime. Do not copy the sibling React utilities, dark-theme classes, or component architecture.

```svg
<svg viewBox="0 0 100 100" fill="none" aria-hidden="true">
  <defs>
    <linearGradient id="sa-grad" x1="20" y1="14" x2="78" y2="84" gradientUnits="userSpaceOnUse">
      <stop stop-color="#2BB6A3" />
      <stop offset="0.5" stop-color="#2F6FB0" />
      <stop offset="1" stop-color="#16325A" />
    </linearGradient>
  </defs>
  <path d="M67 25 C59 18 43 18 36 25 C27 34 32 45 46 49 C60 53 65 61 59 69 C52 78 37 78 29 71" stroke="url(#sa-grad)" stroke-width="11" stroke-linecap="round" />
  <path d="M64 84 L79 33 L94 84 M69 70 H89" stroke="#16325A" stroke-width="9.5" stroke-linecap="round" stroke-linejoin="round" />
  <g stroke="#2F6FB0" stroke-width="2.4" stroke-linecap="round">
    <path d="M55 47 L62 40 L62 28" />
    <path d="M62 40 L72 34 L72 24" />
    <path d="M62 40 L74 44 L82 40" />
    <path d="M55 47 L52 38" />
  </g>
  <g fill="#2BB6A3">
    <circle cx="62" cy="26" r="3.4" />
    <circle cx="72" cy="22" r="3.4" />
  </g>
  <g fill="#2F6FB0">
    <circle cx="52" cy="36" r="3.4" />
    <circle cx="83" cy="40" r="3.4" />
  </g>
</svg>
```

## Implementation constraints

- Preserve React, Vite, TypeScript, Tailwind, the feature-sliced directory boundaries, centralized API utilities, native media controls, and current backend/edge separation.
- Add no router, global state manager, component library, icon package, animation library, font package, or sibling dependency solely for this redesign.
- Keep polling page-local and active-page-only, retain last-good values, avoid overlapping requests, abort on cleanup, and reject stale/out-of-order completion.
- Do not add backend/edge routes, retained alerts, WebSocket/SSE, RTSP publishing/emulation, inference changes, or committed media/model artifacts.
- Do not treat synthetic browser media as production camera proof. Runtime integration and transport-level stream closure remain outside this visual contract.

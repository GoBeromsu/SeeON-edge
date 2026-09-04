# Edge dashboard (Vite SPA)

Frontend-wide rules for `front/`. Slice layout lives in `src/AGENTS.md`.

## Vite / pnpm

- Package manager is `pnpm@10.32.1`. Commit `pnpm-lock.yaml`. Install with `pnpm --dir front install --frozen-lockfile`.
- Scripts: `dev` (Vite `--host 0.0.0.0`), `build` (`tsc --noEmit && vite build`), `test` (Vitest), `test:e2e` (Playwright).
- `@/` maps to `src/`. Skip deep `../../../` imports.
- Same-origin `/api/v1` is the default. Point the Vite proxy with server env `ML_API_PROXY_TARGET`. That is not a `VITE_` bake.
- `VITE_ML_API_BASE_URL` is cross-origin hot-reload QA only. It rewrites fetch, MJPEG, clip, and thumbnail URLs together. Leave it unset in Docker and reverse-proxy deploys.
- `allowedHosts: ['.ts.net']` is tailnet-dev only. Production never reads `vite.config.ts`. FastAPI StaticFiles serves `front/dist` at `/`.
- One bundler. No CRA, no frontend-owned Node API, no second lockfile.

## API / session

- All traffic goes through `shared/api`: `requestJson`, `getApiBase()`, URL helpers in `session.ts`.
- Session is a server-set HttpOnly cookie. `fetch` uses `credentials: 'same-origin'`. Do not keep tokens in `localStorage`, `sessionStorage`, or a JS bearer.
- Login `POST /auth/session`. Probe `GET /auth/session`. Logout `DELETE /auth/session`. Rotate with `PUT /auth/credentials` on an already-authed session.
- `AuthGate` owns `checking` / `unavailable` / `unauthorized` / `authorized`. Mid-flight 401/403 must hit `subscribeUnauthorized`. A frozen authorized wall is a bug.
- Worker relay tokens, RTSP secrets, clip-store filesystem paths, and Hub credentials stay out of assets and UI copy.
- Connection form collects site-local values only: facility code, facility token, installation ref. Hub address is image/env. Show it read-only. Do not make it an editable field.
- Camera registry writes go through the API. No client-side seed, YAML import, or silent `cameras` pull.
- Normalize every JSON payload in `shared/api/*Normalizer.ts`. Reject unknown shapes. Do not render raw worker or backend internals.
- Incident and clip views show API-projected ids and states only.

## Media

- Clip video: `getClipVideoUrl(id)` → `/api/v1/clips/{id}/video`. Thumbnails: `getClipThumbnailUrl`. Live: `getCameraStreamUrl` / `getCameraSnapshotUrl`.
- Browser never opens the clip-store directory. Camera RTSP credentials never land in a `<video>` or `<img>` src.
- `AutoplayVideo` keeps native `controls`. Classify autoplay-block vs media-fail. Do not set `playbackRate` to hide producer timestamp bugs.
- Honor `video_available`, `video_error`, `thumbnail_available`. Missing media is unavailable, not an endless spinner and not a fake poster.
- Live wall uses `useMjpegStream`: `fetch` plus canvas, Content-Length framed parts, stall reconnect after 3s, exponential backoff. No `<img multipart>`. No timed remount `key`.
- Offline or offscreen tiles pass `baseUrl: null` and fall back to snapshot. Live fetch streams do not need `_r=` cache-busting.


## Verification

- UI change: focused Vitest beside the module (`*.test.ts` / `*.test.tsx`), then `pnpm --dir front build`.
- `pnpm --dir front test` is jsdom Vitest. `e2e/` is excluded there. Keep Playwright specs out of the Vitest tree.
- `pnpm --dir front test:e2e` is a real local stack. Root CI already deselects real-stack. Playwright is not the merge gate.
- Tests must not pass by sleep. Subscribe to the state change, act, then await with a bound timeout.
- `src/test/setup.ts` stubs `HTMLMediaElement.play`. Assert UI states and API URLs, not pixel dumps of frames.
- Do not pin Korean prose unless the value is a machine-consumed sentinel or `data-testid`.

## Anti-patterns

- Baking `VITE_*` deploy URLs or relay secrets into the production bundle.
- A second HTTP client or ad-hoc `fetch` that skips `credentials: 'same-origin'` and the 401/403 bus.
- Reading `document.cookie` or putting session tokens in query strings.
- Compensating bad clip duration with `playbackRate`, muted loops, or custom seek bars that hide native controls.
- Polling `edge.sqlite3`, worker ports, or clip-store paths from the browser.
- Silent empty media frames. Keep loading, live/stalled, unavailable, or error visible.
- `crypto.randomUUID()` on LAN HTTP. That API is missing on insecure origins and will crash the tree. Use `generateUuidV4`.
- Client-only wizard progress in `localStorage`. Resume from server-known connection and camera state.
- Guessing clip size, duration, or storage gauges when the API omits the field. Show guidance. Do not estimate.

# Dashboard src (feature-sliced)

Feature-sliced SPA (Vite + React). Import across files via the `@/*` alias
(→ `src/*`); avoid deep relative paths.

## Layout (2-depth ownership)
- `app/` — App shell, entry wiring, URL/location state (App, dashboardLocation). `main.tsx` (Vite entry) stays at `src/`.
- `features/<slice>/` — each slice owns its components + view logic; a slice may add its own `AGENTS.md`:
  - `cameras/` — camera cards, add/delete camera.
  - `camera-management/` — management panel and page.
  - `clips/` — clip grid + labeling (see `features/clips/AGENTS.md` for clip privacy rules).
  - `events/` — live event panels, event logic, status feed.
- `shared/` — cross-slice building blocks:
  - `shared/ui/` — reusable presentational components (StatusBadge, DashboardShell, SystemPanels, AuthGate).
  - `shared/api/` — HTTP client, session, DTO types, normalizers.

Tests live beside their module (`*.test.tsx`). `pnpm test` (vitest) is the green bar.

# Dashboard src (feature-sliced)

Feature-sliced SPA (Vite + React). Import across files via the `@/*` alias
(→ `src/*`); avoid deep relative paths.

## Layout (2-depth ownership)
- `app/` — App shell, entry wiring, URL/location state (App, dashboardLocation). `app/pages/` holds the three top-level route pages (Operations/Events/Settings). `main.tsx` (Vite entry) stays at `src/`.
- `features/<slice>/` — each slice owns its components + view logic; a slice may add its own `AGENTS.md`:
  - `account-settings/` — the single-admin account settings modal (username + new password).
  - `connection/` — device/backend connection settings panel.
- `shared/` — cross-slice building blocks:
  - `shared/ui/` — reusable presentational components (NavBar, AccessibleDialog, AuthGate, Toast, StatusBadge).
  - `shared/api/` — HTTP client, session, DTO types, normalizers.

Tests live beside their module (`*.test.tsx`). `pnpm test` (vitest) is the green bar.

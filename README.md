# eldercare-fall-ml

Edge ML runtime for fall and bed-exit detection, organized as three deployable
instances plus a shared library:

- **`backend/`** — FastAPI control/status/relay gateway (`ml-api` image).
- **`worker/`** — RTSP inference worker that relays facts to the backend (`ml-worker` image).
- **`front/`** — React/Vite dashboard SPA served by the backend.
- **`shared/`** — `shared.events` (the backend↔worker wire code); `contracts` is a
  top-level vendored leaf (ADR-0004).

Training is maintained separately in `SeniorAILab/eldercare-dataset-ops`.

## License notice

This project is licensed under the GNU Affero General Public License v3.0
(`AGPL-3.0-only`); see [`LICENSE`](LICENSE) for the complete terms.

The `ultralytics` worker dependency is also licensed under AGPL-3.0. This
project accepts the obligations of that dependency's AGPL-3.0 license,
including the applicable source-disclosure requirements.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The development interpreter is pinned
to Python 3.12 by `.python-version`, which is what `uv sync` below resolves.

```bash
uv sync
uv run pytest -q
uvx ruff check .
uv run --group lint lint-imports   # architecture-boundary enforcement
```

`pyproject.toml` keeps `requires-python = ">=3.11"` because that floor is the
union of two images that are deliberately not on the same interpreter:
`Dockerfile.backend` builds `ml-api` on 3.11, `Dockerfile.edge` builds
`ml-worker` on 3.12. Only the worker actually needs 3.12 — it uses
`typing.override` (`worker/domains/base.py`), which is 3.12+. A single `uv sync`
venv has to serve both instances plus the test suite (which imports `worker`),
so the development pin takes the higher of the two. Neither Dockerfile copies
`.python-version`, and both set `UV_PYTHON_DOWNLOADS=never`, so this pin does
not reach either image build.

Local model artifacts are intentionally ignored. Place them under `models/` and
copy `worker/ml-worker.example.yaml` to `worker/ml-worker.local.yaml` before
configuring a real worker-reachable RTSP URL. Never commit RTSP credentials or
relay tokens.

## Run

### Local state directories

Two storage paths default to absolute container paths that a non-root local
user cannot create. The defaults are correct for the edge images — where
`compose.edge.yaml` bind-mounts them — but a local run fails on both unless
they are redirected:

| Env var | Default | Owner |
| --- | --- | --- |
| `CLIP_STORE_DIR` | `/var/lib/clip-store` | worker writes clips, `ml-api` reads them |
| `API_CONNECTION_SETTINGS_PATH` | `/var/lib/ml-api/connection-settings.sqlite3` | `ml-api` connection settings |

(`API_LABEL_STORE` -- `ml-api` clip labels + audit log -- defaults to
`resolve_state_dir("ml-api")` (`~/.local/state/ml-api/labels`,
`backend/app/shared/state_dir.py`) instead, which a native dev process can
already write; it only needs redirecting if you want labels/audit history
kept somewhere else.)

Left at their defaults locally, the failures are not obvious: connection
settings degrade to `unable to open database file`, and the worker logs
`clip recorder failed to start; clips disabled` while otherwise running
normally. Export a shared local root once:

```bash
export ML_DEV_STATE="$HOME/.local/state/eldercare-dev"
mkdir -p "$ML_DEV_STATE"/{clip-store,ml-api}
```

`CLIP_STORE_DIR` must be the *same* path for both instances — the worker writes
evidence clips there and `ml-api` reads them back.

### Backend

```bash
API_EDGE_RELAY_TOKEN=local-edge-relay-token \
CLIP_STORE_DIR="$ML_DEV_STATE/clip-store" \
API_CONNECTION_SETTINGS_PATH="$ML_DEV_STATE/ml-api/connection-settings.sqlite3" \
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

`API_EDGE_RELAY_TOKEN` must equal `relay.token` in the worker's YAML — the
worker sends it as `X-Edge-Relay-Token` and `ml-api` compares against this env
var. `GET /api/v1/health` reports `relay.token_configured` so the pairing is
observable without sending a relay call.

### Worker

Validate and run the worker with a local configuration:

```bash
uv run python -m worker --config worker/ml-worker.local.yaml --check-config

CLIP_STORE_DIR="$ML_DEV_STATE/clip-store" \
ML_WORKER_PROFILE=cpu \
ML_WORKER_DEV_MJPEG=true \
ML_WORKER_DEV_MJPEG_HOST=127.0.0.1 \
ML_WORKER_DEV_MJPEG_PORT=8090 \
uv run python -m worker --config worker/ml-worker.local.yaml
```

`ML_WORKER_DEV_MJPEG` serves the live view that the dashboard's room detail
reads; port 8090 matches `Settings.worker_stream_origin`'s default, so leaving
both at these values is what makes the live panel resolve. `cpu` is the only
profile whose device check passes without a GPU.

Run the front dev server:

```bash
pnpm --dir front install --frozen-lockfile
ML_API_PROXY_TARGET=http://127.0.0.1:8000 pnpm --dir front dev
```

## Edge deployment

Copy `.env.edge.prod.example` to `.env.edge.prod`, then replace the real
per-site values: the `.example` backend URLs, `<facility-id>` /
`<relay-token>` / dashboard credentials, `ML_WORKER_PROFILE` (production uses
`cuda`; CPU-only nodes use the overlay below), the host clip-store directory,
and the digest-pinned GHCR image references
(`docs/runbooks/edge-image-publish.md`). Every other variable
`compose.edge.yaml` requires already ships with a working default, so a
single `.env.edge.prod` is enough — no second env file needed.
`ML_WORKER_EVENT_CLIP_EXPORT_ENABLED` defaults to `1` (it gates all alert
delivery, not just clips — see #194); `ML_API_EVENT_CLIP_EXPORT_ENABLED`
stays off by default until backend clip capability is verified for this
deployment.

Before `up`, verify the env file actually renders and has no leftover
`<placeholder>` values:

```bash
scripts/edge-preflight/check-env.sh .env.edge.prod
```

Then start the edge-only stack from this repository root (add
`-f compose.edge.cpu.yaml` on nodes without an NVIDIA GPU):

```bash
docker compose --env-file .env.edge.prod -f compose.edge.yaml up -d
```

The images are published as
`ghcr.io/seniorailab/eldercare-fall-ml/{ml-api,ml-worker}` (deployment identity;
these map to `Dockerfile.backend` / `Dockerfile.edge`). `compose.edge.yaml`
uses `models/` as the default host model-artifact path.

## Operations

- [`docs/operations/config-pitfalls.md`](docs/operations/config-pitfalls.md) —
  settings that are silently ineffective when set on the wrong process or in
  the wrong place (backend vs. worker env, YAML-vs-env precedence).
- [`docs/operations/soak-test-plan.md`](docs/operations/soak-test-plan.md) —
  24h+ continuous-run soak test scenario, metrics, and pass/fail thresholds.
- [`docs/operations/clip-retention-policy.md`](docs/operations/clip-retention-policy.md) —
  clip storage and retention policy.
- [`docs/runbooks/`](docs/runbooks/) — incident runbooks (worker rollback,
  driver/CUDA alignment, local e2e RTSP source, Intel iGPU/VAAPI decode).

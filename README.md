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

Run the backend:

```bash
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Validate and run the worker with a local configuration:

```bash
uv run python -m worker --config worker/ml-worker.local.yaml --check-config
uv run python -m worker --config worker/ml-worker.local.yaml
```

Run the front dev server:

```bash
pnpm --dir front install --frozen-lockfile
ML_API_PROXY_TARGET=http://127.0.0.1:8000 pnpm --dir front dev
```

## Edge deployment

Copy `.env.edge.prod.example` to `.env.edge.prod`, set the backend URLs,
facility credentials, relay token, `ML_WORKER_PROFILE` (production uses `cuda`),
host clip-store directory, and digest-pinned GHCR image references. Event clip export is off by default; set both
`ML_API_EVENT_CLIP_EXPORT_ENABLED=1` and
`ML_WORKER_EVENT_CLIP_EXPORT_ENABLED=1` only after backend capabilities are
verified. Start the edge-only stack from this repository root:

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
  driver/CUDA alignment, local e2e RTSP source).

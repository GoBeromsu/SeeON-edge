# eldercare-fall-ml

Edge ML runtime for fall and bed-exit detection, organized as three deployable
instances plus a shared library:

- **`backend/`** — FastAPI control/status/relay gateway (`ml-api` image).
- **`worker/`** — RTSP inference worker that relays facts to the backend (`ml-worker` image).
- **`front/`** — React/Vite dashboard SPA served by the backend.
- **`shared/`** — `shared.events` (the backend↔worker wire code); `contracts` is a
  top-level vendored leaf (ADR-0006).

Historical training lived in the archived `SeniorAILab/eldercare-dataset-ops`.

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
uv run --group lint mypy contracts shared   # type-boundary enforcement
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

### Rust 1.90.0 builder

`Dockerfile.rust-builder` pins the official Rust 1.90.0 linux/amd64 toolchain
and the DeepStream SDK base by digest. Checksum-verified `clippy` and `rustfmt`
components and the compiler are copied into the SDK builder so native CUDA/
TensorRT linking uses the actual libraries. `rust-toolchain.toml` declares
the matching toolchain.
This prerequisite does not replace `Dockerfile.edge` or install a host compiler.

On the authorized onsite build host, use an explicit two-file context:

```bash
tar -cf - Dockerfile.rust-builder rust-toolchain.toml \
  | docker build \
      --platform linux/amd64 \
      --file Dockerfile.rust-builder \
      --tag seeon-rust-builder:1.90.0-deepstream9.1 \
      -
```

Only these reviewed files belong in the build context: never send `.env`,
credentials, agent runtime state, or the whole working tree. When controlled
remotely, transfer the content-addressed context and verify its digest onsite
before building. Check the resulting image without mounting any host files:

```bash
docker run --rm --network none --read-only \
  seeon-rust-builder:1.90.0-deepstream9.1 rustc --version
docker run --rm --network none --read-only \
  seeon-rust-builder:1.90.0-deepstream9.1 rustfmt --version
docker run --rm --network none --read-only \
  seeon-rust-builder:1.90.0-deepstream9.1 cargo clippy --version
```

The builder workdir is `/usr/src/myapp`. Workspace builds require the exact
canonical native-library directories in `SEEON_GPU_LIB_DIR`,
`SEEON_MEDIA_LIB_DIR`, `SEEON_CLIPDEC_LIB_DIR`, and `SEEON_ORT_LIB_DIR`, plus
their runtime loader paths. `Dockerfile.edge` prepares these libraries in its
`native-build` stage. Missing libraries are refused; there is no automatic
provider fallback.

The separate `worker/adapters/model/onnxruntime` Rust owner executes captured
ONNX bytes through the existing ONNX Runtime CPU library. The role owners in
`worker/runtime/rust/src/cpu` connect it to existing RGB preprocessing, ordered
person-box decoding, bed tensors and fall windows. Thread-confined actors in
`worker/bin/src/inference/cpu.rs` now warm and execute all three roles, retain
the loaded ORT version and fatal failures independently of reply queues, and
return CPU results without accelerator claims. Startup provider selection and
model admission are not yet connected to these actors; the Rust camera loop
still starts TensorRT owners. The native Makefile requires `ORT_INCLUDE_DIR`
with three digest-checked API-29 headers; their immutable source and hashes are
listed in that Makefile. `make test-ort` additionally requires a Python
environment with ONNX, NumPy, and ONNX Runtime 1.29.0.
The image sets `ORT_DISABLE_TELEMETRY=1` before process startup to prevent
vendor telemetry initialization; the CPU adapter refuses admission without it.

The identity reader/publisher APIs distinguish schema 1 (four TensorRT engines)
from schema 2 (one live-pose TensorRT engine and three explicitly tagged CPU
ONNX source hashes). Hybrid admission retains the live engine, Flow, image and
batch checks without requiring unused auxiliary engines. The `engine-build`
CLI selects schema 2 with `--auxiliary-runtime=onnxruntime-cpu`: it builds only
the live-pose GPU engine and fingerprints captured pose, bed and admitted fall
ONNX bytes. Omit `--stored-pose-engine`, `--bed-engine` and `--fall-engine` in
this mode; supplying any of them is refused. All common Flow, model, image and
batch inputs remain required. The default `--auxiliary-runtime=tensorrt`
retains schema 1 and requires all three auxiliary engine outputs. Cache reuse
must match the selected provider; neither mode falls back to the other.

Run startup still selects TensorRT owners. The offline hybrid route does not
establish an end-to-end mixed-provider worker, and native CPU shutdown
qualification remains blocked by observed worker-thread execution stalls.

CPU reference values are recorded independently by
`python tests_support/record_ort_cpu_outputs.py --models <models> --fixtures <gpu-v2a> --output <new-directory>`,
using the pinned Python environment and `ORT_DISABLE_TELEMETRY=1`. Existing
references are never overwritten. The Rust `native_cpu` integration tests take
explicit `SEEON_TEST_ORT_RUNTIME`, `SEEON_TEST_ORT_MODELS`,
`SEEON_TEST_ORT_FIXTURES`, and `SEEON_TEST_ORT_CPU_RECEIPT` paths; select them with
`cargo test -p seeon-onnxruntime-native --test native_cpu -- --include-ignored`.
They compare every output element in order, including its float32 bits.

`cargo test -p seeon-ml-worker --test cpu_owners -- --include-ignored --test-threads=1`
tests actual CPU actor readiness, input recovery, replies and shutdown. It uses
`SEEON_TEST_ORT_RUNTIME`, `SEEON_TEST_ORT_MODELS`, `ORT_DISABLE_TELEMETRY=1`,
and `SEEON_TEST_PYTHON` pointing to Python with ONNX installed. Its generated
fault model tests terminal failure retention under full response queues, not
product-model numerical parity.

Production remains `python -m worker`. Ignored GPU integration tests require
explicit engine/model inputs and actual GPU access. CPU-adapter comparisons
do not replace the GPU parity gate or qualify a complete Rust worker.

## Run

### Local state

The backend stores durable state only in PostgreSQL. A local run needs a
PostgreSQL 18 server whose schemas were created by
`python -m backend.app.edge_db.migration provision`
(`docs/runbooks/postgresql-cutover.md`), and three environment values
(`backend/app/postgres_root.py`):

| Env var | Meaning |
| --- | --- |
| `API_POSTGRES_DSN_FILE` | file holding the runtime connection string |
| `API_POSTGRES_AUTHORITY_FILE` | file holding the persistence authority |
| `API_POSTGRES_SCHEMA` | schema name, default `seeon_edge` |

The clip store is fixed at `/var/lib/clip-store` for both the worker and
`ml-api`; `compose.edge.yaml` mounts it. `CLIP_STORE_DIR`,
`API_CONNECTION_SETTINGS_PATH` and `API_LABEL_STORE` are retired: the backend
refuses to start while any of them is set.

### Backend

```bash
API_POSTGRES_DSN_FILE=/path/to/runtime.dsn \
API_POSTGRES_AUTHORITY_FILE=/path/to/authority.json \
API_EDGE_RELAY_TOKEN=local-edge-relay-token \
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

ML_WORKER_PROFILE=cpu \
uv run python -m worker --config worker/ml-worker.local.yaml
```

`cpu` is the only profile whose device check passes without a GPU.
`ML_WORKER_DEV_MJPEG*` and `CLIP_STORE_DIR` are retired; the worker refuses to
start while any of them is set.

Run the front dev server:

```bash
pnpm --dir front install --frozen-lockfile
ML_API_PROXY_TARGET=http://127.0.0.1:8000 pnpm --dir front dev
```

## Edge deployment

Copy `.env.edge.prod.example` to `.env.edge.prod`, then replace the real
per-site values: the `.example` backend URL, relay token, dashboard
credentials, Flow batch size, the host clip-store directory, and the
digest-pinned GHCR image references
(`docs/runbooks/edge-image-publish.md`). Every other variable
`compose.edge.yaml` requires already ships with a working default, so a
single `.env.edge.prod` is enough — no second env file needed.
Event delivery is always active once relay credentials are valid. Clip export
is a persisted Edge dashboard setting that defaults OFF and applies live without
a worker restart; backend capability checks still gate actual clip relay.

Before `up`, verify the env file actually renders and has no leftover
`<placeholder>` values:

```bash
scripts/edge-preflight/check-env.sh .env.edge.prod
```

Then start the edge-only DeepStream Flow stack from this repository root:

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

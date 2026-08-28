#!/bin/bash
#
# Dev-host wrapper around the worker-owned model provisioner. The compose
# stack runs the same module as the one-shot `edge-model-fetch` service into
# the `worker-models` named volume; on a checkout this populates ./models so
# `python -m worker` and the test-suite find the packaged default artifacts.
#
# Every file comes from a pinned upstream listed in
# worker/tools/fetch_models/manifest.json and is verified against its
# committed SHA-256. Idempotent: re-running with everything present is a
# no-op. Weights stay gitignored (see .gitignore).
#
# Usage: scripts/fetch-models.sh [--force] [--check]
# Environment:
#   ML_WORKER_FETCH_MODELS_DEST      models root (default: <repo>/models)
#   ML_WORKER_FETCH_MODELS_ATTEMPTS  max download attempts per file (default 6)
#   HF_TOKEN                         optional Hugging Face token, never logged
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${ML_WORKER_FETCH_MODELS_DEST:-${repo_root}/models}"

cd "$repo_root"
if command -v uv >/dev/null 2>&1; then
  exec uv run python -m worker.tools.fetch_models --dest "$dest" "$@"
fi
exec python3 -m worker.tools.fetch_models --dest "$dest" "$@"

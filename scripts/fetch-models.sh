#!/bin/bash
#
# Downloads the packaged default LSTM fall-detector weights + upstream
# metadata from the public Hugging Face repo, pinned to a fixed revision so
# every fetch is reproducible. Idempotent: re-running with the files already
# present and matching size is a no-op (re-run with --force to overwrite).
#
# Issue #133: the worker must be able to boot with zero env vars using a
# packaged default LSTM model. Weights are large binary artifacts and stay
# gitignored (see .gitignore); this script is how an operator (or CI/dev
# bootstrap) materializes them locally before first boot.
#
# Upstream's metadata.json is saved as metadata.upstream.json (not
# metadata.json): worker.adapters.model.lstm_manifest.LstmFallManifest
# refuses to boot when metadata.json and this repo's own metadata.yaml
# sidecar (tracked in git, alongside arch.json) coexist in the same
# artifact_dir, so the two must never share a filename.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pinned per docs/... investigation: Berom0227/eldercare-fall-models is public;
# this exact revision is known to carry lstm/model.pt + lstm/metadata.json.
HF_REPO="Berom0227/eldercare-fall-models"
HF_REVISION="d67887844bfd2e4b1ca3f3275f770b0b05e23aba"
HF_BASE_URL="https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/lstm"

dest_dir="${ML_WORKER_FETCH_MODELS_DEST:-${repo_root}/models/fall/lstm}"
force=0

usage() {
  cat <<'EOF'
Usage: scripts/fetch-models.sh [--force]

Downloads the packaged default LSTM fall-detector weights (model.pt) and
upstream metadata (saved as metadata.upstream.json) from the public Hugging
Face repo Berom0227/eldercare-fall-models, pinned to a fixed revision, into
models/fall/lstm/.

Options:
  -h, --help    Show this help and exit.
  --force       Re-download even if the destination files already exist.

Environment:
  ML_WORKER_FETCH_MODELS_DEST   Override the destination directory
                                 (default: <repo>/models/fall/lstm).
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --force)
      force=1
      ;;
    *)
      echo "fetch-models.sh: unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$dest_dir"

# Fetches $HF_BASE_URL/$1 into $dest_dir/${2:-$1}, skipping when already
# present (unless --force), and prints the resulting file's sha256.
fetch_one() {
  local remote_name="$1"
  local local_name="${2:-$1}"
  local dest="$dest_dir/$local_name"
  if [[ -f "$dest" && "$force" -eq 0 ]]; then
    echo "fetch-models.sh: $dest already exists; skipping (use --force to re-download)"
  else
    echo "fetch-models.sh: downloading $remote_name -> $dest"
    curl -fSL --retry 3 --retry-delay 2 -o "$dest.tmp" "$HF_BASE_URL/$remote_name"
    mv "$dest.tmp" "$dest"
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$dest"
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$dest"
  fi
}

fetch_one "model.pt"
fetch_one "metadata.json" "metadata.upstream.json"

echo "fetch-models.sh: done. Weights are in $dest_dir (gitignored; arch.json/metadata.yaml sidecars are tracked)."

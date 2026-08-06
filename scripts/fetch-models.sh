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
  ML_WORKER_FETCH_MODELS_DEST       Override the destination directory
                                     (default: <repo>/models/fall/lstm).
  ML_WORKER_FETCH_MODELS_ATTEMPTS   Max download attempts per file
                                     (default: 6). Each retry backs off
                                     exponentially with jitter and honours a
                                     Retry-After header when the server sends
                                     one.
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
# Issue #188: retries are driven here rather than by `curl --retry`, because
# `--retry-delay` replaces curl's exponential backoff with a flat delay. The
# old `--retry 3 --retry-delay 2` gave up roughly six seconds in, while the
# rate-limit window Hugging Face applies to anonymous downloads is measured in
# minutes -- so CI went red on a 429 that had nothing to do with the change
# under test.
#
# The jitter matters as much as the backoff. This repo runs the same commit on
# both `push` and `pull_request`, so two jobs fetch the same file at the same
# moment as a matter of course. Without jitter they also retry in lockstep and
# keep knocking each other out; that is how one job failed while its twin on
# the identical commit passed.
max_attempts="${ML_WORKER_FETCH_MODELS_ATTEMPTS:-6}"
max_backoff_sec=120

# Downloads $1 into $2, retrying on failure. Honours a Retry-After header when
# the server sends one -- the server's own number beats anything we can guess.
fetch_with_retry() {
  local url="$1"
  local dest_tmp="$2"
  local headers attempt=1 status wait_sec

  headers="$(mktemp)"
  # shellcheck disable=SC2064  # expand $headers now, not at trap time
  trap "rm -f '$headers'" RETURN

  while :; do
    status="$(curl -sSL --connect-timeout 20 -D "$headers" -o "$dest_tmp" \
                   -w '%{http_code}' "$url")" || status=000
    if [[ "$status" == 2* ]]; then
      return 0
    fi

    # Only statuses that can plausibly succeed later are worth waiting on:
    # 429/408 plus 5xx, and 000 for a transport error curl never got a status
    # for. A 404 means the pinned revision or filename is wrong, and retrying
    # that just turns a clear five-second failure into a slow, confusing one.
    case "$status" in
      429|408|5*|000) ;;
      *)
        echo "fetch-models.sh: $url returned HTTP $status; not retryable" >&2
        rm -f "$dest_tmp"
        return 1
        ;;
    esac

    if (( attempt >= max_attempts )); then
      echo "fetch-models.sh: giving up on $url after $attempt attempts (last HTTP status: $status)" >&2
      rm -f "$dest_tmp"
      return 1
    fi

    wait_sec="$(sed -n 's/^[Rr]etry-[Aa]fter:[[:space:]]*\([0-9]\{1,\}\).*/\1/p' "$headers" | tail -1)"
    if [[ -z "$wait_sec" ]]; then
      # 4, 8, 16, 32, 64 seconds, plus up to 5 seconds of jitter.
      wait_sec=$(( 2 ** (attempt + 1) + RANDOM % 6 ))
    fi
    (( wait_sec > max_backoff_sec )) && wait_sec=$max_backoff_sec

    echo "fetch-models.sh: attempt $attempt for $url failed (HTTP $status); retrying in ${wait_sec}s" >&2
    sleep "$wait_sec"
    attempt=$(( attempt + 1 ))
  done
}

# present (unless --force), and prints the resulting file's sha256.
fetch_one() {
  local remote_name="$1"
  local local_name="${2:-$1}"
  local dest="$dest_dir/$local_name"
  if [[ -f "$dest" && "$force" -eq 0 ]]; then
    echo "fetch-models.sh: $dest already exists; skipping (use --force to re-download)"
  else
    echo "fetch-models.sh: downloading $remote_name -> $dest"
    fetch_with_retry "$HF_BASE_URL/$remote_name" "$dest.tmp"
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

#!/bin/sh
# Preflight: verify an edge env file actually renders through
# `docker compose config` and has no leftover <placeholder> values before
# `up` (#179, #182). Missing/empty `${VAR:?...}` vars fail render outright;
# placeholder text like <facility-id> renders FINE (it's a non-empty string)
# but is still wrong, so it needs a separate scan of the rendered output.
#
# Usage (from the repo root, on the edge device):
#   scripts/edge-preflight/check-env.sh [env-file] [-f extra-compose.yaml ...]
#
# Examples:
#   scripts/edge-preflight/check-env.sh
#   scripts/edge-preflight/check-env.sh .env.edge.prod
#   scripts/edge-preflight/check-env.sh .env.edge.prod -f compose.edge.cpu.yaml
set -eu

prefix='[edge-preflight]'
die() { printf '%s %s\n' "$prefix" "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "missing required command: docker"
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is not available"

env_file="${1:-.env.edge.prod}"
[ -f "$env_file" ] || die "env file not found: $env_file (cp .env.edge.prod.example $env_file first)"
shift 2>/dev/null || true

# Remaining args ("$@") are passed straight through as extra docker compose
# arguments, e.g. `-f compose.edge.cpu.yaml` for a CPU-only node.
set -- -f compose.edge.yaml "$@"

if ! render=$(docker compose --env-file "$env_file" "$@" config -q 2>&1); then
  printf '%s compose failed to render from %s:\n' "$prefix" "$env_file" >&2
  printf '%s\n' "$render" >&2
  die "fix the missing/empty variable(s) named above in $env_file, then re-run this check"
fi

placeholders=$(docker compose --env-file "$env_file" "$@" config 2>/dev/null \
  | grep -oE '<[a-zA-Z0-9_-]+>' | sort -u || true)
if [ -n "$placeholders" ]; then
  printf '%s %s renders, but still has unresolved placeholder(s):\n' "$prefix" "$env_file" >&2
  printf '%s\n' "$placeholders" | sed 's/^/  /' >&2
  die "fill in the real value(s) above in $env_file before starting the stack"
fi

printf '%s %s renders cleanly with no leftover placeholders.\n' "$prefix" "$env_file"

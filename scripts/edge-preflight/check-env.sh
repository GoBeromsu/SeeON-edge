#!/bin/sh
# Validate the complete edge env-file authority before Docker Compose sees it.
set -eu

prefix='[edge-preflight]'
die() { printf '%s %s\n' "$prefix" "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "missing required command: docker"
command -v python3 >/dev/null 2>&1 || die "missing required command: python3"
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is not available"

env_file="${1:-.env.edge.prod}"
[ -f "$env_file" ] || die "env file not found: $env_file (cp .env.edge.prod.example $env_file first)"
shift 2>/dev/null || true

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
inventory="$repo_root/edge-env-inventory.json"
[ -f "$inventory" ] || die "environment inventory not found: $inventory"

# Docker Compose ignores unreferenced env-file keys. Validate the file itself so
# a retired authority cannot survive merely because compose stopped projecting it.
if ! validation=$(python3 - "$env_file" "$inventory" <<'PY' 2>&1
import json
import re
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
inventory_path = Path(sys.argv[2])
entries = json.loads(inventory_path.read_text(encoding="utf-8"))["variables"]
by_name = {entry["name"].upper(): entry for entry in entries}
seen: dict[str, tuple[str, int]] = {}
errors: list[str] = []

for line_number, original in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
    stripped = original.strip()
    if not stripped or stripped.startswith("#"):
        continue
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=.*", stripped)
    if match is None:
        errors.append(f"line {line_number}: malformed env assignment")
        continue
    key = match.group(1)
    normalized = key.upper()
    previous = seen.get(normalized)
    if previous is not None:
        errors.append(
            f"line {line_number}: duplicate key {key} "
            f"(case-insensitive duplicate of {previous[0]} on line {previous[1]})"
        )
        continue
    seen[normalized] = (key, line_number)
    entry = by_name.get(normalized)
    value = stripped.split("=", 1)[1].strip()
    if normalized == "ML_WORKER_PROFILE":
        allowed_profiles = set(entry["canonical_choices"]) if entry is not None else set()
        if entry is not None:
            allowed_profiles.update(entry.get("legacy_aliases", {}))
        if value not in allowed_profiles:
            errors.append(
                f"line {line_number}: unsupported profile ML_WORKER_PROFILE={value!r}; "
                f"set one of {'|'.join(sorted(allowed_profiles))}"
            )
            continue
    if entry is None:
        errors.append(f"line {line_number}: unsupported key {key}; add an inventory disposition first")
        continue
    if not (entry.get("compose") is True and entry.get("example") is True):
        errors.append(
            f"line {line_number}: retired key {key} "
            f"({entry['category']}): {entry['behavior']}"
        )

if errors:
    print("\n".join(errors))
    raise SystemExit(1)
PY
); then
  printf '%s %s contains invalid environment authority:\n' "$prefix" "$env_file" >&2
  printf '%s\n' "$validation" | sed 's/^/  /' >&2
  die "remove retired/duplicate/unsupported keys, then re-run this check"
fi

profile=$(python3 - "$env_file" <<'PY'
import sys
from pathlib import Path

for original in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    stripped = original.strip()
    if stripped.startswith("ML_WORKER_PROFILE="):
        print(stripped.split("=", 1)[1].strip())
        break
PY
)

# Classify caller-supplied infrastructure overlays before adding the deterministic
# profile default. Docker Compose otherwise accepts a contradictory NVIDIA
# overlay for a host-only profile and silently reserves a GPU.
nvidia_overlay_supplied=0
igpu_overlay_supplied=0
expect_compose_file=0
record_compose_file() {
  compose_file=${1##*/}
  case "$compose_file" in
    compose.edge.nvidia.yaml) nvidia_overlay_supplied=1 ;;
    compose.edge.igpu.yaml) igpu_overlay_supplied=1 ;;
  esac
}
for argument in "$@"; do
  if [ "$expect_compose_file" -eq 1 ]; then
    record_compose_file "$argument"
    expect_compose_file=0
    continue
  fi
  case "$argument" in
    -f|--file) expect_compose_file=1 ;;
    --file=*) record_compose_file "${argument#*=}" ;;
    -f?*)
      compose_file_argument=${argument#-f}
      record_compose_file "${compose_file_argument#=}"
      ;;
  esac
done
[ "$expect_compose_file" -eq 0 ] || die "missing compose file after -f/--file"

set -- -f compose.edge.yaml "$@"
case "$profile" in
  cuda|nvidia-host-bridge|nvidia-device-experimental)
    if [ "$nvidia_overlay_supplied" -eq 0 ]; then
      set -- "$@" -f compose.edge.nvidia.yaml
    fi
    ;;
  igpu|intel-vaapi-host)
    [ "$nvidia_overlay_supplied" -eq 0 ] || \
      die "ML_WORKER_PROFILE=$profile cannot use compose.edge.nvidia.yaml"
    if [ "$igpu_overlay_supplied" -eq 0 ]; then
      set -- "$@" -f compose.edge.igpu.yaml
    fi
    ;;
  cpu|cpu-host|mps|apple-mps-host)
    [ "$nvidia_overlay_supplied" -eq 0 ] || \
      die "ML_WORKER_PROFILE=$profile cannot use compose.edge.nvidia.yaml"
    ;;
  *)
    [ "$nvidia_overlay_supplied" -eq 0 ] || \
      die "compose.edge.nvidia.yaml requires an NVIDIA ML_WORKER_PROFILE"
    ;;
esac
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

# Hub transport gate: production base must be https:// (no cleartext public Hub).
hub_base=$(python3 - "$env_file" <<'PY'
import sys
from pathlib import Path
for original in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    stripped = original.strip()
    if stripped.startswith("API_BACKEND_BASE_URL="):
        print(stripped.split("=", 1)[1].strip().strip('"').strip("'"))
        break
PY
)
case "$hub_base" in
  "")
    die "API_BACKEND_BASE_URL is required and must be an https:// Hub base URL"
    ;;
  https://*)
    ;;
  *)
    die "API_BACKEND_BASE_URL must use https:// (got: $hub_base). Cleartext public Hub URLs are forbidden."
    ;;
esac
allow_insecure=$(python3 - "$env_file" <<'PY'
import sys
from pathlib import Path
for original in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    stripped = original.strip()
    if stripped.startswith("API_BACKEND_ALLOW_INSECURE_HTTP="):
        print(stripped.split("=", 1)[1].strip().strip('"').strip("'"))
        break
PY
)
case "$allow_insecure" in
  1|true|TRUE|yes|YES)
    die "API_BACKEND_ALLOW_INSECURE_HTTP=$allow_insecure is a local-fixture contract only; unset it for production"
    ;;
esac

env_get() {
  key=$1
  python3 - "$env_file" "$key" <<'PY'
import sys
from pathlib import Path
key = sys.argv[2]
value = ""
for original in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    stripped = original.strip()
    if stripped.startswith(f"{key}="):
        value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
print(value)
PY
}

dash_user=$(env_get API_DASHBOARD_USERNAME)
dash_pass=$(env_get API_DASHBOARD_PASSWORD)
if [ -z "$dash_user" ] || [ -z "$dash_pass" ]; then
  die "API_DASHBOARD_USERNAME and API_DASHBOARD_PASSWORD must both be set to a deployment-unique bootstrap pair"
fi
if [ "$dash_user" = "admin" ] && [ "$dash_pass" = "admin" ]; then
  die "API_DASHBOARD_USERNAME/PASSWORD must not be the known default admin/admin; set a random per-site bootstrap pair"
fi

relay_token=$(env_get API_EDGE_RELAY_TOKEN)
if [ -z "$relay_token" ]; then
  die "API_EDGE_RELAY_TOKEN must be set to a deployment-unique random relay secret (e.g. openssl rand -hex 32)"
fi
# Reject the shipped sample, obvious placeholders, and weak throwaway values so a
# copy-paste of the example (or a lazy value) can never become the live secret.
relay_lower=$(printf '%s' "$relay_token" | tr '[:upper:]' '[:lower:]')
case "$relay_lower" in
  eldercare-internal-edge-relay \
  | "<random-relay-token>" \
  | "<"*">" \
  | changeme | change-me | changethis | change-this | placeholder \
  | secret | token | relay | relay-token | relaytoken | password | default \
  | test | testing | example | sample | admin | dev | development)
    die "API_EDGE_RELAY_TOKEN is a known sample/placeholder/weak value; set a deployment-unique random relay secret"
    ;;
esac
if [ "${#relay_token}" -lt 16 ]; then
  die "API_EDGE_RELAY_TOKEN is too short (<16 chars); use a high-entropy random secret (e.g. openssl rand -hex 32)"
fi

rtsp_private=$(env_get ML_RTSP_ALLOW_PRIVATE_DESTINATIONS)
rtsp_local=$(env_get ML_RTSP_ALLOW_LOCAL_DESTINATIONS)
case "$rtsp_private" in
  ""|0|1) ;;
  *) die "ML_RTSP_ALLOW_PRIVATE_DESTINATIONS must be 0 or 1 (got: $rtsp_private)" ;;
esac
case "$rtsp_local" in
  ""|0|1) ;;
  *) die "ML_RTSP_ALLOW_LOCAL_DESTINATIONS must be 0 or 1 (got: $rtsp_local)" ;;
esac
case "$rtsp_local" in
  1)
    die "ML_RTSP_ALLOW_LOCAL_DESTINATIONS=1 is fixture-only; keep 0 in production"
    ;;
esac

printf '%s %s passes inventory, compose, Hub HTTPS, dashboard, and RTSP gates.\n' \
  "$prefix" "$env_file"

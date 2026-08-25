#!/usr/bin/env bash
set -euo pipefail

# C6's locally qualified OCI image (label revision 4da96c969897371ab01b57416e853011b377c6bc).
readonly C6_WORKER_IMAGE='seeon-edge@sha256:d87cc3eb29abc1fc79a83f6a1001aff51889614a514f73cb58f9e90a18d3dd6d'
readonly ENV_FILE="${ENV_FILE:-.env.edge.prod}"

: "${COMPOSE_PROJECT_NAME:?COMPOSE_PROJECT_NAME must name the existing SeeON compose project}"
docker image inspect "${C6_WORKER_IMAGE}" >/dev/null

command=(
  docker compose
  --project-name "${COMPOSE_PROJECT_NAME}"
  --env-file "${ENV_FILE}"
  -f compose.edge.yaml
  -f compose.edge.nvidia.yaml
  up -d
  --pull never
  --no-deps
  --force-recreate
  ml-worker
)

if [[ "${DRY_RUN:-0}" == '1' ]]; then
  printf 'ML_WORKER_IMAGE=%q ' "${C6_WORKER_IMAGE}"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

ML_WORKER_IMAGE="${C6_WORKER_IMAGE}" "${command[@]}"

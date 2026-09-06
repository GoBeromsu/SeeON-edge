# worker/runtime/config: versioned worker config

Own load, pull, LKG, restart identity, and typed models. YAML is a developer hatch.
Production authority is the backend relay pull plus numeric policies.
No env roster.
`ML_WORKER_PROFILE=flow` is infrastructure selection only. Its DeepStream
artifact paths are image-owned wiring, never camera/config authority.

`--config` is the only local path. `resolve_config_path` reads no env, so a roster cannot arrive through compose or Git.
JSON is refused. Mapping only. Hatch fields: `relay`, `runtime`, empty `cameras`, `dev_mjpeg`.
`--check-config` with YAML validates that file alone. No pull, no model, no camera open.

`load_worker_config` rejects a non-empty `cameras` list before `WorkerConfig` parses.
Static camera roster is retired. Register cameras in the dashboard.
Omitted or empty roster is valid. Zero cameras is a legal boot.

`detection_policies` is refused by field presence, even when valid, out of range, or forged.
`models`, `domains`, and `clip` are retired the same way. Numeric policy never comes from YAML.

Production uses baked `http://ml-api:8000` plus `RELAY_TOKEN`.
`RELAY_URL` is retired and fails closed via `reject_retired_worker_environment`.
GET `/api/v1/cameras/worker-config` with `X-Edge-Relay-Token`.

`ConfigSource`: `PULLED`, `LKG`, `YAML`. Fresh pull writes `WorkerConfigLkgStore` (`stale=False`).
Failed or malformed pull falls back to LKG (`stale=True`). No pull and no LKG refuses boot (exit 2).
YAML path calls `resolve_startup_config`: a live pull wins; a miss may keep the zero-camera hatch.

`BackendWorkerConfigPayload` is the wire model. One bad camera or window drops that row and logs.
A payload that declared cameras and parsed none is corrupt; raise so LKG stays in charge.
Cameras without `rtsp_url` boot as an empty usable roster. Missing `facility_id` becomes `"local"`.

Pulled payload carries fleet state: cameras, domains, windows, `detection_policies`, clip store subdir, clip-export flags.
`models`, `clip.enabled`, and `dev_mjpeg` are local overlays from `resolve_local_overrides`.
Payload `models`/`clip` keys are rejected until Todo 9.

Pulled `detection_policies` is one fail-closed bundle via `parse_policy_bundle`.
`PolicyDocumentError` becomes `WorkerConfigError` (`detection policy refused`).
A missing field uses `default_policy_bundle` for the resolved roster.
Restart polls parse too, so a higher revision with broken semantics does not stop the running LKG worker.

`night_window` maps to `bed_exit` only. `domains.enabled` is a legacy replace-list: listed on, every other known domain off.
Prefer `domains.<name>.enabled`. `BedExitDomainConfig.night_window` loses to an explicit `detection_windows["bed_exit"]`.

If the pull supplied any window info (non-empty `detection_windows` or `night_window`), it owns every domain.
An omitted domain is ALWAYS, 24/7. YAML windows apply only when the pull said nothing.
Degenerate start==end is invalid. Per-domain bad windows fail open to ALWAYS.

Retired env: `RELAY_URL`, both legacy Edge camera-config keys, `CLIP_STORE_DIR`,
`ML_WORKER_FALL_MODEL_*`, `ML_WORKER_CLIP_*`, `ML_WORKER_DEV_MJPEG*`, `ML_WORKER_EVENT_CLIP_EXPORT_ENABLED`.
Present keys refuse boot.

`RestartDirective(generation, version)` is `restart_epoch` then `config_version`.
Newer generation wins even at version 0. Tracker is monotonic. Equal or older is ignored.
`make_restart_check` polls every 60s. Pull fail or `None` keeps the process up.
LKG save refuses a lower `(directive, registry_version)`. Corrupt LKG that would win the race is cleared.
`LiveClipExportPolicy` can apply from a poll without process restart.
Roster, domain, and numeric-policy changes take effect on the next process start after the directive advances.

Frozen pydantic, `extra="forbid"` on local models. Wire payload uses `extra="ignore"` and per-row degrade.
Public types: `WorkerConfig`, `CameraRuntimeConfig`, `RelayConfig`, `DomainsConfig`, `FallModelConfig`,
`BackendWorkerConfigPayload`, `ConfigSnapshot`. `camera_id` is opaque. Duplicate ids raise.
Decode backends: `auto`, `nvdec`, `opencv`, `cpu`. `KNOWN_DOMAIN_NAMES` comes from `DOMAIN_REGISTRY`.
`WorkerConfigError` is the CLI config fault (exit 2). `ConfigValidationError` is field-level.

Focused tests: `tests/test_ml_worker_yaml_config.py`, `tests/test_worker_config_residue.py`,
`tests/test_worker_static_detection_policy_authority.py`, `tests/test_worker_startup_config_resolution.py`,
`tests/test_worker_config_lifecycle.py`, `tests/test_worker_config_pull_lkg.py`,
`tests/test_worker_config_pull_models.py`, `tests/test_worker_config_contract.py`,
`tests/test_worker_restart_directive.py`, `tests/test_worker_policy_resolution.py`,
`tests/test_worker_config_local_overrides.py`. Boundary: `uv run --group lint lint-imports`.

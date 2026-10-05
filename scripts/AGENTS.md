# scripts

Leaf executables for release, edge-host operations, and QA measurement. Invoked
by path, by CI workflows, or by tests; never imported by product packages.
Earned a file as a distinct domain (score 8): 57 source files plus 29 immutable
spike receipts.

## Where to look

| Task | Location | Notes |
| --- | --- | --- |
| Release tag vs tree identity | `release_guard.py`, `release_notes.py` | Run by `release.yml`. Carrier list pinned by `tests/test_release_workflow_contract.py`. |
| Per-image build or reuse | `edge_image_plan.py` (`plan`, `decide`, `retag`, `previous-tag`) | The only owner of that decision. Workflows call it; they never use `paths:` filters. |
| Scope-fidelity gate | `verify_scope_fidelity.py --fixture` / `--repo` | CI lint job runs both. PEP 723 `uv run --script` header. |
| Deleted-tree leak scan | `deletion_closure_scan.py` | `DELETED_TREES` and `EXCLUDED` are the policy. |
| Clip catalog audit and rebuild | `catalog_verify.py`, `catalog_backfill.py` | `--clip-store <dir> [--catalog <db>] [--cameras <json>]`. Tests run them through `sys.executable`. |
| Sidecar archival transaction | `archive_preserved_sidecars.py` | Imported by `tests/test_sidecar_archival_transaction.py`. |
| Model download | `fetch-models.sh [--force] [--check]` | Wraps `worker.tools.fetch_models`. |
| Host preflight and diagnosis | `edge-preflight/*.sh` | `check-env.sh .env.edge.prod` gates deploy. |
| Host updater daemon | `edge-updater/update-edge.sh`, `test.sh`, `README.md` | `EDGE_UPDATER_*` env. Unsent reports land in `$EDGE_UPDATER_DATA_DIR/outbox`. |
| One-shot operator tools | `ops/` | Refused-evidence review, clip-consistency repair, replay analysis, enrollment smoke, alert-amplification diagnostic (918 lines). |
| Serving QA ladder | `qa/trace_continuity.py`, `ort_batch_parity.py`, `batch_probe.py`, `batch_probe_compare.py` | Diagnostic only. |
| Golden episode toolchain | `qa/golden_worksheet.py`, `golden_from_worksheet.py`, `golden_labeller_html.py`, `export_incident_corpus.py` | Imported by tests as `scripts.qa.*`. |
| Canary stack | `qa/deepstream-canary/`, `qa/deepstream_canary_browser.mjs` | `CANARY_*` env; gate policy is JSON. |
| DeepStream spike | `qa/pyservicemaker-spike/` | Recorded measurements and the letterbox fixture capture. |

## Conventions

- No `__init__.py` anywhere. `scripts.qa.*` resolves as a namespace package from the repo root.
- Python tools: `argparse`, a `main()` returning an int, `raise SystemExit(main())`. CI-facing failures print `::error::` to stderr.
- A script that imports repo packages prepends the repo root to `sys.path` and marks the late import `# noqa: E402`.
- Read-only tools say so in the docstring and enforce it (`flow_alert_rate.py` opens SQLite `mode=ro`).
- `ops/repair-clip-consistency.py` is dry-run unless `--apply`, and needs a quiescence receipt.
- Shell is POSIX `sh` by default. `bash` only where arrays or `readonly` are needed (`diagnose-edge.sh`, `rollback-ml-worker-c6.sh`).
- `ml-front-tailscale-serve.sh` pins `#!/bin/bash`, not `env bash`: Homebrew bash 5.3 hangs on a heredoc over `PIPE_BUF` (#9). `tests/test_shell_script_heredoc_contract.py` guards it.
- Fixed-argv subprocess calls carry `# noqa: S603 - fixed argv, no shell`.
- Scripts that reach `backend.app.*` (`catalog_*`, `migrate_camera_floor_to_int.py`, `ops/repair-clip-consistency.py`) are backend-side tools. Worker-side tools reach `shared`, `contracts`, and `worker` (`qa/fall_model_recall_at_gate.py`, `ops/review-refused-evidence.py`). Only `ops/alert-amplification-diagnostic.py` spans both, through `tests_support`.

## Anti-patterns

- Editing `release_guard.py` carriers to match a tag, or adding `front/src/shared/releaseIdentity.ts` or `worker/runtime/provenance/environment.py` to them. The tree decides; the tag follows.
- Rewriting anything under `qa/pyservicemaker-spike/receipts/`. Recorded measurements; scans exclude them.
- Committing ONNX or TensorRT engine outputs from the spike.
- `cp` plus digest in `archive_preserved_sidecars.py`. It uses `link(2)`, never overwrites, and destroys no source that drifted or vanished.
- JNU targets or hostnames in `ops/cloud-enrollment-smoke.sh`. Only `happy-nursing-home-raw` is approved.
- Cleartext public Hub URLs or default `admin/admin` passing `check-env.sh`.
- Mutable image tags (`latest`, `:dev`), destructive SQL, or an env-provisioned camera roster. `verify_scope_fidelity.py` flags all three.
- Product code importing a QA or diagnostic script.
- One-shot operator tools that auto-start the stack.
- Citing `scripts/edge-preflight/gpu-stability-install.sh` or `scripts/rtsp_sweep.sh` as present. Neither is tracked.

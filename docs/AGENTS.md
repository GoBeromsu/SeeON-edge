# docs

Prose contracts: architecture, decision records, operator runbooks. Earned a
file as a distinct domain (score 8): 43 files, 7 subdirectories, and the
strongest written source for boot gates and refusal semantics.

## Structure

```
docs/
├── architecture.md   # canonical; ownership map, failure matrix, parity ledger, open gaps
├── decisions/        # ADR 0001-0008 + README index
├── runbooks/         # operator procedures; some are dated handoffs
├── operations/       # clip retention (Korean), config pitfalls, soak plan
├── research/         # host baselines, spikes, private-repo-baseline.sha256
├── releases/         # one note per seeon-edge-v<semver> tag
├── assets/           # readme-hero.webp + ARTWORK.md provenance
├── exec-plan/        # active/ handoff; archive/ empty
└── rules/            # empty placeholder
```

## Where to look

| Question | Location |
| --- | --- |
| Who owns a behavior, what is still open | `architecture.md` "Source-to-target ownership", "Open gaps" |
| Evidence clip must stay source-faithful | `decisions/0001` |
| Fail-fast, explicit fallback only | `decisions/0002` (Korean body), `decisions/0003` |
| Worker holds no database | `decisions/0005` |
| `contracts/` vendoring rules | `decisions/0006` |
| Overlay is not persisted with clip media | `decisions/0008` |
| Schema 19 bootstrap, restore order | `runbooks/edge-database-schema-19.md` |
| Image publish, digest pinning, per-image reuse | `runbooks/edge-image-publish.md` |
| Execution-record vocabulary and budgets | `runbooks/observability-diagnostics.md` |
| Redeploy without losing identity | `runbooks/edge-redeploy-identity-continuity.md` |
| Driver and CUDA alignment | `runbooks/driver-cuda-alignment.md` |
| Env key dispositions | `operations/config-pitfalls.md` + root `edge-env-inventory.json` |
| 60-day retention floor | `operations/clip-retention-policy.md` |

## Conventions

- An ADR is written only on explicit user request. Each opens with `- Status:`; the README index row must match it.
- ADR numbers are stable. A reversed decision is corrected in place with a status line, not renumbered.
- Claims cite `file:line` anchors. Re-check the anchor when the cited code moves.
- Tests read these paths: `runbooks/edge-image-publish.md`, `assets/readme-hero.webp` (the one approved art path), `architecture.md` "Entrypoint", `runbooks/post-redeploy-event-readout.md`. Renaming one means updating the test.
- A release is an annotated tag `seeon-edge-v<semver>` plus a note in `releases/`. Nobody drafts a release in the GitHub UI.
- Runbook commands pin images by `@sha256:` digest and pass `--pull never`.
- English by default. Korean is deliberate where it already exists.

## Known drift

- `decisions/README.md` has no row for 0007. That ADR is superseded by its own correction: the current decision is to build the backend-owned replay path. Do not cite its original reasoning.
- 0004 is superseded for the mapper push path; roster publication is a `TopologyClient` snapshot.
- 0003 still cites "ADR-0004 (vendored `contracts/`)". 0006 leaves that citation in place knowingly.
- `architecture.md` and 0006 cite `tests/test_worker_architecture_docs.py`; `edge-image-publish.md` cites `tests/test_edge_image_isolation.py`; `clip-retention-policy.md` cites `backend/app/features/clips/audit_log.py`. None exists on disk.
- Many parity-ledger test names in `architecture.md` are legacy and absent.

## Anti-patterns

- Edge addresses, accounts, tokens, camera IPs, RTSP URLs, or `user:password@` userinfo in any doc or issue.
- A runbook step that runs `docker compose down -v`, deletes the `edge-state` volume, or repairs `edge.sqlite3` with direct SQL.
- Substituting `latest`, a branch tag, or a hand-written digest for a sealed digest.
- Describing a local RTSP serving surface (MediaMTX, FFmpeg publisher) as part of this repository.
- Rendering a missing execution record as "no person" or "no fall". The vocabulary is `AVAILABLE | MISSING_NOT_RECORDED | DELETED_BY_CAPACITY | UNKNOWN_COARSENED | UNKNOWN`.
- Quoting measured budgets from a runbook as product defaults.
- Fixing a stale claim by editing the test name into the doc without checking the file exists.

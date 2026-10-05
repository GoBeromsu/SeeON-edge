# tests_support

Importable test-only package: builders, served fixtures, and metric reducers
shared across suites and operator scripts. Earned a file as a distinct domain
(score 12): 13 modules, 60 importing test modules, 4 importing scripts.

## Where to look

| Need | Module | Importers |
| --- | --- | --- |
| Externally prepared schema-18 database for durable-authority tests | `compact_authority_db.py` | 25 |
| Packaged pose-bbox56 fall bundle artifact | `pose_bbox56_bundle_artifact.py` | 10 |
| `BedPoseFeatures` builders matching the detector's `hip_depth >= 0.10` arming rule | `bed_pose_fixtures.py` | 7 |
| Versioned golden episode corpus: load and validate | `golden_episodes.py` | 6 |
| Alert-amplification measurement over the real relay and evidence chain | `alert_amplification_harness.py` (1100 lines) | 5 |
| Loopback Hub fixture plus real relay client | `alert_amplification_runtime.py` | 4 |
| Clip-analysis publication hook doubles | `clip_analysis.py` | 5 |
| Thumbnail helpers | `thumbnail.py` | 5 |
| Local backend over real HTTP | `local_backend_fixture.py` | 3 |
| Connection API server double | `connection_api.py` | 3 |
| Exact golden-episode metrics from replay traces; has a CLI `main()` | `episode_metric.py` | 2 |
| Production callables that own each audit action | `audit_production_owners.py` | 1 |

## Conventions

- Unlike `tests/`, this is a real package (`__init__.py`). Import as `tests_support.<module>`.
- It may import `backend`, `worker`, `shared`, and `contracts` together. That is legal only because it is test-only; it is the one place both sides of the relay meet in-process.
- Product packages never import it. `scripts/ops/alert-amplification-diagnostic.py` and `scripts/qa/golden_*.py` do, so those scripts run from a repo checkout, not from a deployed image.
- Served fixtures (`alert_amplification_runtime.py`, `local_backend_fixture.py`, `connection_api.py`) bind loopback only: no credentials, no RTSP, no media.
- A helper moves here once a second suite or a script needs it. Single-suite helpers stay beside the test in `tests/`.
- Fixtures mirror measured product conventions. When a detector constant or schema CHECK changes, change the builder here, not each test.
- Ruff relaxes `RUF012`, `PERF401`, `SIM102` for this tree only.

## Anti-patterns

- Hand-built SQLite rows. `compact_authority_db.py` goes through `bootstrap_database` so fixtures obey the schema.
- Timing or relay stubs that patch product code. Wrap from the test side.
- Importing `torch` at module top level in a new helper. `pose_bbox56_bundle_artifact.py` already does, and its 10 importers pay for it at collection.
- Numeric pass/fail thresholds inside a harness. Harnesses record measurements; tests and operators decide.
- Growing `alert_amplification_harness.py` further. Split by concern before adding a new one.

# CONTRACTS KNOWLEDGE BASE

Define every cross-layer **protocol, constant, enum, and shared data shape** here — and nowhere else. `contracts` is the single framework-free L0 home for the ML package's interface vocabulary; keep it dependency-light (no pydantic/cv2/torch, no model loading, no I/O) and additive, because every higher layer imports it.

## Local Ownership

- `frame.py`: `Frame` and `FrameSource`.
- `observation.py`: boxes, labels, detection results, and `FrameObservation`.
- `model.py`: model module protocol and shared confidence defaults.
- `artifacts.py`: model/weight path helpers.
- `tracker.py`: shared tracker protocol surface.
- `event.py`: event severity, levels, `EVENT_TYPE_REGISTRY`, and frontend event-type mapping.
- `relay.py`: `AlertEventType`, `EventApiPayload`, relay alert/heartbeat payloads.
- `runner.py`: runner result kinds, `<X>Output` aliases, `RunnerProtocol`.
- `worker_config.py`: `PulledWorkerConfig` family, `WORKER_CONFIG_PATH` / `WORKER_RESTART_PATH`, version keys.
- `replay_trace.py`: versioned replay rows + `encode_*` / `decode_*` (document and JSONL).
- `model_selection.py`: desired/applied model selection, canonical digest, receipt identity validators.
- `edge_provisioning_*.py` (v1, models, parse, codec, enrollment, response, validation): Hub enrollment and topology vocabulary. `edge_provisioning_v1.py` is the entry.
- `edge-provisioning-v1/`: byte-frozen `contract-fixtures.json` + `provenance.json`. Synthetic tokens keep the upstream digest (`.gitleaksignore`).
- `decode_diagnostics.py` / `encode_diagnostics.py`: backend names, fallback reasons, selection shapes.
- `__init__.py` re-exports 34 names; `tests/test_contract_symbol_exports.py` pins them.

## Imports

Allowed: standard library and local `contracts` modules. Import-linter leaf contract forbids `backend`, `worker`, `shared.events`. mypy `strict` on `contracts.*`.

Forbidden: `features`, `sources`, `runners`, `perception`, `domains`, `runtime`, `events`, `api`, `demo`, `training`, model loading, camera I/O, network I/O.

## Naming convention

- **Module**: lowercase singular concept noun, one bounded concept per file (`frame`, `observation`, `runner`, `event`, `model`, `artifacts`). A new concept gets a new module — never a `common`/`types`/`misc` dump.
- **Data shapes**: `@dataclass(frozen=True, slots=True)`, PascalCase noun (`Frame`, `BoundingBox`, `FrameObservation`); domain-prefix when ambiguous (`BedRegionDebugSnapshot`). A mutable variant is `Mutable<Name>` (`MutableEventPayload`).
- **Protocols**: PascalCase with a `Protocol` suffix (`RunnerProtocol`, `RunRunnerProtocol`).
- **Type aliases**: PascalCase; runner/boundary I/O uses an `<X>Output` suffix (`PoseOutput`, `BoxOutput`, `RunnerOutput`); composites get a domain noun (`Detections`, `Regions`, `Image`).
- **Enums**: `StrEnum`, PascalCase with an axis suffix that reads at the call site (`DetectionEventType`, `Level`, `<Concept>State`); members are `UPPER_SNAKE` with lowercase string values.
- **Debug/telemetry**: `<Concept>DebugSnapshot`.
- **Constants**: `UPPER_SNAKE_CASE` (`FALL_LABEL_TEXT`, `DEFAULT_FALL_CONFIDENCE_THRESHOLD`).
- New modules end with an explicit `__all__`. `frame.py`, `observation.py`, `model.py`, `artifacts.py` predate the rule and have none.

## Focused Tests

- `tests/test_contract.py`
- `tests/test_frame_observation_contract.py`
- `tests/test_events_schema.py`
- `tests/test_contract_symbol_exports.py`, `tests/test_worker_config_contract.py`, `tests/test_replay_trace_contract.py`
- `tests/test_edge_provisioning_contract.py`, `tests/test_edge_topology_contract.py`
- `uv run --group lint lint-imports` and `uv run --group lint mypy contracts shared` (the AST ladder test is gone)

## Gotchas

Contracts are consumed across every layer. Prefer additive fields or new dataclasses over changing existing constructor semantics.
`PulledWorkerConfig` gained optional `registry_version` for restart identity; mirroring this
contract to `eldercare-dataset-ops` is follow-up work.

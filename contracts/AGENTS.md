# CONTRACTS KNOWLEDGE BASE

Define every cross-layer **protocol, constant, enum, and shared data shape** here — and nowhere else. `contracts` is the single framework-free L0 home for the ML package's interface vocabulary; keep it dependency-light (no pydantic/cv2/torch, no model loading, no I/O) and additive, because every higher layer imports it.

## Local Ownership

- `frame.py`: `Frame` and `FrameSource`.
- `observation.py`: boxes, labels, detection results, and `FrameObservation`.
- `model.py`: model module protocol and shared confidence defaults.
- `artifacts.py`: model/weight path helpers.
- `tracker.py`: shared tracker protocol surface.
- `event.py`: event severity, levels, and frontend event-type mapping.

## Imports

Allowed: standard library and local `contracts` modules.

Forbidden: `features`, `sources`, `runners`, `perception`, `domains`, `runtime`, `events`, `api`, `demo`, `training`, model loading, camera I/O, network I/O.

## Naming convention

- **Module**: lowercase singular concept noun, one bounded concept per file (`frame`, `observation`, `runner`, `event`, `model`, `artifacts`). A new concept gets a new module — never a `common`/`types`/`misc` dump.
- **Data shapes**: `@dataclass(frozen=True, slots=True)`, PascalCase noun (`Frame`, `BoundingBox`, `FrameObservation`); domain-prefix when ambiguous (`BedRegionDebugSnapshot`). A mutable variant is `Mutable<Name>` (`MutableEventPayload`).
- **Protocols**: PascalCase with a `Protocol` suffix (`RunnerProtocol`, `RunRunnerProtocol`).
- **Type aliases**: PascalCase; runner/boundary I/O uses an `<X>Output` suffix (`PoseOutput`, `BoxOutput`, `RunnerOutput`); composites get a domain noun (`Detections`, `Regions`, `Image`).
- **Enums**: `StrEnum`, PascalCase with an axis suffix that reads at the call site (`DetectionEventType`, `Level`, `<Concept>State`); members are `UPPER_SNAKE` with lowercase string values.
- **Debug/telemetry**: `<Concept>DebugSnapshot`.
- **Constants**: `UPPER_SNAKE_CASE` (`FALL_LABEL_TEXT`, `DEFAULT_FALL_CONFIDENCE_THRESHOLD`).
- Every module ends with an explicit `__all__`.

## Focused Tests

- `tests/test_contract.py`
- `tests/test_frame_observation_contract.py`
- `tests/test_events_schema.py`
- `tests/test_import_dependency_ladder.py`

## Gotchas

Contracts are consumed across every layer. Prefer additive fields or new dataclasses over changing existing constructor semantics.
`PulledWorkerConfig` gained optional `registry_version` for restart identity; mirroring this
contract to `eldercare-dataset-ops` is follow-up work.

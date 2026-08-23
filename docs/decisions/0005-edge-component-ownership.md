# 0005 — Edge Component Ownership

- Status: Accepted
- Date: 2026-08-21
- Relates to: `backend/app/edge_db/`, `backend/app/`, `worker/`, `front/`,
  `compose.edge.yaml`, [0006](0006-vendored-contracts-typed-vocabulary.md).

## Context

The edge stack has three runtime components and one backend-owned SQLite
database. The database is mounted only into the migrator and backend API, so
the inference runtime cannot open it. Responsibility for repair, migration, and
integrity is therefore a backend boundary rather than a convention.

The inference runtime is also expected to be replaced. A future DeepStream or
Service Maker implementation is a C++ GStreamer process that cannot import the
backend database package at all. A rule written against "the worker" would therefore
lapse precisely when the component it names is swapped out.

## Decision

Ownership attaches to the **inference-runtime slot**, not to whichever
implementation currently occupies it.

The slot is defined by role: the component that consumes camera streams, runs
detection, and produces evidence media. Today that is the Python worker. Any
successor that occupies the same role inherits every rule below unchanged.

### Forbidden-dependency matrix

The subject of each row is the *slot*, not a module path.

| Subject | May depend on | Must never depend on |
|---|---|---|
| Inference-runtime slot | `contracts/` typed vocabulary; its own package; the backend's authenticated HTTP API | SQLite, `backend.app.edge_db`, any schema name, any migration, any backend table |
| Backend | `contracts/`; `backend.app.edge_db`; its own package | The inference runtime's internal modules; writing media bytes |
| Frontend | The backend's HTTP API | The database; the inference runtime; any server-side module |

The backend is the sole writer of every application table family. The migrator
retains ownership of `schema_*`.

### Permitted on-disk state for the slot

The slot may keep on disk exactly:

1. media files it produces,
2. a publish-once delivery queue,
3. a verified bounded config read cache,
4. a media-integrity sidecar,
5. zero-payload lock inodes,
6. startup-purged in-progress scratch.

Anything else — canonical state, a query API, an index, a second state database,
or any backend schema name — belongs to the backend. The queue directory is its
own single capacity authority; there is no separate persisted ledger.

## Consequences

- The rule survives replacement of the Python worker, which is the point.
- Enforcement is structural rather than advisory: the slot's container receives
  no database mount, and the boundary is checked semantically across both
  writable volumes rather than by directory naming.
- Evidence is never silently dropped. Because the slot cannot write to the
  database during a backend outage, the publish-once queue is the mechanism that
  keeps exactly one replayable event per detection until the backend
  acknowledges it.

## Status note

The ownership boundary is structurally enforced: the migrator and API mount
`edge-state`; the inference runtime mounts `worker-local-state` and no
`edge-state` volume.

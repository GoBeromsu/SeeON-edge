# SeeON Edge v0.1.0

The first tagged release of the edge stack: the FastAPI gateway (`ml-api`) and
the DeepStream inference worker (`ml-worker`), built from one commit and
published to GHCR as two digest-pinned images.

## Highlights

**Detection and alerting**

- The worker reports the native detection producer instead of falling through
  to `disabled`, and the epoch it stamps on events is the one the detector
  actually ran under (#427).
- The relay names a local acceptance explicitly, which breaks the worker's
  retry loop on alerts the relay had already taken (#433). A terminal receipt
  is never issued for an alert that was not persisted.
- Packets are released when geometry telemetry fails, instead of being held
  (#398).

**Clips**

- `GET /clips` is served from the catalogue and walks the clip store once per
  request, rather than re-hashing every clip's media on each call (#439, #442).
- `HEAD` is answered on the clip and snapshot media routes (#457).

**Edge database**

- The v1–v18 migration ledger is retired; schema 18 is create-only (#437).
  `EDGE_DATABASE_FORMAT_IDENTITY` remains the on-disk format lineage and is
  independent of this product version.

**Structure**

- Backend and worker each own their runtime interface, with a contract test
  guarding drift between them (#436).
- `useMjpegStream` is lifted into `shared/api`, breaking the
  settings↔operations import cycle (#435).
- A lint-driven simplification pass across the worker (#438), and the shell
  e2e harnesses and their bound contract asserts are gone (#425).

**Model provisioning**

- Models are provisioned by a pinned, hash-verified `edge-model-fetch` one-shot
  into a named volume; `ml-worker` starts only after it exits 0 (#441). No
  weights are baked into either image.

**CI and supply chain**

- One build path for both edge images, with staging tags and digest job
  outputs (#440).
- One CI run per pull request, parallel jobs, and a 4-way pytest shard
  (19m37s → 7m47s) (#443).
- Every action in a `pull_request`-reachable workflow is pinned to a 40-hex
  commit SHA, every job is bounded, and the shard is asserted to cover the
  whole suite (#444).

## Cutting this release

Tag scheme, the version-carrier guard, and the rehearsal dispatch are
documented in `docs/runbooks/edge-image-publish.md`.

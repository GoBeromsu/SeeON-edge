# 0007. Deterministic replay has no backend-owned entry point

- Status: **Superseded by its own correction, see below. Do not cite the original reasoning.**
- Date: 2026-08-22

## What this ADR originally claimed, and why it was wrong

The first version of this decision argued that a backend-owned replay command
could not be built, and used that to justify shipping none. It rested on two
claims about the codebase. **Both were false**, and neither was verified before
being written down.

**False claim 1: the durable schema is lossy.** The original text said the
`runtime_analysis_*` tables "omit the per-person keypoints and bed polygons"
needed to reconstruct a `FrameObservation`. They do not. Schema 13 creates
`runtime_analysis_keypoints` and `runtime_analysis_bed_points`, and schema 6
already provides traces, persons, beds, components, decisions, values and
truncation cursors. Schema 17 preserves all of them and reassigns the `runtime_`
writer family to the API. The tables are empty in practice, but that is a
*missing backend ingest path*, not a missing representation. No new DDL is
required and schema 17 does not need widening.

**False claim 2: the backend cannot reach the engine.** The original text said
`backend/` imports no `worker/` code and the API image does not ship it, and
concluded replay was therefore unreachable. The premises are true and the
conclusion does not follow. An authenticated backend-to-worker control channel
already exists on `ml-worker:8090`, gated by the same relay token, and the
backend already uses it for stream, snapshot and bed-zone operations. A
backend-owned command can reconstruct the exact recovered trace from the durable
tables and hand it to a worker-side replay endpoint that owns the engine and the
model. That crosses no import boundary and needs no repackaging.

## How the error happened

A delegated investigation reported these two blockers, and they were accepted
into an ADR without independent verification. A single query against a migrated
database would have shown `runtime_analysis_keypoints` and
`runtime_analysis_bed_points`; a single search for `8090` would have shown the
control channel.

This is the same failure this effort has corrected repeatedly elsewhere:
an artifact that looks authoritative while resting on unchecked input. Recording
it here rather than quietly deleting the file is deliberate, because an ADR that
justified *not* building something is exactly the kind of document nobody
re-examines later.

## Current decision

Build the backend-owned replay path. It requires:

- API-owned ingest and query over the existing `runtime_analysis_*` tables,
  reconstructing a replay-complete `RecoveredCameraTrace` including ordering and
  `TraceTruncation`;
- a packaged `scripts/ops` command that sends the exact reconstructed input to an
  authenticated worker replay endpoint over the existing `:8090` channel;
- refusal rather than approximation when input is missing or truncated. A replay
  result that does not correspond to the decision actually made must never be
  persisted through `QaStore.record_run()`, because it would look authoritative
  while being wrong.

The only claim from the original version that survives is that last one: faking
a result is worse than having no command. Everything used to argue that one could
not be built was mistaken.

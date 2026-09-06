# ADR 0008: Evidence Overlay Is Not Persisted with Clip Media

- Status: Accepted
- Date: 2026-09-03
- Scope: Evidence clip analysis display

## Decision

The scene sidecar is retired. The original `clip.mp4` stays untouched. Live
overlays remain an operator-view capability; burned pixels are permitted only
in the snapshot JPEG derivative.

## Drivers

- Preserve the one-way decision-to-media data plane and ADR-0001 source evidence.
- Keep the NVIDIA path image-free.
- Keep clip persistence limited to immutable source media and its manifest.

## Alternatives Considered

- Child OSD burn-in violates ADR-0001, consumes NVENC sessions, prevents a toggle,
  and reverses the data flow.
- Backend rendering crosses slice boundaries, competes for edge CPU, and introduces
  derivative-cache lifecycle.
- Snapshot-only evidence cannot show the temporal basis of a decision.

## Why This Decision

Removing the sidecar preserves auditable source bytes while eliminating a
second persisted clip representation. Snapshot JPEGs may contain overlays
because they are explicitly derived review artifacts.

## Consequences

Clip storage no longer includes per-frame overlay metadata. The frontend does
not time-align persisted overlay frames, and the worker does not retain a
sidecar ring.

## Follow-ups

- Move live overlays to the client and remove the CPU renderer candidate.

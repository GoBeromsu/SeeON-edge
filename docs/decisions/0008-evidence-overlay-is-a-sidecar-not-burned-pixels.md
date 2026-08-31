# ADR 0008: Evidence Overlay Is a Sidecar, Not Burned Pixels

- Status: Accepted
- Date: 2026-09-01
- Scope: Evidence clip analysis display

## Decision

Fall and bed-exit evidence overlays are packaged as a PTS-keyed scene sidecar at
`clips/<clip_id>/scene-index.json`: canonical JSON with a content-addressed
manifest claim. The frontend renders it only in a sibling `<svg>` overlay. The
source `clip.mp4` bytes remain immutable.

## Drivers

- Preserve the one-way decision-to-media data plane and ADR-0001 source evidence.
- Keep the NVIDIA path image-free.
- Reuse one scene vocabulary for live and clip display.

## Alternatives Considered

- Child OSD burn-in violates ADR-0001, consumes NVENC sessions, prevents a toggle,
  and reverses the data flow.
- Backend rendering crosses slice boundaries, competes for edge CPU, and introduces
  derivative-cache lifecycle.
- Snapshot-only evidence cannot show the temporal basis of a decision.

## Why This Decision

The sidecar requires no re-encoding, permits an operator toggle, preserves
auditable evidence bytes, and behaves identically across runtime profiles.

## Consequences

Each clip can add up to 8 MiB of storage (about 9.6 GB at 20 daily clips over 60
days); existing rotation covers it. The frontend owns time alignment and hides
frames when VFR jitter exceeds tolerance. The worker owns a bounded 48 MiB scene
ring.

## Follow-ups

- Export sidecars to Hub.
- Move live overlays to the client and remove the CPU renderer candidate.

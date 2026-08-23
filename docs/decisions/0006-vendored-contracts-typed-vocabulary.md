# 0006 — Vendored `contracts/` Typed Vocabulary

- Status: Accepted
- Date: 2026-08-21
- Relates to: `contracts/`, `contracts/AGENTS.md`, `tests/test_vendor_drift.py`,
  `tests/test_worker_event_payload_boundary.py`, `pyproject.toml` (import-linter
  layer contracts).

## Context

`contracts/` is the canonical typed vocabulary for data that crosses an instance
boundary. It is mirrored byte-for-byte into a sibling repository, and
`tests/test_vendor_drift.py` snapshots every file beneath it so the two copies
cannot silently diverge.

That arrangement has been load-bearing since the vendor-drift remediation, but it
was never written down. Thirteen citations across ten files referenced
"ADR-0004" as the decision that established it. ADR 0004 is
[Camera Roster Sync Contract Assumptions](0004-camera-roster-sync-contract-assumptions.md)
and says nothing about vendoring.

The mistake originated in ADR 0003, which forward-cited a not-yet-written
ADR 0004 as "vendored `contracts/`". When 0004 was actually written it took a
different subject, and every later citation copied the forward reference rather
than the real record. The decision has therefore been enforced by tests and
tooling while its record did not exist.

This ADR is that missing record. It does not change behavior; it states the rule
the tests already enforce.

## Decision

`contracts/` is the canonical, vendored, typed-vocabulary leaf for
cross-instance L0 data.

1. **Canonical and mirrored.** `contracts/` is the source of truth and is
   mirrored byte-for-byte into its sibling. Drift in either direction is a test
   failure, not a merge conflict to resolve by hand.
2. **Framework-free.** `contracts/` carries data vocabulary only. It must not
   import a web framework, a database, an inference runtime, or any transport.
3. **Never shadowed.** No component may duplicate, re-declare, or shadow a
   vendored type. Component-internal ports and envelopes live under their own
   package, not in `contracts/`.
4. **Vendor-neutral.** The vocabulary is defined by the data that crosses the
   boundary, never by whichever vendor, runtime, or consumer happens to read it.
   Vendor-specific names — DeepStream, GStreamer, NVIDIA, or any successor — must
   not appear in `contracts/`, and no test enforcing this decision may narrow its
   assertions to a single vendor or consumer beyond the existing byte-mirror
   mechanism.
5. **Edited at the source.** Because the mirror is byte-compared, `contracts/`
   is edited deliberately and re-synced, never patched on one side.

## Scope

This record covers vendoring and the typed vocabulary only.

It deliberately does **not** cover the training-repository split. Some citations
that named "ADR-0004" meant "training moved to the dataset-ops repository",
which is a third, separate subject. Recording it here would widen a
citation-repair decision into an unrelated ownership decision, so those
references have had their incorrect ADR citation removed and the split is routed
as its own follow-up.

## Consequences

- The twelve incorrect citations outside ADR 0002 and ADR 0003 now resolve to
  this record.
- **One incorrect citation is knowingly left in place.** ADR
  [0003](0003-explicit-fallback-only.md) still reads "Consistent with ADR-0004
  (vendored `contracts/`)". That line is the origin of the phantom, but the
  minimal-repair decision governing this work forbids editing the bodies of
  ADR 0002 and ADR 0003. Correcting it is a separate, deliberate act on an
  accepted record; it is recorded here as known and routed as a follow-up rather
  than silently patched.
- ADR numbers are stable. `tests/test_worker_architecture_docs.py` pins ADR
  0003's filename, status, and prose, so no existing record is renumbered.
- No behavior changes. `tests/test_vendor_drift.py` already enforced this rule
  before it had a record.

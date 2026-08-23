"""Doc contract for the worker architecture documentation.

Lightweight, pure-Python text assertions (no subprocess): the migration's
source-to-target ownership map must stay complete and unambiguous, the docs this
migration owns must not carry an executable legacy entrypoint, and the canonical
entrypoint plus the ADR-0001 follow-up gap must stay documented.

The authoritative public-repository scanner is
``tests/test_public_repository_privacy.py``; the path-policy mirror below is a
fast local guard so a new doc cannot be introduced under a protected path.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "docs" / "architecture.md"
ROLLBACK_RUNBOOK = ROOT / "docs" / "runbooks" / "worker-migration-rollback.md"
EDGE_TREE = ROOT / "edge"

CANONICAL_ENTRYPOINT = "python -m worker"

# Docs this migration todo owns. Every other doc is retargeted by the operator
# surface cutover todo, which owns its own stale-path gate.
OWNED_DOCS = (ARCHITECTURE, ROLLBACK_RUNBOOK)

# Migration sources are cited as table rows; those citations are historical and
# are not operator instructions.
_SOURCE_ROW = re.compile(r"^\|\s*`(edge/[^`]+)`\s*\|\s*(.+?)\s*\|\s*$")

# An "executable" legacy path is one an operator or process would run, not a
# `path:line` citation of a file being migrated.
_EXECUTABLE_LEGACY_PATTERNS = (
    re.compile(r"python\s+-m\s+edge\b"),
    re.compile(r"-m\s+edge\b"),
    re.compile(r"\bedge\.runtime\."),
    re.compile(r"\bedge\.__main__\b"),
    re.compile(r"--config\s+edge/ml-worker"),
    re.compile(r"\bedge/ml-worker\.local\.yaml\b"),
)

_STALE_PROSE = (
    "edge is runtime package",
    "edge is the runtime package",
    "python -m edge",
)

_SOURCE_SUFFIXES = frozenset({".py", ".md", ".toml", ".yaml"})

# Mirror of the privacy policy's protected path vocabulary.
_PROHIBITED_PATH_PARTS = frozenset(
    {
        "annotations",
        "checkpoints",
        "data",
        "dataset",
        "datasets",
        "eval",
        "evaluation",
        "exports",
        "labels",
        "linkage",
        "models",
        "weights",
    }
)
_PROHIBITED_SUFFIXES = frozenset({".csv", ".jsonl", ".mp4", ".png", ".pt", ".sqlite"})


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _legacy_source_paths() -> tuple[str, ...]:
    """Non-``__init__`` source paths still present in the legacy edge tree.

    Returns an empty tuple once the tree is deleted, at which point the
    architecture table is purely historical and no longer constrained.
    """
    if not EDGE_TREE.is_dir():
        return ()
    return tuple(
        sorted(
            path.relative_to(ROOT).as_posix()
            for path in EDGE_TREE.rglob("*")
            if path.is_file()
            and path.suffix in _SOURCE_SUFFIXES
            and path.name != "__init__.py"
            and "__pycache__" not in path.parts
        )
    )


def _mapped_source_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in _read(ARCHITECTURE).splitlines():
        match = _SOURCE_ROW.match(line)
        if match is not None:
            rows.append((match.group(1), match.group(2)))
    return rows


def _executable_legacy_hits(text: str) -> list[str]:
    hits: list[str] = []
    for line in text.splitlines():
        if _SOURCE_ROW.match(line):
            continue
        hits.extend(
            pattern.pattern
            for pattern in _EXECUTABLE_LEGACY_PATTERNS
            if pattern.search(line)
        )
    return hits


PARITY_BASELINE_SHA = "aeed6a8"
_PARITY_DISPOSITIONS = ("ported", "tracked-deferred", "out-of-scope (uncommitted)")


def _parity_ledger_rows() -> list[tuple[str, str, str, str]]:
    """Capability rows of the feature parity ledger.

    The ledger is a four-column table under ``## Feature parity ledger``. Rows
    are collected until the next heading so the baseline-uncommitted table that
    follows is not mistaken for a capability row.
    """
    rows: list[tuple[str, str, str, str]] = []
    in_section = False
    for line in _read(ARCHITECTURE).splitlines():
        if line.startswith("## "):
            in_section = line.strip() == "## Feature parity ledger"
            continue
        if not in_section or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        if cells[0] in {"Capability", "---"} or set(cells[0]) <= {"-", " "}:
            continue
        rows.append((cells[0], cells[1], cells[2], cells[3]))
    return rows


def test_architecture_documents_a_non_empty_feature_parity_ledger() -> None:
    text = _read(ARCHITECTURE)
    assert "## Feature parity ledger" in text
    assert _parity_ledger_rows() != []


def test_parity_ledger_pins_the_committed_baseline_commit() -> None:
    text = _read(ARCHITECTURE)
    assert PARITY_BASELINE_SHA in text
    assert "committed" in text.lower()


def test_every_parity_row_names_an_owner_and_a_known_disposition() -> None:
    for capability, owner, behaviour_test, disposition in _parity_ledger_rows():
        assert capability, "a parity row must name a capability"
        matched = [
            known for known in _PARITY_DISPOSITIONS if disposition.startswith(known)
        ]
        assert matched, f"{capability} has an unknown disposition: {disposition}"
        if disposition.startswith("ported"):
            assert owner not in {"", "—"}, f"{capability} is ported without an owner"
            assert behaviour_test not in {
                "",
                "—",
            }, f"{capability} is ported without a behaviour test"


def test_ported_parity_rows_cite_behaviour_tests_that_exist() -> None:
    missing: list[str] = []
    for capability, _owner, behaviour_test, disposition in _parity_ledger_rows():
        if not disposition.startswith("ported"):
            continue
        for cited in re.findall(r"`([^`]+)`", behaviour_test):
            if not (ROOT / cited).exists():
                missing.append(f"{capability} -> {cited}")
    assert missing == [], f"parity ledger cites tests that do not exist: {missing}"


def test_ported_parity_rows_cite_owners_that_exist() -> None:
    missing: list[str] = []
    for capability, owner, _behaviour_test, disposition in _parity_ledger_rows():
        if not disposition.startswith("ported"):
            continue
        for cited in re.findall(r"`([^`]+)`", owner):
            if not (ROOT / cited).exists():
                missing.append(f"{capability} -> {cited}")
    assert missing == [], f"parity ledger cites owners that do not exist: {missing}"


def test_parity_ledger_states_the_missing_capability_rule() -> None:
    text = _read(ARCHITECTURE)
    assert "Missing-capability rule" in text
    for expected in ("runtime feature", "script or tool", "behaviour-coverage test"):
        assert expected in text, f"the missing-capability rule omits {expected}"


def test_platform_limited_parity_evidence_names_the_exact_cases() -> None:
    """The row whose evidence cannot run locally must be listed, not implied.

    One ``ported`` row is proven by tests that only run on Linux, because the
    capability itself reads ``/proc``. A developer on macOS sees those fail and,
    without this list, has no way to tell a genuine runtime floor from a parity
    gap -- which is the precise confusion ``Open gaps`` exists to prevent.

    Pinning the individual test IDs rather than a summary sentence is
    deliberate: if one of these is fixed, renamed, or deleted, this fails and
    forces the note to be corrected instead of quietly rotting.
    """
    text = _read(ARCHITECTURE)
    platform_limited = (
        "tests/test_clip_recorder.py::"
        "test_clip_recorder_finalizes_atomic_manifest_with_pre_and_post_window",
        "tests/test_clip_recorder.py::"
        "test_clip_recorder_fsyncs_media_and_manifest_before_staging_cleanup",
    )
    for case in platform_limited:
        assert case in text, f"Open gaps omits the platform-limited case {case}"
        module, _, name = case.partition("::")
        assert (ROOT / module).is_file(), f"{module} no longer exists"
        assert name in _read(ROOT / module), (
            f"{name} no longer exists in {module}; the Open gaps note is stale"
        )
    assert "/proc/self/fd" in text, "the note omits why these cannot run"
    assert "ffprobe" in text, "the note omits the ffprobe dependency"


def test_the_linux_only_parity_row_names_the_production_code_that_makes_it_so() -> None:
    """The floor must be justified by production code, not asserted.

    A row is only legitimately Linux-only if something in the shipped worker
    actually requires Linux. If that citation stops being true, the row should
    become portable rather than stay excused, so this checks the cited file
    still contains the dependency the note blames.
    """
    text = _read(ARCHITECTURE)
    cited = "worker/pipeline/output/evidence/evidence_media.py"
    assert cited in text, "the note does not say which production code needs /proc"
    assert "/proc/self/fd" in _read(ROOT / cited), (
        f"{cited} no longer reads /proc/self/fd; the Linux-only excuse is stale"
    )


def test_snapshot_store_evidence_is_not_claimed_to_be_platform_limited() -> None:
    """Snapshot store was on that list and must not silently return to it.

    Its tests read ``/proc`` only as instrumentation; the capability never did.
    They now work on macOS too, so listing the row as platform-limited would be
    wrong -- and re-breaking their portability should be caught here rather than
    by someone rediscovering it on a Mac.
    """
    instrumentation = _read(ROOT / "tests" / "test_snapshot_store.py")
    assert "F_GETPATH" in instrumentation, (
        "snapshot-store tests no longer have a macOS descriptor-resolution path"
    )
    assert "/dev/fd" in instrumentation, (
        "snapshot-store tests no longer have a macOS descriptor-enumeration path"
    )
    production = _read(ROOT / "worker" / "pipeline" / "output" / "evidence" / "snapshot_store.py")
    assert "/proc" not in production, (
        "snapshot_store.py now reads /proc, so its row really is Linux-only "
        "and the Open gaps note needs updating"
    )

def test_open_gaps_absence_claims_are_still_true() -> None:
    """A gap that says a file is missing must be checked, not trusted.

    ``Open gaps`` claimed "No ``worker/pipeline/camera_pipeline.py`` ... it does
    not exist" while that file was tracked, 163 lines, and had been on disk since
    ``6ce0bbc``. Nobody noticed because a prose claim about absence is exactly
    the kind of thing that rots silently: the file appears, and the sentence
    saying it has not stays put.

    This scans the section for ``No `path`` / ``no `path`` claims and fails if
    any of those paths now exists. Fixing a gap should require deleting the
    entry, not leaving a stale denial behind.
    """
    text = _read(ARCHITECTURE)
    _, _, gaps = text.partition("## Open gaps")
    assert gaps, "the Open gaps section is missing"

    claimed_absent = re.findall(r"\b[Nn]o `([^`]+\.(?:py|sh|ya?ml|md))`", gaps)
    assert claimed_absent, "no absence claims found; this guard would be vacuous"

    still_present = [path for path in claimed_absent if (ROOT / path).exists()]
    assert still_present == [], (
        f"Open gaps says these do not exist, but they do: {still_present}. "
        "Delete the entry rather than leaving a stale denial."
    )

def test_explicit_fallback_adr_exists_and_is_indexed() -> None:
    adr = ROOT / "docs" / "decisions" / "0003-explicit-fallback-only.md"
    assert adr.is_file()
    body = _read(adr)
    assert "Status: Accepted" in body
    assert "Implicit fallback is" in body
    index = _read(ROOT / "docs" / "decisions" / "README.md")
    assert "0003-explicit-fallback-only.md" in index


def test_architecture_doc_exists_and_is_not_a_placeholder() -> None:
    text = _read(ARCHITECTURE)
    assert len(text.splitlines()) > 50
    assert "Project architecture notes live here" not in text


def test_every_legacy_source_has_exactly_one_documented_owner() -> None:
    sources = _legacy_source_paths()
    if not sources:
        pytest.skip("legacy edge tree is deleted; the ownership table is historical")

    rows = _mapped_source_rows()
    mapped = [source for source, _ in rows]

    duplicates = sorted({path for path in mapped if mapped.count(path) > 1})
    assert duplicates == [], f"duplicate ownership rows: {duplicates}"

    unmapped = sorted(set(sources) - set(mapped))
    assert unmapped == [], f"edge sources with no documented owner: {unmapped}"

    stale = sorted(set(mapped) - set(sources))
    assert stale == [], f"ownership rows for paths that do not exist: {stale}"


def test_every_ownership_row_states_a_target_or_deletion_reason() -> None:
    rows = _mapped_source_rows()
    if not rows:
        pytest.skip("legacy edge tree is deleted; the ownership table is historical")

    empty = sorted(source for source, owner in rows if len(owner.strip()) < 5)
    assert empty == [], f"ownership rows without a target or reason: {empty}"

    for source, owner in rows:
        has_target = "worker/" in owner or "worker." in owner
        has_disposition = "delete" in owner.lower() or "folded into" in owner
        assert has_target or has_disposition, f"{source} has no resolvable owner: {owner}"


def test_canonical_worker_entrypoint_is_documented() -> None:
    assert CANONICAL_ENTRYPOINT in _read(ARCHITECTURE)
    assert CANONICAL_ENTRYPOINT in _read(ROOT / "worker" / "AGENTS.md")


@pytest.mark.parametrize("doc", OWNED_DOCS, ids=lambda path: path.name)
def test_owned_docs_name_no_executable_legacy_entrypoint(doc: Path) -> None:
    assert _executable_legacy_hits(_read(doc)) == []


@pytest.mark.parametrize("doc", OWNED_DOCS, ids=lambda path: path.name)
def test_owned_docs_carry_no_stale_runtime_package_prose(doc: Path) -> None:
    lowered = _read(doc).lower()
    assert [phrase for phrase in _STALE_PROSE if phrase in lowered] == []


def test_doc_contract_detects_a_stale_executable_path() -> None:
    assert _executable_legacy_hits("run `python -m edge.runtime.edge_worker`") != []
    assert _executable_legacy_hits("| `edge/runtime/edge_worker.py` | `worker/x.py` |") == []


def test_architecture_documents_the_five_layers_in_order() -> None:
    text = _read(ARCHITECTURE)
    positions = [
        text.index(marker)
        for marker in ("1. INGEST", "2. FRAME BUS", "3. ANALYTICS", "4. DECISION", "5a. OUTPUT")
    ]
    assert positions == sorted(positions)


def test_architecture_documents_state_scope_and_failure_matrix() -> None:
    text = _read(ARCHITECTURE)
    assert "## Per-camera state vs shared state" in text
    assert "## Failure matrix" in text
    assert "shared, one per task per process" in text
    state_table = text.split("## Per-camera state vs shared state", 1)[1]
    state_table = state_table.split("## Source-to-target ownership", 1)[0]
    for per_camera in ("Tracker", "SceneState", "Fall latch", "IncidentManager"):
        row = next(
            (line for line in state_table.splitlines() if per_camera in line), ""
        )
        assert row.endswith("|"), f"{per_camera} is not a state-table row"
        assert "per camera" in row, f"{per_camera} is not scoped per camera"


def test_architecture_records_raw_frame_fanout_limits() -> None:
    text = _read(ARCHITECTURE)
    assert "only envelope permitted to carry an image" in text
    assert "`DecisionInput` carries exactly the seven" in text


def test_rollback_runbook_matches_central_database_lifecycle_contract() -> None:
    text = _read(ROLLBACK_RUNBOOK)
    commands = "\n".join(re.findall(r"```sh\n(.*?)```", text, flags=re.DOTALL))

    assert "/var/lib/seeon-state/edge.sqlite3" in text
    assert "edge-state" in text
    assert "edge-db-migrator" in text
    assert "EDGE_DB_IMPORT_OK" in text
    assert "central cutover traffic boundary" in text
    assert "Before central cutover traffic" in text
    assert "After central cutover traffic" in text
    assert "image digests" in text
    assert "mutable image tag" in text
    assert "binary-only rollback" in text
    assert "current `edge.sqlite3` schema" in text
    for legacy in (
        "catalog.sqlite3",
        "connection-settings.sqlite3",
        "worker-state.sqlite3",
        "ml-api-state",
        "ml-worker-state",
    ):
        assert legacy in text

    migrator = "$DC up --pull always --no-deps edge-db-migrator"
    api = "$DC up -d --wait ml-api"
    worker = "$DC up -d --wait ml-worker"
    assert migrator in commands
    assert api in commands
    assert worker in commands
    assert commands.index(migrator) < commands.index(api) < commands.index(worker)
    assert "down -v" in text
    assert "delete `edge-state`" in text


@pytest.mark.parametrize(
    "relative",
    [
        "worker/AGENTS.md",
        "worker/types/AGENTS.md",
        "worker/interfaces/AGENTS.md",
        "worker/adapters/AGENTS.md",
        "worker/pipeline/AGENTS.md",
        "worker/domains/AGENTS.md",
        "worker/runtime/AGENTS.md",
    ],
)
def test_scoped_worker_agents_files_state_an_ownership_rule(relative: str) -> None:
    text = _read(ROOT / relative)
    assert "## Ownership rule" in text or "## Layers" in text
    assert "lint-imports" in text


def test_scoped_agents_files_declare_their_import_ceiling() -> None:
    expectations = {
        "worker/types/AGENTS.md": "imports only the standard library and `contracts`",
        "worker/interfaces/AGENTS.md": (
            "imports only the standard library, `contracts`, and `worker.types`"
        ),
        "worker/adapters/AGENTS.md": "must not import `worker.pipeline`",
        "worker/pipeline/AGENTS.md": "must not import `worker.runtime`",
        "worker/domains/AGENTS.md": "must not import `worker.pipeline.ingest`",
        "worker/runtime/AGENTS.md": "the only package that may import everything",
    }
    for relative, expected in expectations.items():
        collapsed = " ".join(_read(ROOT / relative).split())
        assert expected in collapsed, relative


def test_worker_docs_scope_internal_vocabulary_against_vendored_contracts() -> None:
    """The worker/contracts split is documented by a NON-vendored owner.

    ``contracts/`` is ADR-0006 vendored byte-for-byte and
    ``tests/test_vendor_drift.py`` snapshots every file beneath it, including
    ``contracts/AGENTS.md``. The clarification therefore lives in
    ``worker/AGENTS.md`` and ``docs/architecture.md``.
    """
    worker_text = " ".join(_read(ROOT / "worker" / "AGENTS.md").split())
    assert "cross-instance L0 data only" in worker_text
    assert "Worker-internal ports and envelopes live under `worker/`" in worker_text
    assert "never duplicate or shadow a vendored type" in worker_text
    assert "including `contracts/AGENTS.md`" in worker_text

    architecture = " ".join(_read(ARCHITECTURE).split())
    assert "Worker-internal ports and envelopes live under `worker/`" in architecture
    assert "cross-instance L0 data stays in `contracts`" in architecture


def test_vendored_contracts_docs_are_untouched_by_this_migration() -> None:
    """Guard the mistake this todo actually hit: do not annotate vendored docs."""
    text = _read(ROOT / "contracts" / "AGENTS.md")
    assert "cross-instance L0 data only" not in text
    assert "worker/interfaces" not in text


@pytest.mark.parametrize(
    "relative",
    [
        "docs/architecture.md",
        "docs/runbooks/worker-migration-rollback.md",
        "worker/AGENTS.md",
        "worker/types/AGENTS.md",
        "worker/interfaces/AGENTS.md",
        "worker/adapters/AGENTS.md",
        "worker/pipeline/AGENTS.md",
        "worker/domains/AGENTS.md",
        "worker/runtime/AGENTS.md",
    ],
)
def test_new_docs_avoid_privacy_protected_paths(relative: str) -> None:
    path = Path(relative)
    assert {part.lower() for part in path.parts} & _PROHIBITED_PATH_PARTS == set()
    assert {suffix.lower() for suffix in path.suffixes} & _PROHIBITED_SUFFIXES == set()
    assert (ROOT / path).is_file()

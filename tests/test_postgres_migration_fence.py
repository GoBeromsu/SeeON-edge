"""Fence the old SQLite file by its bytes, and put those bytes back only on an ALLOW.

Every oracle here reads the file itself: the 100-byte header (user_version is the
big-endian word at offset 60) and SHA-256 digests. What an old runtime would see is
read by the standard library from copies of the main file and its WAL, so no
SQLite connection touches the fenced file after the fence closes its own.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.app.edge_db.migration import sqlite_fence
from backend.app.edge_db.migration.cli import main
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.rollback import ALLOW, DENY, REPORT_FORMAT
from backend.app.edge_db.migration.snapshot import (
    copy_database,
    create_private_file,
    export_snapshot,
    sidecar_paths,
)
from backend.app.edge_db.migration.sqlite_fence import (
    FENCE_RECEIPT_FORMAT,
    FenceReceipt,
    fence_sqlite,
    inspect_fence,
    preserved_path,
    read_fence_receipt,
)
from backend.app.edge_db.migration.unfence import RESTORED, UNFENCE_REPORT_FORMAT, unfence_sqlite
from tests_support.postgres_migration import (
    add_incident,
    open_source_writer,
    source_and_destination,
)
from tests_support.sqlite_source import hold_runtime_lock

GENERATION = 2
SENTINEL = 1_000_002
SCHEMA_19 = 19
REPO = Path(__file__).resolve().parents[1]
MAGIC = b"SQLite format 3\x00"
# A WAL commit that writes page 1 also rewrites the change counter and its
# version-valid-for copy; the stamp itself is the user_version word.
ALLOWED_HEADER_CHANGES = frozenset({*range(24, 28), *range(60, 64), *range(92, 96)})

# Holds a read transaction open until its stdin closes.
_READER = """
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute("BEGIN")
connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
print("ready", flush=True)
sys.stdin.read()
connection.close()
"""

# Tries to commit without waiting; exits 3 when SQLite refuses the write lock.
_WRITER = """
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0)
try:
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("PRAGMA user_version = 7")
    connection.execute("COMMIT")
except sqlite3.OperationalError:
    sys.exit(3)
finally:
    connection.close()
"""

# Runs the fence and dies at the live stamp: before it, or after its COMMIT and
# before the checkpoint folds it into the main file.
_CRASH = """
import os, sys
from pathlib import Path
from backend.app.edge_db.migration import sqlite_fence

source, snapshot, receipt = (Path(argument) for argument in sys.argv[1:4])
moment = sys.argv[4]
stamp = sqlite_fence._stamp

def crash(connection, user_version):
    if connection.execute("PRAGMA database_list").fetchone()[2] != str(source.resolve()):
        stamp(connection, user_version)
        return
    if moment == "after":
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(f"PRAGMA user_version = {user_version:d}")
        connection.execute("COMMIT")
    os._exit(9)

sqlite_fence._stamp = crash
sqlite_fence.fence_sqlite(source, snapshot=snapshot, generation=2, receipt=receipt)
"""


@dataclass(frozen=True)
class _Layout:
    source: Path
    snapshot: Path
    receipt: Path


def _layout(root: Path) -> _Layout:
    source, destination = source_and_destination(root)
    receipts = root / "receipts"
    receipts.mkdir(mode=0o700)
    snapshot = export_snapshot(source, destination).path
    return _Layout(source, snapshot, receipts / "fence.json")


def _absent_layout(root: Path) -> tuple[Path, Path]:
    state = root / "state"
    state.mkdir(mode=0o700)
    receipts = root / "receipts"
    receipts.mkdir(mode=0o700)
    return state / "edge.sqlite3", receipts / "fence.json"


def _fence(layout: _Layout, generation: int = GENERATION) -> FenceReceipt:
    return fence_sqlite(
        layout.source, snapshot=layout.snapshot, generation=generation, receipt=layout.receipt
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_or_none(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _header(path: Path) -> bytes:
    with path.open("rb") as handle:
        header = handle.read(100)
    assert header[:16] == MAGIC
    return header


def _user_version(path: Path) -> int:
    return int.from_bytes(_header(path)[60:64], "big")


def _size(path: Path) -> int:
    try:
        return path.lstat().st_size
    except FileNotFoundError:
        return 0


def _side_content(source: Path) -> tuple[int, int, bool]:
    wal, shm, journal = sidecar_paths(source)
    return _size(wal), _size(shm), journal.exists() or journal.is_symlink()


def _temps(*directories: Path) -> list[str]:
    names = (path.name for directory in directories for path in directory.iterdir())
    return sorted(name for name in names if ".tmp" in name)


def _seen_user_version(scratch: Path, source: Path) -> int:
    """What an old runtime opening the file now would read: the main file plus its WAL."""
    scratch.mkdir()
    copy = scratch / source.name
    shutil.copyfile(source, copy)
    wal = sidecar_paths(source)[0]
    if wal.exists():
        shutil.copyfile(wal, sidecar_paths(copy)[0])
    with closing(sqlite3.connect(copy)) as connection:
        return connection.execute("PRAGMA user_version").fetchone()[0]


def _allow_payload(fence: FenceReceipt, section: dict[str, object]) -> dict[str, object]:
    """An ALLOW shaped like rollback-check's report for the fenced section it inspected."""
    return {
        "format": REPORT_FORMAT,
        "result": ALLOW,
        "reasons": [],
        "snapshot": {"sha256": fence.snapshot_sha256},
        "sqlite": section,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _allow_report(path: Path, fence: FenceReceipt, source: Path) -> Path:
    section, reasons = inspect_fence(source, fence)
    assert reasons == []
    return _write_json(path, _allow_payload(fence, section))


def _unchanged(source: Path) -> None:
    del source


def _flip_last_byte(source: Path) -> None:
    data = bytearray(source.read_bytes())
    data[-1] ^= 0xFF
    source.write_bytes(bytes(data))


def _restamp_schema_19(source: Path) -> None:
    with source.open("r+b") as handle:
        handle.seek(60)
        handle.write(SCHEMA_19.to_bytes(4, "big"))


def _write_wal(source: Path) -> None:
    sidecar_paths(source)[0].write_bytes(b"\x00" * 32)


def _write_shm(source: Path) -> None:
    sidecar_paths(source)[1].write_bytes(b"\x00" * 32)


def _create_journal(source: Path) -> None:
    sidecar_paths(source)[2].touch()


def _remove(source: Path) -> None:
    source.unlink()


def _replace_with_symlink(source: Path) -> None:
    """The same bytes behind a link, so only the file type can refuse it."""
    target = source.with_name("elsewhere.sqlite3")
    source.rename(target)
    source.symlink_to(target)


def _remove_database(source: Path) -> None:
    for path in (source, *sidecar_paths(source)):
        path.unlink(missing_ok=True)


def test_fence_stamps_the_sentinel_and_records_the_bytes_it_left(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    before = layout.source.read_bytes()
    assert _user_version(layout.source) == SCHEMA_19

    fence = _fence(layout)

    after = layout.source.read_bytes()
    assert _user_version(layout.source) == SENTINEL
    assert len(after) == len(before)
    changed = {
        offset for offset, (old, new) in enumerate(zip(before, after, strict=True)) if old != new
    }
    assert changed <= ALLOWED_HEADER_CHANGES
    preserved = preserved_path(layout.receipt)
    assert json.loads(layout.receipt.read_text(encoding="utf-8")) == {
        "format": FENCE_RECEIPT_FORMAT,
        "generation": GENERATION,
        "user_version": SENTINEL,
        "source_present": True,
        "snapshot_sha256": _sha(layout.snapshot),
        "pre_fence_sha256": hashlib.sha256(before).hexdigest(),
        "fenced_sha256": hashlib.sha256(after).hexdigest(),
    }
    assert read_fence_receipt(layout.receipt) == fence
    assert preserved.read_bytes() == before
    assert _user_version(preserved) == SCHEMA_19
    assert stat.S_IMODE(layout.receipt.stat().st_mode) == 0o600
    assert stat.S_IMODE(preserved.stat().st_mode) == 0o600
    assert _side_content(layout.source) == (0, 0, False)
    assert _temps(layout.source.parent, layout.receipt.parent, layout.snapshot.parent) == []
    assert _seen_user_version(tmp_path / "seen", layout.source) == SENTINEL


def test_fence_refuses_a_source_written_after_its_snapshot(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    with closing(open_source_writer(layout.source)) as writer:
        add_incident(writer, 900)
    written = _sha(layout.source)

    with pytest.raises(MigrationError, match="^source changed after the snapshot$"):
        _fence(layout)

    assert _sha(layout.source) == written
    assert _user_version(layout.source) == SCHEMA_19
    assert not layout.receipt.exists()
    assert not preserved_path(layout.receipt).exists()
    assert _temps(layout.source.parent, layout.receipt.parent, layout.snapshot.parent) == []


def test_fence_refuses_while_another_connection_is_open(tmp_path: Path) -> None:
    """A connection that skipped the deployment lock still keeps the fence out."""
    layout = _layout(tmp_path)
    before = _sha(layout.source)
    reader = subprocess.Popen(
        [sys.executable, "-c", _READER, str(layout.source)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert reader.stdout is not None
        assert reader.stdout.readline() == "ready\n"
        with pytest.raises(MigrationError, match="^source database is in use$"):
            _fence(layout)
    finally:
        assert reader.stdin is not None
        reader.stdin.close()
        assert reader.wait(timeout=60) == 0

    assert _sha(layout.source) == before
    assert not layout.receipt.exists()
    assert not preserved_path(layout.receipt).exists()
    assert _fence(layout) == read_fence_receipt(layout.receipt)


def test_fence_holds_the_file_from_its_check_to_its_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing commits between the snapshot check and the stamp.

    The fence's connection takes the write lock with locking_mode EXCLUSIVE, which
    it keeps across transactions until close, and it reads the live bytes through
    a descriptor it holds, so no close drops that lock early. A writer arriving
    after the receipt digest was computed is refused, and the stamp lands.
    """
    layout = _layout(tmp_path)
    stamped = sqlite_fence._stamped_sha256
    attempts: list[int] = []

    def stamped_then_write(preserved: Path, user_version: int) -> str:
        digest = stamped(preserved, user_version)
        writer = subprocess.run(
            [sys.executable, "-c", _WRITER, str(layout.source)], check=False, timeout=60
        )
        attempts.append(writer.returncode)
        return digest

    monkeypatch.setattr(sqlite_fence, "_stamped_sha256", stamped_then_write)

    fence = _fence(layout)

    assert attempts == [3]
    assert _user_version(layout.source) == SENTINEL
    assert _sha(layout.source) == fence.fenced_sha256
    assert _side_content(layout.source) == (0, 0, False)


def test_fence_refuses_while_the_old_runtime_holds_its_lock(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    before = _sha(layout.source)

    with (
        hold_runtime_lock(layout.source),
        pytest.raises(MigrationError, match="^source database is in use by a running runtime$"),
    ):
        _fence(layout)

    assert _sha(layout.source) == before
    assert not layout.receipt.exists()


INSPECT_CASES = [
    pytest.param(_unchanged, [], SENTINEL, (0, 0, False), id="clean"),
    pytest.param(_flip_last_byte, ["sqlite:live_changed"], SENTINEL, (0, 0, False), id="bytes"),
    pytest.param(
        _restamp_schema_19, ["sqlite:live_changed"], SCHEMA_19, (0, 0, False), id="user_version"
    ),
    pytest.param(_write_wal, ["sqlite:wal_content"], SENTINEL, (32, 0, False), id="wal"),
    pytest.param(_write_shm, ["sqlite:shm_content"], SENTINEL, (0, 32, False), id="shm"),
    pytest.param(_create_journal, ["sqlite:journal"], SENTINEL, (0, 0, True), id="journal"),
    pytest.param(_remove, ["sqlite:source_missing"], None, (0, 0, False), id="missing"),
    pytest.param(
        _replace_with_symlink, ["sqlite:source_not_regular"], None, (0, 0, False), id="symlink"
    ),
]


@pytest.mark.parametrize(("change", "reasons", "user_version", "sides"), INSPECT_CASES)
def test_inspect_fence_compares_the_live_bytes_with_the_receipt(
    tmp_path: Path,
    change: Callable[[Path], None],
    reasons: list[str],
    user_version: int | None,
    sides: tuple[int, int, bool],
) -> None:
    layout = _layout(tmp_path)
    fence = _fence(layout)
    change(layout.source)

    section, found = inspect_fence(layout.source, fence)

    assert found == reasons
    assert section == {
        "generation": GENERATION,
        "source_present": True,
        "fenced_sha256": fence.fenced_sha256,
        "live_sha256": None if user_version is None else _sha(layout.source),
        "user_version": user_version,
        "wal_bytes": sides[0],
        "shm_bytes": sides[1],
        "journal": sides[2],
    }


def test_unfence_restores_the_pre_fence_bytes(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.source.chmod(0o640)
    before = layout.source.read_bytes()
    fence = _fence(layout)
    report = _allow_report(tmp_path / "rollback.json", fence, layout.source)

    restored = unfence_sqlite(layout.source, receipt=layout.receipt, rollback_report=report)

    assert restored == {
        "format": UNFENCE_REPORT_FORMAT,
        "result": RESTORED,
        "generation": GENERATION,
        "snapshot_sha256": fence.snapshot_sha256,
        "fenced_sha256": fence.fenced_sha256,
        "restored_sha256": hashlib.sha256(before).hexdigest(),
    }
    assert layout.source.read_bytes() == before
    assert stat.S_IMODE(layout.source.stat().st_mode) == 0o640
    assert _side_content(layout.source) == (0, 0, False)
    assert preserved_path(layout.receipt).read_bytes() == before
    assert _sha(layout.snapshot) == fence.snapshot_sha256
    assert _temps(layout.source.parent, layout.receipt.parent) == []
    assert _seen_user_version(tmp_path / "seen", layout.source) == SCHEMA_19
    # A rerun after the replace landed finds the preserved bytes already live.
    assert unfence_sqlite(layout.source, receipt=layout.receipt, rollback_report=report) == restored
    assert layout.source.read_bytes() == before


REPORT_GUARDS = [
    pytest.param(
        lambda report: report.update(result=DENY), "^rollback check did not ALLOW$", id="deny"
    ),
    pytest.param(
        lambda report: report.update(reasons=["authority_not_fenced"]),
        "^rollback check did not ALLOW$",
        id="reasons",
    ),
    pytest.param(
        lambda report: report.update(sqlite=None),
        "^rollback report did not check the fenced source$",
        id="unchecked",
    ),
    pytest.param(
        lambda report: report.update(snapshot={"sha256": "0" * 64}),
        "^rollback report checked a different snapshot$",
        id="snapshot",
    ),
    pytest.param(
        lambda report: report["sqlite"].update(generation=GENERATION + 1),
        "^rollback report checked a different fence$",
        id="generation",
    ),
    pytest.param(
        lambda report: report["sqlite"].update(fenced_sha256="0" * 64),
        "^rollback report checked a different fence$",
        id="fenced",
    ),
    pytest.param(
        lambda report: report.update(format="seeon-edge-pg-rollback/0"),
        "^rollback report is malformed$",
        id="format",
    ),
]


@pytest.mark.parametrize(("edit", "message"), REPORT_GUARDS)
def test_unfence_needs_an_allow_for_this_fence(
    tmp_path: Path, edit: Callable[[dict[str, object]], None], message: str
) -> None:
    layout = _layout(tmp_path)
    fence = _fence(layout)
    section, reasons = inspect_fence(layout.source, fence)
    assert reasons == []
    payload = _allow_payload(fence, section)
    edit(payload)
    report = _write_json(tmp_path / "rollback.json", payload)

    with pytest.raises(MigrationError, match=message):
        unfence_sqlite(layout.source, receipt=layout.receipt, rollback_report=report)

    assert _sha(layout.source) == fence.fenced_sha256
    assert _user_version(layout.source) == SENTINEL


REPORT_FILES = [
    pytest.param(None, "^rollback report is unreadable$", id="missing"),
    pytest.param(b"\xff\xfe", "^rollback report is unreadable$", id="not-utf8"),
    pytest.param(b"{", "^rollback report is malformed$", id="not-json"),
    pytest.param(b"[]", "^rollback report is malformed$", id="not-object"),
]


@pytest.mark.parametrize(("content", "message"), REPORT_FILES)
def test_unfence_refuses_an_unreadable_rollback_report(
    tmp_path: Path, content: bytes | None, message: str
) -> None:
    layout = _layout(tmp_path)
    fence = _fence(layout)
    report = tmp_path / "rollback.json"
    if content is not None:
        report.write_bytes(content)

    with pytest.raises(MigrationError, match=message):
        unfence_sqlite(layout.source, receipt=layout.receipt, rollback_report=report)

    assert _sha(layout.source) == fence.fenced_sha256


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(_flip_last_byte, "sqlite:live_changed", id="bytes"),
        pytest.param(_write_wal, "sqlite:wal_content", id="wal"),
        pytest.param(_remove, "sqlite:source_missing", id="missing"),
    ],
)
def test_unfence_refuses_a_fenced_file_that_changed(
    tmp_path: Path, change: Callable[[Path], None], reason: str
) -> None:
    layout = _layout(tmp_path)
    fence = _fence(layout)
    report = _allow_report(tmp_path / "rollback.json", fence, layout.source)
    change(layout.source)
    changed = _bytes_or_none(layout.source)

    with pytest.raises(MigrationError, match=f"^fenced source changed: {reason}$"):
        unfence_sqlite(layout.source, receipt=layout.receipt, rollback_report=report)

    assert _bytes_or_none(layout.source) == changed
    assert _sha(preserved_path(layout.receipt)) == fence.pre_fence_sha256


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(_flip_last_byte, id="bytes"),
        pytest.param(_replace_with_symlink, id="symlink"),
        pytest.param(_remove, id="missing"),
    ],
)
def test_unfence_refuses_a_preserved_copy_that_changed(
    tmp_path: Path, tamper: Callable[[Path], None]
) -> None:
    layout = _layout(tmp_path)
    fence = _fence(layout)
    report = _allow_report(tmp_path / "rollback.json", fence, layout.source)
    tamper(preserved_path(layout.receipt))

    with pytest.raises(
        MigrationError, match="^preserved source copy does not match the fence receipt$"
    ):
        unfence_sqlite(layout.source, receipt=layout.receipt, rollback_report=report)

    assert _sha(layout.source) == fence.fenced_sha256
    assert _temps(layout.source.parent) == []


def test_fence_creates_a_stamped_empty_file_for_an_absent_source(tmp_path: Path) -> None:
    source, receipt = _absent_layout(tmp_path)

    fence = fence_sqlite(source, snapshot=None, generation=GENERATION, receipt=receipt)

    header = _header(source)
    assert header[18:20] == b"\x02\x02"
    assert _user_version(source) == SENTINEL
    assert fence == FenceReceipt(
        generation=GENERATION,
        source_present=False,
        snapshot_sha256=None,
        pre_fence_sha256=None,
        fenced_sha256=_sha(source),
    )
    assert json.loads(receipt.read_text(encoding="utf-8")) == {
        "format": FENCE_RECEIPT_FORMAT,
        "generation": GENERATION,
        "user_version": SENTINEL,
        "source_present": False,
        "snapshot_sha256": None,
        "pre_fence_sha256": None,
        "fenced_sha256": _sha(source),
    }
    assert stat.S_IMODE(source.stat().st_mode) == 0o600
    assert not preserved_path(receipt).exists()
    assert _side_content(source) == (0, 0, False)
    assert _temps(source.parent, receipt.parent) == []
    assert _seen_user_version(tmp_path / "seen", source) == SENTINEL
    assert fence_sqlite(source, snapshot=None, generation=GENERATION, receipt=receipt) == fence
    assert _sha(source) == fence.fenced_sha256


def test_unfence_refuses_a_receipt_with_no_source(tmp_path: Path) -> None:
    source, receipt = _absent_layout(tmp_path)
    fence = fence_sqlite(source, snapshot=None, generation=GENERATION, receipt=receipt)
    report = _allow_report(tmp_path / "rollback.json", fence, source)

    with pytest.raises(
        MigrationError,
        match="^fence receipt records no SQLite source; PostgreSQL stays authoritative$",
    ):
        unfence_sqlite(source, receipt=receipt, rollback_report=report)

    assert _sha(source) == fence.fenced_sha256
    assert _user_version(source) == SENTINEL


def _without_snapshot(layout: _Layout) -> dict[str, object]:
    del layout
    return {"snapshot": None}


def _missing_after_snapshot(layout: _Layout) -> dict[str, object]:
    _remove_database(layout.source)
    return {}


def _orphan_side_file(layout: _Layout) -> dict[str, object]:
    _remove_database(layout.source)
    sidecar_paths(layout.source)[0].touch()
    return {"snapshot": None}


def _rollback_journal(layout: _Layout) -> dict[str, object]:
    _create_journal(layout.source)
    return {}


def _delete_journal_mode(layout: _Layout) -> dict[str, object]:
    with closing(sqlite3.connect(layout.source, isolation_level=None)) as connection:
        assert connection.execute("PRAGMA journal_mode = DELETE").fetchone() == ("delete",)
    return {}


def _symlinked_source(layout: _Layout) -> dict[str, object]:
    _replace_with_symlink(layout.source)
    return {}


def _generation_zero(layout: _Layout) -> dict[str, object]:
    del layout
    return {"generation": 0}


def _other_generation(layout: _Layout) -> dict[str, object]:
    _fence(layout)
    return {"generation": GENERATION + 1}


def _other_snapshot(layout: _Layout) -> dict[str, object]:
    _fence(layout)
    other = layout.snapshot.with_name("other.snapshot.sqlite3")
    shutil.copyfile(layout.snapshot, other)
    _flip_last_byte(other)
    return {"snapshot": other}


def _missing_receipt_directory(layout: _Layout) -> dict[str, object]:
    return {"receipt": layout.receipt.parent / "missing" / layout.receipt.name}


FENCE_REFUSALS = [
    pytest.param(
        _without_snapshot, "^a present SQLite source needs its snapshot$", id="no-snapshot"
    ),
    pytest.param(
        _missing_after_snapshot, "^SQLite source is missing after its snapshot$", id="vanished"
    ),
    pytest.param(
        _orphan_side_file, "^SQLite side files exist without their database$", id="orphan-wal"
    ),
    pytest.param(_rollback_journal, "^source database has a rollback journal$", id="journal"),
    pytest.param(
        _delete_journal_mode, "^source database is not in WAL mode$", id="delete-journal-mode"
    ),
    pytest.param(_symlinked_source, "^source database is not a regular file$", id="symlink"),
    pytest.param(_generation_zero, "^fence generation is out of range$", id="generation-zero"),
    pytest.param(
        _other_generation,
        "^fence receipt already records a different fence$",
        id="other-generation",
    ),
    pytest.param(
        _other_snapshot, "^fence receipt already records a different fence$", id="other-snapshot"
    ),
    pytest.param(
        _missing_receipt_directory, "^fence directory does not exist$", id="receipt-directory"
    ),
]


@pytest.mark.parametrize(("arrange", "message"), FENCE_REFUSALS)
def test_fence_refuses_what_it_cannot_prove(
    tmp_path: Path, arrange: Callable[[_Layout], dict[str, object]], message: str
) -> None:
    layout = _layout(tmp_path)
    arguments: dict[str, object] = {
        "snapshot": layout.snapshot,
        "generation": GENERATION,
        "receipt": layout.receipt,
    } | arrange(layout)
    source = _bytes_or_none(layout.source)
    receipt = _bytes_or_none(layout.receipt)
    preserved = _bytes_or_none(preserved_path(layout.receipt))

    with pytest.raises(MigrationError, match=message):
        fence_sqlite(layout.source, **arguments)  # type: ignore[arg-type]

    assert _bytes_or_none(layout.source) == source
    assert _bytes_or_none(layout.receipt) == receipt
    assert _bytes_or_none(preserved_path(layout.receipt)) == preserved
    assert _temps(layout.source.parent, layout.receipt.parent, layout.snapshot.parent) == []


@pytest.mark.parametrize("moment", ["before", "after"])
def test_a_crash_at_the_stamp_leaves_one_authority_until_the_fence_reruns(
    tmp_path: Path, moment: str
) -> None:
    """The receipt is durable before the stamp, so a crash leaves one of two files.

    Before the stamp, the main file is untouched and SQLite stays the only
    authority. After the stamp's COMMIT and before its checkpoint, an old runtime
    would replay the WAL and refuse the sentinel. Either way PostgreSQL has not
    committed generation 2, and unfence refuses the unsettled file. Recovery is
    to rerun the fence, which settles on the receipt, and then either continue
    forward with the generation 2 transfer (PostgreSQL becomes the only authority)
    or abort with rollback-check ALLOW and unfence-sqlite (SQLite is the only
    authority and PostgreSQL stays frozen at generation 1).
    """
    layout = _layout(tmp_path)
    before = _sha(layout.source)

    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            _CRASH,
            str(layout.source),
            str(layout.snapshot),
            str(layout.receipt),
            moment,
        ],
        cwd=REPO,
        check=False,
        timeout=120,
    )

    assert crashed.returncode == 9
    fence = read_fence_receipt(layout.receipt)
    assert fence.pre_fence_sha256 == before
    assert _user_version(layout.source) == SCHEMA_19
    section, reasons = inspect_fence(layout.source, fence)
    if moment == "before":
        assert reasons == ["sqlite:live_changed"]
        assert _sha(layout.source) == before
        assert _side_content(layout.source) == (0, 0, False)
        assert _seen_user_version(tmp_path / "seen", layout.source) == SCHEMA_19
    else:
        assert reasons == ["sqlite:live_changed", "sqlite:wal_content"]
        assert _seen_user_version(tmp_path / "seen", layout.source) == SENTINEL
        stale = _write_json(tmp_path / "stale.json", _allow_payload(fence, section))
        with pytest.raises(
            MigrationError,
            match="^fenced source changed: sqlite:live_changed,sqlite:wal_content$",
        ):
            unfence_sqlite(layout.source, receipt=layout.receipt, rollback_report=stale)

    assert _fence(layout) == fence
    assert _sha(layout.source) == fence.fenced_sha256
    assert _side_content(layout.source) == (0, 0, False)
    assert _seen_user_version(tmp_path / "settled", layout.source) == SENTINEL
    report = _allow_report(tmp_path / "rollback.json", fence, layout.source)
    unfence_sqlite(layout.source, receipt=layout.receipt, rollback_report=report)
    assert _sha(layout.source) == before
    assert _seen_user_version(tmp_path / "restored", layout.source) == SCHEMA_19


def test_a_fence_rerun_removes_the_index_a_read_only_probe_left(tmp_path: Path) -> None:
    """A read-only open cannot delete its -shm on close, so the rerun settles it."""
    layout = _layout(tmp_path)
    fence = _fence(layout)
    probe_uri = layout.source.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(probe_uri, uri=True)) as probe:
        assert probe.execute("PRAGMA user_version").fetchone()[0] == SENTINEL
    assert inspect_fence(layout.source, fence)[1] == ["sqlite:shm_content"]

    assert _fence(layout) == fence

    assert _sha(layout.source) == fence.fenced_sha256
    assert _side_content(layout.source) == (0, 0, False)
    assert inspect_fence(layout.source, fence)[1] == []


def test_cli_unfence_sqlite_reports_the_restore(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _layout(tmp_path)
    before = _sha(layout.source)
    fence = _fence(layout)
    report = _allow_report(tmp_path / "rollback.json", fence, layout.source)
    written = tmp_path / "unfence.json"

    code = main(
        [
            "unfence-sqlite",
            "--source",
            str(layout.source),
            "--fence-receipt",
            str(layout.receipt),
            "--rollback-report",
            str(report),
            "--report",
            str(written),
        ]
    )

    captured = capsys.readouterr()
    assert (code, captured.err) == (0, "")
    assert captured.out == (
        f"EDGE_PG_MIGRATION_UNFENCE_SQLITE_OK result=RESTORED generation=2 sha256={before}\n"
    )
    assert json.loads(written.read_text(encoding="utf-8")) == {
        "format": UNFENCE_REPORT_FORMAT,
        "result": RESTORED,
        "generation": GENERATION,
        "snapshot_sha256": fence.snapshot_sha256,
        "fenced_sha256": fence.fenced_sha256,
        "restored_sha256": before,
    }
    assert _sha(layout.source) == before


def test_cli_unfence_sqlite_fails_without_an_allow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _layout(tmp_path)
    fence = _fence(layout)
    section, _ = inspect_fence(layout.source, fence)
    denied = _allow_payload(fence, section) | {"result": DENY}
    report = _write_json(tmp_path / "rollback.json", denied)

    code = main(
        [
            "unfence-sqlite",
            "--source",
            str(layout.source),
            "--fence-receipt",
            str(layout.receipt),
            "--rollback-report",
            str(report),
        ]
    )

    captured = capsys.readouterr()
    assert (code, captured.out) == (1, "")
    assert captured.err == "EDGE_PG_MIGRATION_UNFENCE_SQLITE_FAILED: rollback check did not ALLOW\n"
    assert _sha(layout.source) == fence.fenced_sha256


def test_export_refuses_a_source_locked_by_its_own_writer(tmp_path: Path) -> None:
    """The backup would retry a busy source forever; the copy refuses it instead."""
    layout = _layout(tmp_path)
    copy = tmp_path / "copy.sqlite3"
    create_private_file(copy)
    with closing(open_source_writer(layout.source)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        add_incident(writer, 901)
        with pytest.raises(MigrationError, match="^source database is locked$"):
            copy_database(writer, copy)
        writer.execute("ROLLBACK")

"""Reconcile after transfer: the gates may be open, and the seed is the only difference allowed.

The fenced check keeps failing an opened target, so the report still proves when
a copy was taken too late. The after-transfer check instead needs the authority
to have moved on, tolerates only the ``edge_site`` row activation seeds into a
target the snapshot gave none, and reads a source the fence stamped by its bytes.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest

from backend.app.edge_db.authority import AuthorityToken
from backend.app.edge_db.migration.cli import main
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.load import import_snapshot
from backend.app.edge_db.migration.mapping import diagnostics_schema_name
from backend.app.edge_db.migration.reconcile import FAIL, PASS, reconcile
from backend.app.edge_db.migration.snapshot import export_snapshot, sidecar_paths
from backend.app.edge_db.migration.sqlite_fence import fence_sqlite, read_fence_receipt
from backend.app.edge_db.migration.transfer import freeze, transfer
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.connection.repository import ConnectionValue
from backend.app.features.connection.store import ConnectionSettingsStore
from tests_support.postgres_migration import (
    MigrationTarget,
    add_incident,
    append_audit,
    open_source_writer,
    runtime_role_database,
    source_and_destination,
)

pytest_plugins = ("tests_support.postgres_migration",)

GENERATION = 2
SENTINEL = 1_000_000 + GENERATION
_NO_SITE = "DELETE FROM edge_site"
_ROUTES = [
    pytest.param(None, ["authority:not_fenced"], False, id="site"),
    pytest.param(_NO_SITE, ["table:edge_site", "authority:not_fenced"], True, id="no-site"),
]
_ENROLLMENT: dict[str, ConnectionValue] = {
    "facility_code": "FAC-AFTER-01",
    "client_installation_ref": "client-after-01",
    "facility_id": "facility-after-01",
    "facility_token": "token-after-01",
    "edge_installation_id": "edge-after-01",
    "enrollment_generation": 1,
}
# Values seeded into the synthetic source that no report or output line may carry.
_SEEDED_VALUES = ("rtsp", "camera.invalid", "operator", bytes(range(64)).hex())


def _export(root: Path, statement: str | None = None) -> tuple[Path, Path]:
    """The seeded old database after one optional raw statement, and its snapshot."""
    root.mkdir()
    source, destination = source_and_destination(root)
    if statement is not None:
        with closing(open_source_writer(source)) as writer:
            writer.execute(statement)
    return source, export_snapshot(source, destination).path


def _fence(source: Path, snapshot: Path) -> Path:
    receipts = source.parent.parent / "receipts"
    receipts.mkdir(mode=0o700)
    receipt = receipts / "fence.json"
    fence_sqlite(source, snapshot=snapshot, generation=GENERATION, receipt=receipt)
    return receipt


def _activate(target: MigrationTarget, snapshot: Path) -> AuthorityToken:
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)
    return transfer(target.database, target.authority_path, schema=target.schema)


def _reconcile(target: MigrationTarget, snapshot: Path, **options: object) -> dict[str, object]:
    return reconcile(target.database, schema=target.schema, snapshot_path=snapshot, **options)


def _verdict(report: dict[str, object]) -> tuple[object, object, object]:
    return report["mode"], report["result"], report["failures"]


def _edge_site_rows(report: dict[str, object]) -> tuple[int, int]:
    (table,) = [table for table in report["tables"] if table["table"] == "edge_site"]
    return table["source"]["rows"], table["target"]["rows"]


@pytest.mark.parametrize(("statement", "fenced_failures", "seeded"), _ROUTES)
def test_a_transferred_target_fails_the_fenced_check_and_passes_the_after_transfer_check(
    migration_target: MigrationTarget,
    tmp_path: Path,
    statement: str | None,
    fenced_failures: list[str],
    seeded: bool,
) -> None:
    target = migration_target
    _, snapshot = _export(tmp_path / "old", statement)
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)
    imported = _reconcile(target, snapshot)
    transfer(target.database, target.authority_path, schema=target.schema)

    fenced = _reconcile(target, snapshot)
    after = _reconcile(target, snapshot, after_transfer=True)

    assert _verdict(imported) == ("before_transfer", PASS, [])
    assert _verdict(fenced) == ("before_transfer", FAIL, fenced_failures)
    assert _verdict(after) == ("after_transfer", PASS, [])
    assert after["authority"] == {"generation": 2, "accepting": True, "egress_enabled": True}
    assert (fenced["activation_seed"], after["activation_seed"]) == (False, seeded)
    assert _edge_site_rows(after) == ((0, 1) if seeded else (1, 1))
    assert after["reconciliation_sha256"] == imported["reconciliation_sha256"]


def test_the_after_transfer_check_fails_a_target_still_at_generation_one(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    _, snapshot = _export(tmp_path / "old", _NO_SITE)
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)

    after = _reconcile(target, snapshot, after_transfer=True)

    assert _verdict(after) == ("after_transfer", FAIL, ["authority:not_transferred"])
    assert after["authority"] == {"generation": 1, "accepting": False, "egress_enabled": False}
    assert after["activation_seed"] is False
    assert _edge_site_rows(after) == (0, 0)


def test_a_target_frozen_after_transfer_passes_the_after_transfer_check(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    _, snapshot = _export(tmp_path / "old")
    _activate(target, snapshot)
    freeze(target.database, target.authority_path)

    after = _reconcile(target, snapshot, after_transfer=True)

    assert _verdict(after) == ("after_transfer", PASS, [])
    assert after["authority"] == {"generation": 2, "accepting": False, "egress_enabled": False}


def test_the_runtime_audit_verifier_accepts_the_transferred_chain(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    _, snapshot = _export(tmp_path / "old")
    with closing(sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)) as old:
        chain = old.execute(
            "SELECT audit_id, record_hash FROM audit_events ORDER BY audit_id"
        ).fetchall()
    token = _activate(target, snapshot)

    # The runtime verifies the whole chain before it serves, so the old tail must survive the copy.
    checkpoint = PostgresAuditStore(target.database, token).verify()

    assert (checkpoint.row_count, checkpoint.audit_id, checkpoint.record_hash) == (
        len(chain),
        *chain[-1],
    )


@pytest.mark.parametrize("statement", [None, _NO_SITE], ids=["site", "no-site"])
def test_a_site_the_runtime_wrote_after_activation_is_not_taken_for_the_seed(
    migration_target: MigrationTarget, tmp_path: Path, statement: str | None
) -> None:
    target = migration_target
    _, snapshot = _export(tmp_path / "old", statement)
    token = _activate(target, snapshot)
    with runtime_role_database(target.dsn, target.schema, target.runtime_role) as database:
        ConnectionSettingsStore(database, token).save(_ENROLLMENT)

    after = _reconcile(target, snapshot, after_transfer=True)

    assert _verdict(after) == ("after_transfer", FAIL, ["table:edge_site"])
    assert after["activation_seed"] is False


def test_a_source_written_after_the_snapshot_fails_the_after_transfer_check(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    source, snapshot = _export(tmp_path / "old")
    with closing(open_source_writer(source)) as writer:
        add_incident(writer, 2)
        append_audit(writer)
    _activate(target, snapshot)

    after = _reconcile(target, snapshot, after_transfer=True, source_path=source)

    assert _verdict(after) == (
        "after_transfer",
        FAIL,
        ["live_source:incidents", "live_source:audit_events"],
    )
    assert after["boundary"]["live_source"]["result"] == FAIL


def test_a_fenced_source_matches_its_receipt_before_and_after_transfer(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    source, snapshot = _export(tmp_path / "old")
    receipt = _fence(source, snapshot)
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)
    options = {"source_path": source, "fence_receipt": receipt}

    fenced = _reconcile(target, snapshot, **options)
    transfer(target.database, target.authority_path, schema=target.schema)
    after = _reconcile(target, snapshot, after_transfer=True, **options)

    fenced_sha256 = read_fence_receipt(receipt).fenced_sha256
    assert _verdict(fenced) == ("before_transfer", PASS, [])
    assert _verdict(after) == ("after_transfer", PASS, [])
    for report in (fenced, after):
        live_source = report["boundary"]["live_source"]
        assert (live_source["result"], live_source["reasons"]) == (PASS, [])
        fence = live_source["fence"]
        assert (fence["generation"], fence["user_version"]) == (GENERATION, SENTINEL)
        assert fence["live_sha256"] == fence["fenced_sha256"] == fenced_sha256


def _flip_last_byte(source: Path) -> None:
    data = bytearray(source.read_bytes())
    data[-1] ^= 0xFF
    source.write_bytes(bytes(data))


def _write_wal(source: Path) -> None:
    sidecar_paths(source)[0].write_bytes(b"\x00" * 32)


@pytest.mark.parametrize(
    ("drift", "reasons"),
    [
        pytest.param(_flip_last_byte, ["sqlite:live_changed"], id="bytes"),
        pytest.param(_write_wal, ["sqlite:wal_content"], id="wal"),
    ],
)
def test_a_fenced_source_that_drifted_from_its_receipt_fails_the_after_transfer_check(
    migration_target: MigrationTarget,
    tmp_path: Path,
    drift: Callable[[Path], None],
    reasons: list[str],
) -> None:
    target = migration_target
    source, snapshot = _export(tmp_path / "old")
    receipt = _fence(source, snapshot)
    _activate(target, snapshot)
    drift(source)

    after = _reconcile(
        target, snapshot, after_transfer=True, source_path=source, fence_receipt=receipt
    )

    assert _verdict(after) == ("after_transfer", FAIL, reasons)
    live_source = after["boundary"]["live_source"]
    assert (live_source["result"], live_source["reasons"]) == (FAIL, reasons)


def test_a_receipt_that_fenced_another_snapshot_fails_the_after_transfer_check(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    _, snapshot = _export(tmp_path / "old")
    other_source, other_snapshot = _export(tmp_path / "other", _NO_SITE)
    other_receipt = _fence(other_source, other_snapshot)
    _activate(target, snapshot)

    after = _reconcile(
        target,
        snapshot,
        after_transfer=True,
        source_path=other_source,
        fence_receipt=other_receipt,
    )

    assert _verdict(after) == ("after_transfer", FAIL, ["sqlite:snapshot_mismatch"])
    assert after["boundary"]["live_source"]["fence"]["live_sha256"] == (
        read_fence_receipt(other_receipt).fenced_sha256
    )


def test_a_fence_receipt_without_its_source_is_refused(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    source, snapshot = _export(tmp_path / "old")
    receipt = _fence(source, snapshot)

    with pytest.raises(MigrationError, match="^a fence receipt needs the source it fenced$"):
        _reconcile(migration_target, snapshot, after_transfer=True, fence_receipt=receipt)


def _owner_args(root: Path, target: MigrationTarget) -> list[str]:
    directory = root / "owner"
    directory.mkdir(mode=0o700)
    path = directory / "owner.dsn"
    path.write_text(f"{target.dsn}\n", encoding="utf-8")
    path.chmod(0o600)
    return ["--owner-dsn-file", str(path), "--schema", target.schema]


def test_cli_reconciles_a_transferred_target_against_the_fenced_source(
    migration_target: MigrationTarget, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = migration_target
    source, snapshot = _export(tmp_path / "old")
    receipt = _fence(source, snapshot)
    _activate(target, snapshot)
    reports = tmp_path / "reports"
    reports.mkdir()
    after_report, fenced_report = reports / "after.json", reports / "fenced.json"
    checked = [
        "reconcile",
        *_owner_args(tmp_path, target),
        "--snapshot",
        str(snapshot),
        "--source",
        str(source),
        "--fence-receipt",
        str(receipt),
        "--report",
    ]

    after_code = main([*checked, str(after_report), "--after-transfer"])
    after_output = capsys.readouterr()
    fenced_code = main([*checked, str(fenced_report)])
    fenced_output = capsys.readouterr()

    assert (after_code, after_output.out, after_output.err) == (
        0,
        (
            f"EDGE_PG_MIGRATION_RECONCILE_OK result=PASS report={after_report} "
            f"diagnostics_schema={diagnostics_schema_name(target.schema)}\n"
        ),
        "",
    )
    assert (fenced_code, fenced_output.out, fenced_output.err) == (
        1,
        "",
        f"EDGE_PG_MIGRATION_RECONCILE_FAILED: result=FAIL report={fenced_report}\n",
    )
    after = json.loads(after_report.read_text(encoding="utf-8"))
    fenced = json.loads(fenced_report.read_text(encoding="utf-8"))
    assert _verdict(after) == ("after_transfer", PASS, [])
    assert _verdict(fenced) == ("before_transfer", FAIL, ["authority:not_fenced"])
    assert after["boundary"]["live_source"]["result"] == PASS
    text = "".join(
        (
            after_output.out,
            after_output.err,
            fenced_output.out,
            fenced_output.err,
            after_report.read_text(encoding="utf-8"),
            fenced_report.read_text(encoding="utf-8"),
        )
    )
    leaked = [index for index, value in enumerate((*_SEEDED_VALUES, target.dsn)) if value in text]
    assert leaked == []


def test_cli_refuses_a_fence_receipt_without_its_source_before_reading_the_dsn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "owner" / "owner.dsn"

    with pytest.raises(SystemExit) as refused:
        main(
            [
                "reconcile",
                "--owner-dsn-file",
                str(missing),
                "--snapshot",
                str(tmp_path / "edge.snapshot.sqlite3"),
                "--report",
                str(tmp_path / "report.json"),
                "--after-transfer",
                "--fence-receipt",
                str(tmp_path / "fence.json"),
            ]
        )

    assert refused.value.code == 2
    assert capsys.readouterr().err.endswith("error: reconcile --fence-receipt requires --source\n")
    assert not (tmp_path / "report.json").exists()

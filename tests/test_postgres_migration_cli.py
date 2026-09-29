"""The cutover command line on real PG18: exit codes, one-line output and no leaked secrets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from psycopg.conninfo import make_conninfo

from backend.app.edge_db.migration.cli import main
from backend.app.edge_db.migration.sqlite_fence import fence_sqlite
from backend.app.edge_db.migration.transfer import pending_authority_path
from tests_support.postgres_migration import (
    MigrationNames,
    authority_file_token,
    authority_row,
    make_worker_state,
    runtime_verifier,
    scram_verifier_accepts,
    source_and_destination,
)

pytest_plugins = ("tests_support.postgres_migration",)

_PASSWORD = "synthetic-runtime-password-1"
# Stands in for the per-test runtime role, which parametrize cannot see.
_RUNTIME = "<runtime role>"
_NOT_OWNER_ONLY = "must be a regular file readable only by its owner"
_NO_PASSWORD = "must carry a printable ASCII password"

# Values seeded into the synthetic source that no report or output line may carry.
_SEEDED_VALUES = ("rtsp", "camera.invalid", "operator", bytes(range(64)).hex())


def _owner_dsn_file(root: Path, dsn: str, mode: int) -> Path:
    directory = root / "owner"
    directory.mkdir(mode=0o700)
    path = directory / "owner.dsn"
    path.write_text(f"{dsn}\n", encoding="utf-8")
    path.chmod(mode)
    return path


def _runtime_dsn_file(root: Path, dsn: str, mode: int) -> Path:
    directory = root / "runtime"
    directory.mkdir(mode=0o700)
    path = directory / "runtime.dsn"
    path.write_text(f"{dsn}\n", encoding="utf-8")
    path.chmod(mode)
    return path


def _authority_file(root: Path) -> Path:
    directory = root / "authority"
    directory.mkdir(mode=0o700)
    return directory / "authority.json"


def _owner_args(root: Path, names: MigrationNames) -> list[str]:
    return [
        "--owner-dsn-file",
        str(_owner_dsn_file(root, names.dsn, 0o600)),
        "--schema",
        names.schema,
        "--statement-timeout-ms",
        "5000",
        "--lock-timeout-ms",
        "2000",
    ]


def test_cli_runs_the_cutover_and_reports_failure_by_exit_code(
    migration_names: MigrationNames, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    names = migration_names
    owner = _owner_args(tmp_path, names)
    authority = str(_authority_file(tmp_path))
    source, snapshot = source_and_destination(tmp_path)
    state = str(make_worker_state(tmp_path))
    reports = tmp_path / "reports"
    reports.mkdir()
    bound, unfenced = reports / "reconcile.json", reports / "reconcile-after-transfer.json"
    receipts = tmp_path / "receipts"
    receipts.mkdir(mode=0o700)
    receipt = str(receipts / "fence.json")
    output: list[str] = []

    def run(*argv: str) -> tuple[int, str, str]:
        code = main(list(argv))
        captured = capsys.readouterr()
        output.extend((captured.out, captured.err))
        return code, captured.out, captured.err

    provisioned = run(
        "provision",
        *owner,
        "--runtime-role",
        names.runtime_role,
        "--authority-file",
        authority,
    )
    exported = run("export", "--source", str(source), "--snapshot", str(snapshot))
    # The operator stamps the stopped old database at the provisioned generation.
    generation, _ = authority_file_token(Path(authority))
    fence_sqlite(source, snapshot=snapshot, generation=generation, receipt=Path(receipt))
    digested = run("queue-digest", "--worker-state-dir", state)
    queue_sha256 = digested[1].rsplit("sha256=", 1)[1].strip()
    imported = run("import", *owner, "--snapshot", str(snapshot))
    reconciled = run(
        "reconcile",
        *owner,
        "--snapshot",
        str(snapshot),
        "--report",
        str(bound),
        "--source",
        str(source),
        "--fence-receipt",
        receipt,
        "--worker-state-dir",
        state,
        "--expect-delivery-queue-sha256",
        queue_sha256,
    )
    transferred = run(
        "transfer", *owner, "--authority-file", authority, "--worker-state-dir", state
    )
    stale = run("reconcile", *owner, "--snapshot", str(snapshot), "--report", str(unfenced))
    rollback = ("--snapshot", str(snapshot), "--source", str(source), "--fence-receipt", receipt)
    live = run("rollback-check", *owner, *rollback)
    frozen = run("freeze", *owner, "--authority-file", authority)
    fenced = run("rollback-check", *owner, *rollback)

    report_text = bound.read_text(encoding="utf-8") + unfenced.read_text(encoding="utf-8")
    text = "".join(output) + report_text
    leaked = [index for index, value in enumerate((*_SEEDED_VALUES, names.dsn)) if value in text]
    assert leaked == []
    assert [step[0] for step in (provisioned, exported, digested, imported)] == [0, 0, 0, 0]
    assert provisioned[1:] == (
        (
            f"EDGE_PG_MIGRATION_PROVISION_OK schema={names.schema} schema_created=true "
            f"diagnostics_schema={names.schema}_diagnostics diagnostics_schema_created=true "
            "role_created=true authority_created=true generation=1 runtime_password_set=false\n"
        ),
        "",
    )
    assert exported[1].startswith(f"EDGE_PG_MIGRATION_EXPORT_OK snapshot={snapshot} sha256=")
    assert digested[1].startswith(
        "EDGE_PG_MIGRATION_QUEUE_DIGEST_OK queued=2 temporary=1 dead_lettered=1 sha256="
    )
    assert imported[1].startswith("EDGE_PG_MIGRATION_IMPORT_OK tables=")
    assert reconciled == (
        0,
        (
            f"EDGE_PG_MIGRATION_RECONCILE_OK result=PASS report={bound} "
            f"diagnostics_schema={names.schema}_diagnostics\n"
        ),
        "",
    )
    bound_report = json.loads(bound.read_text(encoding="utf-8"))
    assert bound_report["delivery_queue"]["result"] == "PASS"
    assert bound_report["diagnostics"] == {
        "schema": f"{names.schema}_diagnostics",
        "mode": "live",
        "reconciled": False,
        "ledger": "PASS",
        "tables": "PASS",
        "rows": {
            "execution_batches": 0,
            "execution_coverage": 0,
            "execution_provenance": 0,
            "execution_records": 0,
            "execution_segments": 0,
            "execution_units": 0,
        },
        "result": "PASS",
    }
    assert transferred == (
        0,
        "EDGE_PG_MIGRATION_TRANSFER_OK generation=2 accepting=true egress_enabled=true\n",
        "",
    )
    assert stale == (
        1,
        "",
        f"EDGE_PG_MIGRATION_RECONCILE_FAILED: result=FAIL report={unfenced}\n",
    )
    assert json.loads(unfenced.read_text(encoding="utf-8"))["failures"] == ["authority:not_fenced"]
    assert live == (
        1,
        "",
        "EDGE_PG_MIGRATION_ROLLBACK_CHECK_FAILED: result=DENY reasons=authority_not_fenced\n",
    )
    assert frozen == (
        0,
        "EDGE_PG_MIGRATION_FREEZE_OK generation=2 accepting=false egress_enabled=false\n",
        "",
    )
    assert fenced == (0, "EDGE_PG_MIGRATION_ROLLBACK_CHECK_OK result=ALLOW reasons=none\n", "")


def test_cli_refuses_a_readable_dsn_file_without_printing_it(
    migration_names: MigrationNames, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    names = migration_names
    authority = _authority_file(tmp_path)
    owner_dsn = _owner_dsn_file(tmp_path, names.dsn, 0o644)

    code = main(
        [
            "provision",
            "--owner-dsn-file",
            str(owner_dsn),
            "--schema",
            names.schema,
            "--runtime-role",
            names.runtime_role,
            "--authority-file",
            str(authority),
        ]
    )

    captured = capsys.readouterr()
    leaked = names.dsn in captured.out + captured.err
    (schema_exists,) = names.admin.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = %s)",
        (names.schema,),
    ).fetchone()
    assert not leaked
    assert (code, captured.out, captured.err) == (
        1,
        "",
        (
            f"EDGE_PG_MIGRATION_PROVISION_FAILED: owner DSN file {owner_dsn} must be a "
            "regular file readable only by its owner\n"
        ),
    )
    assert not schema_exists
    assert not authority.exists()


def _provision_with_runtime_dsn(
    names: MigrationNames, root: Path, runtime_dsn: Path, authority: Path
) -> list[str]:
    return [
        "provision",
        "--owner-dsn-file",
        str(_owner_dsn_file(root, names.dsn, 0o600)),
        "--schema",
        names.schema,
        "--runtime-role",
        names.runtime_role,
        "--runtime-dsn-file",
        str(runtime_dsn),
        "--authority-file",
        str(authority),
    ]


def test_cli_provision_sets_the_runtime_password_from_its_dsn_file(
    migration_names: MigrationNames, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    names = migration_names
    password = "synthetic-runtime-password-1"
    runtime_dsn = _runtime_dsn_file(
        tmp_path, make_conninfo(names.dsn, user=names.runtime_role, password=password), 0o600
    )
    argv = _provision_with_runtime_dsn(names, tmp_path, runtime_dsn, _authority_file(tmp_path))

    first = main(argv)
    first_out = capsys.readouterr()
    second = main(argv)
    second_out = capsys.readouterr()

    verifier = runtime_verifier(names.admin, names.runtime_role)
    text = first_out.out + first_out.err + second_out.out + second_out.err
    assert password not in text
    assert (first, first_out.err, second, second_out.err) == (0, "", 0, "")
    assert first_out.out.endswith(" generation=1 runtime_password_set=true\n")
    assert second_out.out.endswith(" generation=1 runtime_password_set=false\n")
    assert verifier is not None
    assert scram_verifier_accepts(verifier, password)


@pytest.mark.parametrize(
    ("mode", "fields", "message"),
    [
        (0o600, {"password": _PASSWORD}, "user must be the runtime role"),
        (0o600, {"user": _RUNTIME, "password": ""}, _NO_PASSWORD),
        (0o600, {"user": _RUNTIME}, _NO_PASSWORD),
        (0o600, {"user": _RUNTIME, "password": "synthetic-pässword"}, _NO_PASSWORD),
        (0o600, {"user": _RUNTIME, "password": "synthetic\tpassword"}, _NO_PASSWORD),
        (0o644, {"user": _RUNTIME, "password": _PASSWORD}, _NOT_OWNER_ONLY),
        (0o640, {"user": _RUNTIME, "password": _PASSWORD}, _NOT_OWNER_ONLY),
        (None, {}, "cannot be opened (No such file or directory)"),
    ],
    ids=[
        "another-role",
        "empty-password",
        "no-password",
        "non-ascii-password",
        "control-character-password",
        "world-readable",
        "group-readable",
        "missing-file",
    ],
)
def test_cli_refuses_a_runtime_dsn_file_before_touching_the_target(
    migration_names: MigrationNames,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: int | None,
    fields: dict[str, str],
    message: str,
) -> None:
    names = migration_names
    values = {
        key: names.runtime_role if value == _RUNTIME else value for key, value in fields.items()
    }
    password = values.get("password") or _PASSWORD
    if mode is None:
        runtime_dsn = tmp_path / "runtime" / "runtime.dsn"
    else:
        runtime_dsn = _runtime_dsn_file(tmp_path, make_conninfo(names.dsn, **values), mode)
    authority = _authority_file(tmp_path)

    code = main(_provision_with_runtime_dsn(names, tmp_path, runtime_dsn, authority))

    captured = capsys.readouterr()
    (schema_exists,) = names.admin.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = %s)",
        (names.schema,),
    ).fetchone()
    assert password not in captured.out + captured.err
    assert names.dsn not in captured.out + captured.err
    assert (code, captured.out, captured.err) == (
        1,
        "",
        f"EDGE_PG_MIGRATION_PROVISION_FAILED: runtime DSN file {runtime_dsn} {message}\n",
    )
    assert not schema_exists
    assert not authority.exists()


def _fresh_install_paths(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """The product's SQLite path, moved under the test root, and an operator-named one.

    Both directories exist and neither database does.
    """
    canonical = root / "state" / "edge.sqlite3"
    source = root / "fresh-state" / "edge.sqlite3"
    for path in (canonical, source):
        path.parent.mkdir()
    monkeypatch.setattr("backend.app.edge_db.migration.cli.EDGE_DATABASE_PATH", canonical)
    return canonical, source


def _provision_for_transfer(names: MigrationNames, root: Path, owner: list[str]) -> Path:
    authority = _authority_file(root)
    argv = ["provision", *owner, "--runtime-role", names.runtime_role]
    assert main([*argv, "--authority-file", str(authority)]) == 0
    return authority


def test_cli_fresh_install_opens_an_empty_target_from_the_default_source(
    migration_names: MigrationNames,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = migration_names
    canonical, source = _fresh_install_paths(tmp_path, monkeypatch)
    owner = _owner_args(tmp_path, names)
    authority = _provision_for_transfer(names, tmp_path, owner)
    provisioned = authority_row(names.admin, names.schema)
    transfer = ["transfer", *owner, "--authority-file", str(authority)]

    # --source names a fresh install's legacy path; without --fresh-install it is a usage error.
    with pytest.raises(SystemExit) as misused:
        main([*transfer, "--source", str(source)])
    misused_row = authority_row(names.admin, names.schema)
    capsys.readouterr()
    opened = main([*transfer, "--fresh-install"])
    captured = capsys.readouterr()

    generation, token = authority_file_token(authority)
    assert provisioned[0] == 1
    assert provisioned[2:] == (False, False)
    assert misused.value.code == 2
    assert misused_row == provisioned
    assert (opened, captured.out, captured.err) == (
        0,
        "EDGE_PG_MIGRATION_TRANSFER_OK generation=2 accepting=true egress_enabled=true\n",
        "",
    )
    assert generation == 2
    assert authority_row(names.admin, names.schema) == (2, token, True, True)
    assert not pending_authority_path(authority).exists()
    # A fresh install leaves no SQLite file behind at either path.
    assert [*canonical.parent.iterdir(), *source.parent.iterdir()] == []


@pytest.mark.parametrize(
    ("location", "artifact"),
    [
        ("canonical", "edge.sqlite3"),
        ("canonical", "edge.sqlite3-wal"),
        ("canonical", "edge.sqlite3-journal"),
        ("canonical", "dangling-symlink"),
        ("source", "edge.sqlite3"),
    ],
    ids=[
        "canonical-database",
        "canonical-wal-only",
        "canonical-journal-only",
        "canonical-dangling-symlink",
        "source-database",
    ],
)
def test_cli_fresh_install_refuses_legacy_data_at_either_path(
    migration_names: MigrationNames,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    artifact: str,
) -> None:
    names = migration_names
    canonical, source = _fresh_install_paths(tmp_path, monkeypatch)
    owner = _owner_args(tmp_path, names)
    authority = _provision_for_transfer(names, tmp_path, owner)
    capsys.readouterr()
    legacy = canonical if location == "canonical" else source
    if artifact == "dangling-symlink":
        planted = legacy
        planted.symlink_to(legacy.with_name("moved-away.sqlite3"))
    else:
        planted = legacy.with_name(artifact)
        planted.write_bytes(b"")
    provisioned = authority_row(names.admin, names.schema)
    authority_bytes = authority.read_bytes()
    argv = [
        "transfer",
        *owner,
        "--authority-file",
        str(authority),
        "--fresh-install",
        "--source",
        str(source),
    ]

    refused = main(argv)
    captured = capsys.readouterr()
    refused_row = authority_row(names.admin, names.schema)
    refused_bytes = authority.read_bytes()
    refused_pending = pending_authority_path(authority).exists()
    # The counterfactual: the same command once the legacy artifact is gone.
    planted.unlink()
    accepted = main(argv)
    capsys.readouterr()

    named = f" at {canonical}" if location == "canonical" else ""
    assert (refused, captured.out, captured.err) == (
        1,
        "",
        (
            f"EDGE_PG_MIGRATION_TRANSFER_FAILED: legacy SQLite source exists{named}; "
            "export and import it instead\n"
        ),
    )
    assert provisioned[0] == 1
    assert provisioned[2:] == (False, False)
    assert refused_row == provisioned
    assert refused_bytes == authority_bytes
    assert not refused_pending
    assert accepted == 0
    assert authority_row(names.admin, names.schema) == (
        2,
        authority_file_token(authority)[1],
        True,
        True,
    )

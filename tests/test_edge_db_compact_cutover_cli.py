from __future__ import annotations

from pathlib import Path

import pytest
from compact_cutover_fixtures import cutover_request

from backend.app.edge_db import compact_cutover
from backend.app.edge_db.compact_cutover import main

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")


def test_cli_rejects_sqlite_version_override_before_artifacts(
    tmp_path: Path,
) -> None:
    request = cutover_request(tmp_path)

    with pytest.raises(SystemExit) as failure:
        main(
            [
                "--source",
                str(request.source),
                "--live",
                str(request.live),
                "--archive",
                str(request.archive),
                "--candidate",
                str(request.candidate),
                "--receipt",
                str(request.receipt),
                "--clip-store",
                str(request.clip_store),
                "--worker-state",
                str(request.worker_state),
                "--sqlite-version",
                "3.51.3",
            ]
        )

    assert failure.value.code == 2
    assert not request.archive.exists()
    assert not request.candidate.exists()
    assert not request.receipt.exists()


def test_actual_old_sqlite_is_bounded_failure_without_replacement(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = cutover_request(tmp_path)
    before = request.live.read_bytes()
    monkeypatch.setattr(compact_cutover, "_runtime_sqlite_version", lambda: (3, 45, 1))

    status = main(
        [
            "--source",
            str(request.source),
            "--live",
            str(request.live),
            "--archive",
            str(request.archive),
            "--candidate",
            str(request.candidate),
            "--receipt",
            str(request.receipt),
            "--clip-store",
            str(request.clip_store),
            "--worker-state",
            str(request.worker_state),
        ]
    )

    output = capsys.readouterr()
    assert status == 1
    assert output.out == ""
    assert output.err.startswith("EDGE_DB_COMPACT_CUTOVER_FAILED:")
    assert request.live.read_bytes() == before
    assert not request.candidate.exists()

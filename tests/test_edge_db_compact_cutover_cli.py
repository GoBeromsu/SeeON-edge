from __future__ import annotations

from pathlib import Path

import pytest
from compact_cutover_fixtures import cutover_request

from backend.app.edge_db.compact_cutover import main


def test_bad_sqlite_cli_is_bounded_failure_without_replacement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = cutover_request(tmp_path)
    before = request.live.read_bytes()

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
            "--sqlite-version",
            "3.45.1",
        ]
    )

    output = capsys.readouterr()
    assert status == 1
    assert output.out == ""
    assert output.err.startswith("EDGE_DB_COMPACT_CUTOVER_FAILED:")
    assert request.live.read_bytes() == before
    assert not request.candidate.exists()

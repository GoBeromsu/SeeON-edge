from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from compact_cutover_fixtures import cutover_request, sha256

from backend.app.edge_db.compact_cutover import CompactCutoverError, run_compact_cutover
from backend.app.edge_db.compact_cutover_files import copy_exclusive

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")


@pytest.mark.parametrize("output_name", ["archive", "candidate"])
def test_existing_output_hardlink_cannot_mutate_source(tmp_path: Path, output_name: str) -> None:
    request = cutover_request(tmp_path)
    output = getattr(request, output_name)
    os.link(request.source, output)
    source_hash = sha256(request.source)

    with pytest.raises(CompactCutoverError, match="HARDLINK|ALIAS|EXISTS"):
        run_compact_cutover(request)

    assert sha256(request.source) == source_hash
    with sqlite3.connect(request.source) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)
    assert output.samefile(request.source)


def test_symlink_swap_during_exclusive_copy_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "candidate"
    source.write_bytes(b"immutable")

    def swap() -> None:
        destination.unlink()
        destination.symlink_to(source)

    with pytest.raises(CompactCutoverError, match="RACE|ALIAS|SYMLINK|HARDLINK"):
        copy_exclusive(source, destination, mode=0o600, on_written=swap)

    assert source.read_bytes() == b"immutable"


def test_source_with_extra_hardlink_is_refused_before_artifacts(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    alias = request.source.with_name("source-alias.sqlite3")
    os.link(request.source, alias)
    before = sha256(request.source)

    with pytest.raises(CompactCutoverError, match="HARDLINK|LINK_COUNT"):
        run_compact_cutover(request)

    assert sha256(request.source) == before
    assert not request.archive.exists()
    assert not request.candidate.exists()
    assert not request.receipt.exists()

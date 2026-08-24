from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from compact_cutover_fixtures import cutover_request

from backend.app.edge_db.compact_cutover import run_compact_cutover
from backend.app.edge_db.compact_receipt_verification import (
    ReceiptVerification,
    verify_receipts,
)

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")


def test_row18_cutover_provenance_must_match_source_and_receipt(tmp_path: Path) -> None:
    request = cutover_request(tmp_path)
    result = run_compact_cutover(request)
    candidate = request.live.with_name("tampered-candidate.sqlite3")
    shutil.copyfile(request.live, candidate)
    with sqlite3.connect(candidate) as connection:
        connection.execute(
            "UPDATE schema_migrations SET source_db_sha256=? WHERE version=18",
            ("0" * 64,),
        )
        connection.commit()
    first_receipt = json.loads(request.receipt.read_text().splitlines()[0])

    with pytest.raises(sqlite3.DatabaseError, match="provenance differs"):
        verify_receipts(
            ReceiptVerification(
                request.source,
                candidate,
                request.receipt,
                result.source_rows,
                first_receipt["inventory_sha256"],
                (),
            )
        )

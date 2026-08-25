from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from compact_cutover_dense_fixture import dense_cutover_request

from backend.app.edge_db.compact_cutover import run_compact_cutover

pytestmark = pytest.mark.usefixtures("supported_compact_cutover_sqlite")


def test_dense_receipts_are_byte_deterministic_across_independent_runs(tmp_path: Path) -> None:
    request = dense_cutover_request(tmp_path)

    first_result = run_compact_cutover(request)
    first_bytes = request.receipt.read_bytes()
    second_result = run_compact_cutover(request)
    second_bytes = request.receipt.read_bytes()

    assert first_bytes == second_bytes
    assert first_result.receipt_sha256 == second_result.receipt_sha256
    assert hashlib.sha256(first_bytes).hexdigest() == first_result.receipt_sha256

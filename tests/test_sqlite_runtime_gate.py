"""Packaged SQLite floor and bounded Docker logs for schema-18 cutover."""

from __future__ import annotations

import re
from pathlib import Path

from backend.app.edge_db import sqlite_runtime

ROOT = Path(__file__).resolve().parents[1]
BACKEND_IMAGE = ROOT / "Dockerfile.backend"
COMPOSE_FILE = ROOT / "compose.edge.yaml"
PINNED_UV_PYTHON = re.compile(
    r"^FROM ghcr\.io/astral-sh/uv:python[0-9.]+-[a-z0-9.-]+@sha256:[0-9a-f]{64}\s*$",
    re.MULTILINE,
)
BOUNDED_LOG_BLOCK = (
    "    logging:\n"
    "      driver: local\n"
    "      options:\n"
    '        max-size: "10m"\n'
    '        max-file: "5"\n'
)
# edge-model-fetch is a one-shot like the migrator; its download log is bounded too.
RUNTIME_SERVICES = ("edge-db-migrator", "edge-model-fetch", "ml-api", "ml-worker")


def test_backend_image_pins_official_uv_python_digest_and_sqlite_floor() -> None:
    dockerfile = BACKEND_IMAGE.read_text(encoding="utf-8")

    assert PINNED_UV_PYTHON.search(dockerfile)
    assert "sqlite3.sqlite_version_info >= (3, 51, 3)" in dockerfile
    assert "pysqlite3" not in dockerfile
    assert "sqlite.org" not in dockerfile
    assert "compile sqlite" not in dockerfile.lower()


def test_edge_runtime_services_bound_local_docker_logs() -> None:
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    assert compose.count(BOUNDED_LOG_BLOCK) == len(RUNTIME_SERVICES)
    for name in RUNTIME_SERVICES:
        service = re.search(rf"^  {re.escape(name)}:\n(?:    .*\n)+", compose, re.MULTILINE)
        assert service is not None
        assert BOUNDED_LOG_BLOCK in service.group(0)
    assert "image: logging" not in compose


def test_schema18_cutover_preflight_refuses_old_sqlite_before_candidate_exists(
    tmp_path: Path,
) -> None:
    source = tmp_path / "edge.sqlite3"
    source.write_bytes(b"v17-source")
    source_bytes = source.read_bytes()
    candidate = tmp_path / "edge.sqlite3.candidate"

    status = sqlite_runtime.main(
        [
            "--source",
            str(source),
            "--candidate",
            str(candidate),
            "--sqlite-version",
            "3.45.1",
        ]
    )

    assert status != 0
    assert not candidate.exists()
    assert source.exists()
    assert source.read_bytes() == source_bytes


def test_schema18_cutover_preflight_accepts_minimum_sqlite_without_creating_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "edge.sqlite3"
    source.write_bytes(b"v17-source")
    candidate = tmp_path / "edge.sqlite3.candidate"

    status = sqlite_runtime.main(
        [
            "--source",
            str(source),
            "--candidate",
            str(candidate),
            "--sqlite-version",
            "3.51.3",
        ]
    )

    assert status == 0
    assert not candidate.exists()
    assert source.read_bytes() == b"v17-source"


def test_schema18_cutover_preflight_rejects_malformed_sqlite_version(
    tmp_path: Path,
) -> None:
    source = tmp_path / "edge.sqlite3"
    source.write_bytes(b"v17-source")
    candidate = tmp_path / "edge.sqlite3.candidate"

    status = sqlite_runtime.main(
        [
            "--source",
            str(source),
            "--candidate",
            str(candidate),
            "--sqlite-version",
            "not-a-version",
        ]
    )

    assert status != 0
    assert not candidate.exists()
    assert source.read_bytes() == b"v17-source"

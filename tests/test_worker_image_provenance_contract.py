from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.edge"
ZERO_REVISION = "0" * 40
IMAGE_REVISION_MARKER = "/opt/seeon/ml-worker-image-revision"
SCHEMA_IDENTITY_MARKER = "/opt/seeon/edge-database-schema-version"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_edge_image_requires_source_revision_without_unsafe_default() -> None:
    source = _dockerfile()

    assert re.search(r"^ARG SOURCE_REVISION$", source, re.MULTILINE)


@pytest.mark.parametrize(
    "revision",
    [None, "", ZERO_REVISION, "1" * 39, "1" * 41, "A" * 40, "g" * 40, "1" * 39 + " ", "1" * 40],
)
def test_revision_shell_guard_executes_without_writing_image_markers(
    revision: str | None,
) -> None:
    source = _dockerfile()
    start = source.index("RUN if ", source.index("ARG SOURCE_REVISION"))
    end = source.index("    fi;", start) + len("    fi;")
    guard = source[start:end].removeprefix("RUN ").replace("\\\n", "")
    assert "install -d" not in guard
    environment = {"PATH": os.defpath}
    if revision is not None:
        environment["SOURCE_REVISION"] = revision
    result = subprocess.run(
        ["/bin/sh", "-c", guard],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if revision == "1" * 40:
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
    else:
        assert result.returncode == 1, result.stderr
        assert result.stderr == (
            "SOURCE_REVISION must be a non-zero 40-character lowercase hexadecimal revision\n"
        )


def test_edge_image_build_rejects_zero_and_malformed_source_revisions() -> None:
    source = _dockerfile()
    validation = source[source.index("ARG SOURCE_REVISION") : source.index("LABEL ")]

    assert f'[ "$SOURCE_REVISION" = "{ZERO_REVISION}" ]' in validation
    assert '"${#SOURCE_REVISION}" -ne 40' in validation
    assert "[^0123456789abcdef]" in validation
    assert "exit 1" in validation


def test_edge_image_bakes_one_revision_into_label_environment_and_marker() -> None:
    source = _dockerfile()

    assert 'org.opencontainers.image.revision="${SOURCE_REVISION}"' in source
    assert 'ML_WORKER_BUILD_REVISION="${SOURCE_REVISION}"' in source
    assert f'"$SOURCE_REVISION" > {IMAGE_REVISION_MARKER}' in source
    assert SCHEMA_IDENTITY_MARKER in source
    assert 'seeon.edge.database.schema-version="${EDGE_DATABASE_SCHEMA_VERSION}"' in source
    assert f"chmod 0444 {IMAGE_REVISION_MARKER} {SCHEMA_IDENTITY_MARKER}" in source

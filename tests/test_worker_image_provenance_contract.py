from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.edge"
ZERO_REVISION = "0" * 40
IMAGE_REVISION_MARKER = "/opt/seeon/ml-worker-image-revision"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_edge_image_requires_source_revision_without_unsafe_default() -> None:
    source = _dockerfile()

    assert re.search(r"^ARG SOURCE_REVISION$", source, re.MULTILINE)
    assert not re.search(r"^ARG SOURCE_REVISION=", source, re.MULTILINE)


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
    assert f"chmod 0444 {IMAGE_REVISION_MARKER}" in source

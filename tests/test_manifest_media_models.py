from __future__ import annotations

import pytest
from pydantic import ValidationError

from worker.pipeline.output.evidence.manifest_media_models import SourceMediaFacts


def _payload() -> dict[str, object]:
    return {
        "timestamp_translation_seconds": "0/1",
        "streams": [
            {
                "index": 0,
                "time_base": "1/90000",
                "packet_count": 1,
                "parser_caps_sha256": "a" * 64,
            }
        ],
        "au_index": {
            "schema": 1,
            "path": "au-index.cbor",
            "sha256": "b" * 64,
            "size_bytes": 10,
            "count": 1,
        },
    }


def test_source_manifest_retains_parser_caps_hash() -> None:
    facts = SourceMediaFacts.model_validate(_payload())

    assert facts.streams[0].parser_caps_sha256 == "a" * 64


def test_au_index_path_is_fixed_basename() -> None:
    payload = _payload()
    index = payload["au_index"]
    assert isinstance(index, dict)
    index["path"] = "../au-index.cbor"

    with pytest.raises(ValidationError):
        SourceMediaFacts.model_validate(payload)

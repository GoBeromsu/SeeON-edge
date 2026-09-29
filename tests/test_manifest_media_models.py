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


def _translated(first_stream_ticks: int) -> dict[str, object]:
    return {
        "configuration_id": "configuration-1",
        "timestamp_translation_seconds": "-1/1536",
        "streams": [
            {
                "index": 0,
                "time_base": "1/15360",
                "packet_count": 25,
                "timestamp_translation_ticks": first_stream_ticks,
            },
            {
                "index": 1,
                "time_base": "1/48000",
                "packet_count": 0,
                "timestamp_translation_ticks": None,
            },
        ],
    }


def test_exact_remux_translation_is_accepted() -> None:
    facts = SourceMediaFacts.model_validate(_translated(-10))

    assert facts.streams[0].timestamp_translation_ticks == -10


def test_nonuniform_remux_translation_is_rejected() -> None:
    with pytest.raises(ValidationError, match="nonuniform remux timestamp translation"):
        SourceMediaFacts.model_validate(_translated(-11))

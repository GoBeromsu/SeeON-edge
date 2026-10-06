from __future__ import annotations

import pytest

from backend.app.features.audit.catalog import AuditAction, AuditDetailError, parse_detail_json


def test_detail_version_is_exact_uncoerced_json_integer() -> None:
    # Given/When/Then: only JSON int 1 is accepted, never bool/float/string/null.
    assert parse_detail_json(AuditAction.CLIP_LIST, '{"version":1}').json == '{"version":1}'
    for encoded in (
        '{"version":true}',
        '{"version":false}',
        '{"version":1.0}',
        '{"version":1.00}',
        '{"version":"1"}',
        '{"version":null}',
        '{"version":2}',
    ):
        with pytest.raises(AuditDetailError, match="version"):
            parse_detail_json(AuditAction.CLIP_LIST, encoded)


def test_nested_detail_payload_does_not_coerce_version() -> None:
    encoded = '{"version":true,"safe":{"version":1}}'
    with pytest.raises(AuditDetailError):
        parse_detail_json(AuditAction.CLIP_LIST, encoded)

from __future__ import annotations

from backend.app.features.connection.store import mask_facility_token


class TestMasking:
    def test_mask_facility_token_short_token_fully_masked(self) -> None:
        assert mask_facility_token("abcd") == "****"
        assert mask_facility_token("ab") == "****"

    def test_mask_facility_token_long_token_shows_last_four(self) -> None:
        assert mask_facility_token("supersecrettoken1234") == "****1234"

    def test_mask_facility_token_none_or_empty(self) -> None:
        assert mask_facility_token(None) is None
        assert mask_facility_token("") is None

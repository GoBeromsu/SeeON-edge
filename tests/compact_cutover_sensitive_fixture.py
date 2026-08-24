"""Distinct secret aliases used by dense cutover redaction tests."""

from __future__ import annotations

import base64
from typing import Final

from pydantic import JsonValue

FACILITY_TOKEN: Final = "dense-facility-token-LEAK"
PASSWORD_HASH: Final = b"dense-password-hash-material".ljust(64, b"h")
SALT: Final = b"dense-salt-value"
NESTED_TOKEN: Final = "dense-nested-token-LEAK"
BEARER_SECRET: Final = "Bearer dense-bearer-LEAK"
COOKIE_SECRET: Final = "session=dense-cookie-LEAK"
DENSE_SECRETS: Final = (
    FACILITY_TOKEN,
    PASSWORD_HASH.hex(),
    PASSWORD_HASH.decode("ascii"),
    base64.b64encode(PASSWORD_HASH).decode("ascii"),
    SALT.hex(),
    NESTED_TOKEN,
    BEARER_SECRET,
    COOKIE_SECRET,
)


def dense_audit_payload() -> dict[str, JsonValue]:
    return {
        "actor_type": "user",
        "actor_id": "operator",
        "target_type": "clip",
        "target_id": "clip:fixture",
        "outcome": "success",
        "safe": {"preserved": [1, "yes"]},
        "FaCiLiTy_ToKeN": FACILITY_TOKEN,
        "nested": {
            "ToKeN": NESTED_TOKEN,
            "innocuous_aliases": [
                FACILITY_TOKEN,
                PASSWORD_HASH.hex(),
                PASSWORD_HASH.decode("ascii"),
                base64.b64encode(PASSWORD_HASH).decode("ascii"),
                SALT.hex(),
                BEARER_SECRET,
                COOKIE_SECRET,
            ],
        },
        "PASSWORD_HASH": PASSWORD_HASH.hex(),
        "resident_name": "protected resident",
    }


__all__ = [
    "BEARER_SECRET",
    "COOKIE_SECRET",
    "DENSE_SECRETS",
    "FACILITY_TOKEN",
    "NESTED_TOKEN",
    "PASSWORD_HASH",
    "SALT",
    "dense_audit_payload",
]

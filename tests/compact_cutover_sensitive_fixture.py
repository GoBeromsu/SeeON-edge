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
OPAQUE_SESSION: Final = "dense-opaque-session-LEAK"
RELATIVE_PATH: Final = "evidence/snapshots/dense-private.jpg"
GENERIC_PATH: Final = "private/nested/dense-file.bin"
_FACILITY_BYTES = FACILITY_TOKEN.encode()
FACILITY_ALIASES: Final = (
    _FACILITY_BYTES.hex(),
    _FACILITY_BYTES.hex().upper(),
    base64.b64encode(_FACILITY_BYTES).decode("ascii"),
    base64.urlsafe_b64encode(_FACILITY_BYTES).decode("ascii").rstrip("="),
)
DENSE_SECRETS: Final = (
    FACILITY_TOKEN,
    PASSWORD_HASH.hex(),
    PASSWORD_HASH.decode("ascii"),
    base64.b64encode(PASSWORD_HASH).decode("ascii"),
    SALT.hex(),
    NESTED_TOKEN,
    BEARER_SECRET,
    COOKIE_SECRET,
    OPAQUE_SESSION,
    RELATIVE_PATH,
    GENERIC_PATH,
    *FACILITY_ALIASES,
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
            "Session_ID": OPAQUE_SESSION,
            "media_relpath": RELATIVE_PATH,
            "generic_path": GENERIC_PATH,
            "innocuous_aliases": [
                FACILITY_TOKEN,
                *FACILITY_ALIASES,
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
    "FACILITY_ALIASES",
    "FACILITY_TOKEN",
    "GENERIC_PATH",
    "NESTED_TOKEN",
    "OPAQUE_SESSION",
    "PASSWORD_HASH",
    "RELATIVE_PATH",
    "SALT",
    "dense_audit_payload",
]

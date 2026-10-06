from __future__ import annotations

import hashlib
import json

import pytest

import backend.app.features.cameras.router as cameras_router


@pytest.mark.xfail(strict=True, reason="Known bug #254: local override hash can decrease version")
def test_local_override_hash_can_produce_lower_config_version() -> None:
    """Characterizes #254: the hash-based offset has no ordering semantics.

    This test searches a small space of start times to find two valid edits
    that produce a lower config_version on the second save.
    """
    pulled_version = 7

    def local_version(start: str) -> int:
        domains = {"fall": {"enabled": True}, "bed_exit": {"enabled": True}}
        windows = {"bed_exit": {"start": start, "end": "06:00", "tz": "Asia/Seoul"}}
        return cameras_router._local_config_version(pulled_version, domains, windows, windows["bed_exit"])  # type: ignore[attr-defined]

    starts = [f"{h:02d}:{m:02d}" for h in range(18, 24) for m in (0, 30)]
    found = False
    for a in starts:
        for b in starts:
            if a == b:
                continue
            if local_version(b) < local_version(a):
                found = True
                break
        if found:
            break
    assert not found, "Expected to find a decreasing pair (documented in #254)"


@pytest.mark.xfail(strict=True, reason="Known bug #591: policy-scaled config_version can exceed INT4")
def test_policy_scaled_config_version_exceeds_int4() -> None:
    """Characterizes #591: base*1e9 + hash part can exceed 32-bit signed int."""
    base = 735_739
    hash_part = int(hashlib.sha256(b"example-policy").hexdigest()[:8], 16) % 1_000_000_000
    scaled = base * 1_000_000_000 + hash_part
    int32_max = 2_147_483_647
    assert scaled <= int32_max


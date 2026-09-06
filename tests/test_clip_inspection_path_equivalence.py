"""Both clip inspection paths must reach identical verdicts.

`inspect_finalized_clip` has two implementations: a pathname one, and a
descriptor-relative one used by legacy recovery so a clip cannot be classified
against a store that vanished mid-scan. They must agree exactly.

They did not. The descriptor variant subscripted `payload["mime_type"]` while
the pathname variant used `.get`, so a manifest missing that field was `CORRUPT`
through one path and `VERIFIED` through the other -- with the pathname variant
recording the literal string `"None"` as verified metadata. On a store of 1053
clips, a divergence like that decides whether evidence is written off.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from backend.app.features.clips.consistency_ops import (
    ClipConsistencyError,
    inspect_finalized_clip,
)

_CLIP_ID = "clip:probe"
_MEDIA = b"\x00\x00\x00\x18ftypmp42" + b"y" * 300


def _manifest(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "manifest_schema_version": 2,
        "state": "READY",
        "clip_id": _CLIP_ID,
        "camera_id": "cam-1",
        "event_refs": ["11111111-1111-4111-8111-111111111111"],
        "event_ref": "11111111-1111-4111-8111-111111111111",
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:01Z",
        "sha256": hashlib.sha256(_MEDIA).hexdigest(),
        "size_bytes": len(_MEDIA),
        "mime_type": "video/mp4",
        "codec": "h264",
        "duration_ms": 1000,
        "path": f"clips/{_CLIP_ID}/clip.mp4",
    }
    for key, value in overrides.items():
        if value is _ABSENT:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


class _Absent:
    """Sentinel meaning 'remove this key entirely'."""


_ABSENT = _Absent()


def _store(tmp_path: Path, payload: dict[str, Any], *, media: bytes = _MEDIA) -> Path:
    store = tmp_path / "store"
    (store / "clips" / ".staging").mkdir(parents=True, exist_ok=True)
    directory = store / "clips" / _CLIP_ID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "clip.mp4").write_bytes(media)
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return store


def _verdict(store: Path, *, use_descriptor: bool) -> str:
    if not use_descriptor:
        try:
            return inspect_finalized_clip(store, _CLIP_ID).local_state
        except ClipConsistencyError as error:
            return f"refused:{error.code}"
    handle = os.open(store / "clips", os.O_RDONLY | os.O_DIRECTORY)
    try:
        return inspect_finalized_clip(store, _CLIP_ID, clips_dir_fd=handle).local_state
    except ClipConsistencyError as error:
        return f"refused:{error.code}"
    finally:
        os.close(handle)


_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("honest", {}),
    ("missing_mime_type", {"mime_type": _ABSENT}),
    ("missing_codec", {"codec": _ABSENT}),
    ("missing_sha256", {"sha256": _ABSENT}),
    ("missing_path", {"path": _ABSENT}),
    ("path_names_another_object", {"path": "clips/other/clip.mp4"}),
    ("wrong_sha256", {"sha256": "0" * 64}),
    ("wrong_size", {"size_bytes": 1}),
    ("wrong_clip_id", {"clip_id": "clip:elsewhere"}),
    ("unavailable_state", {"state": "UNAVAILABLE", "reason": "probe"}),
)


@pytest.mark.parametrize(("name", "overrides"), _CASES, ids=[case[0] for case in _CASES])
def test_both_inspection_paths_agree(tmp_path: Path, name: str, overrides: dict[str, Any]) -> None:
    """A clip must not be judged differently depending on how it was reached."""
    store = _store(tmp_path, _manifest(**overrides))

    pathname = _verdict(store, use_descriptor=False)
    descriptor = _verdict(store, use_descriptor=True)

    assert pathname == descriptor, (
        f"case {name!r}: pathname inspection says {pathname!r} but descriptor "
        f"inspection says {descriptor!r}; identical evidence must reach the same "
        f"verdict through either path"
    )


def test_an_honest_manifest_still_verifies(tmp_path: Path) -> None:
    """Guard the guard: agreement on 'always refuse' would be worthless."""
    store = _store(tmp_path, _manifest())

    assert _verdict(store, use_descriptor=False) == "VERIFIED"
    assert _verdict(store, use_descriptor=True) == "VERIFIED"


def test_a_manifest_missing_metadata_is_refused_not_stringified(tmp_path: Path) -> None:
    """A missing field must refuse, never be recorded as the string 'None'."""
    store = _store(tmp_path, _manifest(mime_type=_ABSENT))

    assert _verdict(store, use_descriptor=False).startswith("refused:")

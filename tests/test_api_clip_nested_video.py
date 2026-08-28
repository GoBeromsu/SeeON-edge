from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO, cast

import pytest
from fastapi.testclient import TestClient
from receipt_helpers import MediaReceiptStore, add_accepted_media_receipts

from backend.app.features.clips.descriptor_files import OpenedRegularFile
from backend.app.features.clips.store import ClipStore, LocatedClip
from backend.app.features.evidence.receipt_store import ArtifactReceipt
from backend.app.main import create_app as _create_app
from backend.app.main import no_lifespan


def create_app(*, lifespan):
    app = _create_app(lifespan=lifespan)
    add_accepted_media_receipts(app)
    return app


VIDEO = b"0123456789"


@pytest.fixture(autouse=True)
def clip_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "clip-store"
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    monkeypatch.setenv("API_LABEL_STORE", str(tmp_path / "label-store"))
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    return root


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


def _write_clip(
    store_root: Path,
    layout: Path,
    clip_id: str,
    manifest_path: str,
) -> Path:
    recording_root = store_root / layout
    clip_dir = recording_root / "clips" / clip_id
    clip_dir.mkdir(parents=True)
    video_path = clip_dir / "clip.mp4"
    video_path.write_bytes(VIDEO)
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": "camera-1",
                "event_ref": f"event-{clip_id}",
                "event_type": "fall",
                "started_at": "2026-08-10T00:00:00Z",
                "duration_s": 10.0,
                "codec": "h264",
                "path": manifest_path,
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    return video_path


@pytest.mark.parametrize(
    "layout",
    (Path(), Path("archive"), Path("external") / "drive-1"),
)
def test_relative_video_path_resolves_from_located_recording_root(
    clip_env: Path,
    layout: Path,
) -> None:
    layout_name = "root" if layout == Path() else layout.as_posix().replace("/", "-")
    clip_id = f"clip-{layout_name}"
    _write_clip(
        clip_env,
        layout,
        clip_id,
        f"clips/{clip_id}/clip.mp4",
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        metadata = client.get(f"/api/v1/clips/{clip_id}/metadata")
        video = client.get(
            f"/api/v1/clips/{clip_id}/video",
            headers={"Range": "bytes=2-5"},
        )

    assert metadata.status_code == 200
    assert metadata.json()["size_bytes"] == len(VIDEO)
    assert video.status_code == 206
    assert video.content == VIDEO[2:6]
    assert video.headers["content-range"] == f"bytes 2-5/{len(VIDEO)}"
    assert video.headers["accept-ranges"] == "bytes"
    assert video.headers["content-length"] == "4"
    assert video.headers["cache-control"] == "private, no-store"


def test_video_full_and_invalid_range_responses_expose_browser_headers(
    clip_env: Path,
) -> None:
    clip_id = "clip-range-contract"
    _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        full = client.get(f"/api/v1/clips/{clip_id}/video")
        malformed = client.get(
            f"/api/v1/clips/{clip_id}/video",
            headers={"Range": "items=broken"},
        )
        unsatisfiable = client.get(
            f"/api/v1/clips/{clip_id}/video",
            headers={"Range": "bytes=99-"},
        )

    assert full.status_code == 200
    assert full.content == VIDEO
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-length"] == str(len(VIDEO))
    assert full.headers["cache-control"] == "private, no-store"
    assert malformed.status_code == 400
    assert malformed.headers["accept-ranges"] == "bytes"
    assert malformed.headers["cache-control"] == "private, no-store"
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["content-range"] == f"bytes */{len(VIDEO)}"
    assert unsatisfiable.headers["accept-ranges"] == "bytes"
    assert unsatisfiable.headers["cache-control"] == "private, no-store"


def test_video_open_descriptor_is_closed_after_response(
    clip_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_id = "clip-close"
    _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")
    opened_handles: list[BinaryIO] = []
    original_open = ClipStore.open_located_video

    def observed_open(self: ClipStore, located: LocatedClip) -> OpenedRegularFile:
        opened = original_open(self, located)
        opened_handles.append(opened.handle)
        return opened

    monkeypatch.setattr(ClipStore, "open_located_video", observed_open)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{clip_id}/video")

    assert response.status_code == 200
    assert len(opened_handles) == 1
    assert opened_handles[0].closed


def test_video_symlink_swap_after_validation_cannot_serve_outside_bytes(
    clip_env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_id = "clip-swap"
    _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside-secret")
    original_resolve = ClipStore.resolve_located_video_path

    def swap_after_validation(self: ClipStore, located: LocatedClip) -> Path:
        path = original_resolve(self, located)
        path.unlink()
        os.symlink(outside, path)
        return path

    monkeypatch.setattr(ClipStore, "resolve_located_video_path", swap_after_validation)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{clip_id}/video")

    assert response.status_code == 404
    assert response.content != outside.read_bytes()


def test_nested_worker_path_ignores_root_level_decoy(clip_env: Path) -> None:
    clip_id = "clip-nested-decoy"
    _write_clip(
        clip_env,
        Path("archive"),
        clip_id,
        f"clips/{clip_id}/clip.mp4",
    )
    decoy_path = clip_env / "clips" / clip_id / "clip.mp4"
    decoy_path.parent.mkdir(parents=True)
    decoy_path.write_bytes(b"root-level-decoy")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{clip_id}/video")

    assert response.status_code == 200
    assert response.content == VIDEO


def test_absolute_contained_video_path_remains_supported(clip_env: Path) -> None:
    clip_id = "clip-absolute"
    video_path = _write_clip(clip_env, Path("archive"), clip_id, "placeholder")
    manifest_path = video_path.parent / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["path"] = str(video_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{clip_id}/video")

    assert response.status_code == 200
    assert response.content == VIDEO


def test_nested_relative_video_path_cannot_escape_physical_store(
    clip_env: Path,
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.mp4"
    secret.write_bytes(b"secret")
    clip_id = "clip-escape"
    _write_clip(
        clip_env,
        Path("archive"),
        clip_id,
        "../../../secret.mp4",
    )

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{clip_id}/video")

    assert response.status_code == 400
    assert response.json()["detail"] == "manifest path escapes clip store"


def test_local_evidence_plays_when_no_backend_receipt_exists(clip_env: Path) -> None:
    """An operator must be able to watch what this box recorded.

    A receipt is only committed after a successful upstream export, which needs
    clip export enabled (Hub-owned config, off by default) and a Hub-issued
    camera id. Refusing playback until one exists made every clip on this
    deployment permanently unplayable: verified HEVC on disk, thumbnail and
    manifest present, listed as video_available, and the browser answering
    "영상을 재생하지 못했습니다" every time.
    """
    clip_id = "clip-no-receipt"
    _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        # A deployment where no receipt was ever committed, because clip
        # export never ran. The suite otherwise injects accepted receipts for
        # every clip, which is exactly why this regression reached production
        # unnoticed.
        client.app.state.artifact_receipt_store = MediaReceiptStore()
        response = client.get(f"/api/v1/clips/{clip_id}/video")

    assert response.status_code == 200
    assert response.content == VIDEO


def test_a_receipt_that_was_refused_still_blocks_playback(clip_env: Path) -> None:
    """Dropping the existence requirement must not admit a refused artifact."""
    clip_id = "clip-refused-receipt"
    video = _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")
    content = video.read_bytes()

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        store = MediaReceiptStore()
        store.commit(
            ArtifactReceipt(
                clip_id,
                hashlib.sha256(content).hexdigest(),
                len(content),
                accepted=False,
            )
        )
        client.app.state.artifact_receipt_store = store
        response = client.get(f"/api/v1/clips/{clip_id}/video")

    assert response.status_code == 404
    assert response.json()["detail"] == "clip video receipt not accepted"


def test_bytes_that_disagree_with_their_receipt_are_still_refused(clip_env: Path) -> None:
    """The integrity guarantee is the part worth keeping, and it stays strict."""
    clip_id = "clip-tampered"
    _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        store = MediaReceiptStore()
        store.commit(
            ArtifactReceipt(clip_id, hashlib.sha256(b"different footage").hexdigest(), len(VIDEO))
        )
        client.app.state.artifact_receipt_store = store
        response = client.get(f"/api/v1/clips/{clip_id}/video")

    assert response.status_code == 409
    assert response.json()["detail"] == "clip video receipt verification failed"


class _CountingHandle:
    """Wraps a clip descriptor and counts the reads served through it."""

    def __init__(self, inner: BinaryIO) -> None:
        self._inner = inner
        self.reads = 0

    def read(self, *args: int) -> bytes:
        self.reads += 1
        return self._inner.read(*args)

    def seek(self, *args: int) -> int:
        return self._inner.seek(*args)

    def close(self) -> None:
        self._inner.close()

    @property
    def closed(self) -> bool:
        return self._inner.closed

    def __enter__(self) -> _CountingHandle:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _count_video_reads(monkeypatch: pytest.MonkeyPatch) -> list[_CountingHandle]:
    handles: list[_CountingHandle] = []
    original_open = ClipStore.open_located_video

    def observed_open(self: ClipStore, located: LocatedClip) -> OpenedRegularFile:
        opened = original_open(self, located)
        handle = _CountingHandle(opened.handle)
        handles.append(handle)
        return OpenedRegularFile(cast(BinaryIO, handle), opened.path, opened.size_bytes)

    monkeypatch.setattr(ClipStore, "open_located_video", observed_open)
    return handles


def test_head_video_answers_with_the_get_header_section_and_no_body(
    clip_env: Path,
) -> None:
    """A player probes the clip before it opens one; HEAD must not 404 (#452)."""
    clip_id = "clip-head-contract"
    _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        head = client.head(f"/api/v1/clips/{clip_id}/video")
        get = client.get(f"/api/v1/clips/{clip_id}/video")

    assert head.status_code == get.status_code == 200
    assert head.content == b""
    assert get.content == VIDEO
    for header in ("content-type", "content-length", "accept-ranges", "cache-control"):
        assert head.headers[header] == get.headers[header]
    assert head.headers["content-type"].startswith("video/mp4")
    assert head.headers["content-length"] == str(len(VIDEO))
    assert head.headers["accept-ranges"] == "bytes"


def test_head_video_reads_no_clip_bytes_and_releases_the_descriptor(
    clip_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HEAD that streamed the file would cost exactly what it exists to avoid."""
    clip_id = "clip-head-cheap"
    _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")
    handles = _count_video_reads(monkeypatch)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        head = client.head(f"/api/v1/clips/{clip_id}/video")
        head_reads = [handle.reads for handle in handles]
        get = client.get(f"/api/v1/clips/{clip_id}/video")

    assert head.status_code == 200
    assert get.status_code == 200
    assert head_reads == [0]
    assert handles[-1].reads > 0
    assert all(handle.closed for handle in handles)


def test_head_video_matches_get_on_missing_clips_and_ranges(clip_env: Path) -> None:
    clip_id = "clip-head-errors"
    _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        unauthorized_client = TestClient(create_app(lifespan=no_lifespan))
        unauthorized = unauthorized_client.head(f"/api/v1/clips/{clip_id}/video")
        missing = client.head("/api/v1/clips/clip-absent/video")
        partial = client.head(
            f"/api/v1/clips/{clip_id}/video",
            headers={"Range": "bytes=2-5"},
        )
        unsatisfiable = client.head(
            f"/api/v1/clips/{clip_id}/video",
            headers={"Range": "bytes=99-"},
        )
        malformed = client.head(
            f"/api/v1/clips/{clip_id}/video",
            headers={"Range": "items=broken"},
        )

    assert unauthorized.status_code == 401
    assert missing.status_code == 404
    assert partial.status_code == 206
    assert partial.content == b""
    assert partial.headers["content-range"] == f"bytes 2-5/{len(VIDEO)}"
    assert partial.headers["content-length"] == "4"
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["content-range"] == f"bytes */{len(VIDEO)}"
    assert malformed.status_code == 400


def test_get_video_range_handling_is_unchanged_by_head_support(clip_env: Path) -> None:
    clip_id = "clip-head-get-regression"
    _write_clip(clip_env, Path(), clip_id, f"clips/{clip_id}/clip.mp4")

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        full = client.get(f"/api/v1/clips/{clip_id}/video")
        partial = client.get(
            f"/api/v1/clips/{clip_id}/video",
            headers={"Range": "bytes=2-5"},
        )
        suffix = client.get(
            f"/api/v1/clips/{clip_id}/video",
            headers={"Range": "bytes=-3"},
        )

    assert full.status_code == 200
    assert full.content == VIDEO
    assert partial.status_code == 206
    assert partial.content == VIDEO[2:6]
    assert partial.headers["content-range"] == f"bytes 2-5/{len(VIDEO)}"
    assert suffix.status_code == 206
    assert suffix.content == VIDEO[-3:]
    assert suffix.headers["content-range"] == f"bytes 7-9/{len(VIDEO)}"

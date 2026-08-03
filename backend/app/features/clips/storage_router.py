"""Clip storage location: usage stats, directory browse, and location selection.

``GET /clips/storage`` -- current disk usage plus the selected subdirectory.
``GET /clips/storage/browse?path=<rel>`` -- one directory's subdirectories,
for the dashboard's folder-picker modal.
``PUT /clips/storage/location`` body ``{"path": ...}`` -- validate and
persist a new selection (``ClipStorageLocationStore``); propagated to the
worker as ``clip_store_subdir`` via ``cameras.router.worker_config_snapshot``.

Scope (see ``.omc/plans/redesign-api-contracts.md`` §5): choosing a
subdirectory *within* the ``CLIP_STORE_DIR`` mount (default
``/var/lib/clip-store``) only -- changing the host mount itself is out of
scope. Every client-supplied path is walked with O_NOFOLLOW dir_fd chaining
(the same pattern ``evidence/router.py``'s ``_verified_media`` uses for the
worker's clip-media reads) so a crafted ``../../etc`` or a symlink planted
inside the store can never resolve outside the configured root.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Annotated, ClassVar

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict

from backend.app.features.clips.storage_location_store import ClipStorageLocationStore
from backend.app.features.clips.store import CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR
from backend.app.shared.dashboard_auth import authorize_dashboard

router = APIRouter(tags=["clips"])


class ClipStorageBrowseEntry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str
    path: str


class ClipStorageBrowseResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    path: str
    parent: str | None
    directories: list[ClipStorageBrowseEntry]


class ClipStorageResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    root: str
    selected_path: str
    total_bytes: int | None
    used_bytes: int | None
    used_pct: float | None


class ClipStorageLocationRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    path: str


@router.get("/clips/storage", response_model=ClipStorageResponse)
def get_clip_storage(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    return _storage_snapshot(request.app)


@router.get("/clips/storage/browse", response_model=ClipStorageBrowseResponse)
def browse_clip_storage(
    request: Request,
    path: Annotated[str, Query()] = "",
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    segments = _validate_relative_path(path)
    try:
        names = _list_subdirectories(_configured_root(), segments)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="path not found"
        ) from exc
    joined = "/".join(segments)
    parent = None if not segments else "/".join(segments[:-1])
    return {
        "path": joined,
        "parent": parent,
        "directories": [
            {"name": name, "path": f"{joined}/{name}" if joined else name} for name in names
        ],
    }


@router.put("/clips/storage/location", response_model=ClipStorageResponse)
def put_clip_storage_location(
    payload: ClipStorageLocationRequest,
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, object]:
    _authorize(request, authorization)
    segments = _validate_relative_path(payload.path)
    try:
        # Existence + directory-ness check only; the listing itself is
        # unused here, but walking it is what proves the target is a real,
        # symlink-free directory inside the store root.
        _list_subdirectories(_configured_root(), segments)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="path not found"
        ) from exc
    selected = "/".join(segments)
    _location_store(request.app).put(selected)
    return _storage_snapshot(request.app)


def _storage_snapshot(app: FastAPI) -> dict[str, object]:
    root = _configured_root()
    selected = _location_store(app).get()
    try:
        usage = shutil.disk_usage(root)
    except OSError:
        return {
            "root": str(root),
            "selected_path": selected,
            "total_bytes": None,
            "used_bytes": None,
            "used_pct": None,
        }
    used_bytes = usage.total - usage.free
    used_pct = (used_bytes / usage.total * 100.0) if usage.total else None
    return {
        "root": str(root),
        "selected_path": selected,
        "total_bytes": usage.total,
        "used_bytes": used_bytes,
        "used_pct": used_pct,
    }


def _configured_root() -> Path:
    return Path(os.environ.get(CLIP_STORE_DIR_ENV, DEFAULT_CLIP_STORE_DIR))


def _validate_relative_path(value: str) -> list[str]:
    """Split and validate a client-supplied relative path into segments.

    Rejects anything that is not a syntactically safe relative path --
    absolute paths and ``..`` segments in particular -- with a 400 before it
    is ever used to construct a filesystem path. ``Path(base) / value``
    silently discards ``base`` and becomes absolute if ``value`` starts with
    ``/``, so this check happens on the raw string, not after joining.
    """
    if "\x00" in value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="path must be relative"
        )
    segments = [part for part in candidate.parts if part not in ("", ".")]
    if any(segment == ".." for segment in segments):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path must not contain .. segments",
        )
    return segments


def _list_subdirectories(root: Path, segments: list[str]) -> list[str]:
    """Walk from ``root`` through ``segments`` via O_NOFOLLOW dir_fd
    chaining, rejecting a symlink at any level, and return the sorted names
    of the target directory's direct (non-symlink) subdirectories.

    Raises ``FileNotFoundError`` if the root or any segment does not exist,
    is not a directory, or is a symlink -- callers translate that to 404.
    """
    fds: list[int] = []
    try:
        fds.append(os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
        for segment in segments:
            fds.append(
                os.open(
                    segment,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=fds[-1],
                )
            )
        directories: list[str] = []
        for name in os.listdir(fds[-1]):
            try:
                info = os.stat(name, dir_fd=fds[-1], follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(info.st_mode):
                directories.append(name)
        return sorted(directories)
    except OSError as exc:
        raise FileNotFoundError(str(root)) from exc
    finally:
        for fd in reversed(fds):
            os.close(fd)


def _location_store(app: FastAPI) -> ClipStorageLocationStore:
    store = getattr(app.state, "clip_storage_location_store", None)
    if not isinstance(store, ClipStorageLocationStore):
        store = ClipStorageLocationStore.from_env()
        app.state.clip_storage_location_store = store
    return store


def _authorize(request: Request, authorization: str | None) -> None:
    authorize_dashboard(request, legacy_token=_bearer_token(authorization))


def _bearer_token(value: str | None) -> str | None:
    if value is None or not value.startswith("Bearer "):
        return None
    return value.removeprefix("Bearer ").strip() or None


__all__ = ["router"]

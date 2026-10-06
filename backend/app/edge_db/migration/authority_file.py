"""Atomic owner-only persistence-authority file shared with the API root."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from uuid import UUID

from backend.app.edge_db.authority import AuthorityToken
from backend.app.edge_db.migration.errors import MigrationError


def read_authority_file(path: Path) -> AuthorityToken:
    """Read the token in the exact format the API root accepts."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise MigrationError("authority file is unreadable") from error
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or set(payload) != {"generation", "writer_token"}:
        raise MigrationError("authority file is malformed")
    try:
        return AuthorityToken(
            generation=payload["generation"], writer_token=UUID(str(payload["writer_token"]))
        )
    except (TypeError, ValueError) as error:
        raise MigrationError("authority file is malformed") from error


def stage_authority_file(path: Path, token: AuthorityToken) -> Path:
    """Write an fsynced 0600 temp beside the destination; publish after commit."""
    body = json.dumps(
        {"generation": token.generation, "writer_token": str(token.writer_token)},
        separators=(",", ":"),
    )
    temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write((body + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        discard(temp)
        raise
    return temp


def publish_authority_file(temp: Path, path: Path, *, replace: bool) -> None:
    """Publish a staged file; provisioning never overwrites, transfer always replaces."""
    if replace:
        os.replace(temp, path)
    else:
        try:
            os.link(temp, path)
        except FileExistsError as error:
            raise MigrationError("authority file already exists; refusing to overwrite") from error
        temp.unlink()
    fsync_directory(path.parent)


def discard(temp: Path) -> None:
    try:
        temp.unlink()
    except FileNotFoundError:
        return


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "discard",
    "fsync_directory",
    "publish_authority_file",
    "read_authority_file",
    "stage_authority_file",
]

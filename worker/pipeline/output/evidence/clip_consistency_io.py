"""Strict JSON and filesystem trust boundaries for clip maintenance."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    FaultHook,
)
from worker.pipeline.output.evidence.durability import fsync_directory

MAX_JSON_BYTES = 256 * 1024


def checkpoint(hook: FaultHook | None, stage: str) -> None:
    if hook is not None:
        hook(stage)


def validate_no_symlink_components(path: Path, *, allow_missing_leaf: bool = False) -> None:
    reject_lexical_parent_components(path)
    absolute = path.absolute()
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise ClipConsistencyError("unsafe_path", "path component is missing") from None
        if stat.S_ISLNK(info.st_mode):
            raise ClipConsistencyError("unsafe_path", "path contains a symlink component")


def validate_directory(
    path: Path,
    *,
    expected_uid: int,
    owner_controlled: bool,
    label: str,
) -> os.stat_result:
    validate_no_symlink_components(path)
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ClipConsistencyError("unsafe_path", f"{label} unavailable") from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != expected_uid:
        raise ClipConsistencyError("unsafe_path", f"{label} owner or type invalid")
    mode = stat.S_IMODE(info.st_mode)
    forbidden = 0o077 if owner_controlled else 0o022
    if mode & forbidden:
        raise ClipConsistencyError("unsafe_path", f"{label} mode is not secure")
    return info


def validate_regular(
    path: Path,
    *,
    expected_uid: int,
    exact_mode: int | None,
    label: str,
) -> os.stat_result:
    validate_no_symlink_components(path)
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ClipConsistencyError("unsafe_path", f"{label} unavailable") from exc
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != expected_uid:
        raise ClipConsistencyError("unsafe_path", f"{label} owner or type invalid")
    if (exact_mode is not None and mode != exact_mode) or (
        exact_mode is None and mode & 0o022
    ):
        raise ClipConsistencyError("unsafe_path", f"{label} mode is not secure")
    return info


def validate_under_root(path: Path, root: Path, *, allow_missing_leaf: bool) -> None:
    validate_no_symlink_components(root)
    validate_no_symlink_components(path, allow_missing_leaf=allow_missing_leaf)
    try:
        resolved_root = root.resolve(strict=True)
        if allow_missing_leaf and not path.exists():
            resolved_path = path.parent.resolve(strict=True) / path.name
        else:
            resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ClipConsistencyError("unsafe_path", "maintenance path escaped root") from exc


def read_strict_json(
    path: Path,
    *,
    expected_uid: int | None = None,
    exact_mode: int | None = None,
    max_bytes: int = MAX_JSON_BYTES,
    error_code: str,
) -> dict[str, object]:
    if expected_uid is not None:
        validate_regular(
            path,
            expected_uid=expected_uid,
            exact_mode=exact_mode,
            label="JSON authority",
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= max_bytes:
                raise OSError("JSON file shape invalid")
            raw = os.read(descriptor, max_bytes + 1)
        finally:
            os.close(descriptor)
        loaded: object = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ClipConsistencyError(error_code, "JSON authority is invalid") from exc
    if not isinstance(loaded, dict):
        raise ClipConsistencyError(error_code, "JSON authority must be an object")
    return cast("dict[str, object]", loaded)


def atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    root: Path,
    expected_uid: int,
    hook: FaultHook | None,
    stage: str,
) -> None:
    validate_directory(
        root,
        expected_uid=expected_uid,
        owner_controlled=True,
        label="maintenance root",
    )
    validate_under_root(path, root, allow_missing_leaf=True)
    validate_directory(
        path.parent,
        expected_uid=expected_uid,
        owner_controlled=True,
        label="maintenance directory",
    )
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = None
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            checkpoint(hook, f"{stage}:write")
            os.fsync(output.fileno())
            checkpoint(hook, f"{stage}:fsync_file")
        os.replace(temporary, path)
        checkpoint(hook, f"{stage}:replace")
        fsync_directory(path.parent)
        checkpoint(hook, f"{stage}:fsync_directory")
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def sha256_regular(path: Path) -> str:
    validate_no_symlink_components(path)
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def ensure_secure_subdirectory(root: Path, name: str, *, expected_uid: int) -> Path:
    path = root / name
    validate_under_root(path, root, allow_missing_leaf=True)
    path.mkdir(mode=0o700, exist_ok=True)
    validate_directory(
        path,
        expected_uid=expected_uid,
        owner_controlled=True,
        label="maintenance subdirectory",
    )
    return path


def reject_lexical_parent_components(path: Path) -> None:
    if ".." in path.parts:
        raise ClipConsistencyError("unsafe_path", "path contains a lexical parent component")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


__all__ = [
    "atomic_write_json",
    "checkpoint",
    "ensure_secure_subdirectory",
    "read_strict_json",
    "reject_lexical_parent_components",
    "sha256_regular",
    "validate_directory",
    "validate_no_symlink_components",
    "validate_regular",
    "validate_under_root",
]

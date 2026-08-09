"""Bounded JPEG validation and descriptor-relative durable publication."""

from __future__ import annotations

import errno
import os
import secrets
import stat
from contextlib import suppress
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from worker.adapters.encode.adapter_errors import ThumbnailSecurityError

THUMBNAIL_WIDTH: Final = 640
THUMBNAIL_HEIGHT: Final = 360
THUMBNAIL_FILENAME: Final = "thumbnail.jpg"
MAX_THUMBNAIL_BYTES: Final = 2 * 1024 * 1024
_JPEG_START: Final = b"\xff\xd8"
_JPEG_END: Final = b"\xff\xd9"
_READ_SIZE: Final = 64 * 1024
_SOF_MARKERS: Final = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def is_valid_thumbnail(path: Path) -> bool:
    payload = _read_bounded_regular_file(path)
    return payload is not None and is_valid_jpeg(payload)


def is_valid_jpeg(payload: bytes) -> bool:
    if (
        len(payload) > MAX_THUMBNAIL_BYTES
        or not payload.startswith(_JPEG_START)
        or not payload.endswith(_JPEG_END)
    ):
        return False
    if _jpeg_dimensions(payload) != (THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH):
        return False
    try:
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
    except cv2.error:
        return False
    return decoded is not None and decoded.shape[:2] == (
        THUMBNAIL_HEIGHT,
        THUMBNAIL_WIDTH,
    )


def _jpeg_dimensions(payload: bytes) -> tuple[int, int] | None:
    index = len(_JPEG_START)
    payload_length = len(payload)
    while index < payload_length:
        if payload[index] != 0xFF:
            return None
        while index < payload_length and payload[index] == 0xFF:
            index += 1
        if index >= payload_length:
            return None
        marker = payload[index]
        index += 1
        if marker == 0xDA:
            return None
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            continue
        if index + 2 > payload_length:
            return None
        segment_length = int.from_bytes(payload[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > payload_length:
            return None
        if marker in _SOF_MARKERS:
            if segment_length < 7:
                return None
            height = int.from_bytes(payload[index + 3 : index + 5], "big")
            width = int.from_bytes(payload[index + 5 : index + 7], "big")
            return height, width
        index += segment_length
    return None


def publish_thumbnail(payload: bytes, path: Path) -> None:
    directory_descriptor: int | None = None
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    temporary_created = False
    operation = "directory open"
    try:
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
        operation = "temporary create"
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CLOEXEC
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        temporary_created = True
        operation = "temporary write"
        _write_all(temporary_descriptor, payload)
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None
        operation = "atomic replace"
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        temporary_name = None
        operation = "directory fsync"
        os.fsync(directory_descriptor)
    except OSError as exc:
        raise ThumbnailSecurityError(operation, type(exc).__name__) from exc
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if (
            temporary_created
            and temporary_name is not None
            and directory_descriptor is not None
        ):
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=directory_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def fsync_existing_thumbnail(path: Path) -> None:
    operation = "existing file fsync"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        operation = "existing directory fsync"
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise ThumbnailSecurityError(operation, type(exc).__name__) from exc


def _read_bounded_regular_file(path: Path) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size <= 0
            or file_stat.st_size > MAX_THUMBNAIL_BYTES
        ):
            return None
        payload = bytearray()
        while len(payload) <= MAX_THUMBNAIL_BYTES:
            chunk = os.read(
                descriptor,
                min(_READ_SIZE, MAX_THUMBNAIL_BYTES + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) != file_stat.st_size:
            return None
        return bytes(payload)
    except OSError:
        return None
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            raise OSError(errno.EIO, "thumbnail temporary write made no progress")
        remaining = remaining[written:]


__all__ = [
    "MAX_THUMBNAIL_BYTES",
    "THUMBNAIL_FILENAME",
    "THUMBNAIL_HEIGHT",
    "THUMBNAIL_WIDTH",
    "fsync_existing_thumbnail",
    "is_valid_jpeg",
    "is_valid_thumbnail",
    "publish_thumbnail",
]

"""HTTP responses that stream one descriptor-pinned media file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from anyio.to_thread import run_sync
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from backend.app.features.clips.descriptor_files import OpenedRegularFile

_CHUNK_SIZE = 64 * 1024
_CACHE_CONTROL = "private, no-store"
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
}


class MalformedRangeHeader(ValueError):
    """The Range header is not one supported byte range."""


class UnsatisfiableRange(ValueError):
    """The requested byte range does not overlap the opened file."""


@dataclass(frozen=True, slots=True)
class ByteRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@final
class OpenedFileResponse(Response):
    def __init__(
        self,
        opened: OpenedRegularFile,
        byte_range: ByteRange,
        *,
        media_type: str,
        partial: bool,
    ) -> None:
        self._opened = opened
        self._byte_range = byte_range
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": _CACHE_CONTROL,
            "Content-Length": str(byte_range.length),
        }
        if partial:
            headers["Content-Range"] = (
                f"bytes {byte_range.start}-{byte_range.end}/{opened.size_bytes}"
            )
        super().__init__(
            content=b"",
            status_code=206 if partial else 200,
            headers=headers,
            media_type=media_type,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        del receive
        if scope["type"] == "http" and scope["method"].upper() == "HEAD":
            # RFC 9110 section 9.3.2: same header section as the GET, no body.
            # A clip is whole-file evidence, so reading it to throw the bytes
            # away would make a HEAD as expensive as playback -- exactly what a
            # player issues one to avoid. Close the pinned descriptor and send
            # the headers the GET path already computed.
            self._opened.handle.close()
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        with self._opened.handle as source:
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            _ = await run_sync(source.seek, self._byte_range.start)
            remaining = self._byte_range.length
            while remaining > 0:
                chunk = await run_sync(
                    source.read,
                    min(_CHUNK_SIZE, remaining),
                )
                if not chunk:
                    break
                remaining -= len(chunk)
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": True,
                    }
                )
            await send({"type": "http.response.body", "body": b"", "more_body": False})


def media_response(
    opened: OpenedRegularFile,
    range_header: str | None,
    media_type: str,
) -> Response:
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": _CACHE_CONTROL,
    }
    try:
        byte_range = _parse_range(range_header, opened.size_bytes)
    except MalformedRangeHeader:
        opened.handle.close()
        return Response(status_code=400, headers=headers)
    except UnsatisfiableRange:
        opened.handle.close()
        headers["Content-Range"] = f"bytes */{opened.size_bytes}"
        return Response(status_code=416, headers=headers)
    return OpenedFileResponse(
        opened,
        byte_range,
        media_type=media_type,
        partial=range_header is not None,
    )


def media_type(filename: str) -> str:
    suffix = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    return _MEDIA_TYPES.get(f".{suffix}", "application/octet-stream")


def _parse_range(value: str | None, size_bytes: int) -> ByteRange:
    if value is None:
        return ByteRange(0, size_bytes - 1)
    if not value.startswith("bytes=") or "," in value:
        raise MalformedRangeHeader
    range_value = value.removeprefix("bytes=").strip()
    start_text, separator, end_text = range_value.partition("-")
    if separator != "-" or not (start_text or end_text):
        raise MalformedRangeHeader
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0 or size_bytes == 0:
                raise UnsatisfiableRange
            start = max(0, size_bytes - suffix_length)
            return ByteRange(start, size_bytes - 1)
        start = int(start_text)
        end = size_bytes - 1 if not end_text else int(end_text)
    except ValueError as exc:
        raise MalformedRangeHeader from exc
    if start < 0 or end < start or start >= size_bytes:
        raise UnsatisfiableRange
    return ByteRange(start, min(end, size_bytes - 1))


__all__ = ["OpenedFileResponse", "media_response", "media_type"]

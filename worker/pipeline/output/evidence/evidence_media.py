"""Strict finalized MP4 inspection from one immutable file descriptor."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode


@dataclass(slots=True)
class ClipEvidenceError(Exception):
    reason_code: EvidenceReasonCode
    detail: str

    def __str__(self) -> str:
        return f"{self.reason_code}: {self.detail}"


@dataclass(frozen=True, slots=True)
class MediaFacts:
    sha256: str
    size_bytes: int
    duration_ms: int
    video_codec: str = "h264"
    audio_codec: str | None = None


def inspect_finalized_media(path: Path, *, ffprobe_bin: str = "ffprobe") -> MediaFacts:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "media cannot be opened") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "media is not a nonempty file")
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise ClipEvidenceError(
                EvidenceReasonCode.FINALIZE_FAILED,
                "media fsync failed",
            ) from exc
        if not _has_faststart(descriptor, info.st_size):
            raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "MP4 is not faststart")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        measured = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            measured += len(chunk)
        if measured != info.st_size:
            raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "media size changed while hashing")
        os.lseek(descriptor, 0, os.SEEK_SET)
        duration_ms, video_codec, audio_codec = _probe_media(descriptor, ffprobe_bin)
    finally:
        os.close(descriptor)
    return MediaFacts(
        digest.hexdigest(),
        measured,
        duration_ms,
        video_codec,
        audio_codec,
    )


def _media_length_ms(payload: dict[str, object], streams: object) -> int:
    """How long the media actually runs, in milliseconds.

    ``format.duration`` is not the answer. RTSP PTS carry the camera's uptime
    origin, and for a clip cut 8237 seconds into such a stream ffprobe reported
    ``format.duration=8271`` -- the distance from timeline zero, not the 34
    seconds of footage. Measured against the ceiling below, that marked every
    clip this deployment produced CORRUPT, which suppressed the manifest, which
    kept every playable clip out of the catalogue: the operator saw nothing
    while good footage sat on disk.

    Subtracting ``start_time`` is not the answer either, because the value is
    not consistently origin-inclusive -- a file written with an output
    timestamp offset reports ``duration=34`` alongside ``start_time=8237``, and
    the subtraction goes negative.

    The video stream's own duration is unambiguous in both cases, so it wins
    whenever ffprobe reports one.
    """
    if isinstance(streams, list):
        for stream in streams:
            if not isinstance(stream, dict) or stream.get("codec_type") != "video":
                continue
            raw = stream.get("duration")
            if raw is None:
                continue
            seconds = float(raw)
            if seconds > 0:
                return round(seconds * 1000)
    container = payload["format"]
    if not isinstance(container, dict):
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "ffprobe metadata invalid")
    total = float(container["duration"])
    origin = float(container.get("start_time") or 0.0)
    remainder = total - origin
    return round((remainder if remainder > 0 else total) * 1000)


def _probe_media(descriptor: int, ffprobe_bin: str) -> tuple[int, str, str | None]:
    descriptor_root = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,pix_fmt,duration:format=duration,start_time",
        "-of",
        "json",
        f"{descriptor_root}/{descriptor}",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            pass_fds=(descriptor,),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ClipEvidenceError(
            EvidenceReasonCode.FINALIZE_FAILED,
            "ffprobe unavailable",
        ) from exc
    if completed.returncode != 0:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "ffprobe rejected media")
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        duration_ms = _media_length_ms(payload, streams)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "ffprobe metadata invalid") from exc
    if not isinstance(streams, list) or not all(isinstance(stream, dict) for stream in streams):
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "ffprobe streams invalid")
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video_streams) != 1 or len(audio_streams) > 1:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "media stream count invalid")
    video_codec = video_streams[0].get("codec_name")
    audio_codec = audio_streams[0].get("codec_name") if audio_streams else None
    if not isinstance(video_codec, str) or not video_codec:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "media video codec invalid")
    if audio_codec is not None and (not isinstance(audio_codec, str) or not audio_codec):
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "media audio codec invalid")
    if not 1 <= duration_ms <= 120_000:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "media duration invalid")
    return duration_ms, video_codec, audio_codec


def _has_faststart(descriptor: int, size_bytes: int) -> bool:
    offset = 0
    moov_offset: int | None = None
    mdat_offset: int | None = None
    while offset + 8 <= size_bytes:
        os.lseek(descriptor, offset, os.SEEK_SET)
        header = os.read(descriptor, 16)
        if len(header) < 8:
            return False
        box_size = int.from_bytes(header[:4], "big")
        box_type = header[4:8]
        header_size = 8
        if box_size == 1:
            if len(header) < 16:
                return False
            box_size = int.from_bytes(header[8:16], "big")
            header_size = 16
        elif box_size == 0:
            box_size = size_bytes - offset
        if box_size < header_size or offset + box_size > size_bytes:
            return False
        if box_type == b"moov":
            moov_offset = offset
        elif box_type == b"mdat":
            mdat_offset = offset
        if moov_offset is not None and mdat_offset is not None:
            return moov_offset < mdat_offset
        offset += box_size
    return False


__all__ = ["ClipEvidenceError", "MediaFacts", "inspect_finalized_media"]

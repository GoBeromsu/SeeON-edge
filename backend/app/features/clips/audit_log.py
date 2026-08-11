"""API-owned clip audit log and best-effort backend backup."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.features.clips.store import (
    API_LABEL_STORE_ENV,
    default_label_store_dir,
    is_valid_clip_id,
)

logger = logging.getLogger(__name__)

API_AUDIT_LOG_ENV = "API_AUDIT_LOG"
API_AUDIT_LOG_MAX_BYTES_ENV = "API_AUDIT_LOG_MAX_BYTES"
API_AUDIT_ARCHIVE_RETENTION_DAYS_ENV = "API_AUDIT_ARCHIVE_RETENTION_DAYS"
API_BACKEND_CLIP_EVENTS_URL_ENV = "API_BACKEND_CLIP_EVENTS_URL"
DEFAULT_BACKEND_TIMEOUT_SEC = 0.5

# Rotate the live audit.jsonl once it passes this size so a single file never
# grows unbounded on a long-running edge box. Audit lines are ~120-160 bytes
# (ts + actor + action + clip_id), so 10 MiB holds roughly 70k-90k entries
# between rotations -- generous for one facility's play/label/list/audit-view
# traffic.
DEFAULT_AUDIT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB

# Archived audit files are kept at least as long as the clip evidence
# retention floor (see worker/pipeline/output/evidence/clip_config.py
# MIN_RETENTION_DAYS=60) so audit history always outlives the clips it
# references. Both the default and the floor are 60 days; an env override
# below the floor is clamped up to it.
MIN_AUDIT_ARCHIVE_RETENTION_DAYS = 60
DEFAULT_AUDIT_ARCHIVE_RETENTION_DAYS = MIN_AUDIT_ARCHIVE_RETENTION_DAYS

# Sentinel clip_id for audit actions that are not scoped to one clip (e.g.
# listing the catalog or viewing the audit log itself). Passes
# ``is_valid_clip_id`` so it round-trips through the same append() path as
# clip-scoped actions.
AUDIT_NO_CLIP_ID = "-"


def _configured_audit_log_max_bytes() -> int:
    raw = os.environ.get(API_AUDIT_LOG_MAX_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_AUDIT_LOG_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_AUDIT_LOG_MAX_BYTES
    return value if value > 0 else DEFAULT_AUDIT_LOG_MAX_BYTES


def _configured_audit_archive_retention_days() -> int:
    raw = os.environ.get(API_AUDIT_ARCHIVE_RETENTION_DAYS_ENV, "").strip()
    if not raw:
        return DEFAULT_AUDIT_ARCHIVE_RETENTION_DAYS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_AUDIT_ARCHIVE_RETENTION_DAYS
    return max(MIN_AUDIT_ARCHIVE_RETENTION_DAYS, value)


class AuditLogStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @classmethod
    def from_env(cls) -> AuditLogStore:
        configured_path = os.environ.get(API_AUDIT_LOG_ENV)
        if configured_path:
            return cls(configured_path)
        label_root = os.environ.get(API_LABEL_STORE_ENV)
        root = Path(label_root) if label_root else default_label_store_dir()
        return cls(root / "audit.jsonl")

    def append(
        self,
        *,
        actor: str,
        action: str,
        clip_id: str,
        backend_token: str | None = None,
    ) -> dict[str, object]:
        """Record ``entry``, best-effort.

        A read action (``list``, ``audit-view``) must not 500 just because
        the audit trail itself couldn't be written (issue #152) -- and
        crashing a write action (``label``) *after* its primary write already
        landed durably would misreport a save that actually succeeded as a
        failed request. So an unwritable audit log degrades the same way
        ``LabelStore.save`` and ``RuntimeStatusStore`` already do: logged, not
        raised. The best-effort backend backup below still runs regardless,
        giving the entry a second (remote) chance to land even when the local
        write failed.
        """
        if not is_valid_clip_id(clip_id):
            raise ValueError("invalid clip_id")
        entry = {
            "ts": utc_now_iso(),
            "actor": actor,
            "action": action,
            "clip_id": clip_id,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError as exc:
            logger.warning("clip audit log unavailable at %s: %s", self.path, exc)
        post_backend_backup("clip_audit", entry, backend_token=backend_token)
        return entry

    def _rotate_if_needed(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size < _configured_audit_log_max_bytes():
            return
        self._rotate()

    def _rotate(self) -> None:
        """Atomically move the live audit file to a timestamped archive.

        ``os.replace`` is an atomic rename on the same filesystem, so a
        concurrent reader either sees the full pre-rotation file at its old
        path or the (already-renamed) archive -- never a half-written file.
        The next ``append()`` call re-creates ``self.path`` from scratch.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archive_path = self.path.with_name(f"{self.path.stem}-{timestamp}.jsonl")
        suffix = 0
        while archive_path.exists():
            suffix += 1
            archive_path = self.path.with_name(f"{self.path.stem}-{timestamp}-{suffix}.jsonl")
        try:
            os.replace(self.path, archive_path)
        except OSError as exc:  # noqa: BLE001 - rotation is best-effort
            print(f"clip audit log rotation failed: {exc}", file=sys.stderr)
            return
        self._prune_archives()

    def _prune_archives(self) -> None:
        retention_seconds = _configured_audit_archive_retention_days() * 86400
        cutoff = time.time() - retention_seconds
        for archive in self.path.parent.glob(f"{self.path.stem}-*.jsonl"):
            try:
                mtime = archive.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                continue
            try:
                archive.unlink()
            except OSError as exc:  # noqa: BLE001 - pruning is best-effort
                print(f"clip audit archive prune failed for {archive}: {exc}", file=sys.stderr)

    def list_entries(self) -> list[dict[str, object]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        entries: list[dict[str, object]] = []
        for line in lines:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
        return entries


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def post_backend_backup(
    event_type: str, payload: dict[str, Any], *, backend_token: str | None = None
) -> bool:
    url = os.environ.get(API_BACKEND_CLIP_EVENTS_URL_ENV)
    if not url or backend_token is None:
        return False
    envelope = {"type": event_type, "payload": payload}
    body = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    request.add_header("Authorization", f"Bearer {backend_token}")
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_BACKEND_TIMEOUT_SEC) as response:
            status = int(getattr(response, "status", 200))
    except Exception as exc:  # noqa: BLE001 - backend backup is explicitly best-effort
        print(f"clip backend backup failed: {exc}", file=sys.stderr)
        return False
    else:
        return 200 <= status < 300


__all__ = [
    "API_AUDIT_ARCHIVE_RETENTION_DAYS_ENV",
    "API_AUDIT_LOG_ENV",
    "API_AUDIT_LOG_MAX_BYTES_ENV",
    "API_BACKEND_CLIP_EVENTS_URL_ENV",
    "AUDIT_NO_CLIP_ID",
    "AuditLogStore",
    "DEFAULT_AUDIT_ARCHIVE_RETENTION_DAYS",
    "DEFAULT_AUDIT_LOG_MAX_BYTES",
    "MIN_AUDIT_ARCHIVE_RETENTION_DAYS",
    "post_backend_backup",
    "utc_now_iso",
]

"""Persisted dashboard login credentials (scrypt-hashed, atomic-write JSON).

Separate from the in-memory ``DashboardSessionStore`` in ``dashboard_auth.py``:
this module only knows how to durably store and verify one username/password
pair on disk. ``dashboard_auth.py`` decides *when* to consult it (persisted
file wins over env vars, which win over the built-in ``admin``/``admin``
default).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from backend.app.shared.state_dir import resolve_state_dir

_ALGORITHM_SCRYPT = "scrypt"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64
_SALT_BYTES = 16


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )


@dataclass(frozen=True, slots=True)
class PersistedDashboardCredentials:
    username: str
    algorithm: str
    salt: bytes
    password_hash: bytes
    updated_at: str

    def verify_password(self, password: str) -> bool:
        if self.algorithm != _ALGORITHM_SCRYPT:
            return False
        candidate = _hash_password(password, self.salt)
        return hmac.compare_digest(candidate, self.password_hash)


class DashboardCredentialsStore:
    """Atomic JSON-backed persistence for dashboard login credentials.

    Reuses the tmp-file + ``os.replace`` + fsync + chmod 0600 atomic-write
    pattern used by ``CameraRegistryStore`` (see
    ``backend/app/features/cameras/store.py``). A corrupt or unreadable file
    is logged to stderr and treated as "no persisted credentials" -- it must
    never crash boot or brick login (the operator can always fall back to
    env vars or the built-in default by deleting the file).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    @classmethod
    def from_env(cls) -> DashboardCredentialsStore:
        return cls(resolve_state_dir("ml-api") / "dashboard_credentials.json")

    def load(self) -> PersistedDashboardCredentials | None:
        with self._lock:
            return self._load_unlocked()

    def save(self, *, username: str, password: str) -> PersistedDashboardCredentials:
        salt = os.urandom(_SALT_BYTES)
        record = PersistedDashboardCredentials(
            username=username,
            algorithm=_ALGORITHM_SCRYPT,
            salt=salt,
            password_hash=_hash_password(password, salt),
            updated_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        )
        with self._lock:
            self._write_unlocked(record)
        return record

    def _load_unlocked(self) -> PersistedDashboardCredentials | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 - never brick login on a corrupt file
            print(
                f"dashboard credentials store unreadable at {self.path}, "
                f"falling back to env/default: {exc!r}",
                file=sys.stderr,
            )
            return None
        try:
            return _parse_persisted(raw)
        except Exception as exc:  # noqa: BLE001 - same fallback guarantee as above
            print(
                f"dashboard credentials store malformed at {self.path}, "
                f"falling back to env/default: {exc!r}",
                file=sys.stderr,
            )
            return None

    def _write_unlocked(self, record: PersistedDashboardCredentials) -> None:
        payload = {
            "version": 1,
            "username": record.username,
            "algorithm": record.algorithm,
            "salt": base64.b64encode(record.salt).decode("ascii"),
            "password_hash": base64.b64encode(record.password_hash).decode("ascii"),
            "updated_at": record.updated_at,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        os.replace(tmp_path, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        try:
            directory_fd = os.open(self.path.parent, os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)


def _parse_persisted(raw: object) -> PersistedDashboardCredentials | None:
    if not isinstance(raw, dict):
        return None
    username = raw.get("username")
    algorithm = raw.get("algorithm")
    salt_b64 = raw.get("salt")
    hash_b64 = raw.get("password_hash")
    updated_at = raw.get("updated_at")
    if not (
        isinstance(username, str)
        and username
        and isinstance(algorithm, str)
        and isinstance(salt_b64, str)
        and isinstance(hash_b64, str)
        and isinstance(updated_at, str)
    ):
        return None
    return PersistedDashboardCredentials(
        username=username,
        algorithm=algorithm,
        salt=base64.b64decode(salt_b64, validate=True),
        password_hash=base64.b64decode(hash_b64, validate=True),
        updated_at=updated_at,
    )


__all__ = [
    "DashboardCredentialsStore",
    "PersistedDashboardCredentials",
]

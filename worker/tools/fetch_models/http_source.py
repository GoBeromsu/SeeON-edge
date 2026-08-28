"""Single-attempt HTTP byte source plus the retry policy the fetcher applies.

Retry policy mirrors the retired ``scripts/fetch-models.sh`` (issue #188):
only 429/408/5xx and transport errors are retried, ``Retry-After`` wins over
the local backoff, and each wait carries jitter so two jobs on the same
commit do not retry in lockstep. A 404 means the pin is wrong and fails fast.

The source itself never retries: a failure part-way through a body must
restart the whole file (hash and temp file included), and only the fetcher
owns those. It raises ``RetryableSourceError`` so the fetcher can decide.
"""

from __future__ import annotations

import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

DEFAULT_ATTEMPTS_ENV: Final = "ML_WORKER_FETCH_MODELS_ATTEMPTS"
DEFAULT_ATTEMPTS: Final = 6
MAX_BACKOFF_SEC: Final = 120.0
CHUNK_BYTES: Final = 1 << 20
_RETRYABLE_STATUSES: Final = frozenset({408, 429})


class SourceError(RuntimeError):
    """A download failed for good; the destination is left untouched."""


class RetryableSourceError(SourceError):
    """A download failed in a way that may succeed on a later attempt."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ByteSource(Protocol):
    def stream(self, url: str, headers: Mapping[str, str]) -> Iterator[bytes]:
        """Yield the body of ``url`` once; raise ``SourceError`` on failure."""


def _is_retryable(status: int) -> bool:
    return status in _RETRYABLE_STATUSES or 500 <= status <= 599


def _retry_after_seconds(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


def attempts_from_env(env: Mapping[str, str] = os.environ) -> int:
    raw = env.get(DEFAULT_ATTEMPTS_ENV, "").strip()
    if not raw:
        return DEFAULT_ATTEMPTS
    try:
        value = int(raw)
    except ValueError as exc:
        raise SourceError(
            f"{DEFAULT_ATTEMPTS_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise SourceError(f"{DEFAULT_ATTEMPTS_ENV} must be a positive integer, got {raw!r}")
    return value


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = DEFAULT_ATTEMPTS
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def wait_seconds(self, attempt: int, retry_after: float | None) -> float:
        """Seconds to wait after failed 1-based ``attempt`` before the next one."""
        if retry_after is not None:
            return min(retry_after, MAX_BACKOFF_SEC)
        return min(float(2 ** (attempt + 1)) + self.rng.uniform(0.0, 5.0), MAX_BACKOFF_SEC)


class UrllibSource:
    """Single-attempt ``ByteSource`` over ``urllib``."""

    def __init__(self, *, connect_timeout_sec: float = 20.0) -> None:
        self._timeout = connect_timeout_sec

    def stream(self, url: str, headers: Mapping[str, str]) -> Iterator[bytes]:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RetryableSourceError(f"{url} returned HTTP {status}")
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        return
                    yield chunk
        except urllib.error.HTTPError as exc:
            if _is_retryable(exc.code):
                raise RetryableSourceError(
                    f"{url} returned HTTP {exc.code}",
                    retry_after=_retry_after_seconds(exc.headers.get("Retry-After")),
                ) from exc
            raise SourceError(f"{url} returned HTTP {exc.code}; not retryable") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RetryableSourceError(f"{url} transport error: {exc}") from exc

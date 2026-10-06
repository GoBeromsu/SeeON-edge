"""Explicitly scheduled native audit publication; no HTTP or lifespan ownership.

Only a normal owned return authorizes publication. The runtime tracks only its
own publication work and never certifies external transaction drain. Before
session closure or shared pool shutdown, callers must drain all outer borrowed
transactions, including rollback and pool exit after failed or cancelled audit
callbacks.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Protocol, TypeVar

import psycopg

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken
from backend.app.edge_db.postgres import (
    CommitOutcomeUnknown,
    PostgresDatabase,
    PostgresError,
    PostgresPoolBusy,
    PostgresStartupError,
    PostgresTransactionStateError,
    PostgresUnavailable,
)
from backend.app.features.audit import postgres_sessions
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.audit.postgres_verification import PostgresAuditCheckpoint
from backend.app.features.audit.sessions import AuditSession
from backend.app.features.audit.store import AuditEvent, AuditRecord
from backend.app.features.audit.verification import AuditVerificationError

_LOGGER = logging.getLogger(__name__)
_Result = TypeVar("_Result")


class AuditMutationOwner(Protocol):
    @property
    def database(self) -> PostgresDatabase: ...

    @property
    def authority(self) -> AuthorityToken: ...


_FAILURE_CODES = (
    (CommitOutcomeUnknown, "commit_outcome_unknown"),
    (AuthorityFenced, "authority_fenced"),
    (PostgresPoolBusy, "database_busy"),
    (PostgresStartupError, "database_startup_failed"),
    (PostgresTransactionStateError, "transaction_state"),
    (PostgresUnavailable, "database_unavailable"),
    (AuditVerificationError, "verification_failed"),
    (PostgresError, "database_error"),
    (psycopg.Error, "database_error"),
    (OSError, "database_unavailable"),
)


class AuditRuntimeUnavailable(RuntimeError):
    """No governed admission or successful publication is available."""


def _always_audit(_result: object) -> bool:
    return True


def _validate_mutation_expectation(expected: bool, has_publication: bool) -> None:
    if type(expected) is not bool:
        raise AuditRuntimeUnavailable("mutation audit expectation is invalid")
    if expected and not has_publication:
        raise AuditRuntimeUnavailable("mutation audit callback was not called")
    if not expected and has_publication:
        raise AuditRuntimeUnavailable("mutation audit callback was unexpected")


@dataclass(frozen=True, slots=True)
class AuditMutation:
    """Carry admission and lazy event creation to the actual transaction owner."""

    runtime: PostgresAuditRuntime
    event_factory: Callable[[], AuditEvent]

    def __post_init__(self) -> None:
        self.require_admission()

    def require_admission(self, owner: AuditMutationOwner | None = None) -> None:
        """Recheck before external effects after any intervening wait."""
        self.runtime.require_mutation_admission(owner)

    def apply(
        self,
        owner: AuditMutationOwner,
        write: Callable[[Callable[[psycopg.Connection], None]], _Result],
        *,
        expects_audit: Callable[[_Result], bool] = _always_audit,
    ) -> _Result:
        return self.runtime.apply_mutation(
            owner, self.event_factory, write, expects_audit=expects_audit
        )


class InvalidAuditPublication(ValueError):
    """The receipt is not live publication work for this runtime and session."""


def _sanitized_failure(error: Exception) -> Exception:
    """Return a redacted ordinary failure, preserving typed commit uncertainty."""
    if isinstance(error, CommitOutcomeUnknown):
        return CommitOutcomeUnknown()
    return AuditRuntimeUnavailable("audit runtime unavailable")


@dataclass(frozen=True, slots=True)
class AuditRuntimeStatus:
    ready: bool
    verification_current: bool
    eligible_to_attempt: bool
    session_established: bool
    failure_code: str | None
    indeterminate: bool
    stopping: bool


class PendingAuditPublication:
    """Opaque, single-use receipt; possession does not establish COMMIT."""

    __slots__ = ("_consumed", "_owner", "_revision", "_session")

    def __init__(
        self,
        owner: PostgresAuditRuntime,
        session: AuditSession,
        revision: object,
    ) -> None:
        self._consumed = False
        self._owner, self._session, self._revision = owner, session, revision

    def __copy__(self) -> PendingAuditPublication:
        return self

    def __deepcopy__(self, memo: dict) -> PendingAuditPublication:
        return self

    def validate(self, owner: PostgresAuditRuntime, session: AuditSession | None) -> None:
        if self._owner is not owner or self._session is not session or self._consumed:
            raise InvalidAuditPublication("invalid audit publication token") from None

    def consume(self, owner: PostgresAuditRuntime, session: AuditSession | None) -> object:
        """Validate one use by this owner and session, then return only the revision."""
        self.validate(owner, session)
        self._consumed = True
        return self._revision


class PostgresAuditRuntime:
    def __init__(
        self,
        store: PostgresAuditStore,
        *,
        maximum_snapshot_age_sec: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if (
            type(maximum_snapshot_age_sec) not in (int, float)
            or not 0 < maximum_snapshot_age_sec < math.inf
        ):
            raise ValueError("maximum_snapshot_age_sec must be finite and positive") from None
        self._store, self._maximum_age, self._clock = store, maximum_snapshot_age_sec, clock
        self._lock = Lock()
        self._revision = object()
        self._verification: tuple[PostgresAuditCheckpoint, float, object] | None = None
        self._session: AuditSession | None = None
        self._failure_code: str | None = None
        self._indeterminate = self._stopping = False
        self._scanning = self._starting = False
        self._start_attempted = self._close_attempted = False
        self._pending: set[PendingAuditPublication] = set()

    @property
    def database(self) -> PostgresDatabase:
        return self._store.database

    @property
    def authority(self) -> AuthorityToken:
        return self._store.authority

    def _current(self) -> bool:
        evidence = self._verification
        return (
            not self._stopping
            and not self._indeterminate
            and evidence is not None
            and evidence[2] is self._revision
            and 0 <= self._clock() - evidence[1] < self._maximum_age
        )

    def snapshot(self) -> AuditRuntimeStatus:
        with self._lock:
            current = self._current()
            established = self._session is not None
            eligible = current and established
            return AuditRuntimeStatus(
                ready=eligible and self._failure_code is None,
                verification_current=current,
                eligible_to_attempt=eligible,
                session_established=established,
                failure_code=self._failure_code,
                indeterminate=self._indeterminate,
                stopping=self._stopping,
            )

    def _fail(self, error: BaseException) -> None:
        self._revision = object()
        self._indeterminate |= isinstance(error, CommitOutcomeUnknown)
        self._failure_code = (
            "commit_outcome_unknown"
            if self._indeterminate
            else next(
                (code for kind, code in _FAILURE_CODES if isinstance(error, kind)),
                "operation_failed",
            )
        )

    def record_failure(self, error: BaseException) -> None:
        """Record fail-open degradation without retaining exception data or doing I/O."""
        with self._lock:
            self._fail(error)

    def verify_once(self) -> bool:
        """Publish one scan, or return False on contention/invalidation; age starts here."""
        with self._lock:
            if self._scanning or self._stopping or self._indeterminate:
                return False
            checkpoint = self._verification[0] if self._verification else None
            revision, started = self._revision, self._clock()
            self._scanning = True
        try:
            candidate = self._store.verify(checkpoint)
        except BaseException as error:
            with self._lock:
                self._scanning = False
                self._fail(error)
            if not isinstance(error, Exception):
                raise error from None
            raise _sanitized_failure(error) from None
        with self._lock:
            self._scanning = False
            if revision is not self._revision or self._stopping or self._indeterminate:
                return False
            # Even an accepted publication may already be too old for admission.
            self._verification = candidate, started, revision
            return True

    def start_session_once(self) -> bool:
        """Attempt establishment once, only with current evidence; never infer a lost result."""
        with self._lock:
            if self._start_attempted or not self._current():
                return False
            self._start_attempted = self._starting = True
        try:
            session = postgres_sessions.start_session(self._store)
        except BaseException as error:
            with self._lock:
                self._starting = False
                self._fail(error)
            if not isinstance(error, Exception):
                raise error from None
            raise _sanitized_failure(error) from None
        with self._lock:
            self._starting = False
            # Retain known COMMIT for closure, even after stop or a newer failure.
            self._session = session
            return True

    def _append(
        self, event: AuditEvent, connection: psycopg.Connection | None
    ) -> tuple[PendingAuditPublication, AuditRecord]:
        with self._lock:
            session = self._session
            if not self._current() or session is None:
                raise AuditRuntimeUnavailable("audit runtime unavailable") from None
            failure_code = self._failure_code
            token = PendingAuditPublication(self, session, self._revision)
            self._pending.add(token)
        try:
            if failure_code is None:
                record = self._store.append(event, connection=connection)
            else:
                record = postgres_sessions.append_with_recovery(
                    self._store, event, session, failure_code, connection
                )
        except BaseException as error:
            self.publish_failed(token, error)
            if not isinstance(error, Exception):
                raise error from None
            raise _sanitized_failure(error) from None
        return token, record

    def append_owned(self, event: AuditEvent) -> AuditRecord:
        token, record = self._append(event, None)
        self.publish_committed(token)
        return record

    def append_borrowed(
        self, connection: psycopg.Connection, event: AuditEvent
    ) -> PendingAuditPublication:
        """Return tentative publication; any failure escapes to the outer transaction.

        Failure or cancellation clears local pending work before the outer
        owner completes rollback and pool exit. Callers must drain all outer
        borrowed transactions, including failed or cancelled audit callbacks
        and their rollback and pool exit, before session closure or shared pool
        shutdown. Local publication accounting never certifies external drain.
        """
        if connection is None:
            raise ValueError("borrowed audit append requires a connection") from None
        return self._append(event, connection)[0]

    def require_mutation_admission(self, owner: AuditMutationOwner | None = None) -> None:
        if owner is not None and (
            owner.database is not self.database or owner.authority != self.authority
        ):
            raise ValueError("mutation and audit must share database and authority")
        if not self.snapshot().eligible_to_attempt:
            raise AuditRuntimeUnavailable("audit runtime unavailable")

    def apply_mutation(
        self,
        owner: AuditMutationOwner,
        event_factory: Callable[[], AuditEvent],
        write: Callable[[Callable[[psycopg.Connection], None]], _Result],
        *,
        expects_audit: Callable[[_Result], bool] = _always_audit,
    ) -> _Result:
        """Publish after complete owner return, checking its declared callback contract.

        Required-once is the default. Optional CRUD owners explicitly classify
        their result: False means no callback, True means exactly one. Owners
        must propagate callback exceptions and complete rollback/pool exit before
        raising. Event construction happens only inside the callback.

        A contract failure discovered after owner return cannot undo its COMMIT.
        This wrapper neither owns nor certifies external drain. Outbox admission
        retains its specialized duplicate protocol. Compound services must apply
        this at their actual write-owner boundary, not around their whole command.
        """
        self.require_mutation_admission(owner)
        pending: PendingAuditPublication | None = None
        owner_returned = False

        def append(connection: psycopg.Connection) -> None:
            nonlocal pending
            if pending is not None:
                raise AuditRuntimeUnavailable("mutation audit callback was repeated")
            try:
                event = event_factory()
            except BaseException as error:
                self.record_failure(error)
                if not isinstance(error, Exception):
                    raise
                raise AuditRuntimeUnavailable("mutation audit event is invalid") from None
            candidate = self.append_borrowed(connection, event)
            try:
                self.validate_publication(candidate)
            except InvalidAuditPublication as error:
                self.record_failure(error)
                raise AuditRuntimeUnavailable("mutation audit publication is invalid") from None
            pending = candidate

        try:
            result = write(append)
            owner_returned = True
            _validate_mutation_expectation(expects_audit(result), pending is not None)
        except BaseException as error:
            if pending is not None:
                self._publish_mutation_failure(pending, error)
            elif owner_returned or isinstance(
                error, (PostgresError, psycopg.Error, OSError, AuditRuntimeUnavailable)
            ):
                self.record_failure(error)
            raise
        if pending is None:
            return result
        try:
            self.publish_committed(pending)
        except InvalidAuditPublication as error:
            self.record_failure(error)
            raise AuditRuntimeUnavailable("mutation audit publication is invalid") from None
        return result

    def _publish_mutation_failure(
        self, pending: PendingAuditPublication, error: BaseException
    ) -> None:
        try:
            self.publish_failed(pending, error)
        except InvalidAuditPublication:
            self.record_failure(error)
            _log_mutation_accounting_failure()

    def _validate_publication(self, token: PendingAuditPublication) -> None:
        if type(token) is not PendingAuditPublication or token not in self._pending:
            raise InvalidAuditPublication("invalid audit publication token") from None
        token.validate(self, self._session)

    def validate_publication(self, token: PendingAuditPublication) -> None:
        """Validate live ownership before COMMIT without consuming or re-admitting.

        A stale revision is still valid work; it simply cannot heal readiness
        when later published. Publication work is not an external lease count.
        """
        with self._lock:
            self._validate_publication(token)

    def _consume(self, token: PendingAuditPublication) -> object:
        self._validate_publication(token)
        revision = token.consume(self, self._session)
        self._pending.remove(token)
        return revision

    def publish_committed(self, token: PendingAuditPublication) -> bool:
        """Caller promises its database.transact returned normally, including pool exit."""
        with self._lock:
            revision = self._consume(token)
            if revision is not self._revision or self._stopping or self._indeterminate:
                return False
            self._failure_code = None
            return True

    def publish_failed(self, token: PendingAuditPublication, error: BaseException) -> None:
        """Consume after rollback/error/unknown outcome; never interpret failure as rollback."""
        with self._lock:
            self._consume(token)
            self._fail(error)

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            self._revision = object()

    def close_session_once(self) -> bool:
        """Close only after local drain; unknown outcomes require external recovery instead.

        Before session closure or shared pool shutdown, callers must drain all
        outer borrowed transactions, including failed or cancelled audit
        callbacks and their rollback and pool exit. These guards track only
        local publication work and never certify external transaction drain.
        """
        with self._lock:
            if (
                not self._stopping
                or self._indeterminate
                or self._close_attempted
                or self._scanning
                or self._starting
                or self._pending
                or self._session is None
            ):
                return False
            self._close_attempted = True
            session = self._session
        try:
            postgres_sessions.close_session(self._store, session)
        except BaseException as error:
            self.record_failure(error)
            if not isinstance(error, Exception):
                raise error from None
            raise _sanitized_failure(error) from None
        return True


def _log_mutation_accounting_failure() -> None:
    _LOGGER.error("mutation audit publication accounting failed after owned failure")

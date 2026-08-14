"""Strict field validation for clip consistency apply journals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import PurePosixPath
from typing import cast

from worker.pipeline.output.evidence.clip_consistency_types import (
    ClipConsistencyError,
    JournalState,
    RepairCounters,
)

JOURNAL_KEYS = frozenset(
    {
        "format_version",
        "state",
        "schema_version",
        "owner_uid",
        "state_db",
        "clip_store",
        "journal_path",
        "source_state_sha256",
        "source_identity_sha256",
        "non_relation_state_sha256",
        "plan_sha256",
        "relations_before_sha256",
        "relations_after_sha256",
        "backup_receipt_path",
        "backup_receipt_sha256",
        "delete_event_ids",
        "insert_rows",
        "quarantine_clip_ids",
        "quarantine",
        "quarantine_sha256",
        "counters",
        "error",
    }
)
_COUNTER_KEYS = frozenset(RepairCounters.__dataclass_fields__)
_STATES = frozenset({"PREPARED", "DB_COMMITTED", "DONE", "ABORTED", "UNKNOWN"})


def counters(value: object) -> RepairCounters:
    if not isinstance(value, dict):
        raise ClipConsistencyError("journal_invalid", "counter key set differs")
    payload = cast("dict[str, object]", value)
    if frozenset(payload) != _COUNTER_KEYS:
        raise ClipConsistencyError("journal_invalid", "counter key set differs")
    return RepairCounters(**{key: integer(payload, key) for key in _COUNTER_KEYS})


def validate_counter_facts(
    state: JournalState,
    value: RepairCounters,
    *,
    deletes: int,
    inserts: int,
    quarantines: int,
) -> None:
    nonnegative = all(number >= 0 for number in asdict(value).values())
    sql_facts = (
        value.sql_relations_deleted == deletes
        and value.sql_relations_inserted == inserts
        and value.relations_after
        == value.relations_before - deletes + inserts
        and value.staging_before == quarantines
    )
    if state == "DONE":
        staging_facts = (
            value.staging_after == 0
            and value.staging_to_delete == 0
            and value.staging_deleted == quarantines
        )
    else:
        staging_facts = (
            value.staging_after == quarantines
            and value.staging_to_delete == quarantines
            and value.staging_deleted == 0
        )
    if not nonnegative or not sql_facts or not staging_facts:
        raise ClipConsistencyError("journal_invalid", "counter facts differ")


def insert_rows(value: object) -> tuple[tuple[str, str, int], ...]:
    if not isinstance(value, list):
        raise ClipConsistencyError("journal_invalid", "insert rows invalid")
    rows: list[tuple[str, str, int]] = []
    for item in cast("list[object]", value):
        if not isinstance(item, list):
            raise ClipConsistencyError("journal_invalid", "insert row invalid")
        row = cast("list[object]", item)
        if (
            len(row) != 3
            or not isinstance(row[0], str)
            or not isinstance(row[1], str)
            or not isinstance(row[2], int)
        ):
            raise ClipConsistencyError("journal_invalid", "insert row invalid")
        rows.append((row[0], row[1], row[2]))
    return tuple(rows)


def quarantine(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ClipConsistencyError("journal_invalid", "quarantine invalid")
    result: list[tuple[str, str]] = []
    for item in cast("list[object]", value):
        if not isinstance(item, list):
            raise ClipConsistencyError("journal_invalid", "quarantine row invalid")
        row = cast("list[object]", item)
        if len(row) != 2 or not all(isinstance(field, str) for field in row):
            raise ClipConsistencyError("journal_invalid", "quarantine row invalid")
        original = cast(str, row[0])
        held = cast(str, row[1])
        if not _safe_relative(original) or not _safe_relative(held):
            raise ClipConsistencyError("journal_invalid", "quarantine path invalid")
        result.append((original, held))
    return tuple(result)


def strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ClipConsistencyError("journal_invalid", "string list invalid")
    values = cast("list[object]", value)
    if not all(isinstance(item, str) for item in values):
        raise ClipConsistencyError("journal_invalid", "string list invalid")
    return tuple(cast(str, item) for item in values)


def journal_state(value: str) -> JournalState:
    if value not in _STATES:
        raise ClipConsistencyError("journal_invalid", "journal state invalid")
    return cast(JournalState, value)


def integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ClipConsistencyError("journal_invalid", "journal field type differs")
    return value


def string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ClipConsistencyError("journal_invalid", "journal field type differs")
    return value


def optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ClipConsistencyError("journal_invalid", "journal field type differs")
    return value


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and str(path) == value

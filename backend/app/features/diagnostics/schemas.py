"""HTTP schemas for execution-record ingest receipts and engineer queries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.features.diagnostics.query import QueryResult

_QUERY_LIMIT_DEFAULT = 100
_QUERY_LIMIT_MAX = 500


class ExecutionRecordReceiptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(...)
    accepted: int = Field(...)
    duplicates: int = Field(...)
    rejected: list[list[str]] = Field(...)
    storage_state: str = Field(...)
    committed_at_ns: int = Field(...)


class ExecutionQueryParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: str = Field(..., min_length=1, max_length=128)
    from_ns: int = Field(..., ge=0)
    to_ns: int = Field(..., ge=0)
    limit: int = Field(default=_QUERY_LIMIT_DEFAULT, ge=1, le=_QUERY_LIMIT_MAX)
    cursor: str | None = Field(default=None)

    @model_validator(mode="after")
    def require_ordered_range(self) -> ExecutionQueryParams:
        if self.from_ns > self.to_ns:
            raise ValueError("from_ns must be <= to_ns")
        return self


class ExecutionRecordView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(...)
    record_kind: str = Field(...)
    producer: str = Field(...)
    producer_sequence: int = Field(...)
    frame_seq: int | None = Field(...)
    source_pts_ns: int | None = Field(...)
    observed_at_ns: int = Field(...)
    time_quality: str = Field(...)
    causal_unit_id: str = Field(...)
    parent_record_id: str | None = Field(...)
    outcome: str = Field(...)
    reason: str | None = Field(...)
    payload: dict[str, Any] = Field(...)
    provenance_id: str = Field(...)


class ExecutionUnitView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    causal_unit_id: str = Field(...)
    causal_state: str = Field(...)
    terminal: bool = Field(...)
    first_observed_ns: int = Field(...)
    last_observed_ns: int = Field(...)
    record_count: int = Field(...)


class ExecutionCoverageView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage_id: int = Field(...)
    coverage_kind: str = Field(...)
    producer: str | None = Field(...)
    from_sequence: int | None = Field(...)
    to_sequence: int | None = Field(...)
    from_ns: int = Field(...)
    to_ns: int = Field(...)
    record_count: int = Field(...)
    exact: bool = Field(...)
    cause: str = Field(...)
    recorded_at_ns: int = Field(...)


class ExecutionAvailabilityView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_ns: int = Field(...)
    to_ns: int = Field(...)
    kind: str = Field(...)


class ExecutionQueryableRangeView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_observed_at_ns: int | None = Field(...)
    max_observed_at_ns: int | None = Field(...)


class ExecutionQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[ExecutionRecordView] = Field(...)
    units: list[ExecutionUnitView] = Field(...)
    coverage: list[ExecutionCoverageView] = Field(...)
    availability: list[ExecutionAvailabilityView] = Field(...)
    queryable_range: ExecutionQueryableRangeView = Field(...)
    next_cursor: str | None = Field(...)


def query_response_from_result(result: QueryResult) -> ExecutionQueryResponse:
    return ExecutionQueryResponse(
        records=[
            ExecutionRecordView(
                record_id=record.record_id,
                record_kind=str(record.record_kind),
                producer=record.producer,
                producer_sequence=record.producer_sequence,
                frame_seq=record.frame_seq,
                source_pts_ns=record.source_pts_ns,
                observed_at_ns=record.observed_at_ns,
                time_quality=record.time_quality,
                causal_unit_id=record.causal_unit_id,
                parent_record_id=record.parent_record_id,
                outcome=record.outcome,
                reason=record.reason,
                payload=dict(record.payload),
                provenance_id=record.provenance_id,
            )
            for record in result.records
        ],
        units=[
            ExecutionUnitView(
                causal_unit_id=unit.causal_unit_id,
                causal_state=str(unit.causal_state),
                terminal=unit.terminal,
                first_observed_ns=unit.first_observed_ns,
                last_observed_ns=unit.last_observed_ns,
                record_count=unit.record_count,
            )
            for unit in result.units
        ],
        coverage=[
            ExecutionCoverageView(
                coverage_id=row.coverage_id,
                coverage_kind=str(row.coverage_kind),
                producer=row.producer,
                from_sequence=row.from_sequence,
                to_sequence=row.to_sequence,
                from_ns=row.from_ns,
                to_ns=row.to_ns,
                record_count=row.record_count,
                exact=row.exact,
                cause=row.cause,
                recorded_at_ns=row.recorded_at_ns,
            )
            for row in result.coverage
        ],
        availability=[
            ExecutionAvailabilityView(
                from_ns=item.from_ns,
                to_ns=item.to_ns,
                kind=str(item.kind),
            )
            for item in result.availability
        ],
        queryable_range=ExecutionQueryableRangeView(
            min_observed_at_ns=result.queryable_range.min_observed_at_ns,
            max_observed_at_ns=result.queryable_range.max_observed_at_ns,
        ),
        next_cursor=result.next_cursor,
    )


__all__ = [
    "ExecutionAvailabilityView",
    "ExecutionCoverageView",
    "ExecutionQueryParams",
    "ExecutionQueryResponse",
    "ExecutionQueryableRangeView",
    "ExecutionRecordReceiptResponse",
    "ExecutionRecordView",
    "ExecutionUnitView",
    "query_response_from_result",
]

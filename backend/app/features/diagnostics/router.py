"""Execution-record ingest (relay) and engineer query HTTP surface."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from starlette.concurrency import run_in_threadpool

from backend.app.features.diagnostics.schemas import (
    ExecutionQueryParams,
    ExecutionQueryResponse,
    ExecutionRecordReceiptResponse,
    query_response_from_result,
)
from backend.app.features.diagnostics.store import ExecutionRecordStore
from backend.app.features.diagnostics.wire import ingest_batch_from_wire, wire_receipt_from_store
from backend.app.features.relay.router import BoundedBodyRoute, require_relay_execution_records
from backend.app.shared.dashboard_auth import authorize_dashboard
from shared.events.execution_records import ExecutionRecordContractError, WireBatch

DISABLED_DETAIL = "execution records disabled"

router = APIRouter(tags=["diagnostics"], route_class=BoundedBodyRoute)


def execution_record_store(request: Request) -> ExecutionRecordStore:
    store = getattr(request.app.state, "execution_record_store", None)
    if not isinstance(store, ExecutionRecordStore):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=DISABLED_DETAIL,
        )
    return store


def backend_build_revision(request: Request) -> str:
    value = getattr(request.app.state, "backend_build_revision", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML_API_BUILD_REVISION is not configured",
        )
    return value


@router.post("/relay/execution-records", response_model=ExecutionRecordReceiptResponse)
async def ingest_execution_records(
    request: Request,
    _: Annotated[None, Depends(require_relay_execution_records)],
) -> dict[str, object]:
    store = execution_record_store(request)
    try:
        batch = WireBatch.from_json(await request.json())
    except ExecutionRecordContractError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    ingest = ingest_batch_from_wire(batch, backend_build_revision=backend_build_revision(request))
    # SQLite work (including bounded capacity pruning) never runs on the event
    # loop: a busy ingest must not stall /health or the dashboard.
    receipt = await run_in_threadpool(store.ingest_batch, ingest)
    return wire_receipt_from_store(receipt).to_json()


@router.get("/diagnostics/executions", response_model=ExecutionQueryResponse)
def query_execution_records(
    request: Request,
    _: Annotated[str, Depends(authorize_dashboard)],
    params: Annotated[ExecutionQueryParams, Query()],
) -> ExecutionQueryResponse:
    store = execution_record_store(request)
    return query_response_from_result(
        store.query(
            params.camera_id,
            params.from_ns,
            params.to_ns,
            params.limit,
            params.cursor,
        )
    )


__all__ = ["backend_build_revision", "execution_record_store", "router"]

"""Cross-record validation for finalized manifests before central publication."""

from __future__ import annotations

import json
import sqlite3
from typing import cast

from worker.pipeline.output.evidence.manifest_models import ClipManifest


def validate_recovery_manifest(
    connection: sqlite3.Connection,
    manifest: ClipManifest,
) -> None:
    """Require every recorded manifest fact to agree with its staged incidents."""
    if (
        connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='evidence_incidents'"
        ).fetchone()
        is None
    ):
        return
    if manifest.event_ref is not None and manifest.event_ref != manifest.event_refs[0]:
        raise ValueError("manifest direct event reference differs from ordered event refs")
    direct_event_ref = manifest.event_ref or manifest.event_refs[0]
    existing_relations = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT edge_event_id FROM clip_events WHERE clip_id=? ORDER BY ordinal",
            (manifest.clip_id,),
        ).fetchall()
    )
    if existing_relations and existing_relations != manifest.event_refs:
        raise ValueError("manifest event refs differ from durable clip relations")

    for event_ref in manifest.event_refs:
        row = connection.execute(
            """
            SELECT event.payload_json, incident.camera_id, incident.event_type,
                   incident.provenance_state, incident.runtime_manifest_sha256,
                   incident.decision_trace_id, direct.decision_trace_id,
                   analysis.worker_boot_id, analysis.camera_id, analysis.stream_epoch,
                   analysis.pts, analysis.source_time_sec
            FROM evidence_events AS event
            JOIN evidence_incidents AS incident USING (edge_event_id)
            LEFT JOIN evidence_event_trace_refs AS direct USING (edge_event_id)
            LEFT JOIN evidence_decision_traces AS decision
              ON decision.trace_id = direct.decision_trace_id
            LEFT JOIN runtime_analysis_traces AS analysis
              ON analysis.trace_id = decision.analysis_trace_id
            WHERE event.edge_event_id = ?
            """,
            (event_ref,),
        ).fetchone()
        if row is None:
            raise ValueError("manifest event is absent from central evidence")
        payload = _payload(row[0])
        camera_id = _required_text(payload, "camera_id")
        event_type = _required_text(payload, "event_type")
        if manifest.camera_id != camera_id or manifest.camera_id != str(row[1]):
            raise ValueError("manifest camera differs from its incident")
        if event_type != str(row[2]):
            raise ValueError("staged event type differs from its incident")
        if manifest.event_type is not None and manifest.event_type != event_type:
            raise ValueError("manifest event type differs from its incident")
        domain = _event_domain(payload)
        if domain is not None and manifest.domain != domain:
            raise ValueError("manifest domain differs from its staged event")

        qualified = str(row[3]) == "QUALIFIED"
        if qualified:
            if manifest.runtime_manifest_sha256 is None:
                raise ValueError("qualified manifest omits runtime manifest reference")
            if manifest.runtime_manifest_sha256 != str(row[4]):
                raise ValueError("manifest runtime reference differs from its incident")
            if event_ref == direct_event_ref:
                if manifest.decision_trace_id is None:
                    raise ValueError("qualified manifest omits direct decision trace reference")
                if manifest.decision_trace_id != str(row[5]) or manifest.decision_trace_id != str(
                    row[6]
                ):
                    raise ValueError(
                        "manifest decision trace differs from its direct event reference"
                    )
                if manifest.event_type is None:
                    raise ValueError("qualified manifest omits event type")

        origin = manifest.time_origin if event_ref == direct_event_ref else None
        if origin is not None:
            if origin.camera_id != manifest.camera_id:
                raise ValueError("manifest time origin camera differs from clip camera")
            if qualified:
                if origin.worker_boot_id != str(row[7]):
                    raise ValueError("manifest time origin boot differs from decision trace")
                if origin.camera_id != str(row[8]) or origin.stream_epoch != int(row[9]):
                    raise ValueError("manifest time origin differs from analysis trace")
                event_pts = row[10] if row[10] is not None else row[11]
                if event_pts is None or abs(origin.event_pts_sec - float(event_pts)) > 1e-9:
                    raise ValueError("manifest event time differs from analysis trace")


def _payload(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError("staged event payload is invalid") from exc
    if not isinstance(parsed, dict):
        raise TypeError("staged event payload is not an object")
    return cast(dict[str, object], parsed)


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"staged event {key} is missing")
    return value


def _event_domain(payload: dict[str, object]) -> str | None:
    direct = payload.get("domain")
    if isinstance(direct, str) and direct:
        return direct
    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        nested = evidence.get("domain")
        if isinstance(nested, str) and nested:
            return nested
    return None


__all__ = ["validate_recovery_manifest"]

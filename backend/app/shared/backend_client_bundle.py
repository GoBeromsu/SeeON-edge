from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import FastAPI

from shared.events.edge_ingest_client import EdgeIngestClient
from shared.events.evidence_export_client import BackendEvidenceClient


@dataclass(frozen=True, slots=True)
class BackendClientBundle:
    facility_code: str
    client_installation_ref: str
    facility_id: str
    edge_installation_id: str
    enrollment_generation: int
    facility_token: str = field(repr=False)
    events_url: str
    config_url: str
    ingest_client: EdgeIngestClient
    evidence_client: BackendEvidenceClient

    @property
    def generation_key(self) -> tuple[str, int]:
        return self.edge_installation_id, self.enrollment_generation


def backend_client_bundle(app: FastAPI) -> BackendClientBundle | None:
    state = getattr(app, "state", None)
    candidate = getattr(state, "backend_client_bundle", None)
    return candidate if isinstance(candidate, BackendClientBundle) else None


__all__ = ["BackendClientBundle", "backend_client_bundle"]

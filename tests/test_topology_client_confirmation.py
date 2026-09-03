from __future__ import annotations

import json

import pytest

from backend.app.features.cameras import topology_client as topology_client_module
from backend.app.features.cameras.edge_topology_sync_state import TopologyPauseReason
from backend.app.features.cameras.topology_client import TopologyClient, TopologyPaused
from contracts.edge_provisioning_v1 import MachinePrincipal, TopologyConfirmation

SNAPSHOT_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81a"
CONFIRMATION_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81b"
TOKEN = "server-side-secret-token"


@pytest.mark.parametrize(
    ("status_code", "reason"),
    [
        (401, TopologyPauseReason.AUTH),
        (403, TopologyPauseReason.FORBIDDEN),
        (409, TopologyPauseReason.CONFLICT),
    ],
)
def test_confirmation_classifies_upstream_auth_and_conflict_statuses(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    reason: TopologyPauseReason,
) -> None:
    # Given
    captured: list[tuple[str, str, dict[str, str], bytes]] = []

    def request(
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes,
        _timeout: float,
    ) -> tuple[int, dict[str, str], bytes]:
        captured.append((url, method, headers, body))
        return status_code, {}, b"{}"

    monkeypatch.setattr(topology_client_module, "bounded_request", request)
    client = TopologyClient(
        "https://product.example/api/v1/events",
        TOKEN,
        MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 3),
        "NH-7H2K9M4QXP",
        "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
        1.0,
    )

    # When
    result = client.confirm(SNAPSHOT_ID, TopologyConfirmation(CONFIRMATION_ID, "a" * 64, 7))

    # Then
    assert result == TopologyPaused(reason, status_code)
    assert captured == [
        (
            f"https://product.example/api/v1/edge/topology-snapshots/{SNAPSHOT_ID}/confirm",
            "POST",
            {
                "Accept": "application/json",
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
            },
            json.dumps(
                {
                    "confirmationId": CONFIRMATION_ID,
                    "digest": "a" * 64,
                    "expectedServerRevision": 7,
                    "schemaVersion": 1,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
        )
    ]

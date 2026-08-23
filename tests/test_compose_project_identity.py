"""The stack must never guess which `edge-state` volume it is binding.

Compose derives the project name from the checkout directory unless told
otherwise, and the volume is named `<project>_edge-state`. This host already
carries `edge_edge-state`, `seeon-edge-wt-alert-api_edge-state` and
`seeon-prod-edge-state`, so a default-derived name is not a cosmetic difference:
running from a checkout named `SeeON-edge` would bind `seeon-edge_edge-state`,
an empty volume. A cutover would then migrate a fresh database, report success,
and leave the live one untouched with its 1143 undelivered events.

`compose.edge.yaml` therefore requires `COMPOSE_PROJECT_NAME` rather than
defaulting it, so the failure is a refusal at config time instead of a silent
bind to the wrong data.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_COMPOSE = _ROOT / "compose.edge.yaml"
_PROD_EXAMPLE = _ROOT / ".env.edge.prod.example"


def test_the_project_name_is_required_and_never_defaulted() -> None:
    """`:?` not `:-`. A default is what makes the wrong bind silent."""
    compose = _COMPOSE.read_text(encoding="utf-8")

    match = re.search(r"^name:\s*\$\{COMPOSE_PROJECT_NAME(?P<expansion>[^}]*)\}", compose, re.M)
    assert match, "compose.edge.yaml does not pin the project name to COMPOSE_PROJECT_NAME"

    expansion = match.group("expansion")
    assert expansion.startswith(":?"), (
        "COMPOSE_PROJECT_NAME is defaulted rather than required; a default binds "
        "a directory-derived volume when the operator forgets to set it"
    )


def test_the_production_example_declares_the_project_name() -> None:
    """An operator copying the example must not have to discover this."""
    example = _PROD_EXAMPLE.read_text(encoding="utf-8")

    assert "COMPOSE_PROJECT_NAME=" in example, (
        ".env.edge.prod.example does not declare COMPOSE_PROJECT_NAME, so a "
        "deployment copied from it cannot start and the operator gets no guidance "
        "on which value is correct"
    )


def test_the_state_volume_is_not_declared_external_or_renamed() -> None:
    """The volume must stay `<project>_edge-state`, derived from the project.

    Pinning an explicit external name here would decouple the volume from the
    project and reintroduce exactly the ambiguity the required project name
    closes.
    """
    compose = _COMPOSE.read_text(encoding="utf-8")

    volumes_section = compose.split("\nvolumes:", 1)
    assert len(volumes_section) == 2, "compose.edge.yaml declares no volumes section"

    body = volumes_section[1]
    edge_state = re.search(r"^  edge-state:(?P<body>.*?)(?=^  \S|\Z)", body, re.M | re.S)
    assert edge_state, "edge-state volume is no longer declared"

    declaration = edge_state.group("body")
    assert "external" not in declaration, (
        "edge-state is declared external, which decouples it from the project "
        "name and makes the bound volume ambiguous again"
    )
    assert "name:" not in declaration, (
        "edge-state carries an explicit name, which overrides the project-derived "
        "identity the required COMPOSE_PROJECT_NAME exists to establish"
    )

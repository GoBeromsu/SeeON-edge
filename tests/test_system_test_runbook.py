from __future__ import annotations

import re
from pathlib import Path

RUNBOOK = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "runbooks"
    / "happy-nursing-home-system-test.md"
)


def test_runbook_pins_configured_happy_nursing_home_alias_without_target_tuple() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    commands = "\n".join(re.findall(r"```sh\n(.*?)```", text, flags=re.DOTALL))

    assert "ssh -G happy-nursing-home" in commands
    assert "ssh happy-nursing-home" in commands
    assert "ssh 'happy nursing home'" not in commands
    assert not {"EDGE_HOST", "EDGE_USER", "EDGE_KEY"}.intersection(commands)
    assert "ssh -i" not in commands
    assert not re.search(r"ssh\s+[^\n]*(?:@|<approved)", commands)


def test_runbook_gates_first_v7_open_and_requires_fix_forward_after_migration() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "EXPECTED_EDGE_INSTALLATION_ID" in text
    assert "ConnectionSettingsStore.from_env().load()" in text
    assert "SELECT facility_token" not in text
    assert "APPROVED_EDGE_REVISION" in text
    assert "FIX_FORWARD_WORKER_IMAGE" in text
    assert "PRAGMA user_version" in text
    assert "docker image inspect" in text
    assert "previously approved image" not in text
    assert "down -v" in text
    assert "fix-forward" in text

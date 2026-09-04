"""Count the alerts a Flow run produced, from the backend's own record.

The backend already stores every relayed alert as an incident, so the
production alert-rate gate is a query rather than a separate measurement rig:

    python scripts/qa/flow_alert_rate.py --database /var/lib/seeon-state/edge.sqlite3

Reports the site rate and the per-camera rate over the observed window and
compares them with the P1b-AC6 / P1a-AC9 bounds (<= 26 alerts/hour site,
<= 2/hour per camera). Read-only: it opens the database in read-only mode and
never writes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SITE_ALERTS_PER_HOUR = 26.0
CAMERA_ALERTS_PER_HOUR = 2.0


def _parse(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def alert_rate(database: Path, *, window_start: str | None = None) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        clause = "" if window_start is None else " WHERE detected_at >= ?"
        arguments = () if window_start is None else (window_start,)
        rows = connection.execute(
            f"SELECT camera_id, event_type, detected_at FROM incidents{clause}"  # noqa: S608
            " ORDER BY detected_at",
            arguments,
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        return {
            "incidents": 0,
            "window_hours": 0.0,
            "site_alerts_per_hour": 0.0,
            "per_camera": {},
            "verdict": "no incidents recorded",
        }

    first, last = _parse(rows[0][2]), _parse(rows[-1][2])
    window_hours = max((last - first).total_seconds() / 3600.0, 1e-9)
    per_camera: dict[str, int] = {}
    per_type: dict[str, int] = {}
    for camera_id, event_type, _detected_at in rows:
        per_camera[camera_id] = per_camera.get(camera_id, 0) + 1
        per_type[event_type] = per_type.get(event_type, 0) + 1

    site_rate = len(rows) / window_hours
    camera_rates = {camera: count / window_hours for camera, count in per_camera.items()}
    worst_camera = max(camera_rates.items(), key=lambda item: item[1])
    return {
        "incidents": len(rows),
        "first_detected_at": rows[0][2],
        "last_detected_at": rows[-1][2],
        "window_hours": round(window_hours, 4),
        "site_alerts_per_hour": round(site_rate, 3),
        "site_bound": SITE_ALERTS_PER_HOUR,
        "site_within_bound": site_rate <= SITE_ALERTS_PER_HOUR,
        "worst_camera": worst_camera[0],
        "worst_camera_alerts_per_hour": round(worst_camera[1], 3),
        "camera_bound": CAMERA_ALERTS_PER_HOUR,
        "every_camera_within_bound": all(
            rate <= CAMERA_ALERTS_PER_HOUR for rate in camera_rates.values()
        ),
        "by_event_type": per_type,
        "per_camera": {camera: round(rate, 3) for camera, rate in camera_rates.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="flow-alert-rate")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--since", default=None, help="ISO8601 Z lower bound on detected_at")
    arguments = parser.parse_args()
    report = alert_rate(arguments.database, window_start=arguments.since)
    print(json.dumps(report, indent=2, sort_keys=True))
    within = report.get("site_within_bound"), report.get("every_camera_within_bound")
    return 0 if all(value is not False for value in within) else 1


if __name__ == "__main__":
    raise SystemExit(main())

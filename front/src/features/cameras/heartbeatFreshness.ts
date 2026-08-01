/**
 * Humanises `heartbeat_age_sec` (seconds since the edge last reported in) into Korean copy for
 * at-a-glance triage, e.g. "마지막 신호 4분 전". Returns null when there is nothing to say —
 * callers must render nothing in that case rather than an "N/A" placeholder.
 */
export function formatHeartbeatAge(ageSec: number | null | undefined): string | null {
  if (typeof ageSec !== 'number' || !Number.isFinite(ageSec)) return null;

  const clamped = Math.max(0, ageSec);
  if (clamped < 60) {
    return `마지막 신호 ${Math.round(clamped)}초 전`;
  }
  if (clamped < 3600) {
    return `마지막 신호 ${Math.round(clamped / 60)}분 전`;
  }
  return `마지막 신호 ${Math.round(clamped / 3600)}시간 전`;
}

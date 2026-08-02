/**
 * The system currently configures exactly two detection domains (침대 이탈 / 낙상 — see
 * design-handoff/README.md's 탐지 설정 카드 section), with canonical `event_type` strings "bed-exit"
 * and "fall" (contracts/event.py). Any other value is shown verbatim rather than hidden, since
 * `Clip.event_type` is a free-form string end to end (clipNormalizer falls back to whatever the
 * backend sends).
 */
const KNOWN_EVENT_TYPE_ORDER = ['bed-exit', 'fall'] as const;

const EVENT_TYPE_LABELS: Record<string, string> = {
  fall: '낙상',
  'bed-exit': '침대 이탈',
};

const EVENT_TYPE_CHIP_CLASSNAMES: Record<string, string> = {
  fall: 'border-status-rejectedFg bg-status-rejectedBg text-status-rejectedFg',
  'bed-exit': 'border-teal bg-card text-teal',
};

const DEFAULT_CHIP_CLASSNAME = 'border-border bg-muted text-muted-foreground';

export function getEventTypeLabel(eventType: string): string {
  return EVENT_TYPE_LABELS[eventType] ?? eventType;
}

export function getEventTypeChipClassName(eventType: string): string {
  return EVENT_TYPE_CHIP_CLASSNAMES[eventType] ?? DEFAULT_CHIP_CLASSNAME;
}

/** Canonical detection-type order first, then any other event_type values actually present in the data. */
export function orderEventTypes(presentTypes: Iterable<string>): string[] {
  const present = new Set(presentTypes);
  const extra = [...present]
    .filter((type) => !(KNOWN_EVENT_TYPE_ORDER as readonly string[]).includes(type))
    .sort();
  return [...KNOWN_EVENT_TYPE_ORDER, ...extra];
}

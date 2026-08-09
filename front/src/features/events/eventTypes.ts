import { toEventFacet, type EventFacet } from '@/shared/api/clipEventFacet';

const EVENT_FACET_ORDER = ['bed-exit', 'fall', 'other'] as const;

const EVENT_TYPE_LABELS: Record<EventFacet, string> = {
  fall: '낙상',
  'bed-exit': '침대 이탈',
  other: '기타',
};

const EVENT_TYPE_CHIP_CLASSNAMES: Record<EventFacet, string> = {
  fall: 'border-status-rejectedFg bg-status-rejectedBg text-status-rejectedFg',
  'bed-exit': 'border-teal bg-card text-teal',
  other: 'border-border bg-muted text-muted-foreground',
};

export function getEventTypeLabel(eventType: string): string {
  return EVENT_TYPE_LABELS[toEventFacet(eventType)];
}

export function getEventTypeChipClassName(eventType: string): string {
  return EVENT_TYPE_CHIP_CLASSNAMES[toEventFacet(eventType)];
}

export function orderEventTypes(_presentTypes: Iterable<string>): EventFacet[] {
  return [...EVENT_FACET_ORDER];
}

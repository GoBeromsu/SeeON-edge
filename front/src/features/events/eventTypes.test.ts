import { describe, expect, it } from 'vitest';
import {
  getEventTypeChipClassName,
  getEventTypeLabel,
  orderEventTypes,
} from '@/features/events/eventTypes';
import { toEventFacet } from '@/shared/api/clipEventFacet';

describe('eventTypes', () => {
  it('labels the two known detection types in Korean', () => {
    expect(getEventTypeLabel('fall')).toBe('낙상');
    expect(getEventTypeLabel('bed-exit')).toBe('침대 이탈');
  });

  it.each([
    ['fall', 'fall'],
    ['bed-exit', 'bed-exit'],
    ['detection-lost', 'other'],
    [null, 'other'],
    [undefined, 'other'],
  ])('maps raw event type %s to the bounded %s facet', (rawEventType, expectedFacet) => {
    expect(toEventFacet(rawEventType)).toBe(expectedFacet);
  });

  it('labels an unrecognized event type as the other facet', () => {
    expect(getEventTypeLabel('detection-lost')).toBe('기타');
  });

  it('gives fall and bed-exit distinct, non-default chip classes', () => {
    expect(getEventTypeChipClassName('fall')).not.toBe(getEventTypeChipClassName('bed-exit'));
    expect(getEventTypeChipClassName('fall')).toContain('status-rejected');
    expect(getEventTypeChipClassName('bed-exit')).toContain('teal');
  });

  it('falls back to a muted chip class for an unrecognized event type', () => {
    expect(getEventTypeChipClassName('unknown-type')).toContain('muted');
  });

  it('always exposes the three bounded facets in the existing visual order', () => {
    expect(orderEventTypes([])).toEqual(['bed-exit', 'fall', 'other']);
  });

  it('does not append unrecognized raw event types as controls', () => {
    expect(orderEventTypes(['fall', 'zzz-unknown', 'bed-exit', 'aaa-unknown'])).toEqual([
      'bed-exit', 'fall', 'other',
    ]);
  });
});

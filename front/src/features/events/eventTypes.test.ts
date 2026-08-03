import { describe, expect, it } from 'vitest';
import { getEventTypeChipClassName, getEventTypeLabel, orderEventTypes } from '@/features/events/eventTypes';

describe('eventTypes', () => {
  it('labels the two known detection types in Korean', () => {
    expect(getEventTypeLabel('fall')).toBe('낙상');
    expect(getEventTypeLabel('bed-exit')).toBe('침대 이탈');
  });

  it('falls back to the raw value for an unrecognized event type', () => {
    expect(getEventTypeLabel('detection-lost')).toBe('detection-lost');
  });

  it('gives fall and bed-exit distinct, non-default chip classes', () => {
    expect(getEventTypeChipClassName('fall')).not.toBe(getEventTypeChipClassName('bed-exit'));
    expect(getEventTypeChipClassName('fall')).toContain('status-rejected');
    expect(getEventTypeChipClassName('bed-exit')).toContain('teal');
  });

  it('falls back to a muted chip class for an unrecognized event type', () => {
    expect(getEventTypeChipClassName('unknown-type')).toContain('muted');
  });

  it('always orders the canonical types first (bed-exit, then fall), even with no data', () => {
    expect(orderEventTypes([])).toEqual(['bed-exit', 'fall']);
  });

  it('appends unrecognized event types present in the data, sorted, after the canonical two', () => {
    expect(orderEventTypes(['fall', 'zzz-unknown', 'bed-exit', 'aaa-unknown'])).toEqual([
      'bed-exit', 'fall', 'aaa-unknown', 'zzz-unknown',
    ]);
  });
});

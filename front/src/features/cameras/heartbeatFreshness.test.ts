import { describe, expect, it } from 'vitest';
import { formatHeartbeatAge } from '@/features/cameras/heartbeatFreshness';

describe('formatHeartbeatAge', () => {
  it('renders nothing for null', () => {
    expect(formatHeartbeatAge(null)).toBeNull();
  });

  it('renders nothing for undefined', () => {
    expect(formatHeartbeatAge(undefined)).toBeNull();
  });

  it.each([
    ['a non-finite number', Number.NaN],
    ['positive infinity', Number.POSITIVE_INFINITY],
  ])('renders nothing for %s instead of crashing or coercing', (_case, value) => {
    expect(formatHeartbeatAge(value)).toBeNull();
  });

  it('formats sub-minute ages in whole seconds', () => {
    expect(formatHeartbeatAge(4)).toBe('마지막 신호 4초 전');
    expect(formatHeartbeatAge(0)).toBe('마지막 신호 0초 전');
    expect(formatHeartbeatAge(59.4)).toBe('마지막 신호 59초 전');
  });

  it('formats sub-hour ages in rounded minutes', () => {
    expect(formatHeartbeatAge(60)).toBe('마지막 신호 1분 전');
    expect(formatHeartbeatAge(245.75)).toBe('마지막 신호 4분 전');
    expect(formatHeartbeatAge(3599)).toBe('마지막 신호 60분 전');
  });

  it('formats ages of an hour or more in rounded hours', () => {
    expect(formatHeartbeatAge(3600)).toBe('마지막 신호 1시간 전');
    expect(formatHeartbeatAge(7500)).toBe('마지막 신호 2시간 전');
  });

  it('clamps a negative age (clock skew) to zero instead of showing a negative duration', () => {
    expect(formatHeartbeatAge(-5)).toBe('마지막 신호 0초 전');
  });
});

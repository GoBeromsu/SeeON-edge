import { describe, expect, it } from 'vitest';
import { formatClipTimestamp, formatDuration, formatResolution } from '@/features/events/formatters';

describe('formatClipTimestamp', () => {
  it('formats an ISO timestamp using the Korean locale', () => {
    const formatted = formatClipTimestamp('2026-08-02T03:12:00Z');
    expect(formatted).not.toBe('시간 정보 없음');
    expect(formatted).toMatch(/2026/);
  });

  it('shows a fallback for a null timestamp', () => {
    expect(formatClipTimestamp(null)).toBe('시간 정보 없음');
  });

  it('shows a fallback for an unparsable timestamp', () => {
    expect(formatClipTimestamp('not-a-date')).toBe('시간 정보 없음');
  });
});

describe('formatDuration', () => {
  it('formats seconds as m:ss', () => {
    expect(formatDuration(12)).toBe('0:12');
    expect(formatDuration(75)).toBe('1:15');
  });

  it('shows a pending indicator when duration is not yet known', () => {
    expect(formatDuration(null)).toBe('확인 중…');
    expect(formatDuration(Number.NaN)).toBe('확인 중…');
  });
});

describe('formatResolution', () => {
  it('formats width x height with a multiplication sign', () => {
    expect(formatResolution(1920, 1080)).toBe('1920×1080');
  });

  it('shows a pending indicator when dimensions are not yet known', () => {
    expect(formatResolution(null, null)).toBe('확인 중…');
    expect(formatResolution(0, 0)).toBe('확인 중…');
  });
});

import { describe, expect, it } from 'vitest';
import { formatBytes } from '@/shared/format/bytes';

describe('formatBytes', () => {
  it('matches the design spec example exactly (8,400,000 bytes -> "8.4 MB")', () => {
    expect(formatBytes(8_400_000)).toBe('8.4 MB');
  });

  it('formats gigabytes with up to one decimal place, dropping a trailing .0', () => {
    expect(formatBytes(1_200_000_000)).toBe('1.2 GB');
    expect(formatBytes(10_000_000_000)).toBe('10 GB');
  });

  it('formats megabytes with up to one decimal place, dropping a trailing .0', () => {
    expect(formatBytes(850_000_000)).toBe('850 MB');
  });

  it('formats kilobytes as a whole number', () => {
    expect(formatBytes(3_000)).toBe('3 KB');
  });

  it('formats sub-kilobyte sizes in whole bytes', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(0)).toBe('0 B');
  });

  it('shows a fallback for invalid input instead of fabricating a value', () => {
    expect(formatBytes(Number.NaN)).toBe('-');
    expect(formatBytes(-1)).toBe('-');
  });
});

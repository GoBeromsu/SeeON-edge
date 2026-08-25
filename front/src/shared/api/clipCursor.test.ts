import { describe, expect, it } from 'vitest';
import {
  compareClipKeysDescending,
  decodeClipCursor,
  encodeClipCursor,
  isAfterClipCursor,
} from '@/shared/api/clipCursor';

const startedAt = '2026-08-02T03:12:00Z';

describe('clip keyset cursor', () => {
  it('round-trips the backend base64url `started_at\\0clip_id` shape', () => {
    const key = { startedAt, clipId: 'clip/한글-1' };
    const cursor = encodeClipCursor(key);

    expect(cursor).not.toMatch(/[+/]/);
    expect(decodeClipCursor(cursor)).toEqual(key);
  });

  it('matches the exact bytes the backend emits for the same key', () => {
    // backend: base64.urlsafe_b64encode(f"{started_at}\0{clip_id}".encode())
    expect(encodeClipCursor({ startedAt: '2026-08-02T03:12:00Z', clipId: 'clip-1' }))
      .toBe('MjAyNi0wOC0wMlQwMzoxMjowMFoAY2xpcC0x');
  });

  it.each([
    ['', 'empty'],
    ['!!!', 'not base64'],
    ['Y2xpcC0x', 'no separator'],
  ])('rejects the %s cursor', (cursor) => {
    expect(decodeClipCursor(cursor)).toBeNull();
  });

  it('orders newer timestamps first and breaks equal timestamps on the descending clip id', () => {
    const older = { startedAt: '2026-08-02T01:00:00Z', clipId: 'clip-9' };
    const newer = { startedAt, clipId: 'clip-1' };

    expect(compareClipKeysDescending(newer, older)).toBeLessThan(0);
    expect(compareClipKeysDescending({ startedAt, clipId: 'clip-2' }, { startedAt, clipId: 'clip-10' }))
      .toBeLessThan(0);
    expect(compareClipKeysDescending(newer, { ...newer })).toBe(0);
  });

  it('excludes the boundary row itself so a page can never repeat an equal-timestamp row', () => {
    const boundary = { startedAt, clipId: 'clip-5' };

    expect(isAfterClipCursor(boundary, boundary)).toBe(false);
    expect(isAfterClipCursor({ startedAt, clipId: 'clip-6' }, boundary)).toBe(false);
    expect(isAfterClipCursor({ startedAt, clipId: 'clip-4' }, boundary)).toBe(true);
    expect(isAfterClipCursor({ startedAt: '2026-08-02T01:00:00Z', clipId: 'clip-9' }, boundary)).toBe(true);
  });
});

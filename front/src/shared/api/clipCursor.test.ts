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

/**
 * Schema 18 allows any 1-128 character NUL-free TEXT clip id, and SQLite's default BINARY collation
 * orders those by UTF-8 bytes. JavaScript relational operators order by UTF-16 code units, which
 * disagrees for every non-BMP (astral) id: U+10000 encodes as a D800/DC00 surrogate pair that sorts
 * BELOW U+E000 in UTF-16, but as F0 90 80 80 that sorts ABOVE U+E000's EE 80 80 in UTF-8. A cursor
 * built on the UTF-16 order can therefore place a boundary the backend predicate disagrees with and
 * skip a row.
 */
describe('clip keyset ordering is SQLite BINARY (UTF-8 byte) identical', () => {
  // Oracle recorded from real SQLite:
  //   CREATE TABLE clips (clip_id TEXT PRIMARY KEY, started_at TEXT) STRICT;
  //   SELECT clip_id FROM clips ORDER BY started_at DESC, clip_id DESC;
  // => ['\u{1F600}', '\u{10000}', '\uE000', '\uAC00', 'zz', 'ascii']
  const SQLITE_BINARY_DESCENDING = ['\u{1F600}', '\u{10000}', '\uE000', '\uAC00', 'zz', 'ascii'];

  function utf8ByteOrderDescending(ids: readonly string[]): string[] {
    const encoder = new TextEncoder();
    return [...ids].sort((left, right) => {
      const leftBytes = encoder.encode(left);
      const rightBytes = encoder.encode(right);
      for (let index = 0; index < Math.min(leftBytes.length, rightBytes.length); index += 1) {
        if (leftBytes[index] !== rightBytes[index]) return rightBytes[index] - leftBytes[index];
      }
      return rightBytes.length - leftBytes.length;
    });
  }

  it('sorts astral and BMP clip ids in the exact SQLite BINARY descending order', () => {
    const sorted = [...SQLITE_BINARY_DESCENDING]
      .reverse()
      .map((clipId) => ({ startedAt, clipId }))
      .sort(compareClipKeysDescending)
      .map((key) => key.clipId);

    expect(sorted).toEqual(SQLITE_BINARY_DESCENDING);
    // Independent byte-order oracle agrees with the recorded SQLite result.
    expect(utf8ByteOrderDescending(SQLITE_BINARY_DESCENDING)).toEqual(SQLITE_BINARY_DESCENDING);
  });

  it('places an astral clip id above U+E000 the way UTF-8 bytes do, not the way UTF-16 does', () => {
    const astral = { startedAt, clipId: '\u{10000}' };
    const privateUse = { startedAt, clipId: '\uE000' };

    // F0 90 80 80 > EE 80 80, so the astral id is the *earlier* descending row.
    expect(compareClipKeysDescending(astral, privateUse)).toBeLessThan(0);
    expect(compareClipKeysDescending(privateUse, astral)).toBeGreaterThan(0);
    // The naive UTF-16 comparison this replaces would have claimed the opposite.
    expect('\u{10000}' < '\uE000').toBe(true);
  });

  it('cannot skip an astral row when it walks an equal-timestamp boundary', () => {
    const boundary = { startedAt, clipId: '\u{10000}' };

    // Everything at or before the astral boundary stays off the next page...
    expect(isAfterClipCursor(boundary, boundary)).toBe(false);
    expect(isAfterClipCursor({ startedAt, clipId: '\u{1F600}' }, boundary)).toBe(false);
    // ...and every byte-smaller id, astral or not, is still reachable after it.
    expect(isAfterClipCursor({ startedAt, clipId: '\uE000' }, boundary)).toBe(true);
    expect(isAfterClipCursor({ startedAt, clipId: '\uAC00' }, boundary)).toBe(true);
    expect(isAfterClipCursor({ startedAt, clipId: 'ascii' }, boundary)).toBe(true);
  });

  it('breaks a common-prefix tie on length the way BINARY does (shorter is less)', () => {
    const shorter = { startedAt, clipId: 'clip' };
    const longer = { startedAt, clipId: 'clip-1' };

    expect(compareClipKeysDescending(longer, shorter)).toBeLessThan(0);
    expect(isAfterClipCursor(shorter, longer)).toBe(true);
    expect(isAfterClipCursor(longer, shorter)).toBe(false);
  });

  it('keeps a multi-byte id ordered by bytes rather than by code point count', () => {
    // 'z' is 0x7A; U+AC00 is EA B0 80. One character can still outrank a longer ASCII id.
    expect(compareClipKeysDescending(
      { startedAt, clipId: '\uAC00' },
      { startedAt, clipId: 'zzzzzzzz' },
    )).toBeLessThan(0);
  });
});

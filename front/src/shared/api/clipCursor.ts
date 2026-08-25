/**
 * Keyset cursor helpers mirroring the backend `clips` page order
 * `ORDER BY started_at DESC, clip_id DESC` with the strict predicate
 * `started_at < ? OR (started_at = ? AND clip_id < ?)`.
 *
 * The backend cursor is base64url(`${started_at}\0${clip_id}`). These helpers exist so a listing
 * response that omits server pagination (a whole-list body) is paged with the exact same total
 * order and the exact same secondary-key tiebreak, and therefore can neither repeat nor skip a row
 * whose `started_at` equals the boundary row's.
 */

export type ClipCursorKey = {
  readonly startedAt: string;
  readonly clipId: string;
};

const SEPARATOR = '\0';

export function encodeClipCursor(key: ClipCursorKey): string {
  const raw = `${key.startedAt}${SEPARATOR}${key.clipId}`;
  const bytes = new TextEncoder().encode(raw);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_');
}

export function decodeClipCursor(cursor: string): ClipCursorKey | null {
  let raw: string;
  try {
    const binary = atob(cursor.replace(/-/g, '+').replace(/_/g, '/'));
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    raw = new TextDecoder().decode(bytes);
  } catch {
    return null;
  }
  const separatorIndex = raw.indexOf(SEPARATOR);
  if (separatorIndex <= 0) return null;
  const startedAt = raw.slice(0, separatorIndex);
  const clipId = raw.slice(separatorIndex + SEPARATOR.length);
  if (!startedAt || !clipId) return null;
  return { startedAt, clipId };
}

const UTF8 = new TextEncoder();

/**
 * Ascending SQLite BINARY comparison: unsigned UTF-8 byte order, shorter-is-less on a common prefix.
 *
 * Schema 18 allows any 1-128 character NUL-free TEXT clip id, and SQLite's default collation orders
 * those by their UTF-8 bytes. JavaScript relational operators instead order by UTF-16 code units,
 * which disagrees for every non-BMP id: U+10000 is the surrogate pair D800/DC00 and sorts BELOW
 * U+E000 in UTF-16, but is F0 90 80 80 and sorts ABOVE U+E000's EE 80 80 in UTF-8. Using the UTF-16
 * order here would let a synthesized boundary disagree with the backend predicate and skip a row.
 */
function compareBinaryAscending(left: string, right: string): number {
  const leftBytes = UTF8.encode(left);
  const rightBytes = UTF8.encode(right);
  const shared = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < shared; index += 1) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] < rightBytes[index] ? -1 : 1;
  }
  if (leftBytes.length === rightBytes.length) return 0;
  return leftBytes.length < rightBytes.length ? -1 : 1;
}

/**
 * Descending total order: newer first, and at an identical timestamp the byte-larger clip id first.
 *
 * `started_at` stays a plain string comparison because the backend admits a row into `clips` only
 * after `_valid_timestamp` parses it as RFC3339 UTC ending in `Z` at 20-30 characters
 * (`backend/app/features/clips/compact_listing.py`), which is ASCII-only -- and over ASCII, UTF-16
 * code-unit order and UTF-8 byte order are identical. The clip id has no such restriction, so it
 * must go through the BINARY comparison.
 */
export function compareClipKeysDescending(left: ClipCursorKey, right: ClipCursorKey): number {
  if (left.startedAt !== right.startedAt) return left.startedAt < right.startedAt ? 1 : -1;
  const byClipId = compareBinaryAscending(left.clipId, right.clipId);
  if (byClipId === 0) return 0;
  return byClipId < 0 ? 1 : -1;
}

/** True when `key` belongs strictly after the cursor boundary in the descending page order. */
export function isAfterClipCursor(key: ClipCursorKey, boundary: ClipCursorKey): boolean {
  return compareClipKeysDescending(key, boundary) > 0;
}

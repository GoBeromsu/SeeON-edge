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

/** Descending total order: newer first, and at an identical timestamp the larger clip id first. */
export function compareClipKeysDescending(left: ClipCursorKey, right: ClipCursorKey): number {
  if (left.startedAt !== right.startedAt) return left.startedAt < right.startedAt ? 1 : -1;
  if (left.clipId === right.clipId) return 0;
  return left.clipId < right.clipId ? 1 : -1;
}

/** True when `key` belongs strictly after the cursor boundary in the descending page order. */
export function isAfterClipCursor(key: ClipCursorKey, boundary: ClipCursorKey): boolean {
  return compareClipKeysDescending(key, boundary) > 0;
}

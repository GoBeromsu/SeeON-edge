import type { Camera } from '@/shared/api/client';

/**
 * Derives the selectable floor chip set for the camera add/edit modals (issue #85).
 *
 * There is no backend floor catalog -- the set is fully client-derived from the camera list's
 * effective floor (`camera.floor ?? camera.floor_name`, same precedence as elsewhere) plus any
 * `extra` values (session-local "+ 층 추가" additions, or the currently-selected floor for a
 * camera being edited, so its own chip always renders even if it doesn't overlap with another
 * camera's floor). Sorted with numeric-aware Korean collation so "2층" sorts before "10층".
 */
export function deriveFloorOptions(cameras: Camera[], extra: readonly string[] = []): string[] {
  const values = new Set<string>();
  for (const camera of cameras) {
    const floor = camera.floor ?? camera.floor_name;
    if (floor) values.add(floor);
  }
  for (const value of extra) {
    const trimmed = value.trim();
    if (trimmed) values.add(trimmed);
  }
  return Array.from(values).sort((a, b) => a.localeCompare(b, 'ko', { numeric: true }));
}

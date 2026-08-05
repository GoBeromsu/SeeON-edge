/**
 * Fixed floor catalog (issue #155): B1 through 10층, encoded as an integer (basement negative,
 * e.g. B1 = -1) instead of a free-text chip label. Previously the floor selector let facilities
 * type in anything ('2층', '2', 'B1', '2 층' all became different values), which broke sorting
 * (string sort puts '10층' before '2층') and made per-floor aggregation unreliable. The display
 * string is generated at render time from the integer (`floorLabel`) rather than persisted, so
 * relabeling never requires a data migration. Mirrors the backend's
 * backend/app/features/cameras/store.py (FLOOR_VALUES/floor_label/is_valid_floor).
 */
export const FLOOR_MIN = -1;
export const FLOOR_MAX = 10;
export const DEFAULT_FLOOR = 1;

export const FLOOR_VALUES: readonly number[] = [FLOOR_MIN, ...Array.from({ length: FLOOR_MAX }, (_, index) => index + 1)];

export function floorLabel(value: number): string {
  return value < 0 ? `B${-value}` : `${value}층`;
}

export function isValidFloor(value: number): boolean {
  return FLOOR_VALUES.includes(value);
}

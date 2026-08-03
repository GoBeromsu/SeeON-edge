import type { Camera } from '@/shared/api/client';

const collator = new Intl.Collator('ko-KR', { numeric: true, sensitivity: 'base' });

function compareNullable(left: string | null, right: string | null): number {
  if (left === null) return right === null ? 0 : 1;
  if (right === null) return -1;
  return collator.compare(left, right);
}

export function sortCameras(cameras: readonly Camera[]): Camera[] {
  return [...cameras].sort(
    (left, right) => compareNullable(left.floor_name, right.floor_name)
      || collator.compare(left.label, right.label)
      || left.id.localeCompare(right.id),
  );
}

/** Dynamic per facility — floors are whatever distinct `floor_name` values the registered cameras carry. */
export function listFloors(cameras: readonly Camera[]): string[] {
  const floors = new Set<string>();
  for (const camera of cameras) {
    if (camera.floor_name) floors.add(camera.floor_name);
  }
  return [...floors].sort(collator.compare);
}

export function filterCamerasByFloor(cameras: readonly Camera[], floor: string | undefined): Camera[] {
  return sortCameras(floor ? cameras.filter((camera) => camera.floor_name === floor) : cameras);
}

export function isCameraOnline(camera: Pick<Camera, 'status'>): boolean {
  return camera.status === 'online';
}

/** Wall's only concern is camera survival — starting/unknown fold into "offline" alongside true offline. */
export function countCamerasByLiveness(cameras: readonly Camera[]): { online: number; offline: number } {
  let online = 0;
  for (const camera of cameras) {
    if (isCameraOnline(camera)) online += 1;
  }
  return { online, offline: cameras.length - online };
}

/** Spreads concurrent snapshot polls across the 5s refresh window instead of bursting together. */
export function snapshotJitterMs(cameraId: string): number {
  let hash = 0;
  for (const character of cameraId) hash = (hash * 31 + character.codePointAt(0)!) >>> 0;
  return hash % 1_000;
}

// Canonical `event_type` strings are "bed-exit" and "fall" (contracts/event.py — see
// features/events/eventTypes.ts, which documents the same contract for the events page).
const EVENT_TYPE_LABELS: Record<string, string> = {
  'bed-exit': '침대 이탈',
  fall: '낙상',
};

export function eventTypeLabel(type: string): string {
  return EVENT_TYPE_LABELS[type] ?? type;
}

export function formatClipDateTime(value: string | null): string {
  if (!value) return '시간 정보 없음';
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString('ko-KR') : '시간 정보 없음';
}

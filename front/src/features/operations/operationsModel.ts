import type { Camera, CameraHeartbeat, HeartbeatStatus, StatusSnapshot } from '@/shared/api/client';

export const WALL_PAGE_SIZE = 12;
export const UNCLASSIFIED = '미분류';

const collator = new Intl.Collator('ko-KR', { numeric: true, sensitivity: 'base' });

export type CameraFilters = { floor?: string; room?: string };
export type CameraRoomGroup = { id: string; label: string; cameras: Camera[] };
export type CameraFloorGroup = { id: string; label: string; rooms: CameraRoomGroup[] };

function compareNullable(left: string | null, right: string | null): number {
  if (left === null) return right === null ? 0 : 1;
  if (right === null) return -1;
  return collator.compare(left, right);
}

export function sortCameras(cameras: readonly Camera[]): Camera[] {
  return [...cameras].sort((left, right) => compareNullable(left.floor_name, right.floor_name)
    || compareNullable(left.space_name ?? left.space_id, right.space_name ?? right.space_id)
    || collator.compare(left.label, right.label)
    || left.id.localeCompare(right.id));
}

export function groupCameras(cameras: readonly Camera[]): CameraFloorGroup[] {
  const floors = new Map<string, CameraFloorGroup>();
  for (const camera of cameras) {
    const floorId = camera.floor_name ?? UNCLASSIFIED;
    const roomId = camera.space_id ?? UNCLASSIFIED;
    let floor = floors.get(floorId);
    if (!floor) {
      floor = { id: floorId, label: floorId, rooms: [] };
      floors.set(floorId, floor);
    }
    let room = floor.rooms.find((entry) => entry.id === roomId);
    if (!room) {
      room = { id: roomId, label: camera.space_name ?? camera.space_id ?? UNCLASSIFIED, cameras: [] };
      floor.rooms.push(room);
    }
    room.cameras.push(camera);
  }
  return [...floors.values()];
}

export function filterAndGroupCameras(cameras: readonly Camera[], filters: CameraFilters): { cameras: Camera[]; groups: CameraFloorGroup[] } {
  const filtered = sortCameras(cameras.filter((camera) => (!filters.floor || camera.floor_name === filters.floor)
    && (!filters.room || camera.space_id === filters.room)));
  return { cameras: filtered, groups: groupCameras(filtered) };
}

export function paginateCameras(cameras: readonly Camera[], requestedPage: number): { cameras: Camera[]; page: number; pageCount: number } {
  const pageCount = Math.max(1, Math.ceil(cameras.length / WALL_PAGE_SIZE));
  const page = Math.min(Math.max(1, Math.trunc(requestedPage) || 1), pageCount);
  return { cameras: cameras.slice((page - 1) * WALL_PAGE_SIZE, page * WALL_PAGE_SIZE), page, pageCount };
}

export function snapshotJitterMs(cameraId: string): number {
  let hash = 0;
  for (const character of cameraId) hash = (hash * 31 + character.codePointAt(0)!) >>> 0;
  return hash % 1_000;
}

function resolveCameraHeartbeat(camera: Camera, cameras: readonly Camera[], status: StatusSnapshot | null): CameraHeartbeat | undefined {
  if (!status) return undefined;
  const heartbeats = Object.values(status.cameras);
  const exact = heartbeats.filter((entry) => entry.camera_id === camera.id);
  if (exact.length === 1) return exact[0];
  if (exact.length > 1 || !camera.backend_camera_id) return undefined;
  const alias = camera.backend_camera_id;
  if (cameras.filter((entry) => entry.id === alias || entry.backend_camera_id === alias).length !== 1) return undefined;
  const matches = heartbeats.filter((entry) => entry.camera_id === alias);
  return matches.length === 1 ? matches[0] : undefined;
}

export function getCameraLiveness(camera: Camera, cameras: readonly Camera[], status: StatusSnapshot | null): HeartbeatStatus | 'unknown' {
  return resolveCameraHeartbeat(camera, cameras, status)?.status ?? 'unknown';
}

export function getCameraHeartbeatAgeSec(camera: Camera, cameras: readonly Camera[], status: StatusSnapshot | null): number | null {
  return resolveCameraHeartbeat(camera, cameras, status)?.age_sec ?? null;
}

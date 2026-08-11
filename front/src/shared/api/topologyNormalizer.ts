import { isNonEmptyString, isNullableString, isRecord } from '@/shared/api/normalizerFields';
import type {
  CameraTopology,
  TopologyConfirmationResult,
  TopologyFloor,
  TopologyPreviewResponse,
  TopologyRoom,
  TopologySyncResult,
} from '@/shared/api/topologyTypes';

const READINESS = ['INVALID_SCHEMA', 'INVALID_TOPOLOGY', 'LEGACY_MAPPING_REQUIRED'] as const;
const SYNC_STATUSES = ['pending', 'synced', 'failed', 'disabled'] as const;
const SYNC_ERRORS = ['unconfigured', 'auth', 'timeout', 'unreachable', 'conflict'] as const;

function isInteger(value: unknown, minimum = 0): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= minimum;
}

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every(isNonEmptyString);
}

function isOneOf<T extends string>(value: unknown, allowed: readonly T[]): value is T {
  return typeof value === 'string' && allowed.some((item) => item === value);
}

function normalizeRoom(value: unknown): TopologyRoom | null {
  if (!isRecord(value) || !isNonEmptyString(value.edge_ref) || !isNonEmptyString(value.name)
    || value.room_type !== 'ROOM' || !isInteger(value.capacity)
    || !('legacy_canonical_space_id' in value) || !isNullableString(value.legacy_canonical_space_id)
    || !Array.isArray(value.cameras)) return null;
  const cameras = value.cameras.map((camera) => isRecord(camera) && isNonEmptyString(camera.edge_ref)
    && isNonEmptyString(camera.label) ? { edge_ref: camera.edge_ref, label: camera.label } : null);
  if (cameras.some((camera) => camera === null)) return null;
  return { edge_ref: value.edge_ref, name: value.name, room_type: 'ROOM', capacity: value.capacity,
    legacy_canonical_space_id: value.legacy_canonical_space_id, cameras: cameras.filter((camera) => camera !== null) };
}

function normalizeFloor(value: unknown): TopologyFloor | null {
  if (!isRecord(value) || !isNonEmptyString(value.edge_ref) || !isNonEmptyString(value.name)
    || !isInteger(value.order_index) || !Array.isArray(value.rooms)) return null;
  const rooms = value.rooms.map(normalizeRoom);
  if (rooms.some((room) => room === null)) return null;
  return { edge_ref: value.edge_ref, name: value.name, order_index: value.order_index,
    rooms: rooms.filter((room) => room !== null) };
}

export function normalizeCameraTopology(value: unknown): CameraTopology {
  if (!isRecord(value) || !isInteger(value.registry_version)
    || !(value.dirty_registry_version === null || isInteger(value.dirty_registry_version, 1))
    || !(value.readiness_error === null || isOneOf(value.readiness_error, READINESS))
    || !isStringArray(value.unmapped_camera_ids) || !Array.isArray(value.floors)) throw new Error('Invalid topology response');
  const floors = value.floors.map(normalizeFloor);
  if (floors.some((floor) => floor === null)) throw new Error('Invalid topology response');
  return { registry_version: value.registry_version, dirty_registry_version: value.dirty_registry_version,
    readiness_error: value.readiness_error,
    unmapped_camera_ids: value.unmapped_camera_ids, floors: floors.filter((floor) => floor !== null) };
}

export function normalizeTopologySync(value: unknown): TopologySyncResult {
  if (!isRecord(value) || !isOneOf(value.status, SYNC_STATUSES)
    || !(value.error_class === null || isOneOf(value.error_class, SYNC_ERRORS))
    || !('detail' in value) || !isNullableString(value.detail)
    || !('last_ok_at' in value) || !isNullableString(value.last_ok_at)
    || !('next_retry_at' in value) || !isNullableString(value.next_retry_at)
    || !isInteger(value.camera_count)) throw new Error('Invalid topology sync response');
  return { status: value.status, error_class: value.error_class,
    detail: value.detail, last_ok_at: value.last_ok_at, next_retry_at: value.next_retry_at, camera_count: value.camera_count };
}

export function normalizeTopologyPreview(value: unknown): TopologyPreviewResponse {
  if (!isRecord(value) || !('preview' in value)) throw new Error('Invalid topology preview response');
  if (value.preview === null) return { preview: null };
  const preview = value.preview;
  if (!isRecord(preview) || !isNonEmptyString(preview.confirmation_id) || !/^[a-f0-9]{64}$/.test(String(preview.digest))
    || !isNonEmptyString(preview.expires_at) || !isNonEmptyString(preview.snapshot_id)
    || !isInteger(preview.client_revision, 1) || !isInteger(preview.server_revision)
    || !isInteger(preview.cameras) || !isInteger(preview.rooms) || !isInteger(preview.floors)
    || typeof preview.confirmed !== 'boolean') throw new Error('Invalid topology preview response');
  return { preview: { confirmation_id: preview.confirmation_id, digest: String(preview.digest), expires_at: preview.expires_at,
    snapshot_id: preview.snapshot_id, client_revision: preview.client_revision, server_revision: preview.server_revision,
    cameras: preview.cameras, rooms: preview.rooms, floors: preview.floors, confirmed: preview.confirmed } };
}

export function normalizeTopologyConfirmation(value: unknown): TopologyConfirmationResult {
  if (!isRecord(value) || !isNonEmptyString(value.snapshot_id)
    || !isInteger(value.client_revision, 1) || !isInteger(value.server_revision)) throw new Error('Invalid topology confirmation response');
  return { snapshot_id: value.snapshot_id, client_revision: value.client_revision, server_revision: value.server_revision };
}

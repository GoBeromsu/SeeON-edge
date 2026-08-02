import {
  hasNullableString,
  isNonEmptyString,
  isNullableBoolean,
  isNullableFiniteNumber,
  isNullableInteger,
  isRecord,
  pickBoolean,
  pickNullableString,
  pickNumber,
  pickString,
} from '@/shared/api/normalizerFields';
import type {
  Camera,
  CameraRegistry,
  CameraStatus,
  CameraSync,
  CameraSyncErrorClass,
  CameraSyncStatus,
  CameraTestResult,
  DecodeBackend,
} from '@/shared/api/types';

const CAMERA_SYNC_STATUSES: readonly CameraSyncStatus[] = ['synced', 'pending', 'failed', 'disabled'];
const CAMERA_SYNC_ERROR_CLASSES: readonly CameraSyncErrorClass[] = ['unreachable', 'timeout', 'auth', 'unconfigured'];

// Permissive by design (unlike the strict connection normalizers): `sync` is a GET-only,
// still-rolling-out field, so a malformed or absent value degrades to null rather than
// failing the whole camera list.
function normalizeCameraSyncField(value: unknown): CameraSync | null {
  if (!isRecord(value)) {
    return null;
  }
  const status = value.status;
  if (typeof status !== 'string' || !CAMERA_SYNC_STATUSES.includes(status as CameraSyncStatus)) {
    return null;
  }
  const errorClass = value.error_class;
  if (errorClass !== null && !(typeof errorClass === 'string' && CAMERA_SYNC_ERROR_CLASSES.includes(errorClass as CameraSyncErrorClass))) {
    return null;
  }
  if (!hasNullableString(value, 'detail') || !hasNullableString(value, 'last_ok_at')) {
    return null;
  }
  return {
    status: status as CameraSyncStatus,
    error_class: (errorClass ?? null) as CameraSyncErrorClass | null,
    detail: pickNullableString(value, ['detail']),
    last_ok_at: pickNullableString(value, ['last_ok_at']),
  };
}

function normalizeStatus(record: Record<string, unknown>): CameraStatus {
  const explicit = pickString(record, ['status']);
  if (explicit === 'online' || explicit === 'offline' || explicit === 'starting' || explicit === 'unknown') {
    return explicit;
  }
  if (record.online === true) return 'online';
  if (record.online === false) return 'offline';
  return 'unknown';
}

function normalizeDecodeBackend(record: Record<string, unknown>): DecodeBackend | string | null {
  for (const key of ['decode_backend', 'decodeBackend']) {
    const value = record[key];
    if (value === null || typeof value === 'string') {
      return value;
    }
  }
  return null;
}

function maskRtsp(value: string | null): string {
  if (!value) {
    return 'RTSP URL 비공개';
  }
  try {
    const url = new URL(value);
    if (!url.hostname) {
      return 'RTSP URL 비공개';
    }
    url.hostname = 'redacted-camera';
    url.username = url.username ? '***' : '';
    url.password = url.password ? '***' : '';
    return url.toString();
  } catch {
    return 'RTSP URL 비공개';
  }
}

export function normalizeCamera(value: unknown): Camera | null {
  if (!isRecord(value)) {
    return null;
  }

  const id = pickString(value, ['id', 'camera_id']);
  if (!id) {
    return null;
  }

  const rtspMasked = pickString(value, ['rtsp_url_masked', 'rtspUrlMasked']);
  const rtspPlain = pickNullableString(value, ['rtsp_url', 'rtspUrl']);
  const domains = isRecord(value.domains) ? Object.fromEntries(Object.entries(value.domains).map(([key, entry]) => [key, Boolean(entry)])) : null;

  return {
    id,
    label: pickString(value, ['label', 'name'], '이름 없는 카메라'),
    rtsp_url_masked: rtspMasked || maskRtsp(rtspPlain),
    space_id: pickNullableString(value, ['space_id', 'spaceId']),
    space_name: pickNullableString(value, ['space_name', 'spaceName']),
    floor_name: pickNullableString(value, ['floor_name', 'floorName']),
    backend_camera_id: pickNullableString(value, ['backend_camera_id', 'backendCameraId', 'facilityId']),
    status: normalizeStatus(value),
    created_at: pickNullableString(value, ['created_at', 'createdAt']),
    threshold: pickNumber(value, ['threshold']),
    domains,
    bed_count: pickNumber(value, ['bed_count', 'bedCount']),
    night_start: pickNullableString(value, ['night_start', 'nightStart']),
    night_end: pickNullableString(value, ['night_end', 'nightEnd']),
    decode_backend: normalizeDecodeBackend(value),
    // Backend freshness fields (never_connected, last_ok_at, last_probed_at) are rolling out
    // separately; consume them defensively so the UI never crashes while they are still absent.
    never_connected: pickBoolean(value, ['never_connected', 'neverConnected']) ?? false,
    last_ok_at: pickNullableString(value, ['last_ok_at', 'lastOkAt']),
    last_probed_at: pickNullableString(value, ['last_probed_at', 'lastProbedAt']),
    // Heartbeat freshness is populated only by GET /cameras; POST/PATCH/test responses omit it
    // or send null, which is normal, not an error — pickNumber already returns null for anything
    // that isn't a finite number, so garbage or missing fields never crash or coerce.
    last_heartbeat_at: pickNumber(value, ['last_heartbeat_at', 'lastHeartbeatAt']),
    heartbeat_age_sec: pickNumber(value, ['heartbeat_age_sec', 'heartbeatAgeSec']),
    sync: normalizeCameraSyncField(value.sync),
  };
}

export function normalizeCameraRegistry(value: unknown): CameraRegistry {
  if (
    !isRecord(value)
    || !Number.isInteger(value.registry_version)
    || (value.registry_version as number) < 0
    || !Array.isArray(value.cameras)
    || !value.cameras.every(isCameraResponse)
  ) {
    throw new Error('Invalid camera registry response');
  }
  return {
    registry_version: value.registry_version as number,
    cameras: value.cameras.map(normalizeCameraResponse),
  };
}

export function normalizeCameraResponse(value: unknown): Camera {
  if (!isCameraResponse(value)) {
    throw new Error('Invalid camera response');
  }
  return normalizeCamera(value) as Camera;
}

function isCameraResponse(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  const status = value.status;
  return isNonEmptyString(value.id)
    && isNonEmptyString(value.label)
    && isNonEmptyString(value.rtsp_url_masked)
    && (status === 'online' || status === 'offline' || status === 'starting' || status === 'unknown')
    && (!('mapping_pending' in value) || typeof value.mapping_pending === 'boolean')
    && (!('never_connected' in value) || isNullableBoolean(value.never_connected))
    && hasNullableString(value, 'space_id')
    && hasNullableString(value, 'backend_camera_id')
    && hasNullableString(value, 'decode_backend')
    && hasNullableString(value, 'created_at')
    && hasNullableString(value, 'space_name')
    && hasNullableString(value, 'floor_name')
    && hasNullableString(value, 'last_ok_at')
    && hasNullableString(value, 'last_probed_at')
    && (!('last_heartbeat_at' in value) || isNullableFiniteNumber(value.last_heartbeat_at))
    && (!('heartbeat_age_sec' in value) || isNullableFiniteNumber(value.heartbeat_age_sec))
    && (!('sync' in value) || value.sync === null || isRecord(value.sync));
}

export function normalizeCameraTestResult(value: unknown): CameraTestResult {
  if (
    !isRecord(value)
    || typeof value.ok !== 'boolean'
    || ('error_class' in value && value.error_class !== null && value.error_class !== 'timeout' && value.error_class !== 'decode' && value.error_class !== 'auth')
    || ('width' in value && !isNullableInteger(value.width))
    || ('height' in value && !isNullableInteger(value.height))
  ) {
    throw new Error('Invalid camera test response');
  }
  const result: CameraTestResult = { ok: value.ok };
  if (value.error_class === 'timeout' || value.error_class === 'decode' || value.error_class === 'auth') result.error_class = value.error_class;
  if (typeof value.width === 'number') result.width = value.width;
  if (typeof value.height === 'number') result.height = value.height;
  return result;
}

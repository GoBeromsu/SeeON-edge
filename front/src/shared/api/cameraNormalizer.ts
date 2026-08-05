import {
  hasNullableInteger,
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
  BedZone,
  BedZonePoint,
  Camera,
  CameraRegistry,
  CameraStatus,
  CameraTestResult,
  DecodeBackend,
} from '@/shared/api/types';

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

function normalizeBedZonePoint(value: unknown): BedZonePoint | null {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const [x, y] = value;
  return typeof x === 'number' && typeof y === 'number' ? [x, y] : null;
}

/**
 * Defensive, never throws: bed-zone recognition (wave 1) may still be rolling out on the backend,
 * so a missing/malformed field normalizes to null ("인식 필요") rather than invalidating the whole
 * camera record.
 */
function normalizeBedZone(value: unknown): BedZone | null {
  if (!isRecord(value) || !Array.isArray(value.polygon)) return null;
  const polygon: BedZonePoint[] = [];
  for (const point of value.polygon) {
    const normalized = normalizeBedZonePoint(point);
    if (!normalized) return null;
    polygon.push(normalized);
  }
  const imageWidth = pickNumber(value, ['image_width', 'imageWidth']);
  const imageHeight = pickNumber(value, ['image_height', 'imageHeight']);
  const recognizedAt = pickNullableString(value, ['recognized_at', 'recognizedAt']);
  if (!polygon.length || imageWidth === null || imageHeight === null || !recognizedAt) return null;
  return { polygon, image_width: imageWidth, image_height: imageHeight, recognized_at: recognizedAt };
}

/** Strict counterpart to normalizeBedZone: throws on a malformed 200 response from the recognize endpoint. */
export function normalizeBedZoneRecognitionResponse(value: unknown): BedZone {
  const bedZone = isRecord(value) ? normalizeBedZone(value.bed_zone) : null;
  if (!bedZone) {
    throw new Error('Invalid bed-zone recognition response');
  }
  return bedZone;
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

  return {
    id,
    label: pickString(value, ['label', 'name'], '이름 없는 카메라'),
    rtsp_url_masked: rtspMasked || maskRtsp(rtspPlain),
    floor_name: pickNullableString(value, ['floor_name', 'floorName']),
    floor: pickNumber(value, ['floor']),
    space_id: pickNullableString(value, ['space_id', 'spaceId']),
    space_name: pickNullableString(value, ['space_name', 'spaceName']),
    // 클라우드 매핑 상태. 검증만 하고 버리면 기사님이 현장에서 "붙었는지"를
    // 볼 방법이 없다. 값이 없으면 아직 안 붙은 것으로 본다(보수적).
    backend_camera_id: pickNullableString(value, ['backend_camera_id', 'backendCameraId']),
    mapping_pending: pickBoolean(value, ['mapping_pending', 'mappingPending']) ?? true,
    status: normalizeStatus(value),
    created_at: pickNullableString(value, ['created_at', 'createdAt']),
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
    bed_zone: normalizeBedZone(value.bed_zone),
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
    && hasNullableString(value, 'decode_backend')
    && hasNullableString(value, 'created_at')
    && hasNullableString(value, 'floor_name')
    && hasNullableInteger(value, 'floor')
    && hasNullableString(value, 'last_ok_at')
    && hasNullableString(value, 'last_probed_at')
    && (!('last_heartbeat_at' in value) || isNullableFiniteNumber(value.last_heartbeat_at))
    && (!('heartbeat_age_sec' in value) || isNullableFiniteNumber(value.heartbeat_age_sec));
}

export function normalizeCameraTestResult(value: unknown): CameraTestResult {
  if (
    !isRecord(value)
    || typeof value.ok !== 'boolean'
    || ('error_class' in value && value.error_class !== null && value.error_class !== 'timeout' && value.error_class !== 'decode' && value.error_class !== 'auth')
    || ('probe_unavailable' in value && typeof value.probe_unavailable !== 'boolean')
    || ('width' in value && !isNullableInteger(value.width))
    || ('height' in value && !isNullableInteger(value.height))
  ) {
    throw new Error('Invalid camera test response');
  }
  const result: CameraTestResult = { ok: value.ok };
  if (value.error_class === 'timeout' || value.error_class === 'decode' || value.error_class === 'auth') result.error_class = value.error_class;
  if (value.probe_unavailable === true) result.probe_unavailable = true;
  if (typeof value.width === 'number') result.width = value.width;
  if (typeof value.height === 'number') result.height = value.height;
  return result;
}

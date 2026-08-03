import { HttpError, requestJson } from '@/shared/api/http';
import {
  normalizeBedZoneRecognitionResponse,
  normalizeCameraRegistry,
  normalizeCameraResponse,
  normalizeCameraTestResult,
  normalizeClipsResponse,
  normalizeClipStorageBrowse,
  normalizeClipStorageInfo,
  normalizeConnectionTestResult,
  normalizeConnectionView,
  normalizeDetectionSettings,
  normalizeStatusSnapshot,
  normalizeSystemSnapshot,
} from '@/shared/api/normalizers';
import { isRecord } from '@/shared/api/normalizerFields';
import type {
  BedZone,
  Camera,
  CameraInput,
  CameraPatchInput,
  CameraRegistry,
  CameraTestResult,
  Clip,
  ClipStorageBrowseResult,
  ClipStorageInfo,
  ConnectionInput,
  ConnectionTestResult,
  ConnectionView,
  DecodeBackend,
  DetectionSettings,
  DetectionSettingsInput,
  OverlayMode,
  StatusSnapshot,
  SystemSnapshot,
} from '@/shared/api/types';

export type {
  BedZone,
  BedZonePoint,
  Camera,
  CameraInput,
  CameraPatchInput,
  CameraRegistry,
  CameraStatus,
  CameraHeartbeat,
  CameraTestResult,
  Clip,
  ClipStorageBrowseEntry,
  ClipStorageBrowseResult,
  ClipStorageInfo,
  ConnectionErrorClass,
  ConnectionInput,
  ConnectionTestResult,
  ConnectionView,
  DecodeBackend,
  DetectionDomainKey,
  DetectionDomainSetting,
  DetectionMode,
  DetectionSettings,
  DetectionSettingsInput,
  HeartbeatStatus,
  OverlayMode,
  RuntimeCameraDiagnostics,
  RuntimeClipRecorder,
  RuntimeDecodeDiagnostics,
  RuntimeDeviceDiagnostics,
  RuntimeLatencyDiagnostics,
  RuntimeWorkerDiagnostics,
  StatusSnapshot,
  SystemSnapshot,
} from '@/shared/api/types';

export {
  getApiBase,
  getCameraSnapshotUrl,
  getCameraStreamUrl,
} from '@/shared/api/session';

export async function loginDashboard(username: string, password: string): Promise<void> {
  await requestJson('/auth/session', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
}

export async function fetchDashboardSession(): Promise<void> {
  await requestJson('/auth/session');
}

export async function logoutDashboard(): Promise<void> {
  await requestJson('/auth/session', { method: 'DELETE' });
}

/** Single admin account: no current-password confirmation — the caller is already an authenticated session. */
export async function updateDashboardCredentials(input: {
  username?: string;
  newPassword: string;
}): Promise<void> {
  const body: Record<string, unknown> = {
    new_password: input.newPassword,
  };
  if (input.username) {
    body.username = input.username;
  }
  await requestJson('/auth/credentials', {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

function cameraBody(input: CameraInput | CameraPatchInput, extra?: Record<string, unknown>): string {
  const body: Record<string, unknown> = {};
  if (input.label !== undefined) {
    body.label = input.label.trim();
  }
  if (input.rtsp_url !== undefined) {
    body.rtsp_url = input.rtsp_url.trim();
  }
  if ('decode_backend' in input && input.decode_backend !== undefined) {
    body.decode_backend = input.decode_backend;
  }
  if ('floor' in input && input.floor !== undefined) {
    body.floor = input.floor;
  }
  if (extra) {
    Object.assign(body, extra);
  }
  return JSON.stringify(body);
}

/**
 * Mirrors cameraBody()'s partial-update convention: a key is included in the request body only
 * when the caller's input explicitly sets it (`!== undefined`). An included key whose value is
 * `null` clears that field back to its env-seed default; an omitted key is left untouched by the
 * backend (see ConnectionSettingsUpdateRequest.model_fields_set in the backend router).
 */
function connectionBody(input: ConnectionInput): string {
  const body: Record<string, unknown> = {};
  if (input.events_url !== undefined) body.events_url = input.events_url;
  if (input.config_url !== undefined) body.config_url = input.config_url;
  if (input.facility_id !== undefined) body.facility_id = input.facility_id;
  if (input.facility_token !== undefined) body.facility_token = input.facility_token;
  return JSON.stringify(body);
}

export async function fetchConnection(signal?: AbortSignal): Promise<ConnectionView> {
  return normalizeConnectionView(await requestJson('/connection', { signal }));
}

export async function saveConnection(input: ConnectionInput): Promise<ConnectionView> {
  return normalizeConnectionView(await requestJson('/connection', { method: 'PUT', body: connectionBody(input) }));
}

/** Tests the current (possibly unsaved) form values via a probe-only body override, or the saved settings when omitted. */
export async function testConnection(input?: ConnectionInput): Promise<ConnectionTestResult> {
  return normalizeConnectionTestResult(
    await requestJson('/connection/test', { method: 'POST', body: input ? connectionBody(input) : undefined }),
  );
}

/** Structured 422 body the backend sends when probe-before-persist rejects a camera (`error_class`: timeout | auth | decode). */
export type CameraProbeFailureDetail = {
  error: 'probe_failed';
  error_class?: string;
};

/** Structured 409 body the backend sends when the normalized stream identity already exists. */
export type CameraDuplicateDetail = {
  error: 'duplicate_camera';
  existing_camera_id: string;
  existing_label: string;
};

function errorDetail(error: unknown): Record<string, unknown> | undefined {
  if (!(error instanceof HttpError) || typeof error.body !== 'object' || error.body === null) return undefined;
  const detail = (error.body as Record<string, unknown>).detail;
  return typeof detail === 'object' && detail !== null ? (detail as Record<string, unknown>) : undefined;
}

/** Narrows a caught `createCamera` error to a probe-before-persist rejection (HTTP 422), or returns undefined. */
export function cameraProbeFailureDetail(error: unknown): CameraProbeFailureDetail | undefined {
  if (!(error instanceof HttpError) || error.status !== 422) return undefined;
  const detail = errorDetail(error);
  if (!detail || detail.error !== 'probe_failed') return undefined;
  return {
    error: 'probe_failed',
    error_class: typeof detail.error_class === 'string' ? detail.error_class : undefined,
  };
}

/** Narrows a caught `createCamera` error to a duplicate-stream rejection (HTTP 409), or returns undefined. */
export function cameraDuplicateDetail(error: unknown): CameraDuplicateDetail | undefined {
  if (!(error instanceof HttpError) || error.status !== 409) return undefined;
  const detail = errorDetail(error);
  if (!detail || detail.error !== 'duplicate_camera' || typeof detail.existing_camera_id !== 'string') return undefined;
  return {
    error: 'duplicate_camera',
    existing_camera_id: detail.existing_camera_id,
    existing_label: typeof detail.existing_label === 'string' ? detail.existing_label : detail.existing_camera_id,
  };
}

export async function fetchCameras(signal?: AbortSignal): Promise<CameraRegistry> {
  return normalizeCameraRegistry(await requestJson('/cameras', { signal }));
}

export async function createCamera(input: CameraInput, options: { forceRegister?: boolean } = {}): Promise<Camera> {
  return normalizeCameraResponse(await requestJson('/cameras', {
    method: 'POST',
    body: cameraBody(input, options.forceRegister ? { force_register: true } : undefined),
  }));
}

export async function updateCamera(cameraId: string, input: CameraPatchInput): Promise<Camera> {
  return normalizeCameraResponse(
    await requestJson(`/cameras/${encodeURIComponent(cameraId)}`, { method: 'PATCH', body: cameraBody(input) }),
  );
}

export async function updateCameraDecodeBackend(cameraId: string, backend: DecodeBackend): Promise<Camera> {
  return updateCamera(cameraId, { decode_backend: backend });
}

export async function deleteCamera(cameraId: string): Promise<void> {
  await requestJson(`/cameras/${encodeURIComponent(cameraId)}`, { method: 'DELETE' });
}

/**
 * 카메라 연결 테스트.
 *
 * `rtspUrl`을 주면 저장된 값이 아니라 그 값을 검사한다. 수정 화면에서
 * 방금 입력한 URL을 검사해야 오타를 저장 전에 잡을 수 있다.
 */
export async function testCamera(cameraId: string, rtspUrl?: string): Promise<CameraTestResult> {
  return normalizeCameraTestResult(
    await requestJson(`/cameras/${encodeURIComponent(cameraId)}/test`, {
      method: 'POST',
      ...(rtspUrl ? { body: JSON.stringify({ rtsp_url: rtspUrl }) } : {}),
    }),
  );
}

/** Structured 422 body the backend sends when bed-zone recognition finds no bed in the current frame. */
export type BedZoneRecognitionFailureDetail = {
  error_class?: string;
};

/** Narrows a caught `recognizeBedZone` error to a "no bed detected" rejection (HTTP 422), or returns undefined. */
export function bedZoneRecognitionFailureDetail(error: unknown): BedZoneRecognitionFailureDetail | undefined {
  if (!(error instanceof HttpError) || error.status !== 422) return undefined;
  const detail = errorDetail(error);
  if (!detail) return undefined;
  return { error_class: typeof detail.error_class === 'string' ? detail.error_class : undefined };
}

/**
 * Runs one-shot YOLO bed segmentation against the camera's latest frame. 422 (bed_not_found) and 503
 * (worker/frame unavailable) are both surfaced as thrown HttpErrors -- callers keep showing "인식 필요".
 */
export async function recognizeBedZone(cameraId: string): Promise<BedZone> {
  return normalizeBedZoneRecognitionResponse(
    await requestJson(`/cameras/${encodeURIComponent(cameraId)}/bed-zone/recognize`, { method: 'POST' }),
  );
}

export async function fetchDetectionSettings(signal?: AbortSignal): Promise<DetectionSettings> {
  return normalizeDetectionSettings(await requestJson('/detection-settings', { signal }));
}

/** Full replace, not a partial update -- send the complete domains map back (see DetectionSettingsInput). */
export async function saveDetectionSettings(input: DetectionSettingsInput): Promise<DetectionSettings> {
  return normalizeDetectionSettings(
    await requestJson('/detection-settings', { method: 'PUT', body: JSON.stringify(input) }),
  );
}

export async function fetchClipStorage(signal?: AbortSignal): Promise<ClipStorageInfo> {
  return normalizeClipStorageInfo(await requestJson('/clips/storage', { signal }));
}

export async function browseClipStorage(path: string, signal?: AbortSignal): Promise<ClipStorageBrowseResult> {
  const query = path ? `?path=${encodeURIComponent(path)}` : '';
  return normalizeClipStorageBrowse(await requestJson(`/clips/storage/browse${query}`, { signal }));
}

/** path: "" selects the CLIP_STORE_DIR mount root itself; otherwise a relative subdirectory. */
export async function saveClipStorageLocation(path: string): Promise<ClipStorageInfo> {
  return normalizeClipStorageInfo(
    await requestJson('/clips/storage/location', { method: 'PUT', body: JSON.stringify({ path }) }),
  );
}

const OVERLAY_MODES: readonly OverlayMode[] = ['none', 'bedexit', 'fall'];

function normalizeOverlayResponse(value: unknown): OverlayMode {
  const mode = isRecord(value) ? value.mode : null;
  if (typeof mode !== 'string' || !OVERLAY_MODES.includes(mode as OverlayMode)) {
    throw new Error('Invalid overlay response');
  }
  return mode as OverlayMode;
}

export async function fetchCameraOverlay(cameraId: string): Promise<OverlayMode> {
  return normalizeOverlayResponse(await requestJson(`/streams/${encodeURIComponent(cameraId)}/pose`));
}

export async function setCameraOverlay(cameraId: string, mode: OverlayMode): Promise<OverlayMode> {
  return normalizeOverlayResponse(
    await requestJson(`/streams/${encodeURIComponent(cameraId)}/pose`, {
      method: 'POST',
      body: JSON.stringify({ mode }),
    }),
  );
}

export async function fetchStatus(signal?: AbortSignal): Promise<StatusSnapshot> {
  return normalizeStatusSnapshot(await requestJson('/status', { signal }));
}

export async function fetchSystem(signal?: AbortSignal): Promise<SystemSnapshot> {
  return normalizeSystemSnapshot(await requestJson('/system', { signal }));
}

export async function fetchClips(cameraId?: string, signal?: AbortSignal): Promise<Clip[]> {
  const query = cameraId?.trim() ? `?camera_id=${encodeURIComponent(cameraId.trim())}` : '';
  const value = await requestJson(`/clips${query}`, { signal });
  return normalizeClipsResponse(value);
}

import { HttpError, requestJson } from '@/shared/api/http';
import {
  normalizeCameraRegistry,
  normalizeCameraResponse,
  normalizeCameraTestResult,
  normalizeClipsResponse,
  normalizeConnectionTestResult,
  normalizeConnectionView,
  normalizeRosterSyncResult,
  normalizeStatusSnapshot,
  normalizeSystemSnapshot,
} from '@/shared/api/normalizers';
import type {
  Camera,
  CameraInput,
  CameraPatchInput,
  CameraRegistry,
  CameraTestResult,
  Clip,
  ClipLabel,
  ConnectionInput,
  ConnectionTestResult,
  ConnectionView,
  DecodeBackend,
  RosterSyncResult,
  StatusSnapshot,
  SystemSnapshot,
} from '@/shared/api/types';

export type {
  Camera,
  CameraInput,
  CameraPatchInput,
  CameraRegistry,
  CameraStatus,
  CameraHeartbeat,
  CameraSync,
  CameraSyncErrorClass,
  CameraSyncStatus,
  CameraTestResult,
  Clip,
  ClipLabel,
  ConnectionErrorClass,
  ConnectionInput,
  ConnectionTestResult,
  ConnectionView,
  DecodeBackend,
  HeartbeatStatus,
  RosterSyncErrorClass,
  RosterSyncResult,
  RosterSyncStatus,
  RuntimeCamera,
  RuntimeClipRecorder,
  RuntimeDecodeDiagnostics,
  RuntimeFacility,
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

export async function updateDashboardCredentials(input: {
  currentPassword: string;
  newUsername?: string;
  newPassword: string;
}): Promise<void> {
  const body: Record<string, unknown> = {
    current_password: input.currentPassword,
    new_password: input.newPassword,
  };
  if (input.newUsername) {
    body.username = input.newUsername;
  }
  await requestJson('/auth/credentials', {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

type ClipLabelOverlay = Pick<Clip, 'label' | 'reviewer' | 'reviewed_at'>;
const clipLabelOverlay = new Map<string, ClipLabelOverlay>();

export function cameraMediaId(camera: Pick<Camera, 'id' | 'backend_camera_id'>): string {
  return camera.backend_camera_id ?? camera.id;
}

function cameraBody(input: CameraInput | CameraPatchInput, extra?: Record<string, unknown>): string {
  const body: Record<string, unknown> = {};
  if (input.label !== undefined) {
    body.label = input.label.trim();
  }
  if (input.rtsp_url !== undefined) {
    body.rtsp_url = input.rtsp_url.trim();
  }
  if (input.space_id !== undefined) {
    body.space_id = input.space_id.trim();
  }
  if ('detectionSettings' in input && input.detectionSettings) {
    body.detectionSettings = input.detectionSettings;
  }
  if ('decode_backend' in input && input.decode_backend !== undefined) {
    body.decode_backend = input.decode_backend;
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

export async function syncCameras(): Promise<RosterSyncResult> {
  return normalizeRosterSyncResult(await requestJson('/connection/sync-cameras', { method: 'POST' }));
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

export async function testCamera(cameraId: string): Promise<CameraTestResult> {
  return normalizeCameraTestResult(await requestJson(`/cameras/${encodeURIComponent(cameraId)}/test`, { method: 'POST' }));
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
  return normalizeClipsResponse(value).map(applyClipLabelOverlay);
}

export async function labelClip(existing: Clip, label: ClipLabel): Promise<Clip> {
  const value = await requestJson(`/clips/${encodeURIComponent(existing.id)}/label`, {
      method: 'PUT',
      body: JSON.stringify({ label: label === 'UNREVIEWED' ? null : label }),
    });
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('Invalid clip response');
  }
  const response = value as Record<string, unknown>;
  if ((response.clip_id !== existing.id && response.id !== existing.id)
    || (response.label !== null && response.label !== 'TRUE_POSITIVE' && response.label !== 'FALSE_POSITIVE')
    || typeof response.reviewer !== 'string' || !response.reviewer
    || typeof response.reviewed_at !== 'string' || !response.reviewed_at) {
    throw new Error('Invalid clip response');
  }
  const overlay: ClipLabelOverlay = {
    label: response.label,
    reviewer: response.reviewer,
    reviewed_at: response.reviewed_at,
  };
  clipLabelOverlay.set(existing.id, overlay);
  return { ...existing, ...overlay, reviewState: 'confirmed' };
}

function applyClipLabelOverlay(clip: Clip): Clip {
  const overlay = clipLabelOverlay.get(clip.id);
  return overlay ? { ...clip, ...overlay, reviewState: 'confirmed' } : clip;
}

export function clearClipLabelOverlay(): void {
  clipLabelOverlay.clear();
}

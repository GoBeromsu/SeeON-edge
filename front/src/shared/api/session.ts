export const DEFAULT_API_BASE = '/api/v1';

export function getApiBase(): string {
  return normalizeApiBase(import.meta.env.VITE_ML_API_BASE_URL);
}

export function getCameraStreamUrl(cameraId: string): string {
  return mediaUrl(cameraId, '');
}

export function getCameraSnapshotUrl(cameraId: string, refreshKey: string | number): string {
  return mediaUrl(cameraId, '/snapshot', refreshKey);
}

export function getClipVideoUrl(clipId: string): string {
  return `${getApiBase()}/clips/${encodeURIComponent(clipId)}/video`;
}

export function getClipThumbnailUrl(clipId: string): string {
  return `${getApiBase()}/clips/${encodeURIComponent(clipId)}/thumbnail`;
}

function normalizeApiBase(value: string | undefined): string {
  const configured = value?.trim() || DEFAULT_API_BASE;
  if (configured === '/') return '';
  return configured.replace(/\/+$/, '');
}

function mediaUrl(cameraId: string, suffix: string, refreshKey?: string | number): string {
  const query = refreshKey === undefined ? '' : `?refresh=${encodeURIComponent(String(refreshKey))}`;
  return `${getApiBase()}/streams/${encodeURIComponent(cameraId)}${suffix}${query}`;
}

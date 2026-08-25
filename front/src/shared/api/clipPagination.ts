import { normalizeClip, normalizeClipPageResponse } from '@/shared/api/clipNormalizer';
import { requestJson } from '@/shared/api/http';
import type { ClipPage, ClipPageQuery } from '@/shared/api/clipPaginationTypes';
import type { Clip } from '@/shared/api/types';

export async function fetchClipPage(query: ClipPageQuery, signal?: AbortSignal): Promise<ClipPage> {
  const params = new URLSearchParams();
  const cameraId = query.cameraId?.trim();
  const eventType = query.eventType?.trim();
  if (cameraId) params.set('camera_id', cameraId);
  if (eventType) params.set('event_type', eventType);
  params.set('limit', String(query.limit));
  if (query.cursor) params.set('cursor', query.cursor);
  const value = await requestJson(`/clips?${params.toString()}`, { signal });
  return normalizeClipPageResponse(value, query);
}

export async function fetchClipMetadata(clipId: string, signal?: AbortSignal): Promise<Clip> {
  const value = await requestJson(`/clips/${encodeURIComponent(clipId)}/metadata`, { signal });
  const clip = normalizeClip(value);
  if (clip === null) throw new Error('Invalid clip metadata response');
  return clip;
}

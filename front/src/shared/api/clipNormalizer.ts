import { getClipVideoUrl } from '@/shared/api/session';
import {
  hasNullableNonNegativeNumber,
  hasNullableString,
  isNonEmptyString,
  isRecord,
  pickBoolean,
  pickNonNegativeNumber,
  pickNullableString,
  pickString,
} from '@/shared/api/normalizerFields';
import { toEventFacet } from '@/shared/api/clipEventFacet';
import type { ClipPage, ClipPageQuery, ClipPagination } from '@/shared/api/clipPaginationTypes';
import type { Clip } from '@/shared/api/types';

export function normalizeClip(value: unknown): Clip | null {
  if (!isRecord(value)) {
    return null;
  }
  const id = pickString(value, ['id', 'clip_id']);
  if (!id) {
    return null;
  }
  const videoAvailable = pickBoolean(value, ['video_available', 'videoAvailable']);
  const normalizedVideoAvailable = videoAvailable === true;
  return {
    id,
    camera_id: pickNullableString(value, ['camera_id', 'cameraId']),
    camera_label: pickString(value, ['camera_label', 'cameraLabel', 'camera'], '카메라 미상'),
    event_type: toEventFacet(pickString(value, ['event_type', 'eventType', 'type', 'event_ref', 'eventRef'])),
    created_at: pickNullableString(value, ['created_at', 'createdAt', 'timestamp', 'started_at', 'startedAt']),
    video_path: getClipVideoUrl(id),
    video_available: normalizedVideoAvailable,
    video_error: normalizedVideoAvailable
      ? null
      : videoAvailable === false
        ? '저장된 영상을 사용할 수 없습니다.'
        : '영상 제공 상태를 확인할 수 없습니다.',
    duration_s: pickNonNegativeNumber(value, ['duration_s', 'durationS']),
    size_bytes: pickNonNegativeNumber(value, ['size_bytes', 'sizeBytes']),
  };
}

export function normalizeClipsResponse(value: unknown): Clip[] {
  if (!isRecord(value) || !Array.isArray(value.clips) || !value.clips.every(isClipManifestResponse)) {
    throw new Error('Invalid clips response');
  }
  return value.clips.map((clip) => {
    const normalized = normalizeClip(clip);
    if (normalized === null) throw new Error('Invalid clips response');
    return normalized;
  });
}

export function normalizeClipPageResponse(value: unknown, query: ClipPageQuery): ClipPage {
  const clips = normalizeClipsResponse(value);
  if (!isRecord(value)) throw new Error('Invalid clips response');

  if ('pagination' in value || 'event_type_counts' in value) {
    const pagination = normalizePagination(value.pagination);
    const eventTypeCounts = normalizeEventTypeCounts(value.event_type_counts);
    return { clips, pagination, event_type_counts: eventTypeCounts, complete_clips: null };
  }

  const cameraClips = query.cameraId
    ? clips.filter((clip) => clip.camera_id === query.cameraId)
    : clips;
  const eventTypeCounts: Record<string, number> = {};
  for (const clip of cameraClips) {
    eventTypeCounts[clip.event_type] = (eventTypeCounts[clip.event_type] ?? 0) + 1;
  }
  const filtered = query.eventType
    ? cameraClips.filter((clip) => clip.event_type === query.eventType)
    : cameraClips;
  const pageClips = filtered.slice(query.offset, query.offset + query.limit);
  return {
    clips: pageClips,
    pagination: {
      limit: query.limit,
      offset: query.offset,
      total: filtered.length,
      has_more: query.offset + pageClips.length < filtered.length,
    },
    event_type_counts: eventTypeCounts,
    complete_clips: clips,
  };
}

function normalizePagination(value: unknown): ClipPagination {
  if (!isRecord(value)
    || !isPositiveInteger(value.limit)
    || !isNonNegativeInteger(value.offset)
    || !isNonNegativeInteger(value.total)
    || typeof value.has_more !== 'boolean') {
    throw new Error('Invalid clips pagination response');
  }
  return {
    limit: value.limit,
    offset: value.offset,
    total: value.total,
    has_more: value.has_more,
  };
}

function normalizeEventTypeCounts(value: unknown): Readonly<Record<string, number>> {
  if (!isRecord(value)) throw new Error('Invalid clips pagination response');
  const counts: Record<string, number> = {};
  for (const [eventType, count] of Object.entries(value)) {
    if (!eventType.trim() || !isNonNegativeInteger(count)) {
      throw new Error('Invalid clips pagination response');
    }
    const facet = toEventFacet(eventType);
    counts[facet] = (counts[facet] ?? 0) + count;
  }
  return counts;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

function isClipManifestResponse(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  return isNonEmptyString(value.clip_id)
    && isNonEmptyString(value.camera_id)
    && isNonEmptyString(value.event_ref)
    && isNonEmptyString(value.started_at)
    && typeof value.duration_s === 'number'
    && Number.isFinite(value.duration_s)
    && value.duration_s >= 0
    && typeof value.video_available === 'boolean'
    && typeof value.finalized === 'boolean'
    && (!('event_type' in value) || value.event_type === null || isNonEmptyString(value.event_type))
    && (!('codec' in value) || typeof value.codec === 'string')
    && hasNullableString(value, 'path')
    && hasNullableString(value, 'video_error')
    && hasNullableNonNegativeNumber(value, 'size_bytes');
}

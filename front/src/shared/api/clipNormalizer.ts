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
    event_type: pickString(value, ['event_type', 'eventType', 'type', 'event_ref', 'eventRef'], '이벤트'),
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
  return value.clips.map((clip) => normalizeClip(clip) as Clip);
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

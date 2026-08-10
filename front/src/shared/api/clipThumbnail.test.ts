import { describe, expect, it } from 'vitest';
import { normalizeClip, normalizeClipsResponse } from '@/shared/api/normalizers';
import { getClipThumbnailUrl } from '@/shared/api/session';

const validClipManifest = {
  clip_id: 'clip-1',
  camera_id: 'cam-1',
  event_ref: 'event-1',
  event_type: 'fall',
  started_at: '2026-08-02T03:12:00Z',
  duration_s: 12,
  codec: 'h264',
  path: null,
  video_available: true,
  video_error: null,
  finalized: true,
};

describe('clip thumbnail contract', () => {
  it('builds a deterministic API URL from the clip id only', () => {
    expect(getClipThumbnailUrl('clip/path 1')).toBe('/api/v1/clips/clip%2Fpath%201/thumbnail');
  });

  it('preserves availability without trusting a server filesystem path', () => {
    const clip = normalizeClip({
      id: 'clip-1',
      thumbnail_available: true,
      thumbnail_path: '/var/lib/clip-store/clip-1/thumbnail.jpg',
    });

    expect(clip?.thumbnail_available).toBe(true);
    expect(clip).not.toHaveProperty('thumbnail_path');
  });

  it.each([
    ['missing', {}],
    ['null', { thumbnail_available: null }],
    ['string', { thumbnail_available: 'true' }],
    ['false', { thumbnail_available: false }],
  ])('uses the placeholder contract for a %s availability value', (_case, availability) => {
    expect(normalizeClip({ id: 'clip-1', ...availability })?.thumbnail_available).toBe(false);
  });

  it('rejects a malformed thumbnail flag in a clip-list success envelope', () => {
    expect(() => normalizeClipsResponse({
      clips: [{ ...validClipManifest, thumbnail_available: 'yes' }],
    })).toThrow('Invalid clips response');
  });

  it('accepts an older clip-list envelope that omits the thumbnail flag', () => {
    expect(normalizeClipsResponse({ clips: [validClipManifest] })[0]?.thumbnail_available).toBe(false);
  });
});

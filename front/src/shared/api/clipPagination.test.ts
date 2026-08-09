import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchClipMetadata, fetchClipPage } from '@/shared/api/clipPagination';

function clipManifest(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
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
    ...overrides,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('clip pagination API', () => {
  it('sends server filters with the explicit page limit and preserves exact facet counts', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        clips: [clipManifest({ clip_id: 'clip-97' })],
        pagination: { limit: 48, offset: 96, total: 145, has_more: true },
        event_type_counts: { fall: 100, 'bed-exit': 45 },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const page = await fetchClipPage({ cameraId: 'cam/1', eventType: 'fall', limit: 48, offset: 96 });

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/clips?camera_id=cam%2F1&event_type=fall&limit=48&offset=96',
      expect.objectContaining({ credentials: 'same-origin' }),
    );
    expect(page.pagination).toEqual({ limit: 48, offset: 96, total: 145, has_more: true });
    expect(page.event_type_counts).toEqual({ fall: 100, 'bed-exit': 45 });
  });

  it('collapses thousands of unknown raw count keys and a missing event type into the other facet', async () => {
    const total = 9_313;
    const unknownTypeCount = 3_000;
    const rawCounts = Object.fromEntries(Array.from(
      { length: unknownTypeCount },
      (_, index) => [`vendor-event-${index}`, index === 0 ? total - unknownTypeCount + 1 : 1],
    ));
    const clips = Array.from({ length: 48 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `unique-event-${index + 1}`,
      event_type: index === 0 ? null : `vendor-event-${index}`,
    }));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        clips,
        pagination: { limit: 48, offset: 0, total, has_more: true },
        event_type_counts: rawCounts,
      }),
    }));

    const page = await fetchClipPage({ limit: 48, offset: 0 });

    expect(page.clips).toHaveLength(48);
    expect(page.clips.every((clip) => clip.event_type === 'other')).toBe(true);
    expect(page.pagination.total).toBe(total);
    expect(page.event_type_counts).toEqual({ other: total });
  });

  it('normalizes and slices a complete legacy response after applying both filters locally', async () => {
    const clips = [
      clipManifest({ clip_id: 'clip-1', camera_id: 'cam-1', event_type: 'fall' }),
      clipManifest({ clip_id: 'clip-2', camera_id: 'cam-1', event_type: 'bed-exit' }),
      clipManifest({ clip_id: 'clip-3', camera_id: 'cam-1', event_type: 'fall' }),
      clipManifest({ clip_id: 'clip-4', camera_id: 'cam-2', event_type: 'fall' }),
      clipManifest({ clip_id: 'clip-5', camera_id: 'cam-1', event_type: 'fall' }),
    ];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ clips }),
    }));

    const page = await fetchClipPage({ cameraId: 'cam-1', eventType: 'fall', limit: 2, offset: 1 });

    expect(page.clips.map((clip) => clip.id)).toEqual(['clip-3', 'clip-5']);
    expect(page.pagination).toEqual({ limit: 2, offset: 1, total: 3, has_more: false });
    expect(page.event_type_counts).toEqual({ fall: 3, 'bed-exit': 1 });
    expect(page.complete_clips).toHaveLength(5);
  });

  it('loads one clip metadata record for an off-page deep link', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => clipManifest({ clip_id: 'historical/clip' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const clip = await fetchClipMetadata('historical/clip');

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/clips/historical%2Fclip/metadata',
      expect.objectContaining({ credentials: 'same-origin' }),
    );
    expect(clip.id).toBe('historical/clip');
    expect(clip.video_path).toBe('/api/v1/clips/historical%2Fclip/video');
  });
});

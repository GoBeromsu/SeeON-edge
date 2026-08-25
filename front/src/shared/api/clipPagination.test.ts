import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchClipMetadata, fetchClipPage } from '@/shared/api/clipPagination';
import { decodeClipCursor, encodeClipCursor } from '@/shared/api/clipCursor';

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
    const cursor = encodeClipCursor({ startedAt: '2026-08-02T03:12:00Z', clipId: 'clip-96' });
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        clips: [clipManifest({ clip_id: 'clip-97' })],
        pagination: { limit: 48, offset: 0, total: 145, has_more: true, next_cursor: 'bmV4dA==' },
        event_type_counts: { fall: 100, 'bed-exit': 45 },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const page = await fetchClipPage({ cameraId: 'cam/1', eventType: 'fall', limit: 48, cursor });

    expect(fetchMock).toHaveBeenCalledWith(
      `/api/v1/clips?camera_id=cam%2F1&event_type=fall&limit=48&cursor=${encodeURIComponent(cursor)}`,
      expect.objectContaining({ credentials: 'same-origin' }),
    );
    expect(page.pagination).toEqual({ limit: 48, total: 145, has_more: true, next_cursor: 'bmV4dA==' });
    expect(page.event_type_counts).toEqual({ fall: 100, 'bed-exit': 45 });
  });

  it('omits the cursor parameter for the newest page and never sends offset', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        clips: [clipManifest()],
        pagination: { limit: 48, offset: 0, total: 1, has_more: false, next_cursor: null },
        event_type_counts: { fall: 1 },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await fetchClipPage({ limit: 48 });

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/clips?limit=48', expect.objectContaining({ credentials: 'same-origin' }));
  });

  it('treats has_more without a next cursor as the last page instead of re-requesting the boundary', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        clips: [clipManifest()],
        pagination: { limit: 48, offset: 0, total: 90, has_more: true, next_cursor: null },
        event_type_counts: { fall: 90 },
      }),
    }));

    const page = await fetchClipPage({ limit: 48 });

    expect(page.pagination.has_more).toBe(false);
    expect(page.pagination.next_cursor).toBeNull();
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

    const page = await fetchClipPage({ limit: 48 });

    expect(page.clips).toHaveLength(48);
    expect(page.clips.every((clip) => clip.event_type === 'other')).toBe(true);
    expect(page.pagination.total).toBe(total);
    expect(page.event_type_counts).toEqual({ other: total });
  });

  it('normalizes and keyset-pages a complete legacy response after applying both filters locally', async () => {
    const clips = [
      clipManifest({ clip_id: 'clip-1', camera_id: 'cam-1', event_type: 'fall', started_at: '2026-08-02T05:00:00Z' }),
      clipManifest({ clip_id: 'clip-2', camera_id: 'cam-1', event_type: 'bed-exit', started_at: '2026-08-02T04:00:00Z' }),
      clipManifest({ clip_id: 'clip-3', camera_id: 'cam-1', event_type: 'fall', started_at: '2026-08-02T03:00:00Z' }),
      clipManifest({ clip_id: 'clip-4', camera_id: 'cam-2', event_type: 'fall', started_at: '2026-08-02T02:00:00Z' }),
      clipManifest({ clip_id: 'clip-5', camera_id: 'cam-1', event_type: 'fall', started_at: '2026-08-02T01:00:00Z' }),
    ];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ clips }),
    }));

    const first = await fetchClipPage({ cameraId: 'cam-1', eventType: 'fall', limit: 2 });

    expect(first.clips.map((clip) => clip.id)).toEqual(['clip-1', 'clip-3']);
    expect(first.pagination).toEqual({
      limit: 2,
      total: 3,
      has_more: true,
      next_cursor: encodeClipCursor({ startedAt: '2026-08-02T03:00:00Z', clipId: 'clip-3' }),
    });
    expect(first.event_type_counts).toEqual({ fall: 3, 'bed-exit': 1 });
    expect(first.complete_clips).toHaveLength(5);

    const second = await fetchClipPage({
      cameraId: 'cam-1', eventType: 'fall', limit: 2, cursor: first.pagination.next_cursor as string,
    });

    expect(second.clips.map((clip) => clip.id)).toEqual(['clip-5']);
    expect(second.pagination.has_more).toBe(false);
    expect(second.pagination.next_cursor).toBeNull();
  });

  it('walks equal-timestamp rows on the clip-id tiebreak without duplicating or skipping any row', async () => {
    // Every clip shares one started_at, so only the secondary key can order the pages.
    const startedAt = '2026-08-02T03:12:00Z';
    const clips = Array.from({ length: 7 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
      started_at: startedAt,
    }));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ clips }) }));

    const seen: string[] = [];
    const boundaries: string[] = [];
    let cursor: string | null = null;
    for (let guard = 0; guard < 10; guard += 1) {
      const page: Awaited<ReturnType<typeof fetchClipPage>> = await fetchClipPage(
        cursor === null ? { limit: 3 } : { limit: 3, cursor },
      );
      seen.push(...page.clips.map((clip) => clip.id));
      if (page.pagination.next_cursor === null) break;
      boundaries.push(page.pagination.next_cursor);
      cursor = page.pagination.next_cursor;
    }

    // Descending clip-id order, every row exactly once: no duplicate, no gap.
    expect(seen).toEqual(['clip-7', 'clip-6', 'clip-5', 'clip-4', 'clip-3', 'clip-2', 'clip-1']);
    expect(new Set(seen).size).toBe(clips.length);
    expect(boundaries.map((value) => decodeClipCursor(value))).toEqual([
      { startedAt, clipId: 'clip-5' },
      { startedAt, clipId: 'clip-2' },
    ]);
  });

  it('restores the previous equal-timestamp boundary to the identical rows', async () => {
    const startedAt = '2026-08-02T03:12:00Z';
    const clips = Array.from({ length: 7 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
      started_at: startedAt,
    }));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ clips }) }));
    const firstCursor = encodeClipCursor({ startedAt, clipId: 'clip-5' });

    const forward = await fetchClipPage({ limit: 3, cursor: firstCursor });
    const restored = await fetchClipPage({ limit: 3, cursor: firstCursor });

    expect(forward.clips.map((clip) => clip.id)).toEqual(['clip-4', 'clip-3', 'clip-2']);
    expect(restored.clips.map((clip) => clip.id)).toEqual(forward.clips.map((clip) => clip.id));
    expect(restored.pagination.next_cursor).toBe(forward.pagination.next_cursor);
  });

  it('walks astral-character clip ids in SQLite BINARY order without skipping a row', async () => {
    // Recorded from real SQLite `ORDER BY started_at DESC, clip_id DESC` over these exact ids.
    const sqliteDescending = ['\u{1F600}', '\u{10000}', '\uE000', '\uAC00', 'zz', 'ascii'];
    const startedAt = '2026-08-02T03:12:00Z';
    const clips = sqliteDescending.map((clipId, index) => clipManifest({
      clip_id: clipId, event_ref: `event-${index}`, started_at: startedAt,
    }));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ clips }) }));

    const seen: string[] = [];
    let cursor: string | null = null;
    for (let guard = 0; guard < 10; guard += 1) {
      const page: Awaited<ReturnType<typeof fetchClipPage>> = await fetchClipPage(
        cursor === null ? { limit: 2 } : { limit: 2, cursor },
      );
      seen.push(...page.clips.map((clip) => clip.id));
      if (page.pagination.next_cursor === null) break;
      cursor = page.pagination.next_cursor;
    }

    expect(seen).toEqual(sqliteDescending);
    expect(new Set(seen).size).toBe(sqliteDescending.length);
  });

  it('rejects a malformed cursor instead of silently returning the newest page', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true, status: 200, json: async () => ({ clips: [clipManifest()] }),
    }));

    await expect(fetchClipPage({ limit: 3, cursor: '!!!not-base64!!!' })).rejects.toThrow('Invalid clips cursor');
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

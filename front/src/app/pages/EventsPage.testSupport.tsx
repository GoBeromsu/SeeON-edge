import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { vi } from 'vitest';
import { EventsPage } from '@/app/pages/EventsPage';
import { decodeClipCursor, encodeClipCursor } from '@/shared/api/clipCursor';

const activeRoots = new Set<Root>();

export function resetLocation(search = '?page=events'): void {
  window.history.replaceState(null, '', `/${search}`);
}

export const cameraRegistry = {
  registry_version: 1,
  cameras: [
    { id: 'cam-1', label: '301호', rtsp_url_masked: 'rtsp://***', status: 'online' },
    { id: 'cam-2', label: '302호', rtsp_url_masked: 'rtsp://***', status: 'online' },
  ],
} as const;

export function clipManifest(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    clip_id: 'clip-1', camera_id: 'cam-1', event_ref: 'event-1', event_type: 'fall',
    started_at: '2026-08-02T03:12:00Z', duration_s: 12, codec: 'h264', path: null,
    video_available: true, thumbnail_available: true, video_error: null, finalized: true, ...overrides,
  };
}

export const allClips = [
  clipManifest({ clip_id: 'clip-1', camera_id: 'cam-1', event_type: 'fall', started_at: '2026-08-02T03:12:00Z' }),
  clipManifest({ clip_id: 'clip-2', camera_id: 'cam-1', event_type: 'bed-exit', started_at: '2026-08-02T02:00:00Z' }),
  clipManifest({
    clip_id: 'clip-3', camera_id: 'cam-2', event_type: 'fall', started_at: '2026-08-02T01:00:00Z',
    video_available: false, thumbnail_available: false, video_error: '저장된 영상을 사용할 수 없습니다.',
  }),
  clipManifest({ clip_id: 'clip-4', camera_id: 'cam-2', event_type: 'bed-exit', started_at: '2026-08-02T00:00:00Z' }),
];

export function installFetchMock(clips: readonly Record<string, unknown>[] = allClips): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/cameras')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
    }
    if (url.includes('/incidents')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ incidents: [] }) });
    }
    if (url.includes('/clips')) {
      const params = new URL(url, 'http://localhost').searchParams;
      const cameraId = params.get('camera_id');
      const eventType = params.get('event_type');
      const limit = Number(params.get('limit') ?? clips.length);
      const cameraClips = cameraId ? clips.filter((clip) => clip.camera_id === cameraId) : clips;
      const eventTypeCounts = cameraClips.reduce<Record<string, number>>((counts, clip) => {
        const type = String(clip.event_type);
        counts[type] = (counts[type] ?? 0) + 1;
        return counts;
      }, {});
      const filtered = eventType ? cameraClips.filter((clip) => clip.event_type === eventType) : cameraClips;
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => keysetBody(filtered, limit, params.get('cursor')),
      });
    }
    return Promise.reject(new Error(`unexpected fetch: ${url}`));
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/**
 * Mirrors the backend `clips` keyset page: `ORDER BY started_at DESC, clip_id DESC` with the strict
 * `started_at < ? OR (started_at = ? AND clip_id < ?)` boundary, so equal timestamps can only be
 * separated by the clip-id tiebreak -- never by a positional slice.
 */
export function keysetBody(
  clips: readonly Record<string, unknown>[],
  limit: number,
  cursor: string | null,
  eventTypeCounts?: Readonly<Record<string, number>>,
): Record<string, unknown> {
  const counts = eventTypeCounts ?? clips.reduce<Record<string, number>>((accumulator, clip) => {
    const type = String(clip.event_type);
    accumulator[type] = (accumulator[type] ?? 0) + 1;
    return accumulator;
  }, {});
  const ordered = [...clips].sort((left, right) => {
    if (left.started_at !== right.started_at) return String(left.started_at) < String(right.started_at) ? 1 : -1;
    return String(left.clip_id) < String(right.clip_id) ? 1 : -1;
  });
  const boundary = cursor === null ? null : decodeClipCursor(cursor);
  if (cursor !== null && boundary === null) throw new Error(`invalid cursor: ${cursor}`);
  const remaining = boundary === null ? ordered : ordered.filter((clip) => {
    const startedAt = String(clip.started_at);
    if (startedAt !== boundary.startedAt) return startedAt < boundary.startedAt;
    return String(clip.clip_id) < boundary.clipId;
  });
  const page = remaining.slice(0, limit);
  const hasMore = remaining.length > page.length;
  const last = page.at(-1);
  return {
    clips: page,
    pagination: {
      limit,
      offset: 0,
      total: clips.length,
      has_more: hasMore,
      next_cursor: hasMore && last
        ? encodeClipCursor({ startedAt: String(last.started_at), clipId: String(last.clip_id) })
        : null,
    },
    event_type_counts: counts,
  };
}

export function clipCursorOf(url: string): string | null {
  return new URL(url, 'http://localhost').searchParams.get('cursor');
}

export function installLegacyFetchMock(clips: readonly Record<string, unknown>[]): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/cameras')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
    }
    if (url.includes('/incidents')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ incidents: [] }) });
    }
    if (url.includes('/clips')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ clips }) });
    }
    return Promise.reject(new Error(`unexpected fetch: ${url}`));
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

export function clipRequestUrls(fetchMock: ReturnType<typeof vi.fn>): string[] {
  return fetchMock.mock.calls
    .map(([input]) => typeof input === 'string' ? input : String(input))
    .filter((url) => url.includes('/clips'));
}

export async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

export async function renderPage(): Promise<{ readonly host: HTMLDivElement; readonly root: Root }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  activeRoots.add(root);
  act(() => root.render(<EventsPage />));
  await flush();
  return { host, root };
}

export function cleanupPages(): void {
  act(() => {
    activeRoots.forEach((root) => root.unmount());
  });
  activeRoots.clear();
}

import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  cameraRegistry,
  cleanupPages,
  clipCursorOf,
  clipManifest,
  clipRequestUrls,
  flush,
  installFetchMock,
  installLegacyFetchMock,
  keysetBody,
  pageOrdinal,
  renderPage,
  resetLocation,
} from '@/app/pages/EventsPage.testSupport';
import { decodeClipCursor } from '@/shared/api/clipCursor';

function cardIds(host: HTMLDivElement): string[] {
  return Array.from(host.querySelectorAll<HTMLElement>('button.rounded-card'))
    .map((card) => card.dataset.clipId ?? '');
}

function clickPager(host: HTMLDivElement, label: string): void {
  const button = host.querySelector(`button[aria-label="${label}"]`) as HTMLButtonElement;
  if (!button) throw new Error(`missing pager control ${label}`);
  act(() => button.click());
}

afterEach(() => {
  cleanupPages();
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('EventsPage keyset pagination', () => {
  it('bounds a complete legacy response to 48 cards and exposes the keyset pager', async () => {
    resetLocation();
    const clips = Array.from({ length: 50 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
      started_at: `2026-08-02T${String(23 - (index % 24)).padStart(2, '0')}:00:00Z`,
    }));
    installLegacyFetchMock(clips);

    const { host } = await renderPage();

    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    const pager = host.querySelector('nav[aria-label="이벤트 페이지"]') as HTMLElement;
    expect(pager).not.toBeNull();
    expect((pager.querySelector('button[aria-label="다음 페이지"]') as HTMLButtonElement).disabled).toBe(false);
    expect((pager.querySelector('button[aria-label="이전 페이지"]') as HTMLButtonElement).disabled).toBe(true);
    // Keyset pages are not addressable, so no numbered jump target may exist.
    expect(pager.querySelector('button[aria-label="2페이지"]')).toBeNull();
    expect(pager.querySelector('[aria-current="page"]')).toBeNull();
  });

  it('resolves an off-page deep link from a complete legacy response without a metadata endpoint', async () => {
    resetLocation('?page=events&clip=clip-50');
    const clips = Array.from({ length: 50 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
    }));
    const fetchMock = installLegacyFetchMock(clips);

    await renderPage();

    expect(window.location.search).toContain('clip=clip-50');
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('낙상');
    expect(clipRequestUrls(fetchMock).filter((url) => url.endsWith('/clips/clip-50/metadata'))).toHaveLength(0);
  });

  it('uses backend facet counts instead of counts from the visible page', async () => {
    resetLocation();
    const clips = Array.from({ length: 96 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
      event_type: index < 70 ? 'fall' : 'bed-exit',
    }));
    installFetchMock(clips);

    const { host } = await renderPage();

    expect(host.textContent).toContain('전체 96');
    expect(host.textContent).toContain('낙상 70');
    expect(host.textContent).toContain('침대 이탈 26');
  });

  it('advances with the server cursor and commits the page only after it succeeds', async () => {
    resetLocation();
    const clips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${String(index + 1).padStart(3, '0')}`,
      event_ref: `event-${index + 1}`,
      started_at: `2026-08-${String(1 + Math.floor(index / 24)).padStart(2, '0')}T${String(index % 24).padStart(2, '0')}:00:00Z`,
    }));
    const fetchMock = installFetchMock(clips);
    const { host } = await renderPage();
    const firstPageIds = cardIds(host);

    clickPager(host, '다음 페이지');
    await flush();

    const secondPageIds = cardIds(host);
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(secondPageIds.some((id) => firstPageIds.includes(id))).toBe(false);
    const lastRequest = clipRequestUrls(fetchMock).at(-1) as string;
    expect(lastRequest).not.toContain('offset=');
    expect(decodeClipCursor(clipCursorOf(lastRequest) as string)?.clipId).toBe(firstPageIds.at(-1));
  });

  it('walks equal-timestamp rows forward and back without duplicating or skipping a clip', async () => {
    resetLocation();
    // 97 clips share one timestamp: only the clip-id tiebreak can separate the pages.
    const startedAt = '2026-08-02T03:12:00Z';
    const clips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${String(index + 1).padStart(3, '0')}`,
      event_ref: `event-${index + 1}`,
      started_at: startedAt,
    }));
    const fetchMock = installFetchMock(clips);
    const { host } = await renderPage();

    const pageOne = cardIds(host);
    clickPager(host, '다음 페이지');
    await flush();
    const pageTwo = cardIds(host);
    clickPager(host, '다음 페이지');
    await flush();
    const pageThree = cardIds(host);

    const walked = [...pageOne, ...pageTwo, ...pageThree];
    expect(walked).toHaveLength(97);
    expect(new Set(walked).size).toBe(97);
    expect(walked).toEqual([...clips].map((clip) => String(clip.clip_id)).sort().reverse());
    expect((host.querySelector('button[aria-label="다음 페이지"]') as HTMLButtonElement).disabled).toBe(true);
    // Every forward request carried an equal-timestamp cursor whose tiebreak is the previous last row.
    const cursors = clipRequestUrls(fetchMock).map(clipCursorOf).filter((value): value is string => value !== null);
    expect(cursors.map((value) => decodeClipCursor(value))).toEqual([
      { startedAt, clipId: pageOne.at(-1) },
      { startedAt, clipId: pageTwo.at(-1) },
    ]);

    clickPager(host, '이전 페이지');
    await flush();
    expect(cardIds(host)).toEqual(pageTwo);

    clickPager(host, '이전 페이지');
    await flush();
    expect(cardIds(host)).toEqual(pageOne);
    expect((host.querySelector('button[aria-label="이전 페이지"]') as HTMLButtonElement).disabled).toBe(true);
  });

  it('keeps the current page and cards when cursor navigation fails', async () => {
    resetLocation();
    const clips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${String(index + 1).padStart(3, '0')}`,
      event_ref: `event-${index + 1}`,
    }));
    const fetchMock = installFetchMock(clips);
    const { host } = await renderPage();
    const firstPageIds = cardIds(host);
    fetchMock.mockRejectedValueOnce(new Error('page unavailable'));

    clickPager(host, '다음 페이지');
    await flush();

    expect(cardIds(host)).toEqual(firstPageIds);
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(host.textContent).toContain('현재 페이지를 유지합니다.');
  });

  it('polls the newest page every 8 seconds and stops polling on a cursor page', async () => {
    vi.useFakeTimers();
    resetLocation();
    const clips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${String(index + 1).padStart(3, '0')}`,
      event_ref: `event-${index + 1}`,
    }));
    let clipsCallCount = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.includes('/cameras')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
      }
      if (url.includes('/incidents')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => ({ incidents: [] }) });
      }
      clipsCallCount += 1;
      if (clipsCallCount === 2) return Promise.reject(new Error('background unavailable'));
      const params = new URL(url, 'http://localhost').searchParams;
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => keysetBody(clips, Number(params.get('limit')), params.get('cursor'), { fall: clips.length }),
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const { host } = await renderPage();

    await act(async () => { await vi.advanceTimersByTimeAsync(8_000); });

    expect(clipsCallCount).toBe(2);
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(host.textContent).toContain('현재 페이지를 유지합니다.');

    clickPager(host, '다음 페이지');
    await flush();
    expect(clipsCallCount).toBe(3);
    expect(pageOrdinal(host)).toBe(2);

    await act(async () => { await vi.advanceTimersByTimeAsync(16_000); });

    expect(clipsCallCount).toBe(3);
  });

  it('restores the newest page once when retention retires the current cursor', async () => {
    resetLocation();
    let clips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${String(index + 1).padStart(3, '0')}`,
      event_ref: `event-${index + 1}`,
    }));
    const fetchMock = installFetchMock(clips);
    const { host } = await renderPage();

    clickPager(host, '다음 페이지');
    await flush();
    expect(pageOrdinal(host)).toBe(2);

    clips = clips.slice(0, 10);
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.includes('/cameras')) return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
      if (url.includes('/incidents')) return Promise.resolve({ ok: true, status: 200, json: async () => ({ incidents: [] }) });
      const params = new URL(url, 'http://localhost').searchParams;
      // The purged boundary row makes the stored cursor unusable.
      if (params.get('cursor')) {
        return Promise.resolve({ ok: false, status: 400, json: async () => ({ detail: 'invalid cursor' }) });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => keysetBody(clips, Number(params.get('limit')), null, { fall: clips.length }),
      });
    });

    clickPager(host, '현재 페이지 새로 고침');
    await flush();
    await flush();

    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(10);
    expect(host.querySelector('nav[aria-label="이벤트 페이지"]')).toBeNull();
    expect(clipRequestUrls(fetchMock).at(-1)).toBe('/api/v1/clips?limit=48');
  });

  it('preserves an off-page other-facet deep link through metadata lookup', async () => {
    resetLocation('?page=events&event=other&clip=clip-49');
    const pageClips = Array.from({ length: 48 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
      event_type: 'vendor-detector',
    }));
    const deepLinkedClip = clipManifest({ clip_id: 'clip-49', event_ref: 'event-49', event_type: 'vendor-detector' });
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.includes('/cameras')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
      }
      if (url.endsWith('/clips/clip-49/metadata')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => deepLinkedClip });
      }
      if (url.includes('/clips')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            clips: pageClips,
            pagination: { limit: 48, offset: 0, total: 50, has_more: true, next_cursor: 'bmV4dA==' },
            event_type_counts: { 'vendor-detector': 50 },
          }),
        });
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);

    await renderPage();

    expect(window.location.search).toContain('event=other');
    expect(window.location.search).toContain('clip=clip-49');
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('기타');
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/clips/clip-49/metadata'))).toBe(true);
  });
});

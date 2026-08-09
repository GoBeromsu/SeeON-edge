import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  cameraRegistry,
  cleanupPages,
  clipManifest,
  clipRequestUrls,
  flush,
  installFetchMock,
  installLegacyFetchMock,
  renderPage,
  resetLocation,
} from '@/app/pages/EventsPage.testSupport';

afterEach(() => {
  cleanupPages();
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('EventsPage pagination', () => {
  it('bounds a complete legacy response to 48 cards and exposes a numbered pager', async () => {
    resetLocation();
    const clips = Array.from({ length: 50 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
      started_at: `2026-08-02T${String(23 - (index % 24)).padStart(2, '0')}:00:00Z`,
    }));
    installLegacyFetchMock(clips);

    const { host } = await renderPage();

    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(host.querySelector('nav[aria-label="이벤트 페이지"]')).not.toBeNull();
    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe('1');
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

  it('commits numbered page navigation only after the requested page succeeds', async () => {
    resetLocation();
    const clips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
    }));
    const fetchMock = installFetchMock(clips);
    const { host } = await renderPage();

    const pageTwo = host.querySelector('button[aria-label="2페이지"]') as HTMLButtonElement;
    act(() => pageTwo.click());
    await flush();

    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe('2');
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(clipRequestUrls(fetchMock)).toContain('/api/v1/clips?limit=48&offset=48');
  });

  it('keeps the current page and cards when historical navigation fails', async () => {
    resetLocation();
    const clips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
    }));
    const fetchMock = installFetchMock(clips);
    const { host } = await renderPage();
    fetchMock.mockRejectedValueOnce(new Error('page unavailable'));

    const pageTwo = host.querySelector('button[aria-label="2페이지"]') as HTMLButtonElement;
    act(() => pageTwo.click());
    await flush();

    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe('1');
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(host.textContent).toContain('현재 페이지를 유지합니다.');
  });

  it('polls page 1 every 8 seconds, keeps its last-good cards on failure, and stops polling on history', async () => {
    vi.useFakeTimers();
    resetLocation();
    const clips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
    }));
    let clipsCallCount = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.includes('/cameras')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
      }
      clipsCallCount += 1;
      if (clipsCallCount === 2) return Promise.reject(new Error('background unavailable'));
      const params = new URL(url, 'http://localhost').searchParams;
      const limit = Number(params.get('limit'));
      const offset = Number(params.get('offset'));
      const page = clips.slice(offset, offset + limit);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          clips: page,
          pagination: { limit, offset, total: clips.length, has_more: offset + page.length < clips.length },
          event_type_counts: { fall: clips.length },
        }),
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const { host } = await renderPage();

    await act(async () => { await vi.advanceTimersByTimeAsync(8_000); });

    expect(clipsCallCount).toBe(2);
    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe('1');
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(host.textContent).toContain('현재 페이지를 유지합니다.');

    const pageTwo = host.querySelector('button[aria-label="2페이지"]') as HTMLButtonElement;
    act(() => pageTwo.click());
    await flush();
    expect(clipsCallCount).toBe(3);
    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe('2');

    await act(async () => { await vi.advanceTimersByTimeAsync(16_000); });

    expect(clipsCallCount).toBe(3);
  });

  it('moves once to the last valid page when retention invalidates the current offset', async () => {
    resetLocation();
    let clips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
    }));
    const fetchMock = installFetchMock(clips);
    const { host } = await renderPage();

    const pageThree = host.querySelector('button[aria-label="3페이지"]') as HTMLButtonElement;
    act(() => pageThree.click());
    await flush();
    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe('3');

    clips = clips.slice(0, 60);
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.includes('/cameras')) return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
      const params = new URL(url, 'http://localhost').searchParams;
      const limit = Number(params.get('limit'));
      const offset = Number(params.get('offset'));
      const page = clips.slice(offset, offset + limit);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          clips: page,
          pagination: { limit, offset, total: clips.length, has_more: offset + page.length < clips.length },
          event_type_counts: { fall: clips.length },
        }),
      });
    });
    const refresh = host.querySelector('button[aria-label="현재 페이지 새로 고침"]') as HTMLButtonElement;
    act(() => refresh.click());
    await flush();
    await flush();

    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe('2');
    expect(clipRequestUrls(fetchMock).slice(-2)).toEqual([
      '/api/v1/clips?limit=48&offset=96',
      '/api/v1/clips?limit=48&offset=48',
    ]);
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
            pagination: { limit: 48, offset: 0, total: 50, has_more: true },
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

import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  cameraRegistry,
  cleanupPages,
  clipManifest,
  clipRequestUrls,
  flush,
  renderPage,
  resetLocation,
} from '@/app/pages/EventsPage.testSupport';

const TOTAL_CLIPS = 9_313;
const UNIQUE_UNKNOWN_TYPES = 3_000;

function unknownFacetCounts(): Record<string, number> {
  return Object.fromEntries(Array.from(
    { length: UNIQUE_UNKNOWN_TYPES },
    (_, index) => [`vendor-event-${index}`, index === 0 ? TOTAL_CLIPS - UNIQUE_UNKNOWN_TYPES + 1 : 1],
  ));
}

function chooseCamera(host: HTMLDivElement, cameraId: string): void {
  const select = host.querySelector('select') as HTMLSelectElement;
  const nativeSetter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
  act(() => {
    nativeSetter?.call(select, cameraId);
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

afterEach(() => {
  cleanupPages();
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('EventsPage bounded event facets', () => {
  it('caps cards, facet controls, and media mounts for 9,313 clips with thousands of raw types', async () => {
    resetLocation();
    const clips = Array.from({ length: 48 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `unique-event-${index + 1}`,
      event_type: index === 0 ? null : `vendor-event-${index}`,
    }));
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          clips,
          pagination: { limit: 48, offset: 0, total: TOTAL_CLIPS, has_more: true },
          event_type_counts: unknownFacetCounts(),
        }),
      });
    }));

    const { host } = await renderPage();

    const facetGroup = host.querySelector('[role="group"][aria-label="이벤트 종류 필터"]') as HTMLElement;
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(facetGroup.querySelectorAll('button').length).toBeLessThanOrEqual(4);
    expect(facetGroup.textContent).toContain(`전체 ${TOTAL_CLIPS}`);
    expect(facetGroup.textContent).toContain(`기타 ${TOTAL_CLIPS}`);
    expect(host.querySelectorAll('video')).toHaveLength(0);

    const firstCard = host.querySelector('button.rounded-card') as HTMLButtonElement;
    act(() => firstCard.click());
    await flush();

    expect(document.querySelectorAll('[role="dialog"] video')).toHaveLength(1);
    expect(document.querySelectorAll('video')).toHaveLength(1);
  });

  it('keeps the camera request and events URL while the other filter resets pagination', async () => {
    resetLocation();
    const clips = Array.from({ length: 96 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `unique-event-${index + 1}`,
      event_type: `vendor-event-${index}`,
    }));
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
      const params = new URL(url, 'http://localhost').searchParams;
      const offset = Number(params.get('offset'));
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          clips: clips.slice(offset, offset + 48),
          pagination: { limit: 48, offset, total: clips.length, has_more: offset + 48 < clips.length },
          event_type_counts: { 'vendor-event': clips.length },
        }),
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const { host } = await renderPage();
    chooseCamera(host, 'cam-1');
    await flush();
    const pageTwo = host.querySelector('button[aria-label="2페이지"]') as HTMLButtonElement;
    act(() => pageTwo.click());
    await flush();

    const otherChip = Array.from(host.querySelectorAll<HTMLButtonElement>('button'))
      .find((button) => button.textContent?.startsWith('기타'));
    act(() => otherChip?.click());
    await flush();

    expect(window.location.search).toBe('?page=events&event=other');
    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe('1');
    expect(clipRequestUrls(fetchMock)).toContain(
      '/api/v1/clips?camera_id=cam-1&event_type=other&limit=48&offset=0',
    );
  });
});

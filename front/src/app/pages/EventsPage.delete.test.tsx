import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  cameraRegistry,
  cleanupPages,
  clipManifest,
  flush,
  renderPage,
  resetLocation,
} from '@/app/pages/EventsPage.testSupport';
import { EVENTS_POLL_INTERVAL_MS } from '@/features/events/paging';

function jsonResponse(body: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

function pageResponse(clips: readonly Record<string, unknown>[]) {
  return jsonResponse({
    clips,
    pagination: { limit: 48, offset: 0, total: clips.length, has_more: false },
    event_type_counts: clips.reduce<Record<string, number>>((counts, clip) => {
      const type = String(clip.event_type);
      counts[type] = (counts[type] ?? 0) + 1;
      return counts;
    }, {}),
  });
}

function typeConfirm(input: HTMLInputElement, value: string): void {
  act(() => {
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
    valueSetter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function findButton(root: ParentNode, label: string): HTMLButtonElement {
  const button = Array.from(root.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

afterEach(() => {
  cleanupPages();
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('EventsPage clip deletion', () => {
  it('does not validate metadata for a selected clip after accepted PURGED deletion and list refresh', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    resetLocation();
    const purgeTarget = clipManifest({
      clip_id: 'clip-purge-me',
      event_ref: 'event-purge-me',
      event_type: 'fall',
      camera_id: 'cam-1',
    });
    const other = clipManifest({
      clip_id: 'clip-keep',
      event_ref: 'event-keep',
      event_type: 'fall',
      camera_id: 'cam-1',
      started_at: '2026-08-02T02:00:00Z',
    });
    let clips: Record<string, unknown>[] = [purgeTarget, other];
    const metadataUrls: string[] = [];
    let clipsCalls = 0;
    let resolveDelete: ((value: ReturnType<typeof jsonResponse>) => void) | null = null;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      if (url.includes('/incidents')) return Promise.resolve(jsonResponse({ incidents: [] }));
      if (url.endsWith('/clips/clip-purge-me/artifacts')) {
        return Promise.resolve(jsonResponse({
          clip_id: 'clip-purge-me',
          clean: 'AVAILABLE',
          analysis: 'MISSING',
          annotated: 'NOT_REQUESTED',
          playback_view: 'clean',
          annotated_fallback_to_clean: true,
        }));
      }
      if (url.endsWith('/clips/clip-purge-me/metadata')) {
        metadataUrls.push(url);
        return Promise.resolve(jsonResponse({ detail: 'not found' }, 404));
      }
      if (url.includes('/clips/clip-purge-me') && init?.method === 'DELETE') {
        // Backend has already purged the row; subsequent list reads converge without the clip.
        clips = [other];
        return new Promise((resolve) => { resolveDelete = resolve; });
      }
      if (url.includes('/clips') && !url.includes('/clips/')) {
        clipsCalls += 1;
        return Promise.resolve(pageResponse(clips));
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();

    const { host } = await renderPage();
    act(() => (host.querySelector('button.rounded-card') as HTMLButtonElement).click());
    await flush();

    expect(window.location.search).toContain('clip=clip-purge-me');
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();

    act(() => findButton(document.body, '클립 삭제').click());
    const confirmDialog = Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"]'))
      .find((dialog) => dialog.textContent?.includes('클립 삭제 확인')) as HTMLElement;
    typeConfirm(confirmDialog.querySelector('#delete-confirm-input') as HTMLInputElement, 'clip-purge-me');
    act(() => { findButton(confirmDialog, '삭제').click(); });
    await flush();

    // List poll can observe the purge before the DELETE response is applied in the UI.
    const clipsBeforePoll = clipsCalls;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(EVENTS_POLL_INTERVAL_MS);
    });
    await flush();
    expect(clipsCalls).toBeGreaterThan(clipsBeforePoll);

    await act(async () => {
      resolveDelete?.(jsonResponse({ clip_id: 'clip-purge-me', status: 'PURGED' }, 202));
      await Promise.resolve();
    });
    await flush();
    await flush();

    // Post-accept list refresh from onDeleted must also not re-validate metadata.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(EVENTS_POLL_INTERVAL_MS);
    });
    await flush();

    expect(document.querySelector('[data-testid="clip-delete-status"]')?.textContent).toContain('삭제했습니다');
    expect(document.querySelector('[data-testid="clip-modal-unavailable"]')?.textContent).toContain('삭제되어');
    expect(window.location.search).toContain('clip=clip-purge-me');
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(1);
    expect(metadataUrls).toEqual([]);
  });

  it('still surfaces genuine metadata failures for a non-deleted off-page clip', async () => {
    resetLocation('?page=events&clip=clip-missing');
    const metadataUrls: string[] = [];
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      if (url.includes('/incidents')) return Promise.resolve(jsonResponse({ incidents: [] }));
      if (url.endsWith('/clips/clip-missing/metadata')) {
        metadataUrls.push(url);
        return Promise.resolve(jsonResponse({ detail: 'not found' }, 404));
      }
      if (url.includes('/clips')) {
        return Promise.resolve(pageResponse([clipManifest({ clip_id: 'clip-on-page', event_ref: 'event-on-page' })]));
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);

    await renderPage();
    await flush();

    expect(metadataUrls).toEqual(['/api/v1/clips/clip-missing/metadata']);
    expect(window.location.search).toBe('?page=events');
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });
});

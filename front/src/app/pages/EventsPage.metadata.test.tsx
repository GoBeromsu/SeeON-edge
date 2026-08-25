import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  allClips,
  cameraRegistry,
  cleanupPages,
  clipManifest,
  flush,
  installFetchMock,
  renderPage,
  resetLocation,
} from '@/app/pages/EventsPage.testSupport';

type Deferred<T> = {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
};

function deferred<T>(): Deferred<T> {
  let resolvePromise = (_value: T): void => undefined;
  const promise = new Promise<T>((resolve) => {
    resolvePromise = resolve;
  });
  return { promise, resolve: resolvePromise };
}

function jsonResponse(body: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

function pageResponse(
  clips: readonly Record<string, unknown>[],
  eventTypeCounts: Readonly<Record<string, number>> = { fall: clips.length + 1 },
) {
  return jsonResponse({
    clips,
    pagination: {
      limit: 48, offset: 0, total: clips.length + 1, has_more: true, next_cursor: 'bmV4dA==',
    },
    event_type_counts: eventTypeCounts,
  });
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

describe('EventsPage metadata validation', () => {
  it('discards an opened clip immediately when the camera filter changes to another camera', async () => {
    resetLocation();
    installFetchMock();
    const replaceSpy = vi.spyOn(window.history, 'replaceState');
    const { host } = await renderPage();
    const cameraTwoCard = Array.from(host.querySelectorAll<HTMLButtonElement>('button.rounded-card'))
      .find((card) => card.textContent?.includes('302호'));

    act(() => cameraTwoCard?.click());
    await flush();
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('302호');

    chooseCamera(host, 'cam-1');
    await flush();

    expect(window.location.search).not.toContain('clip=');
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(replaceSpy).toHaveBeenLastCalledWith(null, '', '/?page=events');
  });

  it('ignores stale metadata and discards a current response outside the selected camera', async () => {
    resetLocation('?page=events&clip=clip-49');
    const pageClips = Array.from({ length: 48 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
      camera_id: 'cam-1',
    }));
    const firstMetadata = deferred<ReturnType<typeof jsonResponse>>();
    const secondMetadata = deferred<ReturnType<typeof jsonResponse>>();
    const metadataSignals: AbortSignal[] = [];
    let metadataCallCount = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      if (url.endsWith('/clips/clip-49/metadata')) {
        if (init?.signal) metadataSignals.push(init.signal);
        metadataCallCount += 1;
        return metadataCallCount === 1 ? firstMetadata.promise : secondMetadata.promise;
      }
      if (url.includes('/clips')) return Promise.resolve(pageResponse(pageClips));
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);
    const { host } = await renderPage();

    chooseCamera(host, 'cam-1');
    await flush();
    expect(metadataSignals[0]?.aborted).toBe(true);

    await act(async () => {
      firstMetadata.resolve(jsonResponse(clipManifest({ clip_id: 'clip-49', camera_id: 'cam-2' })));
      await Promise.resolve();
    });
    expect(document.querySelector('[role="dialog"]')?.textContent).not.toContain('302호');

    await act(async () => {
      secondMetadata.resolve(jsonResponse(clipManifest({ clip_id: 'clip-49', camera_id: 'cam-2' })));
      await Promise.resolve();
    });
    await flush();

    expect(window.location.search).not.toContain('clip=');
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it.each([400, 404])('removes a terminal %i metadata link with replaceState', async (status) => {
    resetLocation('?page=events&clip=clip-invalid');
    const replaceSpy = vi.spyOn(window.history, 'replaceState');
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      if (url.endsWith('/clips/clip-invalid/metadata')) return Promise.resolve(jsonResponse({ detail: 'invalid' }, status));
      if (url.includes('/clips')) return Promise.resolve(pageResponse(allClips));
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);

    await renderPage();

    expect(window.location.search).toBe('?page=events');
    expect(replaceSpy).toHaveBeenLastCalledWith(null, '', '/?page=events');
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it('preserves a transient other-facet metadata link and accepts an unknown raw type after retry', async () => {
    resetLocation('?page=events&event=other&clip=clip-49');
    const pageClips = Array.from({ length: 48 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
      event_type: 'vendor-detector',
    }));
    const firstMetadata = deferred<ReturnType<typeof jsonResponse>>();
    const retryMetadata = deferred<ReturnType<typeof jsonResponse>>();
    let metadataCallCount = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      if (url.endsWith('/clips/clip-49/metadata')) {
        metadataCallCount += 1;
        return metadataCallCount === 1 ? firstMetadata.promise : retryMetadata.promise;
      }
      if (url.includes('/clips')) return Promise.resolve(pageResponse(pageClips, { 'vendor-detector': 49 }));
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);
    await renderPage();

    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('이벤트 정보를 불러오는 중입니다.');

    await act(async () => {
      firstMetadata.resolve(jsonResponse({ detail: 'unavailable' }, 500));
      await Promise.resolve();
    });
    await flush();

    expect(window.location.search).toContain('clip=clip-49');
    const retryButton = Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button'))
      .find((button) => button.textContent === '다시 시도');
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('이벤트 정보를 불러오지 못했습니다.');
    expect(retryButton).toBeDefined();

    act(() => retryButton?.click());
    await flush();
    expect(metadataCallCount).toBe(2);
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('이벤트 정보를 불러오는 중입니다.');

    await act(async () => {
      retryMetadata.resolve(jsonResponse(clipManifest({
        clip_id: 'clip-49', camera_id: 'cam-1', event_type: 'vendor-detector',
      })));
      await Promise.resolve();
    });
    await flush();

    expect(window.location.search).toContain('event=other');
    expect(window.location.search).toContain('clip=clip-49');
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('기타');
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('301호');
  });
});

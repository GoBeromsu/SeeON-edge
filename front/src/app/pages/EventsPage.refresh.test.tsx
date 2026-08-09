import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  cameraRegistry,
  cleanupPages,
  clipManifest,
  clipRequestUrls,
  flush,
  installFetchMock,
  renderPage,
  resetLocation,
} from '@/app/pages/EventsPage.testSupport';

type Deferred<T> = {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (reason: unknown) => void;
};

function deferred<T>(): Deferred<T> {
  let resolvePromise = (_value: T): void => undefined;
  let rejectPromise = (_reason: unknown): void => undefined;
  const promise = new Promise<T>((resolve, reject) => {
    resolvePromise = resolve;
    rejectPromise = reject;
  });
  return { promise, resolve: resolvePromise, reject: rejectPromise };
}

function jsonResponse(body: unknown) {
  return { ok: true, status: 200, json: async () => body };
}

function pagedResponse(clips: readonly Record<string, unknown>[], offset: number, total: number) {
  return jsonResponse({
    clips,
    pagination: { limit: 48, offset, total, has_more: offset + clips.length < total },
    event_type_counts: { fall: total },
  });
}

afterEach(() => {
  cleanupPages();
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('EventsPage refresh state', () => {
  it('falls back to page zero when retention invalidates both the target and reported last page', async () => {
    resetLocation();
    const initialClips = Array.from({ length: 97 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
    }));
    const fetchMock = installFetchMock(initialClips);
    const { host } = await renderPage();
    const pageThree = host.querySelector('button[aria-label="3페이지"]') as HTMLButtonElement;
    act(() => pageThree.click());
    await flush();

    const survivingClips = initialClips.slice(0, 10);
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      const offset = Number(new URL(url, 'http://localhost').searchParams.get('offset'));
      if (offset === 96) return Promise.resolve(pagedResponse([], 96, 60));
      if (offset === 48) return Promise.resolve(pagedResponse([], 48, 10));
      return Promise.resolve(pagedResponse(survivingClips, 0, 10));
    });

    const refresh = host.querySelector('button[aria-label="현재 페이지 새로 고침"]') as HTMLButtonElement;
    act(() => refresh.click());
    await flush();
    await flush();

    expect(clipRequestUrls(fetchMock).slice(-3)).toEqual([
      '/api/v1/clips?limit=48&offset=96',
      '/api/v1/clips?limit=48&offset=48',
      '/api/v1/clips?limit=48&offset=0',
    ]);
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(10);
    expect(host.querySelector('[aria-current="page"]')).toBeNull();
  });

  it('shows last success and background refreshing while page one polls', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-08-09T08:00:00.000Z'));
    resetLocation();
    const clips = Array.from({ length: 48 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
    }));
    const poll = deferred<ReturnType<typeof pagedResponse>>();
    let clipsCallCount = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      clipsCallCount += 1;
      if (clipsCallCount === 1) return Promise.resolve(pagedResponse(clips, 0, clips.length));
      return poll.promise;
    });
    vi.stubGlobal('fetch', fetchMock);
    const { host } = await renderPage();

    await act(async () => { await vi.advanceTimersByTimeAsync(8_000); });

    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(host.textContent).toContain('이벤트 목록을 새로 고치는 중입니다.');
    expect(host.querySelector('time[data-testid="events-last-success"]')?.getAttribute('datetime'))
      .toBe('2026-08-09T08:00:00.000Z');
  });

  it('keeps last-good cards and retries a failed page-one poll immediately', async () => {
    vi.useFakeTimers();
    resetLocation();
    const clips = Array.from({ length: 48 }, (_, index) => clipManifest({
      clip_id: `clip-${index + 1}`,
      event_ref: `event-${index + 1}`,
    }));
    const failedPoll = deferred<ReturnType<typeof pagedResponse>>();
    const retry = deferred<ReturnType<typeof pagedResponse>>();
    let clipsCallCount = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      clipsCallCount += 1;
      if (clipsCallCount === 1) return Promise.resolve(pagedResponse(clips, 0, clips.length));
      return clipsCallCount === 2 ? failedPoll.promise : retry.promise;
    });
    vi.stubGlobal('fetch', fetchMock);
    const { host } = await renderPage();
    await act(async () => { await vi.advanceTimersByTimeAsync(8_000); });

    await act(async () => {
      failedPoll.reject(new Error('background unavailable'));
      await Promise.resolve();
    });
    await flush();

    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);
    expect(host.textContent).toContain('현재 페이지를 유지합니다.');
    const status = host.querySelector('[role="status"]') as HTMLElement;
    const firstCard = host.querySelector('button.rounded-card') as HTMLButtonElement;
    expect(status.compareDocumentPosition(firstCard) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    const retryButton = Array.from(host.querySelectorAll<HTMLButtonElement>('button'))
      .find((button) => button.textContent === '다시 시도');
    expect(retryButton).toBeDefined();

    act(() => retryButton?.click());
    await flush();

    expect(clipsCallCount).toBe(3);
    expect(host.textContent).toContain('이벤트 목록을 새로 고치는 중입니다.');
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(48);

    await act(async () => {
      retry.resolve(pagedResponse(clips, 0, clips.length));
      await Promise.resolve();
    });
    await flush();
    expect(host.textContent).not.toContain('현재 페이지를 유지합니다.');
  });
});

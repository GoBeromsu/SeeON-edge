import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  cleanupPages,
  clipRequestUrls,
  installFetchMock,
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

describe('EventsPage root cleanup', () => {
  it('stops page-one polling when the shared test cleanup unmounts every root', async () => {
    vi.useFakeTimers();
    resetLocation();
    const fetchMock = installFetchMock();
    await renderPage();
    const requestsBeforeCleanup = clipRequestUrls(fetchMock).length;

    cleanupPages();
    await act(async () => { await vi.advanceTimersByTimeAsync(16_000); });

    expect(clipRequestUrls(fetchMock)).toHaveLength(requestsBeforeCleanup);
  });
});

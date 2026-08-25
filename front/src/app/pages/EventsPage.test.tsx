import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  cleanupPages,
  clipRequestUrls,
  flush,
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

describe('EventsPage', () => {
  it('renders the page title and a 4-column grid of clip cards once cameras and clips load', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    expect(host.querySelector('h1')?.textContent).toBe('이벤트');
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(4);
    expect(host.textContent).toContain('301호');
    expect(host.textContent).toContain('302호');
    expect(host.textContent).not.toContain('카메라 미상');
  });

  it('filters the grid by event type on the server and restores the newest keyset page', async () => {
    resetLocation();
    const fetchMock = installFetchMock();
    const { host } = await renderPage();
    const fallChip = Array.from(host.querySelectorAll('button')).find((button) => button.textContent?.startsWith('낙상'));

    act(() => fallChip?.click());
    await flush();

    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(2);
    expect(clipRequestUrls(fetchMock)).toContain('/api/v1/clips?event_type=fall&limit=48');
    expect(window.location.search).toContain('event=fall');
  });

  it('filters the grid by camera and refetches clips scoped to that camera', async () => {
    resetLocation();
    const fetchMock = installFetchMock();
    const { host } = await renderPage();
    const select = host.querySelector('select') as HTMLSelectElement;
    const nativeSetter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;

    act(() => {
      nativeSetter?.call(select, 'cam-1');
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await flush();

    const cards = Array.from(host.querySelectorAll('button.rounded-card'));
    expect(cards).toHaveLength(2);
    expect(cards.every((card) => !card.textContent?.includes('302호'))).toBe(true);
    expect(clipRequestUrls(fetchMock)).toContain('/api/v1/clips?camera_id=cam-1&limit=48');
  });

  it('opens only one modal video via the clip URL and closes it on Escape', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    expect(host.querySelectorAll('video')).toHaveLength(0);
    const firstCard = host.querySelector('button.rounded-card') as HTMLButtonElement;
    act(() => firstCard.click());
    await flush();

    expect(window.location.search).toContain('clip=clip-1');
    const dialog = document.querySelector('[role="dialog"]');
    expect(dialog?.querySelectorAll('video')).toHaveLength(1);
    expect(dialog?.textContent).toContain('낙상');
    expect(dialog?.textContent).toContain('301호');

    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    await flush();

    expect(window.location.search).not.toContain('clip=');
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it('shows explicit unavailable states in the grid and modal', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();
    const unavailableThumb = host.querySelector('[data-testid="clip-thumbnail-unavailable"]');

    expect(unavailableThumb?.textContent).toBe('저장된 영상을 사용할 수 없습니다.');
    const unavailableCard = unavailableThumb?.closest('button') as HTMLButtonElement;
    act(() => unavailableCard.click());
    await flush();

    const dialog = document.querySelector('[role="dialog"]');
    expect(dialog?.querySelector('[data-testid="clip-modal-unavailable"]')?.textContent).toBe('저장된 영상을 사용할 수 없습니다.');
    expect(dialog?.querySelector('video')).toBeNull();
  });

  it('shows the empty-state message when no clips match the current filters', async () => {
    resetLocation();
    installFetchMock([]);
    const { host } = await renderPage();

    expect(host.textContent).toContain('조건에 맞는 이벤트가 없습니다.');
  });
});

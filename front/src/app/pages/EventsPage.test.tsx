import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { EventsPage } from '@/app/pages/EventsPage';

function resetLocation(search = '?page=events'): void {
  window.history.replaceState(null, '', `/${search}`);
}

const cameraRegistry = {
  registry_version: 1,
  cameras: [
    { id: 'cam-1', label: '301호', rtsp_url_masked: 'rtsp://***', status: 'online' },
    { id: 'cam-2', label: '302호', rtsp_url_masked: 'rtsp://***', status: 'online' },
  ],
};

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

// Deliberately omits camera_label (the real backend manifest has no such field), so these tests
// also exercise EventsPage's resolveCameraLabel cross-reference against the cameras resource.
const allClips = [
  clipManifest({ clip_id: 'clip-1', camera_id: 'cam-1', event_type: 'fall', started_at: '2026-08-02T03:12:00Z' }),
  clipManifest({ clip_id: 'clip-2', camera_id: 'cam-1', event_type: 'bed-exit', started_at: '2026-08-02T02:00:00Z' }),
  clipManifest({
    clip_id: 'clip-3',
    camera_id: 'cam-2',
    event_type: 'fall',
    started_at: '2026-08-02T01:00:00Z',
    video_available: false,
    video_error: '저장된 영상을 사용할 수 없습니다.',
  }),
  clipManifest({ clip_id: 'clip-4', camera_id: 'cam-2', event_type: 'bed-exit', started_at: '2026-08-02T00:00:00Z' }),
];

function installFetchMock(clips: Record<string, unknown>[] = allClips): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/cameras')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
    }
    if (url.includes('/clips')) {
      const match = /camera_id=([^&]+)/.exec(url);
      const cameraId = match ? decodeURIComponent(match[1]) : undefined;
      const scoped = cameraId ? clips.filter((clip) => clip.camera_id === cameraId) : clips;
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ clips: scoped }) });
    }
    return Promise.reject(new Error(`unexpected fetch: ${url}`));
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function renderPage(): Promise<{ host: HTMLDivElement; root: Root }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<EventsPage />));
  await flush();
  return { host, root };
}

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('EventsPage', () => {
  it('renders the page title and a 4-column grid of clip cards once cameras and clips load', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    expect(host.querySelector('h1')?.textContent).toBe('이벤트');
    const cards = host.querySelectorAll('button.rounded-card');
    expect(cards.length).toBe(4);
    // resolveCameraLabel cross-references the cameras resource, since the raw clip manifest has
    // no camera_label field at all.
    expect(host.textContent).toContain('301호');
    expect(host.textContent).toContain('302호');
    expect(host.textContent).not.toContain('카메라 미상');
  });

  it('filters the grid by event-type chip without refetching', async () => {
    resetLocation();
    const fetchMock = installFetchMock();
    const { host } = await renderPage();
    const callsBeforeFilter = fetchMock.mock.calls.length;

    const fallChip = Array.from(host.querySelectorAll('button')).find((button) => button.textContent?.startsWith('낙상'));
    act(() => fallChip?.click());
    await flush();

    expect(host.querySelectorAll('button.rounded-card').length).toBe(2);
    expect(fetchMock.mock.calls.length).toBe(callsBeforeFilter);
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
    expect(cards.length).toBe(2);
    expect(cards.every((card) => !card.textContent?.includes('302호'))).toBe(true);
    const clipsCalls = fetchMock.mock.calls.filter(([input]) => {
      const url = typeof input === 'string' ? input : (input as URL | Request).toString();
      return url.includes('/clips');
    });
    expect(clipsCalls.some(([input]) => {
      const url = typeof input === 'string' ? input : (input as URL | Request).toString();
      return url.includes('camera_id=cam-1');
    })).toBe(true);
  });

  it('opens the clip modal via the URL when a card is clicked, and closes it on Escape', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    const firstCard = host.querySelector('button.rounded-card') as HTMLButtonElement;
    act(() => firstCard.click());
    await flush();

    expect(window.location.search).toContain('clip=clip-1');
    const dialog = document.querySelector('[role="dialog"]');
    expect(dialog).not.toBeNull();
    expect(dialog?.textContent).toContain('낙상');
    expect(dialog?.textContent).toContain('301호');

    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    await flush();

    expect(window.location.search).not.toContain('clip=');
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it('shows the explicit unavailable state for a clip whose video is not available, in the grid and the modal', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    const unavailableThumb = host.querySelector('[data-testid="clip-thumbnail-unavailable"]');
    expect(unavailableThumb).not.toBeNull();
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

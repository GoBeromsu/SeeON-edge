import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { OperationsPage } from '@/app/pages/OperationsPage';

function resetLocation(search = '?page=operations'): void {
  window.history.replaceState(null, '', `/${search}`);
}

const cameraRegistry = {
  registry_version: 1,
  cameras: [
    { id: 'cam-1', label: '101호', rtsp_url_masked: 'rtsp://redacted-camera/a', status: 'online', floor_name: '1층' },
    { id: 'cam-2', label: '102호', rtsp_url_masked: 'rtsp://redacted-camera/b', status: 'offline', floor_name: '1층' },
    { id: 'cam-3', label: '201호', rtsp_url_masked: 'rtsp://redacted-camera/c', status: 'online', floor_name: '2층' },
  ],
};

const statusSnapshot = {
  cameras: {},
  runtime: {
    cameras: {
      'cam-1': {
        camera_id: 'cam-1',
        decode: { requested: 'auto', selected: 'cpu', fallback_count: 0, last_reason: null, updated_at_sec: 1 },
        measured_fps: 12.4,
        latency: null,
      },
    },
    worker: null,
    device: null,
    clip_recorder: null,
  },
};

const detectionSettings = {
  domains: {
    fall: { on: true, mode: 'always', start: null, end: null },
    bed_exit: { on: false, mode: 'always', start: null, end: null },
  },
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

const allClips = [
  clipManifest({ clip_id: 'clip-1', camera_id: 'cam-1', event_type: 'fall', started_at: '2026-08-02T03:12:00Z' }),
  clipManifest({ clip_id: 'clip-2', camera_id: 'cam-1', event_type: 'bed-exit', started_at: '2026-08-02T02:00:00Z' }),
];

let overlayMode = 'none';

function installFetchMock(): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/cameras')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
    }
    if (url.includes('/status')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => statusSnapshot });
    }
    if (url.includes('/detection-settings')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => detectionSettings });
    }
    if (url.includes('/streams/') && url.includes('/pose')) {
      if (init?.method === 'POST') {
        const body = init.body ? JSON.parse(init.body as string) : {};
        overlayMode = body.mode ?? overlayMode;
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ mode: overlayMode }) });
    }
    if (url.includes('/clips')) {
      const match = /camera_id=([^&]+)/.exec(url);
      const cameraId = match ? decodeURIComponent(match[1]) : undefined;
      const scoped = cameraId ? allClips.filter((clip) => clip.camera_id === cameraId) : allClips;
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
  act(() => root.render(<OperationsPage />));
  await flush();
  return { host, root };
}

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  overlayMode = 'none';
});

describe('OperationsPage', () => {
  it('renders the page title and the camera wall grid once cameras load', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    expect(host.querySelector('#main-content h1, h1')?.textContent).toBe('관제');
    expect(host.textContent).toContain('101호');
    expect(host.textContent).toContain('102호');
    expect(host.textContent).toContain('201호');
    expect(host.querySelectorAll('[data-camera-id]').length).toBe(3);
  });

  it('filters the wall by floor and updates the URL', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    const floorButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '2층');
    act(() => floorButton?.click());
    await flush();

    expect(window.location.search).toContain('floor=2%EC%B8%B5');
    expect(host.querySelectorAll('[data-camera-id]').length).toBe(1);
    expect(host.textContent).toContain('201호');
    expect(host.textContent).not.toContain('101호');
  });

  it('opens room detail when a tile is clicked, and returns to the wall via the back button', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    const tile = host.querySelector('[data-camera-id="cam-1"]') as HTMLButtonElement;
    act(() => tile.click());
    await flush();

    expect(window.location.search).toContain('camera=cam-1');
    expect(host.querySelector('h2')?.textContent).toBe('101호');
    expect(host.textContent).toContain('12.4 FPS');
    expect(host.textContent).toContain('1층');
    expect(host.textContent).toContain('rtsp://redacted-camera/a');

    const backButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent?.includes('카메라 월로'));
    act(() => backButton?.click());
    await flush();

    expect(window.location.search).not.toContain('camera=');
    expect(host.querySelectorAll('[data-camera-id]').length).toBe(3);
  });

  it('shows the global detection settings summary and lets the overlay mode be changed', async () => {
    resetLocation('?page=operations&camera=cam-1');
    const fetchMock = installFetchMock();
    const { host } = await renderPage();

    expect(host.textContent).toContain('사용 중 · 상시');
    expect(host.textContent).toContain('사용 안 함');

    const fallChip = Array.from(host.querySelectorAll<HTMLButtonElement>('button[aria-pressed]')).find((button) => button.textContent === '낙상');
    expect(fallChip?.getAttribute('aria-pressed')).toBe('false');

    act(() => fallChip?.click());
    await flush();

    expect(fallChip?.getAttribute('aria-pressed')).toBe('true');
    const overlayCalls = fetchMock.mock.calls.filter(([input]) => {
      const url = typeof input === 'string' ? input : (input as URL | Request).toString();
      return url.includes('/pose');
    });
    expect(overlayCalls.some(([, init]) => (init as RequestInit | undefined)?.method === 'POST')).toBe(true);
  });

  it('lists event history for the selected camera and opens the clip player modal', async () => {
    resetLocation('?page=operations&camera=cam-1');
    installFetchMock();
    const { host } = await renderPage();

    expect(host.textContent).toContain('이벤트 히스토리');
    expect(host.textContent).toContain('낙상');
    expect(host.textContent).toContain('침대 이탈');

    const firstClipTile = host.querySelector('section[aria-label="이벤트 히스토리"] button') as HTMLButtonElement;
    act(() => firstClipTile.click());
    await flush();

    const dialog = document.querySelector('[role="dialog"]');
    expect(dialog).not.toBeNull();
    expect(dialog?.textContent).toContain('101호');

    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    await flush();

    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it('shows an explicit error state with retry when the camera list fails to load', async () => {
    resetLocation();
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: false, status: 500, json: async () => ({}) })));
    const { host } = await renderPage();

    expect(host.querySelector('[role="alert"]')).not.toBeNull();
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('카메라 목록을 불러오지 못했습니다.');
  });
});

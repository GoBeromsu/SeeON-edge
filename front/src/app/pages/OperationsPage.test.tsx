import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { OperationsPage } from '@/app/pages/OperationsPage';
import type { BedZone } from '@/shared/api/client';

const savedFourRegionZone: BedZone = {
  regions: [
    { id: 'bed-1', polygon: [[0, 0], [10, 0], [10, 10]], origin: 'model' },
    { id: 'bed-2', polygon: [[20, 0], [30, 0], [30, 10]], origin: 'model' },
    { id: 'bed-3', polygon: [[40, 0], [50, 0], [50, 10]], origin: 'manual' },
    { id: 'bed-4', polygon: [[60, 0], [70, 0], [70, 10]], origin: 'manual' },
  ],
  image_width: 1920,
  image_height: 1080,
  recognized_at: '2026-09-05T00:00:00Z',
};

vi.mock('@/shared/ui/BedZoneRecognitionPanel', () => ({
  BedZoneRecognitionPanel: ({
    bedZone,
    onSaved,
    onCancel,
  }: {
    bedZone: BedZone | null;
    onSaved: (zone: BedZone | null) => void;
    onCancel: () => void;
  }) => (
    <div>
      <output aria-label="저장된 침대 영역 수">{bedZone?.regions.length ?? 0}</output>
      <button type="button">모의 후보 인식</button>
      <button
        type="button"
        onClick={() => onSaved(savedFourRegionZone)}
      >
        모의 저장 성공
      </button>
      <button type="button" onClick={() => onSaved(null)}>모의 지우기 성공</button>
      <button type="button" onClick={onCancel}>모의 취소</button>
    </div>
  ),
}));

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

let overlaySelection = { person: true, bed: true };
let cameraBedZone: BedZone | null = null;

function installFetchMock(): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/cameras')) {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          ...cameraRegistry,
          cameras: cameraRegistry.cameras.map((camera) => (
            camera.id === 'cam-1' ? { ...camera, bed_zone: cameraBedZone } : camera
          )),
        }),
      });
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
        overlaySelection = body;
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => overlaySelection });
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

const mountedRoots: Root[] = [];

async function renderPage(): Promise<{ host: HTMLDivElement; root: Root }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  mountedRoots.push(root);
  act(() => root.render(<OperationsPage />));
  await flush();
  return { host, root };
}

afterEach(() => {
  // Unmount before wiping the body: a still-mounted root whose portal DOM was
  // removed underneath it throws NotFoundError on its next commit, which
  // surfaces as an uncaught error inside the following test.
  for (const root of mountedRoots.splice(0)) act(() => root.unmount());
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  overlaySelection = { person: true, bed: true };
  cameraBedZone = null;
  detectionSettings.domains.bed_exit.on = false;
});

describe('OperationsPage', () => {
  it('renders the page title with online/offline counts and the camera wall grid once cameras load', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    expect(host.querySelector('#main-content h1, h1')?.textContent).toBe('관제');
    expect(host.textContent).toContain('온라인 2');
    expect(host.textContent).toContain('오프라인 1');
    expect(host.textContent).toContain('101호');
    expect(host.textContent).toContain('102호');
    expect(host.textContent).toContain('201호');
    expect(host.querySelectorAll('[data-camera-id]').length).toBe(3);
  });

  it('renders the offline camera tile as a gray placeholder instead of a live snapshot', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    const offlineTile = host.querySelector('[data-camera-id="cam-2"]') as HTMLButtonElement;
    const offlineMedia = offlineTile.querySelector('.bg-muted');
    expect(offlineMedia?.textContent).toBe('오프라인');
    expect(offlineTile.querySelector('img')).toBeNull();
  });

  it('filters the wall by floor via the select and updates the URL', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    const floorSelect = host.querySelector('select[aria-label="층 필터"]') as HTMLSelectElement;
    floorSelect.value = '2층';
    act(() => floorSelect.dispatchEvent(new Event('change', { bubbles: true })));
    await flush();

    expect(window.location.search).toContain('floor=2%EC%B8%B5');
    expect(host.querySelectorAll('[data-camera-id]').length).toBe(1);
    expect(host.textContent).toContain('201호');
    expect(host.textContent).not.toContain('101호');
  });

  it('opens room detail when a tile is clicked, and returns to the wall via the breadcrumb', async () => {
    resetLocation();
    installFetchMock();
    const { host } = await renderPage();

    const tile = host.querySelector('[data-camera-id="cam-1"]') as HTMLButtonElement;
    act(() => tile.click());
    await flush();

    expect(window.location.search).toContain('camera=cam-1');
    expect(host.querySelector('h1')?.textContent).toBe('101호');
    expect(host.textContent).toContain('관제');
    expect(host.textContent).toContain('12.4 FPS');
    expect(host.textContent).toContain('1층');
    expect(host.textContent).toContain('rtsp://redacted-camera/a');

    const breadcrumbLink = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '관제');
    act(() => breadcrumbLink?.click());
    await flush();

    expect(window.location.search).not.toContain('camera=');
    expect(host.querySelectorAll('[data-camera-id]').length).toBe(3);
  });

  it('shows an offline camera detail with the disconnected panel and its reconnect/manage buttons', async () => {
    resetLocation('?page=operations&camera=cam-2');
    installFetchMock();
    const { host } = await renderPage();

    expect(host.textContent).toContain('카메라에 연결할 수 없습니다');
    expect(host.textContent).toContain('탐지가 중단된 상태입니다');
    const buttons = Array.from(host.querySelectorAll('button')).map((button) => button.textContent);
    expect(buttons).toContain('재연결 시도');
    expect(buttons).toContain('연결 관리');
    expect(host.querySelectorAll('svg[aria-label="중단됨"]')).toHaveLength(2);
  });

  it('shows the global detection settings summary and lets overlay targets be changed independently', async () => {
    resetLocation('?page=operations&camera=cam-1');
    const fetchMock = installFetchMock();
    const { host } = await renderPage();

    expect(host.querySelector('svg[aria-label="탐지 중"]')).not.toBeNull();
    expect(host.querySelector('svg[aria-label="꺼짐"]')).not.toBeNull();
    expect(host.querySelector('button[aria-label="침대 영역 편집"]')?.getAttribute('title')).toBe('침대 영역 편집');

    const personToggle = Array.from(host.querySelectorAll<HTMLButtonElement>('button[aria-pressed]')).find((button) => button.textContent === '사람');
    const bedToggle = Array.from(host.querySelectorAll<HTMLButtonElement>('button[aria-pressed]')).find((button) => button.textContent === '침대');
    expect(personToggle?.getAttribute('aria-pressed')).toBe('true');
    expect(bedToggle?.getAttribute('aria-pressed')).toBe('true');

    act(() => personToggle?.click());
    await flush();

    expect(personToggle?.getAttribute('aria-pressed')).toBe('false');
    expect(bedToggle?.getAttribute('aria-pressed')).toBe('true');
    const overlayCalls = fetchMock.mock.calls.filter(([input]) => {
      const url = typeof input === 'string' ? input : (input as URL | Request).toString();
      return url.includes('/pose');
    });
    expect(overlayCalls.some(([, init]) => (
      (init as RequestInit | undefined)?.method === 'POST'
      && (init as RequestInit).body === JSON.stringify({ person: false, bed: true })
    ))).toBe(true);
  });

  it('keeps readiness tied to refreshed persisted geometry, not unsaved recognition candidates', async () => {
    resetLocation('?page=operations&camera=cam-1');
    detectionSettings.domains.bed_exit.on = true;
    const fetchMock = installFetchMock();
    const { host } = await renderPage();
    const cameraRequestCount = () => fetchMock.mock.calls.filter(([input]) => {
      const url = typeof input === 'string' ? input : (input as URL | Request).toString();
      return url.includes('/cameras');
    }).length;

    expect(host.querySelector('svg[aria-label="침대 영역 미설정"]')).not.toBeNull();
    const initialCameraRequests = cameraRequestCount();
    const editAction = host.querySelector<HTMLButtonElement>('button[aria-label="침대 영역 편집"]');
    act(() => editAction?.click());
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('침대 영역 편집');

    const candidate = Array.from(document.querySelectorAll('button'))
      .find((button) => button.textContent === '모의 후보 인식');
    act(() => candidate?.click());
    await flush();
    expect(host.querySelector('svg[aria-label="침대 영역 미설정"]')).not.toBeNull();
    expect(cameraRequestCount()).toBe(initialCameraRequests);

    cameraBedZone = savedFourRegionZone;
    const save = Array.from(document.querySelectorAll('button'))
      .find((button) => button.textContent === '모의 저장 성공');
    act(() => save?.click());
    await flush();
    expect(host.querySelector('svg[aria-label="침대 영역 미설정"]')).toBeNull();
    expect(cameraRequestCount()).toBeGreaterThan(initialCameraRequests);
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    act(() => host.querySelector<HTMLButtonElement>('button[aria-label="침대 영역 편집"]')?.click());
    await flush();
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    expect(document.querySelector('output[aria-label="저장된 침대 영역 수"]')?.textContent).toBe('4');

    cameraBedZone = null;
    const clear = Array.from(document.querySelectorAll('button'))
      .find((button) => button.textContent === '모의 지우기 성공');
    act(() => clear?.click());
    await flush();
    expect(host.querySelector('svg[aria-label="침대 영역 미설정"]')).not.toBeNull();
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

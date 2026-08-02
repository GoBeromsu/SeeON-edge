import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SettingsPage } from '@/app/pages/SettingsPage';

const cameraRegistry = {
  registry_version: 1,
  cameras: [
    { id: 'cam-1', label: '101호', rtsp_url_masked: 'rtsp://redacted-camera/a', status: 'online', floor_name: '1층' },
  ],
};

const systemSnapshot = {
  updated_at: '2026-08-02T00:00:00Z',
  backend: { configured: true, reachable: true, last_ok_at: '2026-08-02T00:00:00Z' },
  version: '2026.07.28',
  image_digests: { ml_api: '8c4e21', ml_worker: '1fa905' },
  storage: { clip_store: { total_bytes: 100, used_bytes: 10, used_pct: 10 } },
};

const detectionSettings = {
  domains: {
    fall: { on: true, mode: 'always', start: null, end: null },
    bed_exit: { on: false, mode: 'always', start: null, end: null },
  },
};

const connectionView = {
  events_url: 'https://facility.example/events',
  config_url: 'https://facility.example/config',
  facility_id: 'facility-42',
  facility_token_set: true,
  facility_token_masked: '••••cd42',
  configured: true,
  reachable: true,
  last_ok_at: '2026-08-02T00:00:00Z',
  updated_at: '2026-08-02T00:00:00Z',
};

const statusSnapshot = {
  cameras: {},
  runtime: { cameras: {}, worker: { alive: true, pid: 1, started_at_sec: 0 }, device: null, clip_recorder: null },
};

const clipStorage = {
  root: '/var/lib/clip-store',
  selected_path: '',
  total_bytes: 10 * 1024 ** 3,
  used_bytes: 2 * 1024 ** 3,
  used_pct: 20,
};

function installFetchMock(): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/cameras')) return Promise.resolve({ ok: true, status: 200, json: async () => cameraRegistry });
    if (url.includes('/system')) return Promise.resolve({ ok: true, status: 200, json: async () => systemSnapshot });
    if (url.includes('/detection-settings')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => detectionSettings });
    }
    if (url.includes('/connection')) return Promise.resolve({ ok: true, status: 200, json: async () => connectionView });
    if (url.includes('/status')) return Promise.resolve({ ok: true, status: 200, json: async () => statusSnapshot });
    if (url.includes('/clips/storage')) return Promise.resolve({ ok: true, status: 200, json: async () => clipStorage });
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
  act(() => root.render(<SettingsPage />));
  await flush();
  return { host, root };
}

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('SettingsPage', () => {
  it('renders the page title and every settings card once all resources load', async () => {
    installFetchMock();
    const { host } = await renderPage();

    expect(host.querySelector('h1')?.textContent).toBe('설정');
    expect(host.textContent).toContain('카메라');
    expect(host.textContent).toContain('101호');
    expect(host.textContent).toContain('탐지 설정');
    expect(host.textContent).toContain('서버 연결');
    expect(host.textContent).toContain('처리 상태');
    expect(host.textContent).toContain('클립 저장 공간');
  });

  it('surfaces data from the system and clip-storage endpoints in their respective cards', async () => {
    installFetchMock();
    const { host } = await renderPage();

    expect(host.textContent).toContain('v2026.07.28');
    expect(host.textContent).toContain('2.0 / 10.0 GB');
  });
});
